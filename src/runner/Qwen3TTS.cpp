#include "Qwen3TTS.hpp"

#include "Qwen3TTSCodePredictor.hpp"
#include "Qwen3TTSCommon.hpp"
#include "Qwen3TTSSpeechTokenizer.hpp"
#include "Qwen3TTSTalker.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "BaseTokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "sample_log.h"

using namespace qwen3_tts;

namespace {

static int language_id(const std::string &language)
{
    std::string key = language;
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (key.empty() || key == "auto") return -1;

    static const std::unordered_map<std::string, int> ids = {
        {"english", 2050},
        {"german", 2053},
        {"spanish", 2054},
        {"chinese", 2055},
        {"japanese", 2058},
        {"french", 2061},
        {"korean", 2064},
        {"russian", 2069},
        {"italian", 2070},
        {"portuguese", 2071},
    };
    auto it = ids.find(key);
    if (it == ids.end()) return -2;
    return it->second;
}

static std::string build_assistant_text(const std::string &text)
{
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

static std::string build_ref_text(const std::string &text)
{
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n";
}

static void apply_talker_processors(std::vector<float> &logits,
                                    const std::vector<int> &history,
                                    int generated_frames,
                                    float repetition_penalty)
{
    for (int i = kTalkerVocab - 1024; i < kTalkerVocab; ++i)
    {
        if (i != kCodecEosId && i >= 0 && i < (int)logits.size())
            logits[(size_t)i] = -std::numeric_limits<float>::infinity();
    }
    if (generated_frames < 2 && kCodecEosId >= 0 && kCodecEosId < (int)logits.size())
        logits[(size_t)kCodecEosId] = -std::numeric_limits<float>::infinity();

    if (repetition_penalty > 1.0f)
    {
        for (int id : history)
        {
            if (id < 0 || id >= (int)logits.size()) continue;
            float &v = logits[(size_t)id];
            v = v < 0.0f ? v * repetition_penalty : v / repetition_penalty;
        }
    }
}

static uint16_t read_u16_le(const unsigned char *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32_le(const unsigned char *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static bool read_wav_mono_f32(const std::string &path,
                              std::vector<float> &waveform,
                              int &sample_rate,
                              std::string &err)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
    {
        err = "failed to open wav: " + path;
        return false;
    }

    unsigned char riff_header[12];
    if (!ifs.read(reinterpret_cast<char *>(riff_header), sizeof(riff_header)) ||
        std::memcmp(riff_header, "RIFF", 4) != 0 ||
        std::memcmp(riff_header + 8, "WAVE", 4) != 0)
    {
        err = "unsupported wav container: " + path;
        return false;
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sr = 0;
    uint16_t bits = 0;
    std::vector<unsigned char> data;

    while (ifs)
    {
        unsigned char chdr[8];
        if (!ifs.read(reinterpret_cast<char *>(chdr), sizeof(chdr))) break;
        const uint32_t sz = read_u32_le(chdr + 4);
        const std::string id(reinterpret_cast<const char *>(chdr), 4);
        if (id == "fmt ")
        {
            std::vector<unsigned char> fmt(sz);
            if (!ifs.read(reinterpret_cast<char *>(fmt.data()), sz) || sz < 16)
            {
                err = "bad wav fmt chunk: " + path;
                return false;
            }
            audio_format = read_u16_le(fmt.data());
            channels = read_u16_le(fmt.data() + 2);
            sr = read_u32_le(fmt.data() + 4);
            bits = read_u16_le(fmt.data() + 14);
        }
        else if (id == "data")
        {
            data.resize(sz);
            if (!ifs.read(reinterpret_cast<char *>(data.data()), sz))
            {
                err = "bad wav data chunk: " + path;
                return false;
            }
        }
        else
        {
            ifs.seekg((std::streamoff)sz, std::ios::cur);
        }
        if ((sz & 1u) != 0) ifs.seekg(1, std::ios::cur);
    }

    if (channels == 0 || sr == 0 || bits == 0 || data.empty())
    {
        err = "wav fmt/data missing: " + path;
        return false;
    }
    if (audio_format != 1 && audio_format != 3)
    {
        err = "unsupported wav format: " + std::to_string(audio_format);
        return false;
    }

    const int bytes_per_sample = std::max(1, (int)bits / 8);
    const int frame_bytes = bytes_per_sample * (int)channels;
    const size_t frames = data.size() / (size_t)frame_bytes;
    waveform.assign(frames, 0.0f);

    auto sample = [&](const unsigned char *p) -> float {
        if (audio_format == 3 && bits == 32)
        {
            float v = 0.0f;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        if (bits == 16)
        {
            int16_t v = (int16_t)read_u16_le(p);
            return (float)v / 32768.0f;
        }
        if (bits == 24)
        {
            int32_t v = (int32_t)p[0] | ((int32_t)p[1] << 8) | ((int32_t)p[2] << 16);
            if (v & 0x00800000) v |= ~0x00FFFFFF;
            return (float)v / 8388608.0f;
        }
        if (bits == 32)
        {
            int32_t v = (int32_t)read_u32_le(p);
            return (float)v / 2147483648.0f;
        }
        if (bits == 8) return ((float)p[0] - 128.0f) / 128.0f;
        return 0.0f;
    };

    for (size_t f = 0; f < frames; ++f)
    {
        const unsigned char *base = data.data() + f * (size_t)frame_bytes;
        double acc = 0.0;
        for (uint16_t c = 0; c < channels; ++c) acc += sample(base + (size_t)c * (size_t)bytes_per_sample);
        waveform[f] = (float)std::clamp(acc / (double)channels, -1.0, 1.0);
    }

    sample_rate = (int)sr;
    return true;
}

static std::vector<float> resample_linear(const std::vector<float> &waveform, int src_rate, int dst_rate)
{
    if (waveform.empty() || src_rate <= 0 || dst_rate <= 0 || src_rate == dst_rate) return waveform;
    const size_t dst_len = std::max<size_t>(1, (size_t)std::llround((double)waveform.size() * (double)dst_rate / (double)src_rate));
    std::vector<float> out(dst_len);
    for (size_t i = 0; i < dst_len; ++i)
    {
        const double pos = (double)i * (double)src_rate / (double)dst_rate;
        const size_t i0 = std::min<size_t>((size_t)pos, waveform.size() - 1);
        const size_t i1 = std::min<size_t>(i0 + 1, waveform.size() - 1);
        const float a = (float)(pos - (double)i0);
        out[i] = waveform[i0] * (1.0f - a) + waveform[i1] * a;
    }
    return out;
}

class TextProjection
{
public:
    bool init(const std::string &talker_dir)
    {
        return proj3_.init(join_path(talker_dir, "talker_resize_mlp_3.axmodel")) &&
               proj28_.init(join_path(talker_dir, "talker_resize_mlp_28.axmodel"));
    }

    void deinit()
    {
        proj3_.deinit();
        proj28_.deinit();
    }

    std::vector<unsigned short> run(const std::vector<unsigned short> &text_bf16, int seq)
    {
        if (seq <= 0) return {};
        if (text_bf16.size() != (size_t)seq * (size_t)kTextHidden)
            throw std::runtime_error("text projection input size mismatch");

        std::vector<unsigned short> out((size_t)seq * (size_t)kTalkerHidden);
        int offset = 0;
        while (offset < seq)
        {
            const int remaining = seq - offset;
            const int chunk = remaining <= 3 ? 3 : 28;
            const int take = std::min(remaining, chunk);
            Runner &model = chunk == 3 ? proj3_ : proj28_;

            std::vector<float> feed((size_t)chunk * (size_t)kTextHidden, 0.0f);
            for (int r = 0; r < take; ++r)
            {
                for (int c = 0; c < kTextHidden; ++c)
                {
                    feed[(size_t)r * kTextHidden + c] =
                        bfloat16(text_bf16[(size_t)(offset + r) * kTextHidden + c]).fp32();
                }
            }

            auto &runner = model.get();
            const auto &in = runner.get_input(0);
            write_tensor(runner, in, feed.data(), feed.size() * sizeof(float));
            if (runner.inference() != 0) throw std::runtime_error("text projection inference failed");
            const auto &ot = runner.get_output(0);
            auto projected = tensor_to_float(runner, ot, (size_t)chunk * kTalkerHidden);
            for (int r = 0; r < take; ++r)
            {
                for (int c = 0; c < kTalkerHidden; ++c)
                {
                    out[(size_t)(offset + r) * kTalkerHidden + c] =
                        fp32_to_bfloat16_rne(projected[(size_t)r * kTalkerHidden + c]);
                }
            }
            offset += take;
        }
        return out;
    }

private:
    Runner proj3_;
    Runner proj28_;
};

class SpeakerEncoder
{
public:
    bool init(const std::string &model_dir)
    {
        return model_.init(join_path(model_dir, "extract_speaker_embedding_3s.axmodel"));
    }

    void deinit() { model_.deinit(); }

    std::vector<unsigned short> run(std::vector<float> wav24)
    {
        fit_audio(wav24, kSpeakerSamples);
        auto &runner = model_.get();
        write_tensor(runner, runner.get_input("audio"), wav24.data(), wav24.size() * sizeof(float));
        if (runner.inference() != 0) throw std::runtime_error("speaker encoder inference failed");
        auto f = tensor_to_float(runner, runner.get_output("speaker_embedding"), kTalkerHidden);
        return fp32_to_bf16_vec(f);
    }

private:
    Runner model_;
};

} // namespace

struct Qwen3TTS::Impl
{
    std::string model_dir;
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector text_embed;
    LLaMaEmbedSelector codec_embed;
    TextProjection text_projection;
    TalkerRunner talker;
    CodePredictorRunner code_predictor;
    SpeechTokenizerRunner speech_tokenizer;
    SpeakerEncoder speaker_encoder;
    bool deinitialized = false;

    std::vector<unsigned short> text_embeddings(const std::vector<int> &ids)
    {
        std::vector<unsigned short> out((size_t)ids.size() * kTextHidden);
        for (size_t i = 0; i < ids.size(); ++i)
            text_embed.getByIndex((unsigned int)ids[i], out.data() + i * kTextHidden);
        return out;
    }

    std::vector<unsigned short> codec_embedding(int id)
    {
        std::vector<unsigned short> out(kTalkerHidden);
        codec_embed.getByIndex((unsigned int)id, out.data());
        return out;
    }

    std::vector<unsigned short> project_ids(const std::vector<int> &ids)
    {
        auto emb = text_embeddings(ids);
        return text_projection.run(emb, (int)ids.size());
    }

};

Qwen3TTS::Qwen3TTS()
{
    impl_ = new Impl();
}

Qwen3TTS::~Qwen3TTS()
{
    Deinit();
    delete impl_;
    impl_ = nullptr;
}

bool Qwen3TTS::Init(const std::string &model_dir)
{
    impl_->deinitialized = false;
    impl_->model_dir = model_dir;
    const std::string talker_dir = join_path(model_dir, "talker");
    const std::string code_dir = join_path(model_dir, "code-predictor");
    const std::string speech_dir = join_path(model_dir, "speech_tokenizer");

    impl_->tokenizer = create_tokenizer("Qwen3");
    if (!impl_->tokenizer || !impl_->tokenizer->load(join_path(talker_dir, "qwen3_tokenizer.txt")))
    {
        ALOGE("failed to load qwen3 tokenizer");
        return false;
    }

    if (!impl_->text_embed.Init(join_path(talker_dir, "talker.model.text_embedding.weight.bfloat16.bin"), 151936, kTextHidden, true))
        return false;
    if (!impl_->codec_embed.Init(join_path(talker_dir, "talker.model.codec_embedding.weight.bfloat16.bin"), kTalkerVocab, kTalkerHidden, true))
        return false;
    if (!impl_->text_projection.init(talker_dir)) return false;
    if (!impl_->talker.init(talker_dir)) return false;
    if (!impl_->code_predictor.init(code_dir)) return false;
    if (!impl_->speech_tokenizer.init(speech_dir)) return false;
    if (!impl_->speaker_encoder.init(model_dir)) return false;
    return true;
}

void Qwen3TTS::Deinit()
{
    if (!impl_) return;
    if (impl_->deinitialized) return;
    impl_->text_embed.Deinit();
    impl_->codec_embed.Deinit();
    impl_->text_projection.deinit();
    impl_->talker.deinit();
    impl_->code_predictor.deinit();
    impl_->speech_tokenizer.deinit();
    impl_->speaker_encoder.deinit();
    impl_->deinitialized = true;
}

bool Qwen3TTS::GenerateVoiceClone(const Qwen3TTSGenerateOptions &options,
                                  std::vector<float> &wav,
                                  int &sample_rate)
{
    try
    {
        ALOGI("Qwen3-TTS GenerateVoiceClone start: text_len=%zu ref_text_len=%zu ref_audio=%s language=%s x_vector_only=%d non_streaming=%d max_new_tokens=%d seed=%u",
              options.text.size(), options.ref_text.size(), options.ref_audio.c_str(), options.language.c_str(),
              options.x_vector_only_mode ? 1 : 0, options.non_streaming_mode ? 1 : 0,
              options.max_new_tokens, options.seed);
        if (options.text.empty()) throw std::runtime_error("text is empty");
        if (options.ref_audio.empty()) throw std::runtime_error("ref_audio is empty");
        if (!options.x_vector_only_mode && options.ref_text.empty()) throw std::runtime_error("ref_text is required for ICL mode");

        int lang_id = language_id(options.language);
        if (lang_id == -2) throw std::runtime_error("unsupported language: " + options.language);
        ALOGI("Qwen3-TTS language resolved: language=%s lang_id=%d", options.language.c_str(), lang_id);

        std::vector<float> ref_wav;
        int ref_sr = 0;
        std::string err;
        if (!read_wav_mono_f32(options.ref_audio, ref_wav, ref_sr, err)) throw std::runtime_error(err);
        std::vector<float> ref_wav24 = resample_linear(ref_wav, ref_sr, kSampleRate);
        ALOGI("Qwen3-TTS ref audio loaded: sr=%d samples=%zu resampled_sr=%d resampled_samples=%zu",
              ref_sr, ref_wav.size(), kSampleRate, ref_wav24.size());

        std::vector<std::array<int, kCodeGroups>> ref_codes;
        if (!options.x_vector_only_mode)
        {
            ALOGI("Qwen3-TTS speech tokenizer encode begin");
            ref_codes = impl_->speech_tokenizer.encode(ref_wav24);
            ALOGI("Qwen3-TTS speech tokenizer encode end: ref_code_len=%zu", ref_codes.size());
        }
        ALOGI("Qwen3-TTS speaker encoder begin");
        auto spk_embed = impl_->speaker_encoder.run(ref_wav24);
        ALOGI("Qwen3-TTS speaker encoder end: embedding_elems=%zu", spk_embed.size());

        const auto input_ids = impl_->tokenizer->encode(build_assistant_text(options.text));
        std::vector<int> ref_ids;
        if (!options.x_vector_only_mode)
            ref_ids = impl_->tokenizer->encode(build_ref_text(options.ref_text));
        ALOGI("Qwen3-TTS tokenizer done: input_ids=%zu ref_ids=%zu", input_ids.size(), ref_ids.size());

        auto tts_special = impl_->project_ids({kTtsBosTokenId, kTtsEosTokenId, kTtsPadTokenId});
        auto tts_bos = slice_bf16_rows(tts_special, 0, 1, kTalkerHidden);
        auto tts_eos = slice_bf16_rows(tts_special, 1, 1, kTalkerHidden);
        auto tts_pad = slice_bf16_rows(tts_special, 2, 1, kTalkerHidden);
        ALOGI("Qwen3-TTS special embeddings projected: elems=%zu", tts_special.size());

        std::vector<int> codec_prefill;
        if (lang_id < 0)
            codec_prefill = {kCodecNoThinkId, kCodecThinkBosId, kCodecThinkEosId};
        else
            codec_prefill = {kCodecThinkId, kCodecThinkBosId, lang_id, kCodecThinkEosId};
        ALOGI("Qwen3-TTS codec prefill ids=%zu", codec_prefill.size());

        std::vector<std::vector<unsigned short>> codec_items;
        for (int id : codec_prefill) codec_items.push_back(impl_->codec_embedding(id));
        codec_items.push_back(spk_embed);
        codec_items.push_back(impl_->codec_embedding(kCodecPadId));
        codec_items.push_back(impl_->codec_embedding(kCodecBosId));

        std::vector<unsigned short> prompt;
        std::vector<unsigned short> trailing;

        auto role = impl_->project_ids(std::vector<int>(input_ids.begin(), input_ids.begin() + std::min<size_t>(3, input_ids.size())));
        append_bf16(prompt, role);

        for (size_t i = 0; i + 1 < codec_items.size(); ++i)
        {
            std::vector<unsigned short> add = (i + 2 == codec_items.size()) ? tts_bos : tts_pad;
            std::vector<float> sum = bf16_to_fp32_vec(add.data(), add.size());
            add_bf16_to_float(codec_items[i].data(), sum.data(), sum.size());
            append_bf16(prompt, fp32_to_bf16_vec(sum));
        }

        if (!options.x_vector_only_mode)
        {
            if (input_ids.size() < 8 || ref_ids.size() < 5) throw std::runtime_error("tokenized text is unexpectedly short");
            std::vector<int> icl_text_ids;
            icl_text_ids.insert(icl_text_ids.end(), ref_ids.begin() + 3, ref_ids.end() - 2);
            icl_text_ids.insert(icl_text_ids.end(), input_ids.begin() + 3, input_ids.end() - 5);
            auto text_embed = impl_->project_ids(icl_text_ids);
            append_bf16(text_embed, tts_eos);
            const int text_lens = (int)text_embed.size() / kTalkerHidden;
            ALOGI("Qwen3-TTS ICL text prompt: icl_text_ids=%zu text_lens=%d", icl_text_ids.size(), text_lens);

            std::vector<unsigned short> codec_embed_seq;
            append_bf16(codec_embed_seq, impl_->codec_embedding(kCodecBosId));
            for (const auto &frame : ref_codes)
            {
                std::vector<std::vector<unsigned short>> parts;
                parts.push_back(impl_->codec_embedding(frame[0]));
                for (int i = 1; i < kCodeGroups; ++i)
                    parts.push_back(impl_->code_predictor.embed(i - 1, frame[(size_t)i]));
                append_bf16(codec_embed_seq, sum_bf16_embeddings(parts));
            }
            const int codec_lens = (int)codec_embed_seq.size() / kTalkerHidden;
            ALOGI("Qwen3-TTS ICL codec prompt: ref_codes=%zu codec_lens=%d", ref_codes.size(), codec_lens);

            if (options.non_streaming_mode)
            {
                ALOGI("Qwen3-TTS prompt pack mode: non_streaming ICL");
                for (int i = 0; i < text_lens; ++i)
                {
                    std::vector<float> s = bf16_to_fp32_vec(text_embed.data() + (size_t)i * kTalkerHidden, kTalkerHidden);
                    add_bf16_to_float(impl_->codec_embedding(kCodecPadId).data(), s.data(), s.size());
                    append_bf16(prompt, fp32_to_bf16_vec(s));
                }
                for (int i = 0; i < codec_lens; ++i)
                {
                    std::vector<float> s = bf16_to_fp32_vec(codec_embed_seq.data() + (size_t)i * kTalkerHidden, kTalkerHidden);
                    add_bf16_to_float(tts_pad.data(), s.data(), s.size());
                    append_bf16(prompt, fp32_to_bf16_vec(s));
                }
                trailing = tts_pad;
            }
            else
            {
                ALOGI("Qwen3-TTS prompt pack mode: streaming ICL");
                const int common = std::min(text_lens, codec_lens);
                for (int i = 0; i < common; ++i)
                {
                    std::vector<float> s = bf16_to_fp32_vec(text_embed.data() + (size_t)i * kTalkerHidden, kTalkerHidden);
                    add_bf16_to_float(codec_embed_seq.data() + (size_t)i * kTalkerHidden, s.data(), s.size());
                    append_bf16(prompt, fp32_to_bf16_vec(s));
                }
                if (text_lens > codec_lens)
                {
                    trailing = slice_bf16_rows(text_embed, codec_lens, text_lens - codec_lens, kTalkerHidden);
                }
                else
                {
                    for (int i = text_lens; i < codec_lens; ++i)
                    {
                        std::vector<float> s = bf16_to_fp32_vec(tts_pad.data(), kTalkerHidden);
                        add_bf16_to_float(codec_embed_seq.data() + (size_t)i * kTalkerHidden, s.data(), s.size());
                        append_bf16(prompt, fp32_to_bf16_vec(s));
                    }
                    trailing = tts_pad;
                }
            }
        }
        else
        {
            if (input_ids.size() < 8) throw std::runtime_error("tokenized text is unexpectedly short");
            ALOGI("Qwen3-TTS prompt pack mode: x_vector_only");
            auto first_text = impl_->project_ids({input_ids[3]});
            std::vector<float> s = bf16_to_fp32_vec(first_text.data(), kTalkerHidden);
            add_bf16_to_float(codec_items.back().data(), s.data(), s.size());
            append_bf16(prompt, fp32_to_bf16_vec(s));

            if (options.non_streaming_mode)
            {
                prompt.resize(prompt.size() - kTalkerHidden);
                std::vector<int> body(input_ids.begin() + 3, input_ids.end() - 5);
                auto body_proj = impl_->project_ids(body);
                append_bf16(body_proj, tts_eos);
                for (int i = 0; i < (int)body_proj.size() / kTalkerHidden; ++i)
                {
                    std::vector<float> ss = bf16_to_fp32_vec(body_proj.data() + (size_t)i * kTalkerHidden, kTalkerHidden);
                    add_bf16_to_float(impl_->codec_embedding(kCodecPadId).data(), ss.data(), ss.size());
                    append_bf16(prompt, fp32_to_bf16_vec(ss));
                }
                std::vector<float> bos_sum = bf16_to_fp32_vec(tts_pad.data(), kTalkerHidden);
                auto bos = impl_->codec_embedding(kCodecBosId);
                add_bf16_to_float(bos.data(), bos_sum.data(), bos_sum.size());
                append_bf16(prompt, fp32_to_bf16_vec(bos_sum));
                trailing = tts_pad;
            }
            else
            {
                std::vector<int> tail(input_ids.begin() + 4, input_ids.end() - 5);
                auto tail_proj = impl_->project_ids(tail);
                append_bf16(tail_proj, tts_eos);
                trailing = tail_proj;
            }
        }

        const int prompt_len = (int)prompt.size() / kTalkerHidden;
        ALOGI("Qwen3-TTS prompt_len=%d trailing_len=%zu ref_code_len=%zu", prompt_len,
              trailing.size() / kTalkerHidden, ref_codes.size());

        std::mt19937 rng(options.seed);
        std::vector<int> first_code_history;
        std::vector<std::array<int, kCodeGroups>> generated_codes;

        ALOGI("Qwen3-TTS talker prefill call begin");
        auto raw_hidden = impl_->talker.prefill(prompt, prompt_len);
        ALOGI("Qwen3-TTS talker prefill call end: raw_hidden_elems=%zu", raw_hidden.size());
        ALOGI("Qwen3-TTS talker post after prefill begin");
        auto talker_post = impl_->talker.post(raw_hidden);
        ALOGI("Qwen3-TTS talker post after prefill end: logits=%zu hidden_norm=%zu",
              talker_post.logits.size(), talker_post.hidden_norm.size());
        auto norm_hidden = talker_post.hidden_norm;
        auto logits = talker_post.logits;
        apply_talker_processors(logits, first_code_history, 0, options.repetition_penalty);
        int next_first = select_from_logits(logits, options.do_sample, options.top_k, options.top_p, options.temperature, rng);
        ALOGI("Qwen3-TTS first code sampled: next_first=%d", next_first);

        for (int step = 0; step < options.max_new_tokens; ++step)
        {
            if (next_first == kCodecEosId && step >= 2) break;
            first_code_history.push_back(next_first);
            ALOGI("Qwen3-TTS generation step=%d begin: first_code=%d history=%zu",
                  step, next_first, first_code_history.size());

            auto first_embed = impl_->codec_embedding(next_first);
            ALOGI("Qwen3-TTS code predictor call begin: step=%d first_embed_elems=%zu", step, first_embed.size());
            auto sub_codes = impl_->code_predictor.generate(norm_hidden, first_embed, options, rng);
            std::string sub_code_text;
            for (size_t i = 0; i < sub_codes.size(); ++i)
            {
                if (i > 0) sub_code_text += ",";
                sub_code_text += std::to_string(sub_codes[i]);
            }
            ALOGI("Qwen3-TTS code predictor call end: step=%d sub_codes=[%s]",
                  step, sub_code_text.c_str());
            std::array<int, kCodeGroups> frame{};
            frame[0] = next_first;
            for (int i = 0; i < kNumSubCodes; ++i) frame[(size_t)i + 1] = sub_codes[(size_t)i];
            generated_codes.push_back(frame);

            std::vector<std::vector<unsigned short>> parts;
            parts.push_back(first_embed);
            for (int i = 0; i < kNumSubCodes; ++i)
                parts.push_back(impl_->code_predictor.embed(i, sub_codes[(size_t)i]));
            auto codec_sum = sum_bf16_embeddings(parts);

            std::vector<unsigned short> text_add;
            const int trailing_len = (int)trailing.size() / kTalkerHidden;
            if (step < trailing_len)
                text_add = slice_bf16_rows(trailing, step, 1, kTalkerHidden);
            else
                text_add = tts_pad;

            std::vector<float> decode_embed = bf16_to_fp32_vec(codec_sum.data(), codec_sum.size());
            add_bf16_to_float(text_add.data(), decode_embed.data(), decode_embed.size());
            ALOGI("Qwen3-TTS talker decode call begin: step=%d trailing_len=%d text_add_elems=%zu",
                  step, trailing_len, text_add.size());
            raw_hidden = impl_->talker.decode_one(fp32_to_bf16_vec(decode_embed));
            ALOGI("Qwen3-TTS talker decode call end: step=%d raw_hidden_elems=%zu", step, raw_hidden.size());
            ALOGI("Qwen3-TTS talker post after decode begin: step=%d", step);
            talker_post = impl_->talker.post(raw_hidden);
            ALOGI("Qwen3-TTS talker post after decode end: step=%d logits=%zu hidden_norm=%zu",
                  step, talker_post.logits.size(), talker_post.hidden_norm.size());
            norm_hidden = talker_post.hidden_norm;
            logits = talker_post.logits;
            apply_talker_processors(logits, first_code_history, (int)generated_codes.size(), options.repetition_penalty);
            next_first = select_from_logits(logits, options.do_sample, options.top_k, options.top_p, options.temperature, rng);
            ALOGI("Qwen3-TTS generation step=%d end: generated=%zu next_first=%d",
                  step, generated_codes.size(), next_first);
        }

        std::vector<std::array<int, kCodeGroups>> decode_codes;
        decode_codes.reserve(ref_codes.size() + generated_codes.size());
        decode_codes.insert(decode_codes.end(), ref_codes.begin(), ref_codes.end());
        decode_codes.insert(decode_codes.end(), generated_codes.begin(), generated_codes.end());
        ALOGI("Qwen3-TTS speech tokenizer decode begin: ref_codes=%zu generated_codes=%zu total_codes=%zu",
              ref_codes.size(), generated_codes.size(), decode_codes.size());
        auto wav_all = impl_->speech_tokenizer.decode(decode_codes);
        ALOGI("Qwen3-TTS speech tokenizer decode end: wav_all_samples=%zu", wav_all.size());

        if (!ref_codes.empty())
        {
            const int ref_len = (int)ref_codes.size();
            const int total_len = (int)decode_codes.size();
            const size_t cut = (size_t)((double)ref_len / std::max(1, total_len) * (double)wav_all.size());
            wav.assign(wav_all.begin() + (ptrdiff_t)std::min(cut, wav_all.size()), wav_all.end());
            ALOGI("Qwen3-TTS ref audio cut: ref_len=%d total_len=%d cut_samples=%zu output_samples=%zu",
                  ref_len, total_len, cut, wav.size());
        }
        else
        {
            wav = std::move(wav_all);
            ALOGI("Qwen3-TTS output without ref cut: output_samples=%zu", wav.size());
        }
        sample_rate = kSampleRate;
        ALOGI("Qwen3-TTS generated_frames=%zu output_samples=%zu", generated_codes.size(), wav.size());
        return true;
    }
    catch (const std::exception &e)
    {
        ALOGE("Qwen3TTS GenerateVoiceClone failed: %s", e.what());
        return false;
    }
}

bool Qwen3TTS::SaveWavFloat32(const std::string &path,
                              const std::vector<float> &wav,
                              int sample_rate)
{
    if (wav.empty() || sample_rate <= 0) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    auto w16 = [&](uint16_t v) {
        f.put((char)(v & 0xff));
        f.put((char)((v >> 8) & 0xff));
    };
    auto w32 = [&](uint32_t v) {
        f.put((char)(v & 0xff));
        f.put((char)((v >> 8) & 0xff));
        f.put((char)((v >> 16) & 0xff));
        f.put((char)((v >> 24) & 0xff));
    };

    const uint16_t channels = 1;
    const uint16_t bits = 32;
    const uint32_t data_bytes = (uint32_t)(wav.size() * sizeof(float));
    const uint32_t byte_rate = (uint32_t)sample_rate * channels * (bits / 8);
    const uint16_t block_align = channels * (bits / 8);
    const uint32_t riff_size = 4 + (8 + 16) + (8 + data_bytes);

    f.write("RIFF", 4);
    w32(riff_size);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    w32(16);
    w16(3);
    w16(channels);
    w32((uint32_t)sample_rate);
    w32(byte_rate);
    w16(block_align);
    w16(bits);
    f.write("data", 4);
    w32(data_bytes);
    f.write(reinterpret_cast<const char *>(wav.data()), (std::streamsize)data_bytes);
    return (bool)f;
}
