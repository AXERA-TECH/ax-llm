#include "Qwen3TTSSpeechTokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace qwen3_tts {

bool SpeechTokenizerRunner::init(const std::string &dir)
{
    model_dir_ = dir;
    if (!encoder_.init(join_path(dir, "speech_tokenizer_encoder.axmodel"))) return false;
    if (!decode_pre_.init(join_path(dir, "speech_tokenizer_decode_pre_inproj.axmodel"))) return false;
    for (int i = 0; i < kSpeechDecoderLayers; ++i)
    {
        layers_.emplace_back(std::make_unique<Runner>());
        if (!layers_.back()->init(join_path(dir, "decoder/qwen3_tts_tokenizer_12hz_p64_l" + std::to_string(i) + "_together.axmodel"))) return false;
    }
    if (!post_.init(join_path(dir, "decoder/qwen3_tts_tokenizer_12hz_post.axmodel"))) return false;
    if (!decode_post_.init(join_path(dir, "speech_tokenizer_decode_post_t64.axmodel"))) return false;
    return true;
}

void SpeechTokenizerRunner::deinit()
{
    encoder_.deinit();
    decode_pre_.deinit();
    for (auto &l : layers_) l->deinit();
    layers_.clear();
    post_.deinit();
    decode_post_.deinit();
}

std::vector<std::array<int, kCodeGroups>> SpeechTokenizerRunner::encode(const std::vector<float> &wav24)
{
    std::vector<float> fixed = wav24;
    fit_audio(fixed, kSpeechEncoderSamples);
    auto &runner = encoder_.get();
    write_tensor(runner, runner.get_input("input_values"), fixed.data(), fixed.size() * sizeof(float));
    if (runner.inference() != 0) throw std::runtime_error("speech tokenizer encoder inference failed");
    const auto &out = runner.get_output("audio_codes");
    const size_t elems = shape_elems(out.vShape);
    std::vector<int32_t> raw(elems);
    read_tensor(runner, out, raw.data(), elems * sizeof(int32_t));
    const int valid_samples = std::min<int>((int)wav24.size(), kSpeechEncoderSamples);
    const int code_len = std::min<int>(38, (valid_samples + kSpeechEncodeDownsample - 1) / kSpeechEncodeDownsample);
    std::vector<std::array<int, kCodeGroups>> codes((size_t)code_len);
    for (int t = 0; t < code_len; ++t)
        for (int q = 0; q < kCodeGroups; ++q)
            codes[(size_t)t][(size_t)q] = (int)raw[(size_t)t * kCodeGroups + q];
    return codes;
}

std::vector<float> SpeechTokenizerRunner::decode(const std::vector<std::array<int, kCodeGroups>> &codes)
{
    int code_len = 0;
    for (const auto &c : codes)
    {
        if (c[0] > 0) ++code_len;
        else break;
    }
    if (code_len <= 0) return {};
    if (code_len > 127) throw std::runtime_error("speech tokenizer decoder code_len exceeds compiled kv cache");

    std::vector<float> hidden512 = decode_pre(codes, code_len);
    auto transformed = run_transformer(hidden512, code_len);
    auto hidden1024 = run_post(transformed, code_len);
    return run_decode_post(hidden1024, code_len);
}

std::vector<float> SpeechTokenizerRunner::decode_pre(const std::vector<std::array<int, kCodeGroups>> &codes, int code_len)
{
    std::vector<int32_t> feed((size_t)kCodeGroups * 325, 0);
    for (int t = 0; t < code_len; ++t)
        for (int q = 0; q < kCodeGroups; ++q)
            feed[(size_t)q * 325 + t] = (int32_t)codes[(size_t)t][(size_t)q];
    auto &runner = decode_pre_.get();
    write_tensor(runner, runner.get_input("codes"), feed.data(), feed.size() * sizeof(int32_t));
    if (runner.inference() != 0) throw std::runtime_error("speech tokenizer decode_pre inference failed");
    auto out = tensor_to_float(runner, runner.get_output("hidden_512"), (size_t)325 * 512);
    out.resize((size_t)code_len * 512);
    return out;
}

std::vector<unsigned short> SpeechTokenizerRunner::run_transformer(const std::vector<float> &hidden512, int code_len)
{
    const int prefill_len = 64;
    const int valid_prefill = std::min(code_len, prefill_len);
    std::vector<unsigned short> data((size_t)prefill_len * 512, 0);
    auto hbf = fp32_to_bf16_vec(hidden512);
    std::memcpy(data.data(), hbf.data(), (size_t)valid_prefill * 512 * sizeof(unsigned short));

    std::vector<unsigned int> indices(64);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<unsigned short> mask((size_t)64 * 64, bfloat16(-65536.0f).data);
    for (int r = 0; r < 64; ++r)
    {
        const int left = std::max(0, r - 72 + 1);
        for (int c = left; c <= r; ++c) mask[(size_t)r * 64 + c] = 0;
    }

    std::vector<std::vector<unsigned short>> kc(kSpeechDecoderLayers, std::vector<unsigned short>((size_t)127 * 1024, 0));
    std::vector<std::vector<unsigned short>> vc(kSpeechDecoderLayers, std::vector<unsigned short>((size_t)127 * 1024, 0));
    for (int layer = 0; layer < kSpeechDecoderLayers; ++layer)
    {
        auto &runner = layers_[(size_t)layer]->get();
        write_tensor(runner, input_tensor(runner, 1, "indices"), indices.data(), indices.size() * sizeof(unsigned int));
        write_tensor(runner, input_tensor(runner, 1, "input"), data.data(), data.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 1, "mask"), mask.data(), mask.size() * sizeof(unsigned short));
        std::vector<unsigned short> zeros_k((size_t)input_tensor(runner, 1, "K_cache").nSize / sizeof(unsigned short), 0);
        std::vector<unsigned short> zeros_v((size_t)input_tensor(runner, 1, "V_cache").nSize / sizeof(unsigned short), 0);
        write_tensor(runner, input_tensor(runner, 1, "K_cache"), zeros_k.data(), zeros_k.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 1, "V_cache"), zeros_v.data(), zeros_v.size() * sizeof(unsigned short));
        if (runner.inference(1) != 0) throw std::runtime_error("speech tokenizer transformer prefill failed");
        read_tensor(runner, output_tensor(runner, 1, "K_cache_out"), kc[(size_t)layer].data(), (size_t)64 * 1024 * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 1, "V_cache_out"), vc[(size_t)layer].data(), (size_t)64 * 1024 * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 1, "output"), data.data(), data.size() * sizeof(unsigned short));
    }

    std::vector<unsigned short> all;
    all.insert(all.end(), data.begin(), data.begin() + (ptrdiff_t)((size_t)valid_prefill * 512));
    for (int pos = 64; pos < code_len; ++pos)
    {
        std::vector<unsigned short> tok(512);
        std::memcpy(tok.data(), hbf.data() + (size_t)pos * 512, (size_t)512 * sizeof(unsigned short));
        std::vector<unsigned short> d = tok;
        std::vector<unsigned short> dmask(128, bfloat16(-65536.0f).data);
        const int left = std::max(0, pos - 72 + 1);
        for (int i = left; i < pos; ++i) dmask[(size_t)i] = 0;
        dmask.back() = 0;
        const unsigned int idx = (unsigned int)pos;
        for (int layer = 0; layer < kSpeechDecoderLayers; ++layer)
        {
            auto &runner = layers_[(size_t)layer]->get();
            write_tensor(runner, input_tensor(runner, 0, "K_cache"), kc[(size_t)layer].data(), kc[(size_t)layer].size() * sizeof(unsigned short));
            write_tensor(runner, input_tensor(runner, 0, "V_cache"), vc[(size_t)layer].data(), vc[(size_t)layer].size() * sizeof(unsigned short));
            write_tensor(runner, input_tensor(runner, 0, "indices"), &idx, sizeof(idx));
            write_tensor(runner, input_tensor(runner, 0, "input"), d.data(), d.size() * sizeof(unsigned short));
            write_tensor(runner, input_tensor(runner, 0, "mask"), dmask.data(), dmask.size() * sizeof(unsigned short));
            if (runner.inference(0) != 0) throw std::runtime_error("speech tokenizer transformer decode failed");
            read_tensor(runner, output_tensor(runner, 0, "K_cache_out"), kc[(size_t)layer].data() + (size_t)pos * 1024, (size_t)1024 * sizeof(unsigned short));
            read_tensor(runner, output_tensor(runner, 0, "V_cache_out"), vc[(size_t)layer].data() + (size_t)pos * 1024, (size_t)1024 * sizeof(unsigned short));
            read_tensor(runner, output_tensor(runner, 0, "output"), d.data(), d.size() * sizeof(unsigned short));
        }
        all.insert(all.end(), d.begin(), d.end());
    }
    return all;
}

std::vector<float> SpeechTokenizerRunner::run_post(const std::vector<unsigned short> &hidden512, int code_len)
{
    std::vector<float> hidden1024((size_t)code_len * 1024, 0.0f);
    auto &runner = post_.get();
    for (int i = 0; i < code_len; ++i)
    {
        write_tensor(runner, runner.get_input("input"), hidden512.data() + (size_t)i * 512, (size_t)512 * sizeof(unsigned short));
        if (runner.inference() != 0) throw std::runtime_error("speech tokenizer post inference failed");
        auto out = tensor_to_float(runner, runner.get_output("output"), 1024);
        std::memcpy(hidden1024.data() + (size_t)i * 1024, out.data(), (size_t)1024 * sizeof(float));
    }
    return hidden1024;
}

std::vector<float> SpeechTokenizerRunner::run_decode_post_chunk(const float *hidden, int valid_len)
{
    std::vector<float> feed((size_t)64 * 1024, 0.0f);
    std::memcpy(feed.data(), hidden, (size_t)valid_len * 1024 * sizeof(float));
    auto &runner = decode_post_.get();
    write_tensor(runner, runner.get_input("hidden"), feed.data(), feed.size() * sizeof(float));
    if (runner.inference() != 0) throw std::runtime_error("speech tokenizer vocoder inference failed");
    return tensor_to_float(runner, runner.get_output("wav"), 122325);
}

std::vector<float> SpeechTokenizerRunner::run_decode_post(const std::vector<float> &hidden1024, int code_len)
{
    const int decode_post_len = 64;
    const int left_context = 25;
    std::vector<float> wav_all;
    int start = 0;
    while (start < code_len)
    {
        const int context = std::min(left_context, start);
        const int capacity = decode_post_len - context;
        const int end = std::min(code_len, start + capacity);
        auto wav = run_decode_post_chunk(hidden1024.data() + (size_t)(start - context) * 1024, end - start + context);
        const int skip = context * kSpeechDecodeUpsample;
        const int take = (end - start) * kSpeechDecodeUpsample;
        if ((int)wav.size() > skip)
        {
            const int n = std::min(take, (int)wav.size() - skip);
            wav_all.insert(wav_all.end(), wav.begin() + skip, wav.begin() + skip + n);
        }
        start = end;
    }
    wav_all.resize((size_t)code_len * kSpeechDecodeUpsample);
    return wav_all;
}

} // namespace qwen3_tts
