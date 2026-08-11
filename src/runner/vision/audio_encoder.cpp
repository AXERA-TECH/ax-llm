// AudioEncoder: per-VLM audio encoders (Gemma4 dual 5s/30s, Qwen3Omni Whisper 30s).
// Extracted verbatim from vision_module.cpp (the audio machinery does not belong in a
// vision module). The multimodal injection (audio_pad placeholders) stays in VisionModule.
#include "audio_encoder.hpp"

#include "vision_runtime.hpp"          // ax_runner_t + v_h2d/v_d2h + V_WADDR/V_RADDR
#include "utils/audio_processor.hpp"   // audio:: profiles + Load*/Infer*/ReadDuration
#include "utils/files.hpp"             // is_file
#include "bfloat16.hpp"
#include "sample_log.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vision {

namespace {

struct AudioEncoderRuntime {
    enum class ProfileKind {
        None = 0,
        Gemma4,
        Whisper,
    };

    ax_runner_t encoder;
    bool encoder_inited = false;
    int encoder_output_is_bf16 = -1;
    ProfileKind profile_kind = ProfileKind::None;
    audio::Gemma4AudioProfile profile;
    audio::WhisperAudioProfile whisper_profile;
    std::string axmodel_path;
};

static bool try_pick_tokens_by_output_bytes(const ax_runner_tensor_t& out0,
                                            int tokens_embed_size,
                                            int& out_is_bf16,
                                            int& out_tokens_per_block)
{
    for (int bytes_per_elem : {4, 2}) {
        if ((out0.nSize % bytes_per_elem) != 0) continue;
        const size_t elem = (size_t)out0.nSize / (size_t)bytes_per_elem;
        if (elem == 0 || (elem % (size_t)tokens_embed_size) != 0) continue;
        out_is_bf16 = (bytes_per_elem == 2) ? 1 : 0;
        out_tokens_per_block = (int)(elem / (size_t)tokens_embed_size);
        return true;
    }
    return false;
}

static bool init_audio_profile(AudioEncoderRuntime& runtime,
                               const std::string& axmodel_path,
                               int devid,
                               int tokens_embed_size,
                               std::string& err)
{
    audio::Gemma4AudioProfile profile;
    if (!audio::InferGemma4AudioProfileFromPath(axmodel_path, profile)) {
        err = "failed to infer Gemma4 audio profile from path: " + axmodel_path;
        return false;
    }

    if (runtime.encoder.init(axmodel_path.c_str(), devid) != 0) {
        err = "init audio encoder axmodel failed: " + axmodel_path;
        return false;
    }
    runtime.encoder_inited = true;
    runtime.profile_kind = AudioEncoderRuntime::ProfileKind::Gemma4;
    runtime.profile = profile;
    runtime.axmodel_path = axmodel_path;

#ifdef USE_AXCL
    runtime.encoder.set_auto_sync_before_inference(true);
    runtime.encoder.set_auto_sync_after_inference(true);
#endif

    const auto& out0 = runtime.encoder.get_output(0);
    int out_is_bf16 = -1;
    int tokens_per_block = 0;
    auto try_pick_expected = [&](int bytes_per_elem) -> bool {
        if ((out0.nSize % bytes_per_elem) != 0) return false;
        const size_t elem = (size_t)out0.nSize / (size_t)bytes_per_elem;
        if (elem == 0 || (elem % (size_t)tokens_embed_size) != 0) return false;
        const int tokens = (int)(elem / (size_t)tokens_embed_size);
        if (tokens != runtime.profile.num_audio_tokens) return false;
        out_is_bf16 = (bytes_per_elem == 2) ? 1 : 0;
        tokens_per_block = tokens;
        return true;
    };

    if (!try_pick_expected(4) && !try_pick_expected(2)) {
        if (!try_pick_tokens_by_output_bytes(out0, tokens_embed_size, out_is_bf16, tokens_per_block)) {
            err = "failed to infer Gemma4 audio output layout: " + axmodel_path;
            return false;
        }
        ALOGW("Gemma4 audio profile token count mismatch: expected=%d inferred=%d from %s",
              runtime.profile.num_audio_tokens, tokens_per_block, axmodel_path.c_str());
    }

    runtime.encoder_output_is_bf16 = out_is_bf16;
    runtime.profile.num_audio_tokens = tokens_per_block;
    ALOGI("Gemma4 audio profile init ok: path=%s duration=%.1fs mel_frames=%d tokens=%d out_dtype=%s",
          axmodel_path.c_str(),
          runtime.profile.duration_sec,
          runtime.profile.num_mel_frames,
          runtime.profile.num_audio_tokens,
          (runtime.encoder_output_is_bf16 ? "bf16" : "fp32"));
    return true;
}

static bool init_whisper_audio_profile(AudioEncoderRuntime& runtime,
                                       const std::string& axmodel_path,
                                       int devid,
                                       int tokens_embed_size,
                                       std::string& err)
{
    if (runtime.encoder.init(axmodel_path.c_str(), devid) != 0) {
        err = "init whisper audio encoder axmodel failed: " + axmodel_path;
        return false;
    }
    runtime.encoder_inited = true;
    runtime.profile_kind = AudioEncoderRuntime::ProfileKind::Whisper;
    runtime.axmodel_path = axmodel_path;

#ifdef USE_AXCL
    runtime.encoder.set_auto_sync_before_inference(true);
    runtime.encoder.set_auto_sync_after_inference(true);
#endif

    const auto& in0 = runtime.encoder.get_input(0);
    if (in0.vShape.size() >= 3) {
        runtime.whisper_profile.feature_size = (int)in0.vShape[in0.vShape.size() - 2];
        runtime.whisper_profile.num_mel_frames = (int)in0.vShape[in0.vShape.size() - 1];
    } else {
        const size_t elem = ((size_t)in0.nSize % sizeof(float) == 0) ? ((size_t)in0.nSize / sizeof(float)) : 0;
        runtime.whisper_profile.feature_size = 128;
        runtime.whisper_profile.num_mel_frames =
            (runtime.whisper_profile.feature_size > 0) ? (int)(elem / (size_t)runtime.whisper_profile.feature_size) : 0;
    }

    if (runtime.whisper_profile.feature_size <= 0 || runtime.whisper_profile.num_mel_frames <= 0) {
        err = "failed to infer whisper audio input layout: " + axmodel_path;
        return false;
    }
    runtime.whisper_profile.duration_sec =
        (float)runtime.whisper_profile.num_mel_frames * (float)runtime.whisper_profile.hop_length /
        (float)runtime.whisper_profile.sampling_rate;

    const auto& out0 = runtime.encoder.get_output(0);
    int out_is_bf16 = -1;
    int tokens_per_block = 0;
    if (!try_pick_tokens_by_output_bytes(out0, tokens_embed_size, out_is_bf16, tokens_per_block)) {
        err = "failed to infer whisper audio output layout: " + axmodel_path;
        return false;
    }

    runtime.encoder_output_is_bf16 = out_is_bf16;
    runtime.whisper_profile.num_audio_tokens = tokens_per_block;
    ALOGI("Whisper audio profile init ok: path=%s duration=%.1fs mel_frames=%d tokens=%d out_dtype=%s",
          axmodel_path.c_str(),
          runtime.whisper_profile.duration_sec,
          runtime.whisper_profile.num_mel_frames,
          runtime.whisper_profile.num_audio_tokens,
          (runtime.encoder_output_is_bf16 ? "bf16" : "fp32"));
    return true;
}

static AudioEncoderRuntime* select_audio_profile(AudioEncoderRuntime* p5,
                                                 AudioEncoderRuntime* p30,
                                                 float duration_sec)
{
    if (p5 && !p5->encoder_inited) p5 = nullptr;
    if (p30 && !p30->encoder_inited) p30 = nullptr;

    if (p5 && duration_sec <= p5->profile.duration_sec + 0.25f) return p5;
    if (p30) return p30;
    if (p5) return p5;
    return nullptr;
}

static bool encode_block_fp32(ax_runner_t& enc, int devid, int out_is_bf16,
                              const std::vector<float>& values,
                              std::vector<unsigned short>& out_bf16,
                              std::string& err)
{
    const auto& in0 = enc.get_input(0);
    const size_t input_bytes = values.size() * sizeof(float);
    if ((size_t)in0.nSize < input_bytes) {
        err = "encoder input tensor too small for float32 input";
        return false;
    }

    if (input_bytes == (size_t)in0.nSize) {
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, values.data(), input_bytes);
        } else {
            v_h2d(V_WADDR(in0), values.data(), input_bytes, devid);
        }
    } else {
        std::vector<unsigned char> tmp((size_t)in0.nSize, 0);
        std::memcpy(tmp.data(), values.data(), input_bytes);
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, tmp.data(), tmp.size());
        } else {
            v_h2d(V_WADDR(in0), tmp.data(), tmp.size(), devid);
        }
    }

    enc.inference();

    const auto& out0 = enc.get_output(0);
    int elem_count = 0;
    if (out_is_bf16) {
        elem_count = (int)((size_t)out0.nSize / sizeof(unsigned short));
    } else {
        elem_count = (int)((size_t)out0.nSize / sizeof(float));
    }
    if (elem_count <= 0) {
        err = "audio encoder output elem_count invalid";
        return false;
    }
    out_bf16.resize(elem_count);

    if (out_is_bf16) {
        if (out0.pVirAddr) {
            std::memcpy(out_bf16.data(), out0.pVirAddr, (size_t)elem_count * sizeof(unsigned short));
        } else {
            v_d2h(out_bf16.data(), V_RADDR(out0), (size_t)elem_count * sizeof(unsigned short), devid);
        }
        return true;
    }

    std::vector<float> tmp(elem_count);
    if (out0.pVirAddr) {
        std::memcpy(tmp.data(), out0.pVirAddr, (size_t)elem_count * sizeof(float));
    } else {
        v_d2h(tmp.data(), V_RADDR(out0), (size_t)elem_count * sizeof(float), devid);
    }
    for (int i = 0; i < elem_count; ++i) out_bf16[i] = bfloat16(tmp[i]).data;
    return true;
}

} // namespace

struct AudioEncoder::Impl {
    Kind kind = Kind::None;
    AudioEncoderRuntime audio_5s;
    AudioEncoderRuntime audio_30s;
    ~Impl() {
        if (audio_5s.encoder_inited) audio_5s.encoder.deinit();
        if (audio_30s.encoder_inited) audio_30s.encoder.deinit();
    }
};

AudioEncoder::AudioEncoder() : impl_(new Impl()) {}
AudioEncoder::~AudioEncoder() = default;

bool AudioEncoder::Init(Kind kind, const std::string& enc_5s, const std::string& enc_30s,
                        int devid, int tokens_embed_size, std::string& err) {
    impl_->kind = kind;
    if (kind == Kind::Gemma4) {
        if (!enc_5s.empty() && is_file(enc_5s)) {
            if (!init_audio_profile(impl_->audio_5s, enc_5s, devid, tokens_embed_size, err)) return false;
        }
        if (!enc_30s.empty() && is_file(enc_30s)) {
            if (!init_audio_profile(impl_->audio_30s, enc_30s, devid, tokens_embed_size, err)) return false;
        }
        if (!impl_->audio_5s.encoder_inited && !impl_->audio_30s.encoder_inited) {
            ALOGW("Gemma4 audio encoders are not configured or missing; AUDIO inputs will be rejected.");
        }
    } else if (kind == Kind::Whisper) {
        if (!enc_30s.empty() && is_file(enc_30s)) {
            if (!init_whisper_audio_profile(impl_->audio_30s, enc_30s, devid, tokens_embed_size, err)) return false;
        }
        if (!impl_->audio_30s.encoder_inited) {
            ALOGW("Qwen3Omni audio encoder is not configured or missing; AUDIO inputs will be rejected.");
        }
    }
    return true;
}

bool AudioEncoder::enabled() const {
    return impl_->audio_5s.encoder_inited || impl_->audio_30s.encoder_inited;
}

bool AudioEncoder::Encode(const std::vector<std::string>& uris,
                          std::vector<unsigned short>& out_block,
                          int& out_num_media_for_tokenizer,
                          int& out_num_media_tokens,
                          std::string& err) {
    if (impl_->kind == Kind::Gemma4) {
        if (uris.size() != 1) { err = "Gemma4 audio expects exactly 1 audio file per message"; return false; }
        float duration_sec = 0.0f;
        if (!audio::ReadAudioDurationSeconds(uris[0], duration_sec, err)) return false;
        auto* runtime = select_audio_profile(&impl_->audio_5s, &impl_->audio_30s, duration_sec);
        if (!runtime) { err = "Gemma4 audio encoder profile is not initialized"; return false; }
        std::vector<float> input_features;
        if (!audio::LoadGemma4AudioInputFeatures(uris[0], runtime->profile, input_features, nullptr, err)) return false;
        std::vector<unsigned short> emb;
        if (!encode_block_fp32(runtime->encoder, runtime->encoder.get_devid(),
                               runtime->encoder_output_is_bf16, input_features, emb, err)) return false;
        out_num_media_for_tokenizer = 1;
        out_num_media_tokens = runtime->profile.num_audio_tokens;
        out_block = std::move(emb);
        if (duration_sec > runtime->profile.duration_sec) {
            ALOGW("Gemma4 audio input %.3fs exceeds selected %.1fs profile; trailing audio will be truncated",
                  duration_sec, runtime->profile.duration_sec);
        } else {
            ALOGI("Gemma4 audio profile selected: %.1fs -> %d tokens (input=%.3fs)",
                  runtime->profile.duration_sec, runtime->profile.num_audio_tokens, duration_sec);
        }
        return true;
    }
    if (impl_->kind == Kind::Whisper) {
        if (uris.size() != 1) { err = "Qwen3Omni audio expects exactly 1 audio file per message"; return false; }
        if (!impl_->audio_30s.encoder_inited) { err = "Qwen3Omni audio encoder is not initialized"; return false; }
        if (std::getenv("AXLLM_DEBUG_QWEN3OMNI_AUDIO")) ALOGI("Qwen3Omni audio begin: uri=%s", uris[0].c_str());
        std::vector<float> input_features;
        if (!audio::LoadWhisperAudioInputFeatures(uris[0], impl_->audio_30s.whisper_profile, input_features, nullptr, err)) return false;
        if (std::getenv("AXLLM_DEBUG_QWEN3OMNI_AUDIO")) ALOGI("Qwen3Omni audio features ready: elems=%zu", input_features.size());
        std::vector<unsigned short> emb;
        if (!encode_block_fp32(impl_->audio_30s.encoder, impl_->audio_30s.encoder.get_devid(),
                               impl_->audio_30s.encoder_output_is_bf16, input_features, emb, err)) return false;
        if (std::getenv("AXLLM_DEBUG_QWEN3OMNI_AUDIO")) ALOGI("Qwen3Omni audio encoder ready: bf16_elems=%zu", emb.size());
        out_num_media_for_tokenizer = 1;
        out_num_media_tokens = impl_->audio_30s.whisper_profile.num_audio_tokens;
        out_block = std::move(emb);
        ALOGI("Qwen3Omni audio profile selected: %.1fs -> %d tokens",
              impl_->audio_30s.whisper_profile.duration_sec, impl_->audio_30s.whisper_profile.num_audio_tokens);
        return true;
    }
    err = "AUDIO not supported for this vlm_type";
    return false;
}

} // namespace vision
