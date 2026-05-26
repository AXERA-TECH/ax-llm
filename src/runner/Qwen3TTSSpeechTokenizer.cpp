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
        const auto path = find_numbered_model_path(
            join_path(dir, "decoder"),
            "qwen3_tts_tokenizer_12hz_p",
            "_l" + std::to_string(i) + "_together.axmodel");
        if (path.empty())
        {
            ALOGE("speech tokenizer decoder layer axmodel not found: dir=%s layer=%d", join_path(dir, "decoder").c_str(), i);
            return false;
        }
        if (!layers_.back()->init(path)) return false;
    }
    if (!post_.init(join_path(dir, "decoder/qwen3_tts_tokenizer_12hz_post.axmodel"))) return false;
    const auto decode_post_path = find_numbered_model_path(dir, "speech_tokenizer_decode_post_t", ".axmodel");
    if (decode_post_path.empty())
    {
        ALOGE("speech tokenizer decode_post axmodel not found: dir=%s", dir.c_str());
        return false;
    }
    if (!decode_post_.init(decode_post_path)) return false;
    try
    {
        const auto &decode_pre_input = decode_pre_.get().get_input("codes");
        decode_pre_code_len_ = tensor_dim(decode_pre_input, 2, "speech tokenizer decode_pre codes");

        auto &first = layers_.front()->get();
        const auto &prefill_input = input_tensor(first, 1, "input");
        const auto &decode_k = input_tensor(first, 0, "K_cache");
        const auto &decode_mask = input_tensor(first, 0, "mask");
        const auto &decode_k_out = output_tensor(first, 0, "K_cache_out");

        transformer_prefill_len_ = tensor_dim(prefill_input, 1, "speech tokenizer transformer prefill input");
        transformer_hidden_size_ = tensor_dim(prefill_input, 2, "speech tokenizer transformer prefill input");
        transformer_kv_cache_len_ = tensor_dim(decode_k, 1, "speech tokenizer transformer decode K_cache");
        transformer_kv_dim_ = tensor_elems_bf16(decode_k_out);
        transformer_decode_mask_len_ = tensor_elems_bf16(decode_mask);

        const auto &decode_post_input = decode_post_.get().get_input("hidden");
        decode_post_len_ = tensor_dim(decode_post_input, 1, "speech tokenizer decode_post hidden");
        decode_post_hidden_size_ = tensor_dim(decode_post_input, 2, "speech tokenizer decode_post hidden");
        decode_post_output_samples_ = shape_elems(decode_post_.get().get_output("wav").vShape);

        if (decode_pre_code_len_ <= 0 || transformer_prefill_len_ <= 0 ||
            transformer_hidden_size_ <= 0 || transformer_kv_cache_len_ <= 0 ||
            transformer_kv_dim_ <= 0 || transformer_decode_mask_len_ <= 1 ||
            decode_post_len_ <= 0 || decode_post_hidden_size_ <= 0)
            throw std::runtime_error("invalid speech tokenizer inferred shape");
        ALOGI("speech tokenizer inferred shapes: decode_pre_len=%d transformer_prefill=%d hidden=%d kv_cache=%d kv_dim=%d decode_mask=%d decode_post_len=%d decode_post_hidden=%d",
              decode_pre_code_len_, transformer_prefill_len_, transformer_hidden_size_,
              transformer_kv_cache_len_, transformer_kv_dim_, transformer_decode_mask_len_,
              decode_post_len_, decode_post_hidden_size_);
    }
    catch (const std::exception &e)
    {
        ALOGE("failed to infer speech tokenizer shapes: %s", e.what());
        return false;
    }
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
    ALOGI("speech tokenizer encode start: input_samples=%zu fixed_samples=%d", wav24.size(), kSpeechEncoderSamples);
    std::vector<float> fixed = wav24;
    fit_audio(fixed, kSpeechEncoderSamples);
    auto &runner = encoder_.get();
    const auto &input = runner.get_input("input_values");
    ALOGI("speech tokenizer encode write input: tensor=%s shape=%s nSize=%d bytes=%zu",
          input.sName.c_str(), shape_to_string(input.vShape).c_str(), input.nSize,
          fixed.size() * sizeof(float));
    write_tensor(runner, input, fixed.data(), fixed.size() * sizeof(float));
    ALOGI("speech tokenizer encode inference begin");
    const int ret = runner.inference();
    ALOGI("speech tokenizer encode inference end ret=%d", ret);
    if (ret != 0) throw std::runtime_error("speech tokenizer encoder inference failed");
    const auto &out = runner.get_output("audio_codes");
    const size_t elems = shape_elems(out.vShape);
    ALOGI("speech tokenizer encode read output: tensor=%s shape=%s nSize=%d elems=%zu",
          out.sName.c_str(), shape_to_string(out.vShape).c_str(), out.nSize, elems);
    std::vector<int32_t> raw(elems);
    read_tensor(runner, out, raw.data(), elems * sizeof(int32_t));
    const int valid_samples = std::min<int>((int)wav24.size(), kSpeechEncoderSamples);
    const int code_len = std::min<int>(38, (valid_samples + kSpeechEncodeDownsample - 1) / kSpeechEncodeDownsample);
    std::vector<std::array<int, kCodeGroups>> codes((size_t)code_len);
    for (int t = 0; t < code_len; ++t)
        for (int q = 0; q < kCodeGroups; ++q)
            codes[(size_t)t][(size_t)q] = (int)raw[(size_t)t * kCodeGroups + q];
    ALOGI("speech tokenizer encode done: valid_samples=%d code_len=%d", valid_samples, code_len);
    return codes;
}

std::vector<float> SpeechTokenizerRunner::decode(const std::vector<std::array<int, kCodeGroups>> &codes)
{
    ALOGI("speech tokenizer decode start: input_code_frames=%zu", codes.size());
    int code_len = 0;
    for (const auto &c : codes)
    {
        if (c[0] > 0) ++code_len;
        else break;
    }
    ALOGI("speech tokenizer decode valid code_len=%d", code_len);
    if (code_len <= 0) return {};
    if (code_len > transformer_kv_cache_len_) throw std::runtime_error("speech tokenizer decoder code_len exceeds compiled kv cache");

    ALOGI("speech tokenizer decode_pre begin");
    std::vector<float> hidden512 = decode_pre(codes, code_len);
    ALOGI("speech tokenizer decode_pre end: hidden512=%zu", hidden512.size());
    ALOGI("speech tokenizer transformer begin");
    auto transformed = run_transformer(hidden512, code_len);
    ALOGI("speech tokenizer transformer end: transformed=%zu", transformed.size());
    ALOGI("speech tokenizer post begin");
    auto hidden1024 = run_post(transformed, code_len);
    ALOGI("speech tokenizer post end: hidden1024=%zu", hidden1024.size());
    ALOGI("speech tokenizer decode_post begin");
    auto wav = run_decode_post(hidden1024, code_len);
    ALOGI("speech tokenizer decode_post end: wav_samples=%zu", wav.size());
    return wav;
}

std::vector<float> SpeechTokenizerRunner::decode_pre(const std::vector<std::array<int, kCodeGroups>> &codes, int code_len)
{
    ALOGI("speech tokenizer decode_pre start: code_len=%d", code_len);
    if (code_len > decode_pre_code_len_) throw std::runtime_error("speech tokenizer decode_pre code_len exceeds compiled input length");
    std::vector<int32_t> feed((size_t)kCodeGroups * (size_t)decode_pre_code_len_, 0);
    for (int t = 0; t < code_len; ++t)
        for (int q = 0; q < kCodeGroups; ++q)
            feed[(size_t)q * (size_t)decode_pre_code_len_ + (size_t)t] = (int32_t)codes[(size_t)t][(size_t)q];
    auto &runner = decode_pre_.get();
    const auto &input = runner.get_input("codes");
    ALOGI("speech tokenizer decode_pre write input: tensor=%s shape=%s nSize=%d bytes=%zu",
          input.sName.c_str(), shape_to_string(input.vShape).c_str(), input.nSize,
          feed.size() * sizeof(int32_t));
    write_tensor(runner, input, feed.data(), feed.size() * sizeof(int32_t));
    ALOGI("speech tokenizer decode_pre inference begin");
    const int ret = runner.inference();
    ALOGI("speech tokenizer decode_pre inference end ret=%d", ret);
    if (ret != 0) throw std::runtime_error("speech tokenizer decode_pre inference failed");
    const auto &output = runner.get_output("hidden_512");
    ALOGI("speech tokenizer decode_pre read output: tensor=%s shape=%s nSize=%d",
          output.sName.c_str(), shape_to_string(output.vShape).c_str(), output.nSize);
    auto out = tensor_to_float(runner, output, (size_t)decode_pre_code_len_ * (size_t)transformer_hidden_size_);
    out.resize((size_t)code_len * (size_t)transformer_hidden_size_);
    ALOGI("speech tokenizer decode_pre done: output_elems=%zu", out.size());
    return out;
}

std::vector<unsigned short> SpeechTokenizerRunner::run_transformer(const std::vector<float> &hidden512, int code_len)
{
    ALOGI("speech tokenizer transformer start: code_len=%d hidden512=%zu", code_len, hidden512.size());
    const int prefill_len = transformer_prefill_len_;
    const int valid_prefill = std::min(code_len, prefill_len);
    if (hidden512.size() < (size_t)code_len * (size_t)transformer_hidden_size_)
        throw std::runtime_error("speech tokenizer transformer hidden input is too small");
    std::vector<unsigned short> data((size_t)prefill_len * (size_t)transformer_hidden_size_, 0);
    auto hbf = fp32_to_bf16_vec(hidden512);
    std::memcpy(data.data(), hbf.data(), (size_t)valid_prefill * (size_t)transformer_hidden_size_ * sizeof(unsigned short));

    std::vector<unsigned int> indices((size_t)prefill_len);
    std::iota(indices.begin(), indices.end(), 0);
    const auto &first_mask = input_tensor(layers_.front()->get(), 1, "mask");
    const int prefill_mask_elems = tensor_elems_bf16(first_mask);
    const int prefill_mask_cols = prefill_mask_elems / prefill_len;
    const int prefill_history_len = std::max(0, prefill_mask_cols - prefill_len);
    std::vector<unsigned short> mask((size_t)prefill_mask_elems, bfloat16(-65536.0f).data);
    for (int r = 0; r < prefill_len; ++r)
    {
        const int abs_q = prefill_history_len + r;
        const int left = std::max(0, abs_q - 72 + 1);
        const int right = std::min(abs_q, prefill_mask_cols - 1);
        for (int c = left; c <= right; ++c) mask[(size_t)r * (size_t)prefill_mask_cols + (size_t)c] = 0;
    }

    std::vector<std::vector<unsigned short>> kc(kSpeechDecoderLayers, std::vector<unsigned short>((size_t)transformer_kv_cache_len_ * (size_t)transformer_kv_dim_, 0));
    std::vector<std::vector<unsigned short>> vc(kSpeechDecoderLayers, std::vector<unsigned short>((size_t)transformer_kv_cache_len_ * (size_t)transformer_kv_dim_, 0));
    for (int layer = 0; layer < kSpeechDecoderLayers; ++layer)
    {
        auto &runner = layers_[(size_t)layer]->get();
        ALOGI("speech tokenizer transformer prefill layer=%d/%d begin", layer, kSpeechDecoderLayers);
        write_tensor(runner, input_tensor(runner, 1, "indices"), indices.data(), indices.size() * sizeof(unsigned int));
        write_tensor(runner, input_tensor(runner, 1, "input"), data.data(), data.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 1, "mask"), mask.data(), mask.size() * sizeof(unsigned short));
        std::vector<unsigned short> zeros_k((size_t)input_tensor(runner, 1, "K_cache").nSize / sizeof(unsigned short), 0);
        std::vector<unsigned short> zeros_v((size_t)input_tensor(runner, 1, "V_cache").nSize / sizeof(unsigned short), 0);
        write_tensor(runner, input_tensor(runner, 1, "K_cache"), zeros_k.data(), zeros_k.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 1, "V_cache"), zeros_v.data(), zeros_v.size() * sizeof(unsigned short));
        ALOGI("speech tokenizer transformer prefill inference begin: layer=%d gid=1", layer);
        const int ret = runner.inference(1);
        ALOGI("speech tokenizer transformer prefill inference end: layer=%d gid=1 ret=%d", layer, ret);
        if (ret != 0) throw std::runtime_error("speech tokenizer transformer prefill failed");
        const auto &out_k = output_tensor(runner, 1, "K_cache_out");
        const auto &out_v = output_tensor(runner, 1, "V_cache_out");
        read_tensor(runner, out_k, kc[(size_t)layer].data(), std::min((size_t)out_k.nSize, kc[(size_t)layer].size() * sizeof(unsigned short)));
        read_tensor(runner, out_v, vc[(size_t)layer].data(), std::min((size_t)out_v.nSize, vc[(size_t)layer].size() * sizeof(unsigned short)));
        read_tensor(runner, output_tensor(runner, 1, "output"), data.data(), data.size() * sizeof(unsigned short));
        ALOGI("speech tokenizer transformer prefill layer=%d/%d end", layer, kSpeechDecoderLayers);
    }

    std::vector<unsigned short> all;
    all.insert(all.end(), data.begin(), data.begin() + (ptrdiff_t)((size_t)valid_prefill * (size_t)transformer_hidden_size_));
    for (int pos = prefill_len; pos < code_len; ++pos)
    {
        std::vector<unsigned short> tok((size_t)transformer_hidden_size_);
        std::memcpy(tok.data(), hbf.data() + (size_t)pos * (size_t)transformer_hidden_size_, (size_t)transformer_hidden_size_ * sizeof(unsigned short));
        std::vector<unsigned short> d = tok;
        std::vector<unsigned short> dmask((size_t)transformer_decode_mask_len_, bfloat16(-65536.0f).data);
        const int left = std::max(0, pos - 72 + 1);
        for (int i = left; i < pos && i + 1 < transformer_decode_mask_len_; ++i) dmask[(size_t)i] = 0;
        dmask.back() = 0;
        const unsigned int idx = (unsigned int)pos;
        for (int layer = 0; layer < kSpeechDecoderLayers; ++layer)
        {
            auto &runner = layers_[(size_t)layer]->get();
            ALOGI("speech tokenizer transformer decode pos=%d layer=%d/%d begin", pos, layer, kSpeechDecoderLayers);
            write_tensor(runner, input_tensor(runner, 0, "K_cache"), kc[(size_t)layer].data(), kc[(size_t)layer].size() * sizeof(unsigned short));
            write_tensor(runner, input_tensor(runner, 0, "V_cache"), vc[(size_t)layer].data(), vc[(size_t)layer].size() * sizeof(unsigned short));
            write_tensor(runner, input_tensor(runner, 0, "indices"), &idx, sizeof(idx));
            write_tensor(runner, input_tensor(runner, 0, "input"), d.data(), d.size() * sizeof(unsigned short));
            write_tensor(runner, input_tensor(runner, 0, "mask"), dmask.data(), dmask.size() * sizeof(unsigned short));
            ALOGI("speech tokenizer transformer decode inference begin: pos=%d layer=%d gid=0", pos, layer);
            const int ret = runner.inference(0);
            ALOGI("speech tokenizer transformer decode inference end: pos=%d layer=%d gid=0 ret=%d", pos, layer, ret);
            if (ret != 0) throw std::runtime_error("speech tokenizer transformer decode failed");
            read_tensor(runner, output_tensor(runner, 0, "K_cache_out"), kc[(size_t)layer].data() + (size_t)pos * (size_t)transformer_kv_dim_, (size_t)transformer_kv_dim_ * sizeof(unsigned short));
            read_tensor(runner, output_tensor(runner, 0, "V_cache_out"), vc[(size_t)layer].data() + (size_t)pos * (size_t)transformer_kv_dim_, (size_t)transformer_kv_dim_ * sizeof(unsigned short));
            read_tensor(runner, output_tensor(runner, 0, "output"), d.data(), d.size() * sizeof(unsigned short));
            ALOGI("speech tokenizer transformer decode pos=%d layer=%d/%d end", pos, layer, kSpeechDecoderLayers);
        }
        all.insert(all.end(), d.begin(), d.end());
    }
    ALOGI("speech tokenizer transformer done: output_elems=%zu", all.size());
    return all;
}

std::vector<float> SpeechTokenizerRunner::run_post(const std::vector<unsigned short> &hidden512, int code_len)
{
    ALOGI("speech tokenizer post start: code_len=%d hidden512=%zu", code_len, hidden512.size());
    std::vector<float> hidden1024((size_t)code_len * (size_t)decode_post_hidden_size_, 0.0f);
    auto &runner = post_.get();
    for (int i = 0; i < code_len; ++i)
    {
        ALOGI("speech tokenizer post frame=%d/%d begin", i, code_len);
        write_tensor(runner, runner.get_input("input"), hidden512.data() + (size_t)i * (size_t)transformer_hidden_size_, (size_t)transformer_hidden_size_ * sizeof(unsigned short));
        const int ret = runner.inference();
        ALOGI("speech tokenizer post frame=%d/%d inference ret=%d", i, code_len, ret);
        if (ret != 0) throw std::runtime_error("speech tokenizer post inference failed");
        auto out = tensor_to_float(runner, runner.get_output("output"), (size_t)decode_post_hidden_size_);
        std::memcpy(hidden1024.data() + (size_t)i * (size_t)decode_post_hidden_size_, out.data(), (size_t)decode_post_hidden_size_ * sizeof(float));
    }
    ALOGI("speech tokenizer post done: hidden1024=%zu", hidden1024.size());
    return hidden1024;
}

std::vector<float> SpeechTokenizerRunner::run_decode_post_chunk(const float *hidden, int valid_len)
{
    ALOGI("speech tokenizer vocoder chunk start: valid_len=%d", valid_len);
    std::vector<float> feed((size_t)decode_post_len_ * (size_t)decode_post_hidden_size_, 0.0f);
    std::memcpy(feed.data(), hidden, (size_t)valid_len * (size_t)decode_post_hidden_size_ * sizeof(float));
    auto &runner = decode_post_.get();
    write_tensor(runner, runner.get_input("hidden"), feed.data(), feed.size() * sizeof(float));
    ALOGI("speech tokenizer vocoder inference begin: valid_len=%d", valid_len);
    const int ret = runner.inference();
    ALOGI("speech tokenizer vocoder inference end: valid_len=%d ret=%d", valid_len, ret);
    if (ret != 0) throw std::runtime_error("speech tokenizer vocoder inference failed");
    auto wav = tensor_to_float(runner, runner.get_output("wav"), decode_post_output_samples_);
    ALOGI("speech tokenizer vocoder chunk done: wav_samples=%zu", wav.size());
    return wav;
}

std::vector<float> SpeechTokenizerRunner::run_decode_post(const std::vector<float> &hidden1024, int code_len)
{
    ALOGI("speech tokenizer decode_post start: code_len=%d hidden1024=%zu", code_len, hidden1024.size());
    const int left_context = 25;
    std::vector<float> wav_all;
    int start = 0;
    while (start < code_len)
    {
        const int context = std::min(left_context, start);
        const int capacity = decode_post_len_ - context;
        const int end = std::min(code_len, start + capacity);
        ALOGI("speech tokenizer decode_post chunk: start=%d end=%d context=%d", start, end, context);
        auto wav = run_decode_post_chunk(hidden1024.data() + (size_t)(start - context) * (size_t)decode_post_hidden_size_, end - start + context);
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
    ALOGI("speech tokenizer decode_post done: wav_samples=%zu", wav_all.size());
    return wav_all;
}

} // namespace qwen3_tts
