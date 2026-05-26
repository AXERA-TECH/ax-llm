#include "Qwen3TTSTalker.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace qwen3_tts {

bool TalkerRunner::init(const std::string &talker_dir)
{
    for (int i = 0; i < kTalkerLayers; ++i)
    {
        auto path = find_numbered_model_path(
            talker_dir,
            "qwen3_tts_talker_p",
            "_l" + std::to_string(i) + "_together.axmodel");
        if (path.empty())
        {
            ALOGE("talker layer axmodel not found: dir=%s layer=%d", talker_dir.c_str(), i);
            return false;
        }
        layers_.emplace_back(std::make_unique<Runner>());
        if (!layers_.back()->init(path)) return false;
    }
    if (!post_.init(join_path(talker_dir, "qwen3_tts_talker_post.axmodel"))) return false;
    try
    {
        auto &first = layers_.front()->get();
        const auto &prefill_input = input_tensor(first, 1, "input");
        const auto &decode_k = input_tensor(first, 0, "K_cache");
        const auto &decode_mask = input_tensor(first, 0, "mask");
        const auto &decode_k_out = output_tensor(first, 0, "K_cache_out");

        prefill_len_ = tensor_dim(prefill_input, 1, "talker prefill input");
        if (tensor_dim(prefill_input, 2, "talker prefill input") != kTalkerHidden)
            throw std::runtime_error("talker hidden size mismatch: " + shape_to_string(prefill_input.vShape));
        kv_cache_len_ = tensor_dim(decode_k, 1, "talker decode K_cache");
        kv_stride_ = tensor_elems_bf16(decode_k_out);
        decode_mask_len_ = tensor_elems_bf16(decode_mask);
        max_prefill_chunks_ = std::max(0, first.get_num_input_groups() - 1);

        if (prefill_len_ <= 0 || kv_cache_len_ <= 0 || kv_stride_ <= 0 || decode_mask_len_ <= 1 || max_prefill_chunks_ <= 0)
            throw std::runtime_error("invalid talker inferred shape");
        ALOGI("talker inferred shapes: prefill_len=%d max_prefill_chunks=%d kv_cache_len=%d kv_stride=%d decode_mask_len=%d",
              prefill_len_, max_prefill_chunks_, kv_cache_len_, kv_stride_, decode_mask_len_);
    }
    catch (const std::exception &e)
    {
        ALOGE("failed to infer talker shapes: %s", e.what());
        return false;
    }
    try
    {
        (void)post_.get().get_output("output_norm");
        ALOGI("talker post exposes output_norm; use it as code_predictor past_hidden");
    }
    catch (const std::exception &e)
    {
        ALOGE("talker post must expose output_norm [1,1,%d] for code_predictor past_hidden: %s",
              kTalkerHidden, e.what());
        return false;
    }
    return true;
}

void TalkerRunner::deinit()
{
    for (auto &l : layers_) l->deinit();
    post_.deinit();
    layers_.clear();
}

int TalkerRunner::kv_cache_len() const
{
    return kv_cache_len_;
}

std::vector<unsigned short> TalkerRunner::prefill(const std::vector<unsigned short> &embeds, int valid_len)
{
    const int chunk_len = prefill_len_;
    const int chunks = (valid_len + chunk_len - 1) / chunk_len;
    if (chunks <= 0 || chunks > max_prefill_chunks_) throw std::runtime_error("talker prompt exceeds compiled prefill capacity");
    const int padded_len = chunks * chunk_len;
    ALOGI("talker prefill start: valid_len=%d padded_len=%d chunks=%d embed_elems=%zu kv_cache_len=%d",
          valid_len, padded_len, chunks, embeds.size(), kv_cache_len());

    std::vector<unsigned short> data((size_t)padded_len * kTalkerHidden, 0);
    std::memcpy(data.data(), embeds.data(), (size_t)valid_len * kTalkerHidden * sizeof(unsigned short));

    std::vector<unsigned int> indices((size_t)3 * padded_len, 1);
    for (int r = 0; r < valid_len; ++r)
    {
        indices[(size_t)r] = (unsigned int)r;
        indices[(size_t)padded_len + r] = (unsigned int)r;
        indices[(size_t)2 * padded_len + r] = (unsigned int)r;
    }

    std::vector<unsigned short> full_mask((size_t)padded_len * padded_len, bfloat16(-65536.0f).data);
    for (int r = 0; r < valid_len; ++r)
        for (int c = 0; c <= r; ++c)
            full_mask[(size_t)r * padded_len + c] = 0;

    k_cache_.assign(kTalkerLayers, std::vector<unsigned short>((size_t)kv_cache_len() * (size_t)kv_stride_, 0));
    v_cache_.assign(kTalkerLayers, std::vector<unsigned short>((size_t)kv_cache_len() * (size_t)kv_stride_, 0));
    ALOGI("talker prefill cache allocated: layers=%d per_cache_elems=%zu",
          kTalkerLayers, k_cache_.empty() ? 0 : k_cache_[0].size());

    for (int layer = 0; layer < kTalkerLayers; ++layer)
    {
        std::vector<unsigned short> layer_out((size_t)padded_len * kTalkerHidden, 0);
        auto &runner = layers_[(size_t)layer]->get();
        ALOGI("talker prefill layer %d/%d begin", layer, kTalkerLayers);
        for (int chunk = 0; chunk < chunks; ++chunk)
        {
            const int gid = chunk + 1;
            const int start = chunk * chunk_len;
            const int end = start + chunk_len;
            ALOGI("talker prefill layer=%d chunk=%d/%d gid=%d start=%d end=%d",
                  layer, chunk + 1, chunks, gid, start, end);

            const auto &t_idx = input_tensor(runner, gid, "indices");
            ALOGI("talker prefill write indices: layer=%d gid=%d tensor=%s shape=%s nSize=%d",
                  layer, gid, t_idx.sName.c_str(), shape_to_string(t_idx.vShape).c_str(), t_idx.nSize);
            if (t_idx.vShape.size() == 2 && t_idx.vShape[0] == 3)
            {
                std::vector<unsigned int> idx3((size_t)3 * chunk_len);
                for (int row = 0; row < 3; ++row)
                    std::memcpy(idx3.data() + (size_t)row * chunk_len,
                                indices.data() + (size_t)row * padded_len + start,
                                (size_t)chunk_len * sizeof(unsigned int));
                write_tensor(runner, t_idx, idx3.data(), idx3.size() * sizeof(unsigned int));
            }
            else
            {
                write_tensor(runner, t_idx, indices.data() + (size_t)start, (size_t)chunk_len * sizeof(unsigned int));
            }

            const auto &t_in = input_tensor(runner, gid, "input");
            ALOGI("talker prefill write input: layer=%d gid=%d tensor=%s shape=%s nSize=%d bytes=%zu",
                  layer, gid, t_in.sName.c_str(), shape_to_string(t_in.vShape).c_str(), t_in.nSize,
                  (size_t)chunk_len * kTalkerHidden * sizeof(unsigned short));
            write_tensor(runner, t_in, data.data() + (size_t)start * kTalkerHidden, (size_t)chunk_len * kTalkerHidden * sizeof(unsigned short));

            const auto &t_mask = input_tensor(runner, gid, "mask");
            const int mask_elems = tensor_elems_bf16(t_mask);
            const int mask_cols = mask_elems / chunk_len;
            std::vector<unsigned short> mask((size_t)mask_elems, bfloat16(-65536.0f).data);
            const int copy_cols = std::min(mask_cols, padded_len);
            for (int r = 0; r < chunk_len; ++r)
                std::memcpy(mask.data() + (size_t)r * mask_cols,
                            full_mask.data() + (size_t)(start + r) * padded_len,
                            (size_t)copy_cols * sizeof(unsigned short));
            ALOGI("talker prefill write mask: layer=%d gid=%d tensor=%s shape=%s nSize=%d bytes=%zu",
                  layer, gid, t_mask.sName.c_str(), shape_to_string(t_mask.vShape).c_str(), t_mask.nSize,
                  mask.size() * sizeof(unsigned short));
            write_tensor(runner, t_mask, mask.data(), mask.size() * sizeof(unsigned short));

            const auto &t_k = input_tensor(runner, gid, "K_cache");
            const auto &t_v = input_tensor(runner, gid, "V_cache");
            ALOGI("talker prefill write cache: layer=%d gid=%d K=%s shape=%s nSize=%d V=%s shape=%s nSize=%d cache_bytes=%zu",
                  layer, gid, t_k.sName.c_str(), shape_to_string(t_k.vShape).c_str(), t_k.nSize,
                  t_v.sName.c_str(), shape_to_string(t_v.vShape).c_str(), t_v.nSize,
                  chunk == 0 ? (size_t)t_k.nSize : (size_t)start * kTalkerHidden * sizeof(unsigned short));
            if (chunk == 0)
            {
                std::vector<unsigned short> zeros_k((size_t)t_k.nSize / sizeof(unsigned short), 0);
                std::vector<unsigned short> zeros_v((size_t)t_v.nSize / sizeof(unsigned short), 0);
                write_tensor(runner, t_k, zeros_k.data(), zeros_k.size() * sizeof(unsigned short));
                write_tensor(runner, t_v, zeros_v.data(), zeros_v.size() * sizeof(unsigned short));
            }
            else
            {
                const size_t cache_bytes = (size_t)start * (size_t)kv_stride_ * sizeof(unsigned short);
                write_tensor(runner, t_k, k_cache_[(size_t)layer].data(), std::min(cache_bytes, (size_t)t_k.nSize));
                write_tensor(runner, t_v, v_cache_[(size_t)layer].data(), std::min(cache_bytes, (size_t)t_v.nSize));
            }

            ALOGI("talker prefill inference begin: layer=%d gid=%d", layer, gid);
            const int ret = runner.inference(gid);
            ALOGI("talker prefill inference end: layer=%d gid=%d ret=%d", layer, gid, ret);
            if (ret != 0) throw std::runtime_error("talker prefill inference failed");

            const auto &out_k = output_tensor(runner, gid, "K_cache_out");
            const auto &out_v = output_tensor(runner, gid, "V_cache_out");
            const auto &out_h = output_tensor(runner, gid, "output");
            ALOGI("talker prefill read outputs: layer=%d gid=%d K=%s shape=%s nSize=%d V=%s shape=%s nSize=%d H=%s shape=%s nSize=%d",
                  layer, gid, out_k.sName.c_str(), shape_to_string(out_k.vShape).c_str(), out_k.nSize,
                  out_v.sName.c_str(), shape_to_string(out_v.vShape).c_str(), out_v.nSize,
                  out_h.sName.c_str(), shape_to_string(out_h.vShape).c_str(), out_h.nSize);
            const size_t cache_off = (size_t)start * (size_t)kv_stride_;
            const size_t cache_space = k_cache_[(size_t)layer].size() - cache_off;
            const size_t out_k_elems = std::min((size_t)tensor_elems_bf16(out_k), cache_space);
            const size_t out_v_elems = std::min((size_t)tensor_elems_bf16(out_v), cache_space);
            read_tensor(runner, out_k, k_cache_[(size_t)layer].data() + cache_off, out_k_elems * sizeof(unsigned short));
            read_tensor(runner, out_v, v_cache_[(size_t)layer].data() + cache_off, out_v_elems * sizeof(unsigned short));
            read_tensor(runner, out_h, layer_out.data() + (size_t)start * kTalkerHidden, (size_t)chunk_len * kTalkerHidden * sizeof(unsigned short));
        }
        data.swap(layer_out);
        ALOGI("talker prefill layer %d/%d end", layer, kTalkerLayers);
    }

    current_len_ = valid_len;
    ALOGI("talker prefill done: current_len=%d", current_len_);
    return slice_bf16_rows(data, valid_len - 1, 1, kTalkerHidden);
}

std::vector<unsigned short> TalkerRunner::decode_one(const std::vector<unsigned short> &embed)
{
    if (embed.size() != (size_t)kTalkerHidden) throw std::runtime_error("talker decode embed size mismatch");
    if (current_len_ >= kv_cache_len()) throw std::runtime_error("talker decode reaches kv cache limit");
    ALOGI("talker decode_one start: current_len=%d embed_elems=%zu", current_len_, embed.size());

    std::vector<unsigned short> data = embed;
    const unsigned int idx = (unsigned int)current_len_;
    std::vector<unsigned short> mask((size_t)decode_mask_len_, bfloat16(-65536.0f).data);
    for (int i = 0; i < current_len_ && i + 1 < decode_mask_len_; ++i) mask[(size_t)i] = 0;
    mask.back() = 0;

    for (int layer = 0; layer < kTalkerLayers; ++layer)
    {
        auto &runner = layers_[(size_t)layer]->get();
        ALOGI("talker decode_one layer=%d/%d write inputs idx=%u", layer, kTalkerLayers, idx);
        write_tensor(runner, input_tensor(runner, 0, "K_cache"), k_cache_[(size_t)layer].data(), k_cache_[(size_t)layer].size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "V_cache"), v_cache_[(size_t)layer].data(), v_cache_[(size_t)layer].size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "indices"), &idx, sizeof(idx));
        write_tensor(runner, input_tensor(runner, 0, "input"), data.data(), data.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "mask"), mask.data(), mask.size() * sizeof(unsigned short));
        ALOGI("talker decode_one inference begin: layer=%d gid=0", layer);
        const int ret = runner.inference(0);
        ALOGI("talker decode_one inference end: layer=%d gid=0 ret=%d", layer, ret);
        if (ret != 0) throw std::runtime_error("talker decode inference failed");
        read_tensor(runner, output_tensor(runner, 0, "K_cache_out"), k_cache_[(size_t)layer].data() + (size_t)current_len_ * (size_t)kv_stride_, (size_t)kv_stride_ * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 0, "V_cache_out"), v_cache_[(size_t)layer].data() + (size_t)current_len_ * (size_t)kv_stride_, (size_t)kv_stride_ * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 0, "output"), data.data(), data.size() * sizeof(unsigned short));
    }

    ++current_len_;
    ALOGI("talker decode_one done: current_len=%d", current_len_);
    return data;
}

TalkerRunner::PostResult TalkerRunner::post(const std::vector<unsigned short> &raw_hidden)
{
    auto &runner = post_.get();
    const auto &input = runner.get_input("input");
    ALOGI("talker post write input: elems=%zu tensor=%s shape=%s nSize=%d",
          raw_hidden.size(), input.sName.c_str(), shape_to_string(input.vShape).c_str(), input.nSize);
    write_tensor(runner, input, raw_hidden.data(), raw_hidden.size() * sizeof(unsigned short));
    ALOGI("talker post inference begin");
    const int ret = runner.inference();
    ALOGI("talker post inference end ret=%d", ret);
    if (ret != 0) throw std::runtime_error("talker post inference failed");

    PostResult result;
    const auto &output = runner.get_output("output");
    const auto &output_norm = runner.get_output("output_norm");
    ALOGI("talker post read outputs: output=%s shape=%s nSize=%d output_norm=%s shape=%s nSize=%d",
          output.sName.c_str(), shape_to_string(output.vShape).c_str(), output.nSize,
          output_norm.sName.c_str(), shape_to_string(output_norm.vShape).c_str(), output_norm.nSize);
    result.logits = tensor_to_float(runner, output, kTalkerVocab);
    result.hidden_norm = tensor_to_bf16(runner, output_norm, kTalkerHidden);
    if (result.hidden_norm.size() != (size_t)kTalkerHidden)
        throw std::runtime_error("talker post output_norm size mismatch");
    ALOGI("talker post done: logits=%zu hidden_norm=%zu", result.logits.size(), result.hidden_norm.size());
    return result;
}

} // namespace qwen3_tts
