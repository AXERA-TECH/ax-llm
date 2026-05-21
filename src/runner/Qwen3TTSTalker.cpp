#include "Qwen3TTSTalker.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace qwen3_tts {

bool TalkerRunner::init(const std::string &talker_dir)
{
    for (int i = 0; i < kTalkerLayers; ++i)
    {
        auto path = join_path(talker_dir, "qwen3_tts_talker_p128_l" + std::to_string(i) + "_together.axmodel");
        layers_.emplace_back(std::make_unique<Runner>());
        if (!layers_.back()->init(path)) return false;
    }
    if (!post_.init(join_path(talker_dir, "qwen3_tts_talker_post.axmodel"))) return false;
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
    return 2048;
}

std::vector<unsigned short> TalkerRunner::prefill(const std::vector<unsigned short> &embeds, int valid_len)
{
    const int chunk_len = 128;
    const int chunks = (valid_len + chunk_len - 1) / chunk_len;
    if (chunks <= 0 || chunks > 5) throw std::runtime_error("talker prompt exceeds compiled prefill capacity");
    const int padded_len = chunks * chunk_len;

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

    k_cache_.assign(kTalkerLayers, std::vector<unsigned short>((size_t)kv_cache_len() * kTalkerHidden, 0));
    v_cache_.assign(kTalkerLayers, std::vector<unsigned short>((size_t)kv_cache_len() * kTalkerHidden, 0));

    for (int layer = 0; layer < kTalkerLayers; ++layer)
    {
        std::vector<unsigned short> layer_out((size_t)padded_len * kTalkerHidden, 0);
        auto &runner = layers_[(size_t)layer]->get();
        for (int chunk = 0; chunk < chunks; ++chunk)
        {
            const int gid = chunk + 1;
            const int start = chunk * chunk_len;
            const int end = start + chunk_len;

            const auto &t_idx = input_tensor(runner, gid, "indices");
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
            write_tensor(runner, t_in, data.data() + (size_t)start * kTalkerHidden, (size_t)chunk_len * kTalkerHidden * sizeof(unsigned short));

            std::vector<unsigned short> mask((size_t)chunk_len * end);
            for (int r = 0; r < chunk_len; ++r)
                std::memcpy(mask.data() + (size_t)r * end,
                            full_mask.data() + (size_t)(start + r) * padded_len,
                            (size_t)end * sizeof(unsigned short));
            const auto &t_mask = input_tensor(runner, gid, "mask");
            write_tensor(runner, t_mask, mask.data(), mask.size() * sizeof(unsigned short));

            const auto &t_k = input_tensor(runner, gid, "K_cache");
            const auto &t_v = input_tensor(runner, gid, "V_cache");
            if (chunk == 0)
            {
                std::vector<unsigned short> zeros_k((size_t)t_k.nSize / sizeof(unsigned short), 0);
                std::vector<unsigned short> zeros_v((size_t)t_v.nSize / sizeof(unsigned short), 0);
                write_tensor(runner, t_k, zeros_k.data(), zeros_k.size() * sizeof(unsigned short));
                write_tensor(runner, t_v, zeros_v.data(), zeros_v.size() * sizeof(unsigned short));
            }
            else
            {
                write_tensor(runner, t_k, k_cache_[(size_t)layer].data(), (size_t)start * kTalkerHidden * sizeof(unsigned short));
                write_tensor(runner, t_v, v_cache_[(size_t)layer].data(), (size_t)start * kTalkerHidden * sizeof(unsigned short));
            }

            if (runner.inference(gid) != 0) throw std::runtime_error("talker prefill inference failed");

            const auto &out_k = output_tensor(runner, gid, "K_cache_out");
            const auto &out_v = output_tensor(runner, gid, "V_cache_out");
            const auto &out_h = output_tensor(runner, gid, "output");
            read_tensor(runner, out_k, k_cache_[(size_t)layer].data() + (size_t)start * kTalkerHidden, (size_t)chunk_len * kTalkerHidden * sizeof(unsigned short));
            read_tensor(runner, out_v, v_cache_[(size_t)layer].data() + (size_t)start * kTalkerHidden, (size_t)chunk_len * kTalkerHidden * sizeof(unsigned short));
            read_tensor(runner, out_h, layer_out.data() + (size_t)start * kTalkerHidden, (size_t)chunk_len * kTalkerHidden * sizeof(unsigned short));
        }
        data.swap(layer_out);
    }

    current_len_ = valid_len;
    return slice_bf16_rows(data, valid_len - 1, 1, kTalkerHidden);
}

std::vector<unsigned short> TalkerRunner::decode_one(const std::vector<unsigned short> &embed)
{
    if (embed.size() != (size_t)kTalkerHidden) throw std::runtime_error("talker decode embed size mismatch");
    if (current_len_ >= kv_cache_len()) throw std::runtime_error("talker decode reaches kv cache limit");

    std::vector<unsigned short> data = embed;
    const unsigned int idx = (unsigned int)current_len_;
    std::vector<unsigned short> mask((size_t)kv_cache_len() + 1, bfloat16(-65536.0f).data);
    for (int i = 0; i < current_len_; ++i) mask[(size_t)i] = 0;
    mask.back() = 0;

    for (int layer = 0; layer < kTalkerLayers; ++layer)
    {
        auto &runner = layers_[(size_t)layer]->get();
        write_tensor(runner, input_tensor(runner, 0, "K_cache"), k_cache_[(size_t)layer].data(), k_cache_[(size_t)layer].size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "V_cache"), v_cache_[(size_t)layer].data(), v_cache_[(size_t)layer].size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "indices"), &idx, sizeof(idx));
        write_tensor(runner, input_tensor(runner, 0, "input"), data.data(), data.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "mask"), mask.data(), mask.size() * sizeof(unsigned short));
        if (runner.inference(0) != 0) throw std::runtime_error("talker decode inference failed");
        read_tensor(runner, output_tensor(runner, 0, "K_cache_out"), k_cache_[(size_t)layer].data() + (size_t)current_len_ * kTalkerHidden, (size_t)kTalkerHidden * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 0, "V_cache_out"), v_cache_[(size_t)layer].data() + (size_t)current_len_ * kTalkerHidden, (size_t)kTalkerHidden * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 0, "output"), data.data(), data.size() * sizeof(unsigned short));
    }

    ++current_len_;
    return data;
}

TalkerRunner::PostResult TalkerRunner::post(const std::vector<unsigned short> &raw_hidden)
{
    auto &runner = post_.get();
    write_tensor(runner, runner.get_input("input"), raw_hidden.data(), raw_hidden.size() * sizeof(unsigned short));
    if (runner.inference() != 0) throw std::runtime_error("talker post inference failed");

    PostResult result;
    result.logits = tensor_to_float(runner, runner.get_output("output"), kTalkerVocab);
    result.hidden_norm = tensor_to_bf16(runner, runner.get_output("output_norm"), kTalkerHidden);
    if (result.hidden_norm.size() != (size_t)kTalkerHidden)
        throw std::runtime_error("talker post output_norm size mismatch");
    return result;
}

} // namespace qwen3_tts
