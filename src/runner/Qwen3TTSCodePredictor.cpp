#include "Qwen3TTSCodePredictor.hpp"

#include <cstring>
#include <numeric>
#include <stdexcept>

#include "Qwen3TTS.hpp"

namespace qwen3_tts {

bool CodePredictorRunner::init(const std::string &dir)
{
    for (int i = 0; i < kCodePredictorLayers; ++i)
    {
        auto path = find_numbered_model_path(
            dir,
            "qwen3_tts_talker_code_predictor_p",
            "_l" + std::to_string(i) + "_together.axmodel");
        if (path.empty())
        {
            ALOGE("code predictor layer axmodel not found: dir=%s layer=%d", dir.c_str(), i);
            return false;
        }
        layers_.emplace_back(std::make_unique<Runner>());
        if (!layers_.back()->init(path)) return false;
    }
    if (!post_.init(join_path(dir, "qwen3_tts_talker_code_predictor_post.axmodel"))) return false;
    for (int i = 0; i < kNumSubCodes; ++i)
    {
        lm_heads_.emplace_back(std::make_unique<Runner>());
        if (!lm_heads_.back()->init(join_path(dir, "code_predictor_lm_head_" + std::to_string(i) + ".axmodel"))) return false;
    }
    for (int i = 0; i < kNumSubCodes; ++i)
    {
        auto p = join_path(dir, "talker.code_predictor.model.codec_embedding." + std::to_string(i) + ".weight.bfloat16.bin");
        auto emb = std::make_unique<LLaMaEmbedSelector>();
        if (!emb->Init(p, kCodecVocab, kTalkerHidden, true)) return false;
        embeddings_.push_back(std::move(emb));
    }
    try
    {
        auto &first = layers_.front()->get();
        const auto &prefill_input = input_tensor(first, 1, "input");
        const auto &decode_k = input_tensor(first, 0, "K_cache");
        const auto &decode_mask = input_tensor(first, 0, "mask");
        const auto &decode_k_out = output_tensor(first, 0, "K_cache_out");

        prefill_len_ = tensor_dim(prefill_input, 1, "code predictor prefill input");
        if (tensor_dim(prefill_input, 2, "code predictor prefill input") != kTalkerHidden)
            throw std::runtime_error("code predictor hidden size mismatch: " + shape_to_string(prefill_input.vShape));
        kv_cache_len_ = tensor_dim(decode_k, 1, "code predictor decode K_cache");
        kv_stride_ = tensor_elems_bf16(decode_k_out);
        decode_mask_len_ = tensor_elems_bf16(decode_mask);

        if (prefill_len_ < 2 || kv_cache_len_ <= 0 || kv_stride_ <= 0 || decode_mask_len_ <= 1)
            throw std::runtime_error("invalid code predictor inferred shape");
        ALOGI("code predictor inferred shapes: prefill_len=%d kv_cache_len=%d kv_stride=%d decode_mask_len=%d",
              prefill_len_, kv_cache_len_, kv_stride_, decode_mask_len_);
    }
    catch (const std::exception &e)
    {
        ALOGE("failed to infer code predictor shapes: %s", e.what());
        return false;
    }
    return true;
}

void CodePredictorRunner::deinit()
{
    for (auto &l : layers_) l->deinit();
    for (auto &h : lm_heads_) h->deinit();
    post_.deinit();
    layers_.clear();
    lm_heads_.clear();
    for (auto &e : embeddings_) e->Deinit();
    embeddings_.clear();
}

std::vector<unsigned short> CodePredictorRunner::embed(int table, int id)
{
    std::vector<unsigned short> out(kTalkerHidden);
    embeddings_[(size_t)table]->getByIndex((unsigned int)id, out.data());
    return out;
}

std::vector<int> CodePredictorRunner::generate(const std::vector<unsigned short> &talker_hidden_norm,
                                               const std::vector<unsigned short> &first_code_embed,
                                               const Qwen3TTSGenerateOptions &opt,
                                               std::mt19937 &rng)
{
    ALOGI("code predictor generate start: hidden_norm_elems=%zu first_code_embed_elems=%zu",
          talker_hidden_norm.size(), first_code_embed.size());
    std::vector<unsigned short> data((size_t)prefill_len_ * kTalkerHidden, 0);
    std::memcpy(data.data(), talker_hidden_norm.data(), (size_t)kTalkerHidden * sizeof(unsigned short));
    std::memcpy(data.data() + kTalkerHidden, first_code_embed.data(), (size_t)kTalkerHidden * sizeof(unsigned short));

    std::vector<unsigned int> indices((size_t)prefill_len_, 0);
    std::iota(indices.begin(), indices.end(), 0);
    for (size_t i = 2; i < indices.size(); ++i) indices[i] = 0;

    const auto &first_mask = input_tensor(layers_.front()->get(), 1, "mask");
    const int prefill_mask_elems = tensor_elems_bf16(first_mask);
    const int prefill_mask_cols = prefill_mask_elems / prefill_len_;
    std::vector<unsigned short> mask((size_t)prefill_mask_elems, bfloat16(-65536.0f).data);
    mask[0] = 0;
    if (prefill_mask_cols >= 2)
    {
        mask[(size_t)prefill_mask_cols] = 0;
        mask[(size_t)prefill_mask_cols + 1] = 0;
    }

    k_cache_.assign(kCodePredictorLayers, std::vector<unsigned short>((size_t)kv_cache_len_ * (size_t)kv_stride_, 0));
    v_cache_.assign(kCodePredictorLayers, std::vector<unsigned short>((size_t)kv_cache_len_ * (size_t)kv_stride_, 0));

    for (int layer = 0; layer < kCodePredictorLayers; ++layer)
    {
        auto &runner = layers_[(size_t)layer]->get();
        ALOGI("code predictor prefill layer=%d/%d write inputs", layer, kCodePredictorLayers);
        const auto &t_idx = input_tensor(runner, 1, "indices");
        const auto &t_in = input_tensor(runner, 1, "input");
        const auto &t_mask = input_tensor(runner, 1, "mask");
        const auto &t_k = input_tensor(runner, 1, "K_cache");
        const auto &t_v = input_tensor(runner, 1, "V_cache");
        ALOGI("code predictor prefill tensors: layer=%d idx=%s/%s/%d input=%s/%s/%d mask=%s/%s/%d K=%s/%s/%d V=%s/%s/%d",
              layer,
              t_idx.sName.c_str(), shape_to_string(t_idx.vShape).c_str(), t_idx.nSize,
              t_in.sName.c_str(), shape_to_string(t_in.vShape).c_str(), t_in.nSize,
              t_mask.sName.c_str(), shape_to_string(t_mask.vShape).c_str(), t_mask.nSize,
              t_k.sName.c_str(), shape_to_string(t_k.vShape).c_str(), t_k.nSize,
              t_v.sName.c_str(), shape_to_string(t_v.vShape).c_str(), t_v.nSize);
        write_tensor(runner, t_idx, indices.data(), indices.size() * sizeof(unsigned int));
        write_tensor(runner, t_in, data.data(), data.size() * sizeof(unsigned short));
        write_tensor(runner, t_mask, mask.data(), mask.size() * sizeof(unsigned short));
        std::vector<unsigned short> zeros_k((size_t)t_k.nSize / sizeof(unsigned short), 0);
        std::vector<unsigned short> zeros_v((size_t)t_v.nSize / sizeof(unsigned short), 0);
        write_tensor(runner, t_k, zeros_k.data(), zeros_k.size() * sizeof(unsigned short));
        write_tensor(runner, t_v, zeros_v.data(), zeros_v.size() * sizeof(unsigned short));
        ALOGI("code predictor prefill inference begin: layer=%d gid=1", layer);
        const int ret = runner.inference(1);
        ALOGI("code predictor prefill inference end: layer=%d gid=1 ret=%d", layer, ret);
        if (ret != 0) throw std::runtime_error("code predictor prefill inference failed");
        const auto &out_k = output_tensor(runner, 1, "K_cache_out");
        const auto &out_v = output_tensor(runner, 1, "V_cache_out");
        const auto &out_h = output_tensor(runner, 1, "output");
        ALOGI("code predictor prefill read outputs: layer=%d K=%s/%s/%d V=%s/%s/%d H=%s/%s/%d",
              layer,
              out_k.sName.c_str(), shape_to_string(out_k.vShape).c_str(), out_k.nSize,
              out_v.sName.c_str(), shape_to_string(out_v.vShape).c_str(), out_v.nSize,
              out_h.sName.c_str(), shape_to_string(out_h.vShape).c_str(), out_h.nSize);
        read_tensor(runner, out_k, k_cache_[(size_t)layer].data(), std::min((size_t)out_k.nSize, k_cache_[(size_t)layer].size() * sizeof(unsigned short)));
        read_tensor(runner, out_v, v_cache_[(size_t)layer].data(), std::min((size_t)out_v.nSize, v_cache_[(size_t)layer].size() * sizeof(unsigned short)));
        read_tensor(runner, out_h, data.data(), std::min((size_t)out_h.nSize, data.size() * sizeof(unsigned short)));
    }

    std::vector<unsigned short> last_hidden(data.begin() + kTalkerHidden, data.begin() + (size_t)2 * kTalkerHidden);
    std::vector<int> ids;
    int current_len = 2;
    for (int step = 0; step < kNumSubCodes; ++step)
    {
        ALOGI("code predictor subcode step=%d current_len=%d begin", step, current_len);
        if (step > 0)
        {
            auto prev_embed = embed(step - 1, ids.back());
            last_hidden = decode_one(prev_embed, current_len);
            ++current_len;
        }
        auto hidden_norm = run_post_norm(last_hidden);
        auto logits = run_lm_head(step, hidden_norm);
        ids.push_back(select_from_logits(logits, opt.subtalker_do_sample, opt.subtalker_top_k,
                                         opt.subtalker_top_p, opt.subtalker_temperature, rng));
        ALOGI("code predictor subcode step=%d sampled=%d", step, ids.back());
    }
    ALOGI("code predictor generate done: ids=%zu", ids.size());
    return ids;
}

std::vector<unsigned short> CodePredictorRunner::decode_one(const std::vector<unsigned short> &embed, int current_len)
{
    ALOGI("code predictor decode_one start: current_len=%d embed_elems=%zu", current_len, embed.size());
    if (current_len >= kv_cache_len_) throw std::runtime_error("code predictor decode reaches kv cache limit");
    std::vector<unsigned short> data = embed;
    std::vector<unsigned short> mask((size_t)decode_mask_len_, bfloat16(-65536.0f).data);
    for (int i = 0; i < current_len && i + 1 < decode_mask_len_; ++i) mask[(size_t)i] = 0;
    mask.back() = 0;
    const unsigned int idx = (unsigned int)current_len;
    for (int layer = 0; layer < kCodePredictorLayers; ++layer)
    {
        auto &runner = layers_[(size_t)layer]->get();
        ALOGI("code predictor decode_one layer=%d/%d write inputs idx=%u", layer, kCodePredictorLayers, idx);
        write_tensor(runner, input_tensor(runner, 0, "K_cache"), k_cache_[(size_t)layer].data(), k_cache_[(size_t)layer].size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "V_cache"), v_cache_[(size_t)layer].data(), v_cache_[(size_t)layer].size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "indices"), &idx, sizeof(idx));
        write_tensor(runner, input_tensor(runner, 0, "input"), data.data(), data.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "mask"), mask.data(), mask.size() * sizeof(unsigned short));
        ALOGI("code predictor decode_one inference begin: layer=%d gid=0", layer);
        const int ret = runner.inference(0);
        ALOGI("code predictor decode_one inference end: layer=%d gid=0 ret=%d", layer, ret);
        if (ret != 0) throw std::runtime_error("code predictor decode inference failed");
        read_tensor(runner, output_tensor(runner, 0, "K_cache_out"), k_cache_[(size_t)layer].data() + (size_t)current_len * (size_t)kv_stride_, (size_t)kv_stride_ * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 0, "V_cache_out"), v_cache_[(size_t)layer].data() + (size_t)current_len * (size_t)kv_stride_, (size_t)kv_stride_ * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 0, "output"), data.data(), data.size() * sizeof(unsigned short));
    }
    ALOGI("code predictor decode_one done: current_len=%d", current_len);
    return data;
}

std::vector<float> CodePredictorRunner::run_post_norm(const std::vector<unsigned short> &hidden)
{
    auto &runner = post_.get();
    const auto &input = runner.get_input("input");
    ALOGI("code predictor post_norm write input: elems=%zu tensor=%s shape=%s nSize=%d",
          hidden.size(), input.sName.c_str(), shape_to_string(input.vShape).c_str(), input.nSize);
    write_tensor(runner, input, hidden.data(), hidden.size() * sizeof(unsigned short));
    ALOGI("code predictor post_norm inference begin");
    const int ret = runner.inference();
    ALOGI("code predictor post_norm inference end ret=%d", ret);
    if (ret != 0) throw std::runtime_error("code predictor post inference failed");
    const auto &output = runner.get_output("output_norm");
    ALOGI("code predictor post_norm read output: tensor=%s shape=%s nSize=%d",
          output.sName.c_str(), shape_to_string(output.vShape).c_str(), output.nSize);
    return tensor_to_float(runner, output, kTalkerHidden);
}

std::vector<float> CodePredictorRunner::run_lm_head(int step, const std::vector<float> &hidden)
{
    auto &runner = lm_heads_[(size_t)step]->get();
    const auto &input = runner.get_input("input");
    ALOGI("code predictor lm_head write input: step=%d elems=%zu tensor=%s shape=%s nSize=%d",
          step, hidden.size(), input.sName.c_str(), shape_to_string(input.vShape).c_str(), input.nSize);
    write_tensor(runner, input, hidden.data(), hidden.size() * sizeof(float));
    ALOGI("code predictor lm_head inference begin: step=%d", step);
    const int ret = runner.inference();
    ALOGI("code predictor lm_head inference end: step=%d ret=%d", step, ret);
    if (ret != 0) throw std::runtime_error("code predictor lm head inference failed");
    const auto &output = runner.get_output("output");
    ALOGI("code predictor lm_head read output: step=%d tensor=%s shape=%s nSize=%d",
          step, output.sName.c_str(), shape_to_string(output.vShape).c_str(), output.nSize);
    return tensor_to_float(runner, output, kCodecVocab);
}

} // namespace qwen3_tts
