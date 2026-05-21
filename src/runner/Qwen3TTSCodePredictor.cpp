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
        auto path = join_path(dir, "qwen3_tts_talker_code_predictor_p64_l" + std::to_string(i) + "_together.axmodel");
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
    std::vector<unsigned short> data((size_t)64 * kTalkerHidden, 0);
    std::memcpy(data.data(), talker_hidden_norm.data(), (size_t)kTalkerHidden * sizeof(unsigned short));
    std::memcpy(data.data() + kTalkerHidden, first_code_embed.data(), (size_t)kTalkerHidden * sizeof(unsigned short));

    std::vector<unsigned int> indices(64, 0);
    std::iota(indices.begin(), indices.end(), 0);
    for (size_t i = 2; i < indices.size(); ++i) indices[i] = 0;

    std::vector<unsigned short> mask((size_t)64 * 64, bfloat16(-65536.0f).data);
    mask[0] = 0;
    mask[64] = 0;
    mask[65] = 0;

    k_cache_.assign(kCodePredictorLayers, std::vector<unsigned short>((size_t)128 * kTalkerHidden, 0));
    v_cache_.assign(kCodePredictorLayers, std::vector<unsigned short>((size_t)128 * kTalkerHidden, 0));

    for (int layer = 0; layer < kCodePredictorLayers; ++layer)
    {
        auto &runner = layers_[(size_t)layer]->get();
        write_tensor(runner, input_tensor(runner, 1, "indices"), indices.data(), indices.size() * sizeof(unsigned int));
        write_tensor(runner, input_tensor(runner, 1, "input"), data.data(), data.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 1, "mask"), mask.data(), mask.size() * sizeof(unsigned short));
        std::vector<unsigned short> zeros_k((size_t)input_tensor(runner, 1, "K_cache").nSize / sizeof(unsigned short), 0);
        std::vector<unsigned short> zeros_v((size_t)input_tensor(runner, 1, "V_cache").nSize / sizeof(unsigned short), 0);
        write_tensor(runner, input_tensor(runner, 1, "K_cache"), zeros_k.data(), zeros_k.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 1, "V_cache"), zeros_v.data(), zeros_v.size() * sizeof(unsigned short));
        if (runner.inference(1) != 0) throw std::runtime_error("code predictor prefill inference failed");
        read_tensor(runner, output_tensor(runner, 1, "K_cache_out"), k_cache_[(size_t)layer].data(), (size_t)64 * kTalkerHidden * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 1, "V_cache_out"), v_cache_[(size_t)layer].data(), (size_t)64 * kTalkerHidden * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 1, "output"), data.data(), (size_t)64 * kTalkerHidden * sizeof(unsigned short));
    }

    std::vector<unsigned short> last_hidden(data.begin() + kTalkerHidden, data.begin() + (size_t)2 * kTalkerHidden);
    std::vector<int> ids;
    int current_len = 2;
    for (int step = 0; step < kNumSubCodes; ++step)
    {
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
    }
    return ids;
}

std::vector<unsigned short> CodePredictorRunner::decode_one(const std::vector<unsigned short> &embed, int current_len)
{
    std::vector<unsigned short> data = embed;
    std::vector<unsigned short> mask(129, bfloat16(-65536.0f).data);
    for (int i = 0; i < current_len; ++i) mask[(size_t)i] = 0;
    mask.back() = 0;
    const unsigned int idx = (unsigned int)current_len;
    for (int layer = 0; layer < kCodePredictorLayers; ++layer)
    {
        auto &runner = layers_[(size_t)layer]->get();
        write_tensor(runner, input_tensor(runner, 0, "K_cache"), k_cache_[(size_t)layer].data(), k_cache_[(size_t)layer].size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "V_cache"), v_cache_[(size_t)layer].data(), v_cache_[(size_t)layer].size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "indices"), &idx, sizeof(idx));
        write_tensor(runner, input_tensor(runner, 0, "input"), data.data(), data.size() * sizeof(unsigned short));
        write_tensor(runner, input_tensor(runner, 0, "mask"), mask.data(), mask.size() * sizeof(unsigned short));
        if (runner.inference(0) != 0) throw std::runtime_error("code predictor decode inference failed");
        read_tensor(runner, output_tensor(runner, 0, "K_cache_out"), k_cache_[(size_t)layer].data() + (size_t)current_len * kTalkerHidden, (size_t)kTalkerHidden * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 0, "V_cache_out"), v_cache_[(size_t)layer].data() + (size_t)current_len * kTalkerHidden, (size_t)kTalkerHidden * sizeof(unsigned short));
        read_tensor(runner, output_tensor(runner, 0, "output"), data.data(), data.size() * sizeof(unsigned short));
    }
    return data;
}

std::vector<float> CodePredictorRunner::run_post_norm(const std::vector<unsigned short> &hidden)
{
    auto &runner = post_.get();
    write_tensor(runner, runner.get_input("input"), hidden.data(), hidden.size() * sizeof(unsigned short));
    if (runner.inference() != 0) throw std::runtime_error("code predictor post inference failed");
    return tensor_to_float(runner, runner.get_output("output_norm"), kTalkerHidden);
}

std::vector<float> CodePredictorRunner::run_lm_head(int step, const std::vector<float> &hidden)
{
    auto &runner = lm_heads_[(size_t)step]->get();
    write_tensor(runner, runner.get_input("input"), hidden.data(), hidden.size() * sizeof(float));
    if (runner.inference() != 0) throw std::runtime_error("code predictor lm head inference failed");
    return tensor_to_float(runner, runner.get_output("output"), kCodecVocab);
}

} // namespace qwen3_tts
