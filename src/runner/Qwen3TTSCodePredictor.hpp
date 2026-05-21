#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "LLMEmbedSelector.hpp"
#include "Qwen3TTSCommon.hpp"

struct Qwen3TTSGenerateOptions;

namespace qwen3_tts {

class CodePredictorRunner
{
public:
    bool init(const std::string &dir);
    void deinit();

    std::vector<unsigned short> embed(int table, int id);
    std::vector<int> generate(const std::vector<unsigned short> &talker_hidden_norm,
                              const std::vector<unsigned short> &first_code_embed,
                              const Qwen3TTSGenerateOptions &opt,
                              std::mt19937 &rng);

private:
    std::vector<unsigned short> decode_one(const std::vector<unsigned short> &embed, int current_len);
    std::vector<float> run_post_norm(const std::vector<unsigned short> &hidden);
    std::vector<float> run_lm_head(int step, const std::vector<float> &hidden);

    std::vector<std::unique_ptr<Runner>> layers_;
    Runner post_;
    std::vector<std::unique_ptr<Runner>> lm_heads_;
    std::vector<std::unique_ptr<LLaMaEmbedSelector>> embeddings_;
    std::vector<std::vector<unsigned short>> k_cache_;
    std::vector<std::vector<unsigned short>> v_cache_;
};

} // namespace qwen3_tts
