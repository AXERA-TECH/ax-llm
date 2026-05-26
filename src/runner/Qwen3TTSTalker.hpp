#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Qwen3TTSCommon.hpp"

namespace qwen3_tts {

class TalkerRunner
{
public:
    struct PostResult
    {
        std::vector<float> logits;
        std::vector<unsigned short> hidden_norm;
    };

    bool init(const std::string &talker_dir);
    void deinit();

    int kv_cache_len() const;
    std::vector<unsigned short> prefill(const std::vector<unsigned short> &embeds, int valid_len);
    std::vector<unsigned short> decode_one(const std::vector<unsigned short> &embed);
    PostResult post(const std::vector<unsigned short> &raw_hidden);

private:
    std::vector<std::unique_ptr<Runner>> layers_;
    Runner post_;
    std::vector<std::vector<unsigned short>> k_cache_;
    std::vector<std::vector<unsigned short>> v_cache_;
    int current_len_ = 0;
    int prefill_len_ = 0;
    int max_prefill_chunks_ = 0;
    int kv_cache_len_ = 0;
    int kv_stride_ = 0;
    int decode_mask_len_ = 0;
};

} // namespace qwen3_tts
