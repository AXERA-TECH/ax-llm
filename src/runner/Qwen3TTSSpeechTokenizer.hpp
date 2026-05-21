#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "Qwen3TTSCommon.hpp"

namespace qwen3_tts {

class SpeechTokenizerRunner
{
public:
    bool init(const std::string &dir);
    void deinit();

    std::vector<std::array<int, kCodeGroups>> encode(const std::vector<float> &wav24);
    std::vector<float> decode(const std::vector<std::array<int, kCodeGroups>> &codes);

private:
    std::vector<float> decode_pre(const std::vector<std::array<int, kCodeGroups>> &codes, int code_len);
    std::vector<unsigned short> run_transformer(const std::vector<float> &hidden512, int code_len);
    std::vector<float> run_post(const std::vector<unsigned short> &hidden512, int code_len);
    std::vector<float> run_decode_post_chunk(const float *hidden, int valid_len);
    std::vector<float> run_decode_post(const std::vector<float> &hidden1024, int code_len);

    std::string model_dir_;
    Runner encoder_;
    Runner decode_pre_;
    std::vector<std::unique_ptr<Runner>> layers_;
    Runner post_;
    Runner decode_post_;
};

} // namespace qwen3_tts
