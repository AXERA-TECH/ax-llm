#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Qwen3TTSGenerateOptions
{
    std::string text;
    std::string language = "Chinese";
    std::string ref_audio;
    std::string ref_text;
    std::string output_wav = "qwen3_tts_cpp_voice_clone.wav";

    bool x_vector_only_mode = false;
    bool non_streaming_mode = false;
    bool do_sample = true;
    bool subtalker_do_sample = true;
    int max_new_tokens = 4096;
    int top_k = 50;
    float top_p = 1.0f;
    float temperature = 0.9f;
    int subtalker_top_k = 50;
    float subtalker_top_p = 1.0f;
    float subtalker_temperature = 0.9f;
    float repetition_penalty = 1.05f;
    unsigned int seed = 1234;
};

class Qwen3TTS
{
public:
    Qwen3TTS();
    ~Qwen3TTS();

    bool Init(const std::string &model_dir);
    void Deinit();

    bool GenerateVoiceClone(const Qwen3TTSGenerateOptions &options,
                            std::vector<float> &wav,
                            int &sample_rate);

    static bool SaveWavFloat32(const std::string &path,
                               const std::vector<float> &wav,
                               int sample_rate);

private:
    struct Impl;
    Impl *impl_ = nullptr;
};
