#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace smolvla {

struct Config {
    std::string image_encoder_axmodel;
    std::string state_proj_axmodel;
    std::string action_embed_axmodel;
    std::string action_out_axmodel;
    std::string llm_template_axmodel;
    std::string llm_post_axmodel;
    std::string tokens_embed;

    int tokens_embed_num = 49280;
    int vlm_hidden_size = 960;
    int expert_hidden_size = 720;
    int kv_cache_size = 320;
    int num_layers = 16;
    int num_images = 3;
    int image_width = 512;
    int image_height = 512;
    int image_tokens = 64;
    int language_tokens = 48;
    int prefix_len = 256;
    int chunk_size = 50;
    int state_dim = 32;
    int action_dim = 32;
    int output_action_dim = 0;
    int num_steps = 10;
    int seed = 0;
    bool use_mmap_embed = false;
    std::vector<float> action_mean;
    std::vector<float> action_std;
};

struct Input {
    // Normalized LeRobot image tensor, flattened as
    // [num_images, 3, image_height, image_width] in RGB/NCHW, range [-1, 1].
    std::vector<float> images;

    // Token ids from the SmolVLA processor. Length is padded/truncated to language_tokens.
    std::vector<int> language_tokens;
    std::vector<uint8_t> language_mask;

    // Normalized and padded robot state. Length is padded/truncated to state_dim.
    std::vector<float> state;

    // Optional initial noise, flattened as [chunk_size, action_dim].
    std::vector<float> noise;
};

class Runner {
public:
    Runner();
    ~Runner();

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    bool Init(const Config& cfg, int devid = -1);
    void Deinit();

    bool Predict(const Input& input, std::vector<float>& out_actions);
    const std::string& LastError() const { return last_error_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    std::string last_error_;
};

bool LoadConfigJson(const std::string& path, Config& cfg, std::string& err);

} // namespace smolvla
