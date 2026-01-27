#ifndef MROPE_QWEN3_H
#define MROPE_QWEN3_H

#include <vector>

// Forward declaration of Config (assume defined in mrope.hpp or utils.hpp)
struct Config {
    struct VisionConfig {
        int temporal_patch_size = 0;
        int tokens_per_second = 0;
        int spatial_merge_size = 0;
        int patch_size = 0;
        int width = 0;
        int height = 0;
        int fps = 0;
    };
    VisionConfig vision_config;
    int image_token_id = 0;
    int video_token_id = 0;
    int vision_start_token_id = 0;

    std::vector<std::vector<int>> image_grid_thw;   // auto calc
    std::vector<std::vector<int>> video_grid_thw;   // auto calc
};

std::vector<std::vector<int>> get_rope_index(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw
);

#endif // MROPE_QWEN3_H
