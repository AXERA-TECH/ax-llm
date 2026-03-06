#pragma once

#include <vector>

namespace mrope {

struct Config {
    struct VisionConfig {
        int temporal_patch_size = 2;
        int tokens_per_second = 1;
        int spatial_merge_size = 2;
        int patch_size = 14;
        int width = 448;
        int height = 448;
        int fps = 1;
    };
    VisionConfig vision_config;
    int image_token_id = -1;
    int video_token_id = -1;
    int vision_start_token_id = -1;

    std::vector<std::vector<int>> image_grid_thw;
    std::vector<std::vector<int>> video_grid_thw;
};

// Qwen2.5-VL style rope index (supports optional time scaling for video grid_t).
std::vector<std::vector<int>> get_rope_index_qwen2_5(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw,
    const std::vector<double>& second_per_grid_ts);

// Qwen3-VL style rope index:
// preprocesses `video_grid_thw` by splitting each {t,h,w} into t rows of {1,h,w}.
std::vector<std::vector<int>> get_rope_index_qwen3(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw);

} // namespace mrope

