#include <vector>
#include <algorithm>
#include <optional>
#include <cassert>
#include "mrope.hpp"

#include <iostream>
#include <vector>
#include <numeric>  // std::iota

#include <vector>
#include <algorithm>
#include <limits>  // 用于std::numeric_limits
#include <stdexcept>  // 用于异常处理

int findMaxIn2DVector(const std::vector<std::vector<int>>& vec) {
    if (vec.empty()) {
        throw std::invalid_argument("输入二维vector为空");
    }

    int max_value = std::numeric_limits<int>::min();  // 初始化为最小值
    bool has_elements = false;

    for (const auto& subvec : vec) {
        if (!subvec.empty()) {
            has_elements = true;
            // 使用std::max_element获取子vector的最大值
            int sub_max = *std::max_element(subvec.begin(), subvec.end());
            if (sub_max > max_value) {
                max_value = sub_max;
            }
        }
    }

    if (!has_elements) {
        throw std::invalid_argument("所有子vector均为空");  // 处理全空子vector
    }

    return max_value;
}

// 生成范围序列 [0, text_len-1]
std::vector<int> generateRange(int text_len, int start) {
    std::vector<int> range(text_len);
    std::iota(range.begin(), range.end(), start);  // 填充从0开始的序列
    return range;
}

// 扩展为多行矩阵
std::vector<std::vector<int>> expandToMatrix(const std::vector<int>& range, int rows) {
    std::vector<std::vector<int>> matrix(rows, range);  // 每一行都是range的副本
    return matrix;
}

// 生成多维索引 (unused in Qwen3, but kept for completeness)
std::vector<std::vector<int>> generateIndices(int grid_t, int grid_h, int grid_w) {
    std::vector<std::vector<int>> indices(3, std::vector<int>(grid_t * grid_h * grid_w));

    int idx = 0;
    for (int t = 0; t < grid_t; ++t) {
        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {
                indices[0][idx] = t;  // 时间索引
                indices[1][idx] = h;  // 高度索引
                indices[2][idx] = w;  // 宽度索引
                ++idx;
            }
        }
    }

    return indices;
}

// Qwen3-VL specific: Preprocess video_grid_thw by repeating each row t times and setting t=1
std::vector<std::vector<int>> preprocessVideoGrid(const std::vector<std::vector<int>>& video_grid_thw) {
    std::vector<std::vector<int>> processed;
    for (const auto& grid : video_grid_thw) {
        if (grid.size() != 3) {
            throw std::invalid_argument("Invalid grid format");
        }
        int t = grid[0];
        // Repeat the row t times
        for (int i = 0; i < t; ++i) {
            std::vector<int> repeated_grid = {1, grid[1], grid[2]};  // Set t=1
            processed.push_back(repeated_grid);
        }
    }
    return processed;
}

std::vector<std::vector<int>> get_rope_index(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw
) {
    const int spatial_merge_size = config.vision_config.spatial_merge_size;
    const int image_token_id = config.image_token_id;
    const int video_token_id = config.video_token_id;
    const int vision_start_token_id = config.vision_start_token_id;

    std::vector<std::vector<int>> position_ids(3);

    // Preprocess video_grid_thw for Qwen3-VL (split into single-frame segments with timestamps)
    auto processed_video_grid = preprocessVideoGrid(video_grid_thw);

    // Handle pure text case
    if (input_ids.empty() || (image_grid_thw.empty() && video_grid_thw.empty())) {
        int b = 0;
        for (int i = 0; i < 3; ++i) {
            std::vector<int> seq(input_ids.size());
            // 手动实现递增序列
            for (size_t j = 0; j < seq.size(); ++j) {
                seq[j] = j;
            }
            position_ids[i].insert(position_ids[i].end(), seq.begin(), seq.end());
        }
        return position_ids;
    }

    // Multimodal case (batch_size=1, single sequence)
    const auto& ids = input_ids;
    // Assume full mask for simplicity (as in Qwen3 fallback)
    const auto mask = std::vector<int>(ids.size(), 1);

    // Filter valid tokens (masked)
    std::vector<int> filtered_ids;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (mask[i]) filtered_ids.push_back(ids[i]);
    }

    int image_nums = 0, video_nums = 0;

    for (size_t i = 0; i < filtered_ids.size() - 1; ++i) {
        if (filtered_ids[i] == vision_start_token_id) {
            if (filtered_ids[i + 1] == config.image_token_id) {
                image_nums++;
            }
            if (filtered_ids[i + 1] == config.video_token_id) {
                video_nums++;
            }
        }
    }

    int image_index = 0, video_index = 0;
    std::vector<std::vector<int>> batch_pos(3);
    int st = 0;
    int remain_images = image_nums;
    int remain_videos = video_nums;
    std::vector<std::vector<std::vector<int>>> llm_pos_ids_list;

    // Loop over vision blocks (images + videos, now with processed_video_grid)
    for (size_t i_ = 0; i_ < static_cast<size_t>(image_nums + video_nums); ++i_) {

        int ed_image = filtered_ids.size() + 1;
        int ed_video = filtered_ids.size() + 1;

        if (remain_images > 0) {
            for (size_t j = st; j < filtered_ids.size(); ++j) {
                if (filtered_ids[j] == config.image_token_id) {
                    ed_image = static_cast<int>(j);
                    break;
                }
            }
        }

        if (remain_videos > 0) {
            for (size_t j = st; j < filtered_ids.size(); ++j) {
                if (filtered_ids[j] == config.video_token_id) {
                    ed_video = static_cast<int>(j);
                    break;
                }
            }
        }

        int t, h, w;
        int ed;

        if (ed_image < ed_video) {
            // Image
            t = image_grid_thw[image_index][0];
            h = image_grid_thw[image_index][1];
            w = image_grid_thw[image_index][2];
            image_index += 1;
            remain_images -= 1;
            ed = ed_image;
        } else {
            // Video (using processed grid, each entry has t=1)
            t = processed_video_grid[video_index][0];  // 1
            h = processed_video_grid[video_index][1];
            w = processed_video_grid[video_index][2];
            video_index += 1;
            remain_videos -= 1;
            ed = ed_video;
        }

        int llm_grid_t = t;  // For videos, t=1
        int llm_grid_h = h / spatial_merge_size;
        int llm_grid_w = w / spatial_merge_size;

        int text_len = ed - st;

        int st_idx;
        if (llm_pos_ids_list.empty()) {
            st_idx = 0;
        } else {
            st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
        }
        auto range = generateRange(text_len, st_idx);
        auto expanded_matrix = expandToMatrix(range, 3);

        llm_pos_ids_list.push_back(expanded_matrix);

        // For Qwen3: t_index always starts from 0 (no scaling, timestamps handle time)
        std::vector<int> t_index;
        for (int ti = 0; ti < llm_grid_t; ++ti) {  // llm_grid_t=1 for videos/images typically
            for (int hw = 0; hw < llm_grid_h * llm_grid_w; ++hw) {
                t_index.push_back(ti + text_len + st_idx);  // No second_per_grid_t scaling; ti is 0 for t=1
            }
        }

        std::vector<int> h_index;
        for (int ti = 0; ti < llm_grid_t; ++ti) {
            for (int hi = 0; hi < llm_grid_h; ++hi) {
                for (int wi = 0; wi < llm_grid_w; ++wi) {
                    h_index.push_back(hi + text_len + st_idx);
                }
            }
        }

        std::vector<int> w_index;
        for (int ti = 0; ti < llm_grid_t; ++ti) {
            for (int hi = 0; hi < llm_grid_h; ++hi) {
                for (int wi = 0; wi < llm_grid_w; ++wi) {
                    w_index.push_back(wi + text_len + st_idx);
                }
            }
        }

        std::vector<std::vector<int>> thw_idx;
        thw_idx.push_back(t_index);
        thw_idx.push_back(h_index);
        thw_idx.push_back(w_index);
        llm_pos_ids_list.push_back(thw_idx);

        st = ed + llm_grid_t * llm_grid_h * llm_grid_w;

        // Append remaining text if any
        if (st < static_cast<int>(filtered_ids.size())) {
            if (llm_pos_ids_list.empty()) {
                st_idx = 0;
            } else {
                st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
            }

            text_len = static_cast<int>(filtered_ids.size()) - st;

            range = generateRange(text_len, st_idx);
            expanded_matrix = expandToMatrix(range, 3);
            llm_pos_ids_list.push_back(expanded_matrix);
        }
    }

    // Concatenate all position lists
    for (const auto& item : llm_pos_ids_list) {
        for (size_t pi = 0; pi < position_ids.size(); ++pi) {
            position_ids[pi].insert(position_ids[pi].end(), item[pi].begin(), item[pi].end());
        }
    }

    return position_ids;
}