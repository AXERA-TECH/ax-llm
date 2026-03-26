#include "mrope.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mrope {

static int findMaxIn2DVector(const std::vector<std::vector<int>>& vec) {
    if (vec.empty()) throw std::invalid_argument("empty 2d vector");
    int max_value = std::numeric_limits<int>::min();
    bool has_elements = false;
    for (const auto& subvec : vec) {
        if (subvec.empty()) continue;
        has_elements = true;
        int sub_max = *std::max_element(subvec.begin(), subvec.end());
        if (sub_max > max_value) max_value = sub_max;
    }
    if (!has_elements) throw std::invalid_argument("all sub vectors empty");
    return max_value;
}

static std::vector<int> generateRange(int text_len, int start) {
    std::vector<int> range(text_len);
    std::iota(range.begin(), range.end(), start);
    return range;
}

static std::vector<std::vector<int>> expandToMatrix(const std::vector<int>& range, int rows) {
    return std::vector<std::vector<int>>(rows, range);
}

static std::vector<std::vector<int>> preprocessVideoGridQwen3(const std::vector<std::vector<int>>& video_grid_thw) {
    std::vector<std::vector<int>> processed;
    for (const auto& grid : video_grid_thw) {
        if (grid.size() != 3) throw std::invalid_argument("invalid grid format");
        int t = grid[0];
        for (int i = 0; i < t; ++i) processed.push_back({1, grid[1], grid[2]});
    }
    return processed;
}

static std::vector<std::vector<int>> get_rope_index_impl(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw,
    const std::vector<double>* second_per_grid_ts,
    bool preprocess_video_grid_qwen3)
{
    const int spatial_merge_size = config.vision_config.spatial_merge_size;
    const int vision_start_token_id = config.vision_start_token_id;

    std::vector<std::vector<int>> position_ids(3);

    if (input_ids.empty() || (image_grid_thw.empty() && video_grid_thw.empty())) {
        for (int i = 0; i < 3; ++i) {
            std::vector<int> seq(input_ids.size());
            for (size_t j = 0; j < seq.size(); ++j) seq[j] = (int)j;
            position_ids[i].insert(position_ids[i].end(), seq.begin(), seq.end());
        }
        return position_ids;
    }

    const auto& ids = input_ids;
    const auto mask = std::vector<int>(ids.size(), 1);

    std::vector<int> filtered_ids;
    filtered_ids.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) if (mask[i]) filtered_ids.push_back(ids[i]);

    // Qwen3 uses processed video grid (each entry is t=1); Qwen2.5 uses original grid.
    std::vector<std::vector<int>> processed_video_grid =
        preprocess_video_grid_qwen3 ? preprocessVideoGridQwen3(video_grid_thw) : video_grid_thw;

    int image_nums = 0, video_nums = 0;
    for (size_t i = 0; i + 1 < filtered_ids.size(); ++i) {
        if (filtered_ids[i] != vision_start_token_id) continue;
        if (filtered_ids[i + 1] == config.image_token_id) image_nums++;
        if (filtered_ids[i + 1] == config.video_token_id) video_nums++;
    }

    int image_index = 0, video_index = 0;
    int st = 0;
    int remain_images = image_nums;
    int remain_videos = video_nums;

    std::vector<std::vector<std::vector<int>>> llm_pos_ids_list;
    llm_pos_ids_list.reserve((image_nums + video_nums) * 2 + 1);

    for (size_t blk = 0; blk < (size_t)(image_nums + video_nums); ++blk) {
        int ed_image = (int)filtered_ids.size() + 1;
        int ed_video = (int)filtered_ids.size() + 1;

        if (remain_images > 0) {
            for (size_t j = (size_t)st; j < filtered_ids.size(); ++j) {
                if (filtered_ids[j] == config.image_token_id) { ed_image = (int)j; break; }
            }
        }
        if (remain_videos > 0) {
            for (size_t j = (size_t)st; j < filtered_ids.size(); ++j) {
                if (filtered_ids[j] == config.video_token_id) { ed_video = (int)j; break; }
            }
        }

        int t = 1, h = 1, w = 1;
        double second_per_grid_t = 0.0;
        int ed = 0;

        if (ed_image < ed_video) {
            t = image_grid_thw[image_index][0];
            h = image_grid_thw[image_index][1];
            w = image_grid_thw[image_index][2];
            second_per_grid_t = 0.0;
            image_index++;
            remain_images--;
            ed = ed_image;
        } else {
            t = processed_video_grid[video_index][0];
            h = processed_video_grid[video_index][1];
            w = processed_video_grid[video_index][2];
            if (second_per_grid_ts && (size_t)video_index < second_per_grid_ts->size()) {
                second_per_grid_t = (*second_per_grid_ts)[video_index];
            } else {
                second_per_grid_t = 1.0;
            }
            video_index++;
            remain_videos--;
            ed = ed_video;
        }

        int llm_grid_t = t;
        int llm_grid_h = h / spatial_merge_size;
        int llm_grid_w = w / spatial_merge_size;
        int text_len = ed - st;

        int st_idx = llm_pos_ids_list.empty() ? 0 : (findMaxIn2DVector(llm_pos_ids_list.back()) + 1);
        auto range = generateRange(text_len, st_idx);
        llm_pos_ids_list.push_back(expandToMatrix(range, 3));

        std::vector<int> t_index;
        t_index.reserve((size_t)llm_grid_t * llm_grid_h * llm_grid_w);
        for (int ti = 0; ti < llm_grid_t; ti++) {
            for (int hw = 0; hw < llm_grid_h * llm_grid_w; hw++) {
                // Qwen2.5 uses time scaling; Qwen3 passes second_per_grid_t=1 and uses t=1 after preprocess.
                int v = (int)(ti * second_per_grid_t * config.vision_config.tokens_per_second) + text_len + st_idx;
                t_index.push_back(v);
            }
        }

        std::vector<int> h_index;
        h_index.reserve((size_t)llm_grid_t * llm_grid_h * llm_grid_w);
        for (int ti = 0; ti < llm_grid_t; ti++) {
            for (int hi = 0; hi < llm_grid_h; hi++) {
                for (int wi = 0; wi < llm_grid_w; wi++) h_index.push_back(hi + text_len + st_idx);
            }
        }

        std::vector<int> w_index;
        w_index.reserve((size_t)llm_grid_t * llm_grid_h * llm_grid_w);
        for (int ti = 0; ti < llm_grid_t; ti++) {
            for (int hi = 0; hi < llm_grid_h; hi++) {
                for (int wi = 0; wi < llm_grid_w; wi++) w_index.push_back(wi + text_len + st_idx);
            }
        }

        llm_pos_ids_list.push_back({std::move(t_index), std::move(h_index), std::move(w_index)});

        st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
    }

    // Append remaining text after all images/videos.
    if (st < (int)filtered_ids.size()) {
        int st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
        int text_len = (int)filtered_ids.size() - st;
        auto range = generateRange(text_len, st_idx);
        llm_pos_ids_list.push_back(expandToMatrix(range, 3));
    }

    // Flatten to position_ids(3 x seq_len)
    for (const auto& blk : llm_pos_ids_list) {
        if (blk.size() != 3) throw std::runtime_error("invalid pos ids block");
        for (int r = 0; r < 3; ++r) {
            position_ids[r].insert(position_ids[r].end(), blk[r].begin(), blk[r].end());
        }
    }

    return position_ids;
}

std::vector<std::vector<int>> get_rope_index_qwen2_5(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw,
    const std::vector<double>& second_per_grid_ts)
{
    return get_rope_index_impl(config, input_ids, image_grid_thw, video_grid_thw, &second_per_grid_ts, false);
}

std::vector<std::vector<int>> get_rope_index_qwen3(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw)
{
    return get_rope_index_impl(config, input_ids, image_grid_thw, video_grid_thw, nullptr, true);
}

} // namespace mrope

