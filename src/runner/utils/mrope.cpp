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
    const bool same_image_video_token = (config.image_token_id == config.video_token_id);

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

    auto expected_media_tokens = [&](const std::vector<int>& grid) -> int {
        if (grid.size() != 3) return -1;
        const int t = grid[0];
        const int h = grid[1];
        const int w = grid[2];
        const int llm_grid_t = t;
        const int llm_grid_h = spatial_merge_size > 0 ? (h / spatial_merge_size) : 0;
        const int llm_grid_w = spatial_merge_size > 0 ? (w / spatial_merge_size) : 0;
        if (llm_grid_t <= 0 || llm_grid_h <= 0 || llm_grid_w <= 0) return -1;
        return llm_grid_t * llm_grid_h * llm_grid_w;
    };

    auto fix_length = [](std::vector<int>& v, int want) {
        if (want < 0) want = 0;
        if ((int)v.size() > want) {
            v.resize((size_t)want);
        } else if ((int)v.size() < want) {
            const int pad = v.empty() ? 0 : v.back();
            v.insert(v.end(), (size_t)(want - (int)v.size()), pad);
        }
    };

    int image_nums = 0, video_nums = 0, media_nums = 0;
    for (size_t i = 0; i + 1 < filtered_ids.size(); ++i) {
        if (filtered_ids[i] != vision_start_token_id) continue;
        const int next = filtered_ids[i + 1];
        if (!same_image_video_token) {
            if (next == config.image_token_id) image_nums++;
            else if (next == config.video_token_id) video_nums++;
        } else {
            if (next == config.image_token_id) media_nums++;
        }
    }

    int image_index = 0, video_index = 0;
    int st = 0;

    std::vector<std::vector<std::vector<int>>> llm_pos_ids_list;
    llm_pos_ids_list.reserve((same_image_video_token ? media_nums : (image_nums + video_nums)) * 2 + 1);

    const int media_token_id = same_image_video_token ? config.image_token_id : -1;
    const int total_blocks = same_image_video_token ? media_nums : (image_nums + video_nums);
    int remain_images = image_nums;
    int remain_videos = video_nums;

    for (int blk = 0; blk < total_blocks; ++blk) {
        int ed_image = (int)filtered_ids.size() + 1;
        int ed_video = (int)filtered_ids.size() + 1;
        int ed_media = (int)filtered_ids.size() + 1;

        if (!same_image_video_token) {
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
        } else {
            for (size_t j = (size_t)st; j < filtered_ids.size(); ++j) {
                if (filtered_ids[j] == media_token_id) { ed_media = (int)j; break; }
            }
        }

        int t = 1, h = 1, w = 1;
        double second_per_grid_t = 0.0;
        int ed = 0;
        int vision_tokens = 0;

        if (!same_image_video_token) {
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
            vision_tokens = expected_media_tokens({t, h, w});
            if (vision_tokens <= 0) vision_tokens = 0;
        } else {
            if (ed_media > (int)filtered_ids.size()) break;

            int run_len = 0;
            for (size_t j = (size_t)ed_media; j < filtered_ids.size(); ++j) {
                if (filtered_ids[j] != media_token_id) break;
                run_len++;
            }

            const int exp_img = (image_index < (int)image_grid_thw.size()) ? expected_media_tokens(image_grid_thw[image_index]) : -1;
            const int exp_vid = (video_index < (int)processed_video_grid.size()) ? expected_media_tokens(processed_video_grid[video_index]) : -1;

            bool pick_video = false;
            if (exp_vid > 0 && run_len == exp_vid && (exp_img <= 0 || run_len != exp_img)) {
                pick_video = true;
            } else if (exp_img > 0 && run_len == exp_img && (exp_vid <= 0 || run_len != exp_vid)) {
                pick_video = false;
            } else if (exp_img <= 0 && exp_vid > 0) {
                pick_video = true;
            } else if (exp_vid <= 0 && exp_img > 0) {
                pick_video = false;
            } else if (exp_img > 0 && exp_vid > 0) {
                // Ambiguous: prefer the one that still has grids remaining and matches length best.
                const int d_img = (exp_img > run_len) ? (exp_img - run_len) : (run_len - exp_img);
                const int d_vid = (exp_vid > run_len) ? (exp_vid - run_len) : (run_len - exp_vid);
                pick_video = d_vid < d_img;
            } else {
                // No grids left; fallback to sequential positions without mRoPE for this block.
                pick_video = false;
            }

            if (!pick_video && image_index < (int)image_grid_thw.size()) {
                t = image_grid_thw[image_index][0];
                h = image_grid_thw[image_index][1];
                w = image_grid_thw[image_index][2];
                second_per_grid_t = 0.0;
                image_index++;
            } else if (pick_video && video_index < (int)processed_video_grid.size()) {
                t = processed_video_grid[video_index][0];
                h = processed_video_grid[video_index][1];
                w = processed_video_grid[video_index][2];
                if (second_per_grid_ts && (size_t)video_index < second_per_grid_ts->size()) {
                    second_per_grid_t = (*second_per_grid_ts)[video_index];
                } else {
                    second_per_grid_t = 1.0;
                }
                video_index++;
            } else if (image_index < (int)image_grid_thw.size()) {
                t = image_grid_thw[image_index][0];
                h = image_grid_thw[image_index][1];
                w = image_grid_thw[image_index][2];
                second_per_grid_t = 0.0;
                image_index++;
            } else if (video_index < (int)processed_video_grid.size()) {
                t = processed_video_grid[video_index][0];
                h = processed_video_grid[video_index][1];
                w = processed_video_grid[video_index][2];
                if (second_per_grid_ts && (size_t)video_index < second_per_grid_ts->size()) {
                    second_per_grid_t = (*second_per_grid_ts)[video_index];
                } else {
                    second_per_grid_t = 1.0;
                }
                video_index++;
            }

            ed = ed_media;
            vision_tokens = run_len;
        }

        const int llm_grid_t = t;
        const int llm_grid_h = h / spatial_merge_size;
        const int llm_grid_w = w / spatial_merge_size;
        const int text_len = ed - st;

        const int st_idx = llm_pos_ids_list.empty() ? 0 : (findMaxIn2DVector(llm_pos_ids_list.back()) + 1);
        auto range = generateRange(text_len, st_idx);
        llm_pos_ids_list.push_back(expandToMatrix(range, 3));

        std::vector<int> t_index;
        t_index.reserve((size_t)std::max(0, vision_tokens));
        for (int ti = 0; ti < llm_grid_t; ti++) {
            for (int hw = 0; hw < llm_grid_h * llm_grid_w; hw++) {
                // Qwen2.5 uses time scaling; Qwen3 passes second_per_grid_t=1 and uses t=1 after preprocess.
                int v = (int)(ti * second_per_grid_t * config.vision_config.tokens_per_second) + text_len + st_idx;
                t_index.push_back(v);
            }
        }

        std::vector<int> h_index;
        h_index.reserve((size_t)std::max(0, vision_tokens));
        for (int ti = 0; ti < llm_grid_t; ti++) {
            for (int hi = 0; hi < llm_grid_h; hi++) {
                for (int wi = 0; wi < llm_grid_w; wi++) h_index.push_back(hi + text_len + st_idx);
            }
        }

        std::vector<int> w_index;
        w_index.reserve((size_t)std::max(0, vision_tokens));
        for (int ti = 0; ti < llm_grid_t; ti++) {
            for (int hi = 0; hi < llm_grid_h; hi++) {
                for (int wi = 0; wi < llm_grid_w; wi++) w_index.push_back(wi + text_len + st_idx);
            }
        }

        // Ensure the media token count matches the actual placeholder run length
        // when image/video placeholders share the same token id (e.g. PaddleOCR-VL).
        if (same_image_video_token) {
            fix_length(t_index, vision_tokens);
            fix_length(h_index, vision_tokens);
            fix_length(w_index, vision_tokens);
        }

        llm_pos_ids_list.push_back({std::move(t_index), std::move(h_index), std::move(w_index)});

        if (!same_image_video_token) {
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
        } else {
            st = ed + vision_tokens;
        }
    }

    // Append remaining text after all images/videos.
    if (st < (int)filtered_ids.size()) {
        int st_idx = llm_pos_ids_list.empty() ? 0 : (findMaxIn2DVector(llm_pos_ids_list.back()) + 1);
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
