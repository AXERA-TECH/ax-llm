#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "BaseTokenizer.hpp"
#include "VLMType.hpp"

namespace vision {

struct RunState {
    // For each position in `input_ids`: -1 for text tokens, otherwise the index into `vision_embed` (in tokens).
    std::vector<int> pos2vision;

    // Flattened bf16 embeddings for all vision placeholder tokens, in placeholder order:
    // [vision_token_count * tokens_embed_size].
    std::vector<unsigned short> vision_embed;

    // Optional "deepstack" features (Qwen3-VL branches): one float array per layer head.
    // Each entry is flattened as [vision_token_count * tokens_embed_size].
    // Applied during prefill: for visual tokens, layer output += deepstack_features[layer][vision_idx].
    std::vector<std::vector<float>> deepstack_features;

    // Optional position ids (e.g. Qwen-VL mRoPE): 3 x seq_len.
    std::vector<std::vector<int>> position_ids;

    // Decode index start override (e.g. Qwen-VL uses max_pos_id + 1). -1 means use sequential default.
    int decode_start = -1;
};

// Media inputs aligned to `history` by index.
// `history[content_index].type` can be IMAGE / VIDEO / AUDIO.
struct MediaInputs {
    size_t content_index = 0;
    std::vector<std::string> uris;
};

struct PromptBudget {
    std::vector<int> last_tokens;
    int precompute_len = 0;
    int prefill_token_num = 0;
    int max_tail_tokens = 0;
    int max_history_tokens = 0;
    int max_total_tokens = 0;
};

struct PrepareMetadata {
    bool auto_reset_for_video = false;
    int current_video_frames = -1;
    int fresh_video_frames = -1;
};

class VisionModule {
public:
    VisionModule();
    ~VisionModule();

    VisionModule(const VisionModule&) = delete;
    VisionModule& operator=(const VisionModule&) = delete;

    VisionModule(VisionModule&&) noexcept;
    VisionModule& operator=(VisionModule&&) noexcept;

    bool enabled() const { return enabled_; }
    int tokens_per_block() const { return tokens_per_block_; }

    // Initialize vision encoder + tokenizer-side token ids.
    // Create by `VLMType` or parse name/int via magic_enum helpers in `src/runner/VLMType.hpp`.
    bool Init(VLMType type,
              const std::string& encoder_axmodel,
              const std::string& cache_dir,
              int tokens_embed_size,
              int devid,
              const std::shared_ptr<BaseTokenizer>& tokenizer,
              int vision_width,
              int vision_height,
              int temporal_patch_size,
              int spatial_merge_size,
              int patch_size,
              int fps,
              int tokens_per_second,
              int video_num_frames,
              bool video_do_sample_frames,
              const std::string& audio_encoder_axmodel_5s,
              const std::string& audio_encoder_axmodel_30s,
              std::string& err);

    void Deinit();

    // Prepare `history_out` by filling `num_media/num_media_tokens`, then:
    // - `input_ids_out = tokenizer->encode(history_out)`
    // - build `state_out` for multimodal injection and (optional) mRoPE position ids.
    bool Prepare(const std::vector<Content>& history_in,
                 const std::vector<MediaInputs>& media_inputs,
                 const PromptBudget* budget,
                 std::vector<Content>& history_out,
                 std::vector<int>& input_ids_out,
                 RunState& state_out,
                 std::string& err,
                 PrepareMetadata* meta = nullptr);

private:
    bool enabled_ = false;
    bool cache_enabled_ = true;
    VLMType type_ = VLMType::None;

    int tokens_embed_size_ = 0;
    int tokens_per_block_ = 0;

    int vision_width_ = 0;
    int vision_height_ = 0;
    int temporal_patch_size_ = 2;
    int spatial_merge_size_ = 2;
    int patch_size_ = 14;
    int fps_ = 1;
    int tokens_per_second_ = 1;
    int video_num_frames_ = 0;
    bool video_do_sample_frames_ = true;

    int deepstack_layers_ = 0;

    std::shared_ptr<BaseTokenizer> tokenizer_;

    // Placeholder token ids to locate vision positions inside input_ids.
    int image_pad_id_ = -1;
    int video_pad_id_ = -1;
    int audio_pad_id_ = -1;
    int vision_start_id_ = -1; // used by mRoPE only

    // A small cache for image -> encoded blocks (bf16). Video is not cached.
    struct CachedImage {
        std::vector<std::vector<unsigned short>> blocks_bf16; // per block
        std::vector<std::vector<float>> deepstack_features;   // per layer (optional)
    };
    std::unordered_map<std::string, CachedImage> image_cache_;
    // Simple LRU bookkeeping for `image_cache_` to prevent unbounded growth in long-running `serve` mode.
    // Controlled by env `AXLLM_VISION_MEM_CACHE_SIZE` (0 disables mem cache; default is small).
    size_t image_cache_max_entries_ = 8;
    std::list<std::string> image_cache_lru_;
    std::unordered_map<std::string, std::list<std::string>::iterator> image_cache_lru_pos_;
    std::string cache_dir_;
    std::string cache_key_prefix_;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool EncodeForContent(const Content& content,
                          const MediaInputs& media,
                          int& out_num_media_for_tokenizer,
                          int& out_num_media_tokens,
                          std::vector<std::vector<unsigned short>>& out_blocks,
                          std::vector<std::vector<float>>* out_deepstack_append,
                          std::vector<std::vector<int>>& out_image_grid_thw,
                          std::vector<std::vector<int>>& out_video_grid_thw,
                          std::string& err);

    bool BuildInjectionState(const std::vector<int>& input_ids,
                             const std::vector<std::vector<unsigned short>>& blocks,
                             const std::vector<std::vector<float>>& deepstack,
                             RunState& state_out,
                             std::string& err);
};

} // namespace vision
