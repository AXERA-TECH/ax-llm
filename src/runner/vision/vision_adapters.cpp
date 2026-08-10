// Concrete IVisionAdapter implementations + factory.
// Slice 1: mRoPE position-id computation extracted from VisionModule::Prepare
// (was `if (type_ == Qwen2_5VL || Qwen3VL) { ... }`). Moved verbatim; the only change
// is member vars -> VisionParams fields. Qwen3Omni is deliberately NOT given an mRoPE
// override (it used sequential position ids before), so it maps to the base no-op.

#include "IVisionAdapter.hpp"
#include "vision_module.hpp"   // for vision::RunState
#include "utils/mrope.hpp"

#include <algorithm>

namespace vision {

namespace {

// Shared mRoPE config builder for the Qwen-VL family.
mrope::Config make_mrope_config(const VisionParams& vp) {
    mrope::Config cfg;
    cfg.vision_config.temporal_patch_size = vp.temporal_patch_size;
    cfg.vision_config.tokens_per_second = vp.tokens_per_second;
    cfg.vision_config.spatial_merge_size = vp.spatial_merge_size;
    cfg.vision_config.patch_size = vp.patch_size;
    cfg.vision_config.width = vp.width;
    cfg.vision_config.height = vp.height;
    cfg.vision_config.fps = vp.fps;
    cfg.image_token_id = vp.image_pad_id;
    cfg.video_token_id = vp.video_pad_id;
    cfg.vision_start_token_id = vp.vision_start_id;
    return cfg;
}

void apply_decode_start_from_max_pos(RunState& state_out) {
    int max_pos = -1;
    for (const auto& row : state_out.position_ids) {
        for (int v : row) max_pos = std::max(max_pos, v);
    }
    if (max_pos >= 0) state_out.decode_start = max_pos + 1;
}

class Qwen2_5VLAdapter : public IVisionAdapter {
public:
    void computePositionIds(const std::vector<int>& input_ids,
                            const std::vector<std::vector<int>>& image_grid_thw,
                            const std::vector<std::vector<int>>& video_grid_thw,
                            const VisionParams& vp,
                            RunState& state_out) const override {
        mrope::Config cfg = make_mrope_config(vp);
        std::vector<double> second_per_grid_ts;
        second_per_grid_ts.reserve(video_grid_thw.size());
        for (size_t i = 0; i < video_grid_thw.size(); ++i) {
            second_per_grid_ts.push_back(double(vp.temporal_patch_size) / double(std::max(1, vp.fps)));
        }
        state_out.position_ids =
            mrope::get_rope_index_qwen2_5(cfg, input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts);
        apply_decode_start_from_max_pos(state_out);
    }
};

class Qwen3VLAdapter : public IVisionAdapter {
public:
    void computePositionIds(const std::vector<int>& input_ids,
                            const std::vector<std::vector<int>>& image_grid_thw,
                            const std::vector<std::vector<int>>& video_grid_thw,
                            const VisionParams& vp,
                            RunState& state_out) const override {
        mrope::Config cfg = make_mrope_config(vp);
        state_out.position_ids = mrope::get_rope_index_qwen3(cfg, input_ids, image_grid_thw, video_grid_thw);
        apply_decode_start_from_max_pos(state_out);
    }
};

} // namespace

std::unique_ptr<IVisionAdapter> make_vision_adapter(VLMType type) {
    switch (type) {
        case VLMType::Qwen2_5VL: return std::make_unique<Qwen2_5VLAdapter>();
        case VLMType::Qwen3VL:   return std::make_unique<Qwen3VLAdapter>();
        // Qwen3Omni + all other types: base no-op (sequential position ids), matching
        // the pre-refactor behavior. Specialized adapters added in later slices.
        default:                 return std::make_unique<IVisionAdapter>();
    }
}

} // namespace vision
