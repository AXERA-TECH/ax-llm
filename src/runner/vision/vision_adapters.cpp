// Concrete IVisionAdapter implementations + factory.
// Migrated from vision_module.cpp's `type_ == VLMType::X` dispatch, one hook at a time.
//   slice 1: computePositionIds (mRoPE)  -- Qwen2_5VL / Qwen3VL
//   slice 2: resolveTokenIds             -- all types (verbatim from the Init switch)
// Each hook is moved verbatim; the only change is member vars -> VisionParams / TokenIds.

#include "IVisionAdapter.hpp"
#include "vision_module.hpp"   // for vision::RunState
#include "utils/mrope.hpp"
#include "sample_log.h"

#include <algorithm>

namespace vision {

namespace {

// ---- mRoPE helpers (shared by the Qwen-VL family) ------------------------------------

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

// ---- Qwen-VL family (Qwen2_5VL / Qwen3VL / Qwen3Omni) --------------------------------

// Shared base: token ids common to the Qwen trio (image_pad, video_pad, vision_start).
// Qwen3Omni overrides to also resolve audio_pad; mRoPE is added by the 2.5/3VL subclasses
// only (Qwen3Omni deliberately keeps the base no-op == sequential position ids).
class QwenVLAdapter : public IVisionAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<|image_pad|>", out.image_pad, err)) return false;
        if (!get_single_token_id(tok, "<|video_pad|>", out.video_pad, err)) return false;
        if (!get_single_token_id(tok, "<|vision_start|>", out.vision_start, err)) return false;
        ALOGI("Qwen token ids: vision_start=%d image_pad=%d video_pad=%d audio_pad=%d",
              out.vision_start, out.image_pad, out.video_pad, out.audio_pad);
        return true;
    }
};

class Qwen2_5VLAdapter : public QwenVLAdapter {
public:
    void computePositionIds(const std::vector<int>& input_ids,
                            const std::vector<std::vector<int>>& image_grid_thw,
                            const std::vector<std::vector<int>>& video_grid_thw,
                            const VisionParams& vp, RunState& state_out) const override {
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

class Qwen3VLAdapter : public QwenVLAdapter {
public:
    void computePositionIds(const std::vector<int>& input_ids,
                            const std::vector<std::vector<int>>& image_grid_thw,
                            const std::vector<std::vector<int>>& video_grid_thw,
                            const VisionParams& vp, RunState& state_out) const override {
        mrope::Config cfg = make_mrope_config(vp);
        state_out.position_ids = mrope::get_rope_index_qwen3(cfg, input_ids, image_grid_thw, video_grid_thw);
        apply_decode_start_from_max_pos(state_out);
    }
};

class Qwen3OmniAdapter : public QwenVLAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<|image_pad|>", out.image_pad, err)) return false;
        if (!get_single_token_id(tok, "<|video_pad|>", out.video_pad, err)) return false;
        if (!get_single_token_id(tok, "<|audio_pad|>", out.audio_pad, err)) return false;
        if (!get_single_token_id(tok, "<|vision_start|>", out.vision_start, err)) return false;
        ALOGI("Qwen token ids: vision_start=%d image_pad=%d video_pad=%d audio_pad=%d",
              out.vision_start, out.image_pad, out.video_pad, out.audio_pad);
        return true;
    }
    // No computePositionIds override: Qwen3Omni uses sequential position ids (base no-op).
};

// ---- Other VLM families -------------------------------------------------------------

class PaddleOCRVLAdapter : public IVisionAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<|IMAGE_PLACEHOLDER|>", out.image_pad, err)) return false;
        // PaddleOCRVL uses the same placeholder token for both image and video blocks.
        out.video_pad = out.image_pad;
        if (!get_single_token_id(tok, "<|IMAGE_START|>", out.vision_start, err)) return false;
        ALOGI("PaddleOCR-VL token ids: vision_start=%d image_pad=%d video_pad=%d",
              out.vision_start, out.image_pad, out.video_pad);
        return true;
    }
};

class InternVL3Adapter : public IVisionAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<IMG_CONTEXT>", out.image_pad, err)) return false;
        out.video_pad = out.image_pad;
        if (!get_single_token_id(tok, "<img>", out.vision_start, err)) {
            // Not required for injection; mRoPE not used.
            out.vision_start = -1;
        }
        return true;
    }
};

class FastVLMAdapter : public IVisionAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<image>", out.image_pad, err)) return false;
        out.video_pad = out.image_pad;
        out.vision_start = -1;
        return true;
    }
};

class SmolVLM2Adapter : public IVisionAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<image>", out.image_pad, err)) return false;
        out.video_pad = out.image_pad;
        out.vision_start = -1;
        return true;
    }
};

class Gemma4VLAdapter : public IVisionAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<|image|>", out.image_pad, err)) return false;
        if (!get_single_token_id(tok, "<|video|>", out.video_pad, err)) {
            out.video_pad = -1;
        }
        if (!get_single_token_id(tok, "<|audio|>", out.audio_pad, err)) return false;
        out.vision_start = -1;
        ALOGI("Gemma4-VL token ids: image_pad=%d video_pad=%d audio_pad=%d",
              out.image_pad, out.video_pad, out.audio_pad);
        return true;
    }
};

class MiniCPMV46VLAdapter : public IVisionAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<|image_pad|>", out.image_pad, err)) return false;
        if (!get_single_token_id(tok, "<|video_pad|>", out.video_pad, err)) {
            out.video_pad = out.image_pad;
        }
        out.vision_start = -1;
        ALOGI("MiniCPM-V-4.6 token ids: image_pad=%d video_pad=%d", out.image_pad, out.video_pad);
        return true;
    }
};

class LocateAnythingVLAdapter : public IVisionAdapter {
public:
    bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& tok, TokenIds& out,
                         std::string& err) const override {
        if (!get_single_token_id(tok, "<IMG_CONTEXT>", out.image_pad, err)) return false;
        out.video_pad = out.image_pad;
        out.vision_start = -1;
        ALOGI("LocateAnything token ids: image_pad(<IMG_CONTEXT>)=%d", out.image_pad);
        return true;
    }
};

} // namespace

std::unique_ptr<IVisionAdapter> make_vision_adapter(VLMType type) {
    switch (type) {
        case VLMType::Qwen2_5VL:       return std::make_unique<Qwen2_5VLAdapter>();
        case VLMType::Qwen3VL:         return std::make_unique<Qwen3VLAdapter>();
        case VLMType::Qwen3Omni:       return std::make_unique<Qwen3OmniAdapter>();
        case VLMType::PaddleOCRVL:     return std::make_unique<PaddleOCRVLAdapter>();
        case VLMType::InternVL3:       return std::make_unique<InternVL3Adapter>();
        case VLMType::FastVLM:         return std::make_unique<FastVLMAdapter>();
        case VLMType::SmolVLM2:        return std::make_unique<SmolVLM2Adapter>();
        case VLMType::Gemma4VL:        return std::make_unique<Gemma4VLAdapter>();
        case VLMType::MiniCPMV46VL:    return std::make_unique<MiniCPMV46VLAdapter>();
        case VLMType::LocateAnythingVL:return std::make_unique<LocateAnythingVLAdapter>();
        // VLMType::None + anything unrecognized: base (no-op) adapter.
        default:                       return std::make_unique<IVisionAdapter>();
    }
}

} // namespace vision
