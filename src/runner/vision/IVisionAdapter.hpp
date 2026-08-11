#pragma once

// Per-VLM strategy for the parts of vision preprocessing that differ by model.
// Extracted incrementally from vision_module.cpp's `type_ == VLMType::X` dispatch.
// The base class provides default (no-op / sequential) behavior; concrete adapters
// override only the hooks where their VLM diverges. This lets hooks migrate one at a
// time while every un-migrated path keeps using the existing vision_module.cpp code.

#include <memory>
#include <string>
#include <vector>

#include "VLMType.hpp"
#include "BaseTokenizer.hpp"
#include "utils/ax_cv.hpp"

namespace vision {

struct RunState; // defined in vision_module.hpp

// Result of the per-VLM image preprocessing: the pixel values to encode + how to encode
// them. The generic encode step (encode_block_*/encode_classic_image, which owns the
// vision encoder) stays in VisionModule, so adapters carry only the divergent part.
struct ImagePreproc {
    enum Mode {
        PixelU8,             // encode_block_u8 on each pixel block
        PixelNormalizedFloat,// encode_block_normalized_float(mean,std) on each pixel block
        ClassicMat,          // encode_classic_image directly from the source Mat (no pixel blocks)
    };
    Mode mode = PixelU8;
    std::vector<std::vector<unsigned char>> pixel_blocks; // for PixelU8 / PixelNormalizedFloat
    float norm_mean = 0.0f;   // PixelNormalizedFloat
    float norm_std = 1.0f;    // PixelNormalizedFloat
    bool collect_deepstack = false; // Qwen2_5VL/Qwen3VL u8 path only
};

// Result of per-VLM video preprocessing. Like ImagePreproc but also carries the media
// counts + optional per-video grid_thw (Qwen2_5VL/Qwen3VL: {nBlocks,..}; PaddleOCR: {nFrames,..}).
struct VideoPreproc {
    ImagePreproc::Mode mode = ImagePreproc::PixelU8; // video uses PixelU8 or PixelNormalizedFloat
    std::vector<std::vector<unsigned char>> pixel_blocks;
    float norm_mean = 0.0f;
    float norm_std = 1.0f;
    bool collect_deepstack = false;   // Qwen2_5VL/Qwen3VL
    int num_media_for_tokenizer = 0;  // frame count, or block count for SmolVLM2/Qwen
    bool emit_video_grid_thw = false;
    int grid_t = 0, grid_h = 0, grid_w = 0;
};

// How this VLM plans video frames under a token budget in VisionModule::Prepare. The
// planning orchestration stays in Prepare (it mutates history/media/meta and cross-refs
// Prepare-local state); this only replaces the per-VLM type dispatch that selects it.
enum class VideoPlanKind {
    None,             // no budget-aware frame planning in Prepare
    Gemma4AutoReset,  // Gemma4's bespoke fresh-history auto-reset plan
    SimpleBudgetFit,  // MiniCPM-V / Qwen3Omni shared per-frame budget fit
};

// Placeholder / special token ids used to locate vision positions in input_ids.
struct TokenIds {
    int image_pad = -1;
    int video_pad = -1;
    int audio_pad = -1;
    int vision_start = -1; // mRoPE only
};

// Resolve a special token that must map to exactly one id. Shared by adapters.
inline bool get_single_token_id(const std::shared_ptr<BaseTokenizer>& tok,
                                const std::string& s, int& out_id, std::string& err) {
    auto ids = tok->encode(s);
    if (ids.size() != 1) {
        err = "special token is not a single id: '" + s + "' size=" + std::to_string(ids.size());
        return false;
    }
    out_id = ids[0];
    return true;
}

// Vision geometry / token-id parameters an adapter hook may need, passed per-call so
// adapters stay free of VisionModule internals.
struct VisionParams {
    int temporal_patch_size = 2;
    int tokens_per_second = 1;
    int spatial_merge_size = 2;
    int patch_size = 14;
    int width = 0;
    int height = 0;
    int fps = 1;
    int image_pad_id = -1;
    int video_pad_id = -1;
    int vision_start_id = -1;
};

class IVisionAdapter {
public:
    virtual ~IVisionAdapter() = default;

    // Multimodal position ids (Qwen-VL mRoPE). Default: no-op — leaves
    // state_out.position_ids empty (== sequential default) and decode_start unchanged.
    // Only Qwen2_5VL / Qwen3VL override this; Qwen3Omni intentionally does NOT (it uses
    // sequential position ids), so it maps to the base no-op.
    virtual void computePositionIds(const std::vector<int>& /*input_ids*/,
                                    const std::vector<std::vector<int>>& /*image_grid_thw*/,
                                    const std::vector<std::vector<int>>& /*video_grid_thw*/,
                                    const VisionParams& /*vp*/,
                                    RunState& /*state_out*/) const {}

    // Resolve placeholder/special token ids for this VLM. Default: leave all at -1
    // (matches the pre-refactor `default:` case). Every concrete adapter overrides this.
    virtual bool resolveTokenIds(const std::shared_ptr<BaseTokenizer>& /*tok*/,
                                 TokenIds& /*out*/, std::string& /*err*/) const { return true; }

    // Per-VLM image preprocessing: run this model's image processor and describe how the
    // result should be encoded. Default: unsupported (matches the pre-refactor `else`).
    virtual bool preprocessImage(axcv::Mat& /*img*/, const VisionParams& /*vp*/,
                                 ImagePreproc& /*out*/, std::string& err) const {
        err = "IMAGE not supported for this vlm_type";
        return false;
    }

    // Whether this VLM emits a per-image grid_thw {1, gridH, gridW} (Qwen family + Paddle).
    virtual bool emitsImageGridThw() const { return false; }

    // Per-VLM video preprocessing: run this model's video/frame processor and describe how
    // to encode + count. Default: unsupported (matches the pre-refactor `else`).
    virtual bool preprocessVideo(std::vector<axcv::Mat>& /*frames*/, const VisionParams& /*vp*/,
                                 VideoPreproc& /*out*/, std::string& err) const {
        err = "VIDEO not supported for this vlm_type";
        return false;
    }

    // Which Prepare video frame-planning path this VLM uses (dispatch only; the planning
    // logic stays in Prepare). Default: none.
    virtual VideoPlanKind videoPlanKind() const { return VideoPlanKind::None; }

    // Human-readable model name for Prepare log/error messages.
    virtual const char* displayName() const { return "VLM"; }

    // Extra suffix appended to the vision cache key (preprocessing variant marker).
    // Default: none. Qwen family = hwc_v1, PaddleOCR = nchw_v2.
    virtual const char* cacheKeySuffix() const { return ""; }

    // Number of deepstack feature layers this VLM's encoder provides, given its output
    // tensor count. Default: 0. Only Qwen3VL (min(3, num_outputs-1)).
    virtual int deepstackLayerCount(int /*num_encoder_outputs*/) const { return 0; }

    // Auto-resolve vision width/height (+ classic NCHW/NHWC flag) from the encoder INPUT
    // tensor. io carries width/height (read old + write new) and patch/temporal params.
    // Default: keep configured geometry (MiniCPM/LocateAnything don't auto-infer).
    virtual bool resolveInputGeometry(size_t /*in_nSize*/, const std::vector<unsigned int>& /*in_vShape*/,
                                      const std::string& /*encoder_path*/, VisionParams& /*io*/,
                                      int& /*io_input_is_nchw*/, std::string& /*err*/) const { return true; }

    // Resolve tokens_per_block + encoder output dtype from the encoder OUTPUT tensor.
    // Returns false to fall back to VisionModule's generic vShape/nSize path.
    virtual bool resolveOutputTokens(size_t /*out_nSize*/, const std::string& /*encoder_path*/,
                                     int /*tokens_embed_size*/, const VisionParams& /*vp*/,
                                     int& /*out_tokens_per_block*/, int& /*out_is_bf16*/) const { return false; }
};

// Selects the adapter for a VLMType. Returns the base (default behavior) for any type
// that has not yet had a specialized adapter extracted.
std::unique_ptr<IVisionAdapter> make_vision_adapter(VLMType type);

} // namespace vision
