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

namespace vision {

struct RunState; // defined in vision_module.hpp

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
};

// Selects the adapter for a VLMType. Returns the base (default behavior) for any type
// that has not yet had a specialized adapter extracted.
std::unique_ptr<IVisionAdapter> make_vision_adapter(VLMType type);

} // namespace vision
