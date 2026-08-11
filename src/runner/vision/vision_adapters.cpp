// Concrete IVisionAdapter implementations + factory.
// Migrated from vision_module.cpp's `type_ == VLMType::X` dispatch, one hook at a time.
//   slice 1: computePositionIds (mRoPE)  -- Qwen2_5VL / Qwen3VL
//   slice 2: resolveTokenIds             -- all types (verbatim from the Init switch)
// Each hook is moved verbatim; the only change is member vars -> VisionParams / TokenIds.

#include "IVisionAdapter.hpp"
#include "vision_module.hpp"   // for vision::RunState
#include "vision_infer.hpp"    // encoder-tensor geometry/token inference helpers
#include "utils/mrope.hpp"
#include "utils/image_processor.hpp"
#include "utils/ax_cv.hpp"
#include "sample_log.h"

#include <algorithm>

namespace vision {

namespace {

// Shared Qwen-family (Qwen2_5VL/Qwen3VL/Qwen3Omni) + PaddleOCR input geometry inference.
// float_input=true for the float32-patch encoders (PaddleOCR, Qwen3Omni).
bool resolve_qwen_family_geometry(size_t in_nSize, bool float_input, VisionParams& io, std::string& err) {
    const int old_w = io.width, old_h = io.height;
    size_t eff_nSize = in_nSize;
    if (float_input && (eff_nSize % sizeof(float)) == 0) eff_nSize /= sizeof(float);
    const size_t cfg_bytes = (size_t)std::max(0, old_h) * (size_t)std::max(0, old_w) *
                             (size_t)std::max(1, io.temporal_patch_size) * (size_t)3;
    int h = old_h, w = old_w;
    std::string note;
    if (try_infer_qwen_hw_from_input_bytes(eff_nSize, std::max(1, io.temporal_patch_size), 3,
                                           std::max(1, io.patch_size), h, w, note)) {
        if (w != old_w || h != old_h) {
            ALOGW("Qwen-VL vision size override: cfg=%dx%d bytes=%zu, model_input_bytes=%zu -> %dx%d (%s).",
                  old_w, old_h, cfg_bytes, in_nSize, w, h, note.c_str());
        }
        io.width = w;
        io.height = h;
    } else {
        if (io.width <= 0 || io.height <= 0) {
            err = "failed to infer Qwen-VL vision_width/vision_height from encoder input";
            return false;
        }
        if (cfg_bytes != in_nSize) {
            ALOGW("Qwen-VL vision size mismatch (cfg=%dx%d bytes=%zu, model_input_bytes=%zu). Will pad/zero input tail.",
                  io.width, io.height, cfg_bytes, in_nSize);
        }
    }
    return true;
}

// Shared Qwen-family + PaddleOCR output tokens_per_block + dtype (expected grid-merge count).
bool resolve_qwen_family_tokens(size_t out_nSize, int tokens_embed, const VisionParams& vp,
                                int& tpb, int& is_bf16) {
    const int grid_h = vp.height / std::max(1, vp.patch_size);
    const int grid_w = vp.width / std::max(1, vp.patch_size);
    const int llm_grid_h = grid_h / std::max(1, vp.spatial_merge_size);
    const int llm_grid_w = grid_w / std::max(1, vp.spatial_merge_size);
    const int expected_tokens = std::max(1, llm_grid_h) * std::max(1, llm_grid_w);
    int b = -1, t = 0;
    if (pick_tokens_by_bytes(out_nSize, tokens_embed, 4, b, t) && t == expected_tokens) { is_bf16 = b; tpb = t; return true; }
    if (pick_tokens_by_bytes(out_nSize, tokens_embed, 2, b, t) && t == expected_tokens) { is_bf16 = b; tpb = t; return true; }
    if (pick_tokens_by_bytes(out_nSize, tokens_embed, 4, b, t)) {
        ALOGW("vision encoder tokens_per_block=%d (expected=%d). Using fp32 by nSize inference (out0.nSize=%zu).",
              t, expected_tokens, out_nSize);
        is_bf16 = b; tpb = t; return true;
    }
    if (pick_tokens_by_bytes(out_nSize, tokens_embed, 2, b, t)) {
        ALOGW("vision encoder tokens_per_block=%d (expected=%d). Using bf16 by nSize inference (out0.nSize=%zu).",
              t, expected_tokens, out_nSize);
        is_bf16 = b; tpb = t; return true;
    }
    return false;
}

// Classic encoder (InternVL3 / FastVLM) input geometry + NCHW/NHWC detection.
bool resolve_classic_geometry(size_t in_nSize, const std::vector<unsigned int>& in_vShape,
                              VisionParams& io, int& io_input_is_nchw, std::string& err) {
    io_input_is_nchw = -1;
    int h = 0, w = 0, is_nchw = -1;
    const bool got_shape = try_infer_hw_from_4d_shape_with_c3(in_vShape, h, w, &is_nchw);
    if (got_shape) {
        io_input_is_nchw = is_nchw;
        if (io.width > 0 && io.height > 0 && (io.width != w || io.height != h)) {
            ALOGW("classic vision size override: cfg=%dx%d -> model=%dx%d (from input shape)", io.width, io.height, w, h);
        }
        io.width = w;
        io.height = h;
    } else {
        if (io.width <= 0 || io.height <= 0) {
            err = "classic vision encoder input shape missing; please provide valid vision_width/vision_height";
            return false;
        }
        const size_t need_nhwc_u8 = (size_t)io.width * (size_t)io.height * (size_t)3;
        const size_t need_nchw_f32 = need_nhwc_u8 * sizeof(float);
        if (in_nSize == need_nchw_f32) io_input_is_nchw = 1;
        else if (in_nSize == need_nhwc_u8) io_input_is_nchw = 0;
        else { err = "classic vision encoder layout not detected from input shape/nSize"; return false; }
        ALOGW("classic vision input shape unavailable; fallback to cfg size %dx%d by nSize=%zu (layout=%s)",
              io.width, io.height, in_nSize, (io_input_is_nchw == 1 ? "NCHW-fp32" : "NHWC-u8"));
    }
    return true;
}

// Shared Qwen-family image processor: Qwen2VideoProcessor on a single image -> exactly one
// pixel block, with the pre-existing min/max sanity log. Encode mode is set by the caller.
bool run_qwen_image_processor(axcv::Mat& img, const VisionParams& vp,
                              std::vector<std::vector<unsigned char>>& pixel_values, std::string& err) {
    std::vector<axcv::Mat> one{img};
    Qwen2VideoProcessor(one, pixel_values, vp.height, vp.width, vp.temporal_patch_size,
                        vp.spatial_merge_size, vp.patch_size);
    if (pixel_values.size() != 1) { err = "Qwen2VideoProcessor(image) returned != 1 block"; return false; }
    const auto& pv = pixel_values[0];
    unsigned char mn = 255, mx = 0;
    for (unsigned char b : pv) { if (b < mn) mn = b; if (b > mx) mx = b; }
    ALOGI("Qwen-VL pixel_values[0] bytes=%zu min=%u max=%u (w=%d h=%d tp=%d ps=%d sm=%d)",
          pv.size(), (unsigned)mn, (unsigned)mx, vp.width, vp.height, vp.temporal_patch_size,
          vp.patch_size, vp.spatial_merge_size);
    return true;
}

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

    // Qwen2_5VL / Qwen3VL image path: u8 encode, collect deepstack (Qwen3VL has layers>0).
    bool preprocessImage(axcv::Mat& img, const VisionParams& vp, ImagePreproc& out,
                         std::string& err) const override {
        if (!run_qwen_image_processor(img, vp, out.pixel_blocks, err)) return false;
        out.mode = ImagePreproc::PixelU8;
        out.collect_deepstack = true;
        return true;
    }

    bool emitsImageGridThw() const override { return true; }

    // Qwen2_5VL / Qwen3VL video: Qwen2VideoProcessor over ALL frames -> temporal blocks,
    // u8 encode + deepstack, video_grid_thw = {nBlocks, gridH, gridW}.
    bool preprocessVideo(std::vector<axcv::Mat>& frames, const VisionParams& vp, VideoPreproc& out,
                         std::string& /*err*/) const override {
        std::vector<std::vector<unsigned char>> pixel_values;
        Qwen2VideoProcessor(frames, pixel_values, vp.height, vp.width, vp.temporal_patch_size,
                            vp.spatial_merge_size, vp.patch_size);
        out.mode = ImagePreproc::PixelU8;
        out.collect_deepstack = true;
        out.num_media_for_tokenizer = (int)pixel_values.size();
        out.emit_video_grid_thw = true;
        out.grid_t = (int)pixel_values.size();
        out.grid_h = vp.height / vp.patch_size;
        out.grid_w = vp.width / vp.patch_size;
        out.pixel_blocks = std::move(pixel_values);
        return true;
    }

    const char* cacheKeySuffix() const override { return "|resize=pillow_bicubic|patch=hwc_v1"; }

    bool resolveInputGeometry(size_t in_nSize, const std::vector<unsigned int>&, const std::string&,
                              VisionParams& io, int&, std::string& err) const override {
        return resolve_qwen_family_geometry(in_nSize, /*float_input=*/false, io, err);
    }
    bool resolveOutputTokens(size_t out_nSize, const std::string&, int tokens_embed,
                             const VisionParams& vp, int& tpb, int& is_bf16) const override {
        return resolve_qwen_family_tokens(out_nSize, tokens_embed, vp, tpb, is_bf16);
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

    int deepstackLayerCount(int nout) const override { return nout > 1 ? std::min(3, nout - 1) : 0; }
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

    VideoPlanKind videoPlanKind() const override { return VideoPlanKind::SimpleBudgetFit; }
    const char* displayName() const override { return "Qwen3Omni"; }
    AudioKind audioKind() const override { return AudioKind::Qwen3OmniAudio; }

    // Qwen3Omni consumes float32 patch input -> divide by sizeof(float). Tokens inherited.
    bool resolveInputGeometry(size_t in_nSize, const std::vector<unsigned int>&, const std::string&,
                              VisionParams& io, int&, std::string& err) const override {
        return resolve_qwen_family_geometry(in_nSize, /*float_input=*/true, io, err);
    }

    // Qwen3Omni image path: same processor, but normalized-float encode, no deepstack.
    bool preprocessImage(axcv::Mat& img, const VisionParams& vp, ImagePreproc& out,
                         std::string& err) const override {
        if (!run_qwen_image_processor(img, vp, out.pixel_blocks, err)) return false;
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.5f;
        out.norm_std = 0.5f;
        out.collect_deepstack = false;
        return true;
    }
    // emitsImageGridThw inherited from QwenVLAdapter (true).

    // Qwen3Omni video: per-frame Qwen2VideoProcessor (1 block/frame), normalized-float, no grid.
    bool preprocessVideo(std::vector<axcv::Mat>& frames, const VisionParams& vp, VideoPreproc& out,
                         std::string& err) const override {
        out.pixel_blocks.reserve(frames.size());
        for (auto& frame : frames) {
            std::vector<axcv::Mat> one{frame};
            std::vector<std::vector<unsigned char>> pv1;
            Qwen2VideoProcessor(one, pv1, vp.height, vp.width, vp.temporal_patch_size,
                                vp.spatial_merge_size, vp.patch_size);
            if (pv1.size() != 1) { err = "Qwen2VideoProcessor(video frame) returned != 1 block"; return false; }
            out.pixel_blocks.push_back(std::move(pv1[0]));
        }
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.5f;
        out.norm_std = 0.5f;
        out.collect_deepstack = false;
        out.num_media_for_tokenizer = (int)frames.size();
        return true;
    }
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

    bool preprocessImage(axcv::Mat& img, const VisionParams& vp, ImagePreproc& out,
                         std::string& /*err*/) const override {
        // PaddleOCR-VL VIT expects patches in [N, C, pH, pW] (channel-first per patch,
        // no spatial merge in preprocessing -- merge happens inside the VIT model).
        std::vector<unsigned char> pv;
        PaddleOCRVLImageProcessor(img, pv, vp.height, vp.width, vp.patch_size);
        {
            unsigned char mn = 255, mx = 0;
            for (unsigned char b : pv) { if (b < mn) mn = b; if (b > mx) mx = b; }
            ALOGI("PaddleOCRVL pixel_values bytes=%zu min=%u max=%u (w=%d h=%d ps=%d)",
                  pv.size(), (unsigned)mn, (unsigned)mx, vp.width, vp.height, vp.patch_size);
        }
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.5f;
        out.norm_std = 0.5f;
        out.pixel_blocks.push_back(std::move(pv));
        return true;
    }

    bool emitsImageGridThw() const override { return true; }

    // PaddleOCR-VL video: each frame independently (same VIT as image), video_grid_thw={nFrames,..}.
    bool preprocessVideo(std::vector<axcv::Mat>& frames, const VisionParams& vp, VideoPreproc& out,
                         std::string& /*err*/) const override {
        out.pixel_blocks.reserve(frames.size());
        for (auto& frame : frames) {
            std::vector<unsigned char> pv;
            PaddleOCRVLImageProcessor(frame, pv, vp.height, vp.width, vp.patch_size);
            out.pixel_blocks.push_back(std::move(pv));
        }
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.5f;
        out.norm_std = 0.5f;
        out.num_media_for_tokenizer = (int)frames.size();
        out.emit_video_grid_thw = true;
        out.grid_t = (int)frames.size();
        out.grid_h = vp.height / vp.patch_size;
        out.grid_w = vp.width / vp.patch_size;
        return true;
    }

    const char* cacheKeySuffix() const override { return "|resize=pillow_bicubic|patch=nchw_v2"; }

    bool resolveInputGeometry(size_t in_nSize, const std::vector<unsigned int>&, const std::string&,
                              VisionParams& io, int&, std::string& err) const override {
        return resolve_qwen_family_geometry(in_nSize, /*float_input=*/true, io, err);
    }
    bool resolveOutputTokens(size_t out_nSize, const std::string&, int tokens_embed,
                             const VisionParams& vp, int& tpb, int& is_bf16) const override {
        return resolve_qwen_family_tokens(out_nSize, tokens_embed, vp, tpb, is_bf16);
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

    bool preprocessImage(axcv::Mat& /*img*/, const VisionParams& /*vp*/, ImagePreproc& out,
                         std::string& /*err*/) const override {
        out.mode = ImagePreproc::ClassicMat; // encoded directly from the Mat by VisionModule.
        return true;
    }

    bool resolveInputGeometry(size_t in_nSize, const std::vector<unsigned int>& in_vShape, const std::string&,
                              VisionParams& io, int& io_input_is_nchw, std::string& err) const override {
        return resolve_classic_geometry(in_nSize, in_vShape, io, io_input_is_nchw, err);
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

    bool preprocessImage(axcv::Mat& /*img*/, const VisionParams& /*vp*/, ImagePreproc& out,
                         std::string& /*err*/) const override {
        out.mode = ImagePreproc::ClassicMat;
        return true;
    }

    bool resolveInputGeometry(size_t in_nSize, const std::vector<unsigned int>& in_vShape, const std::string&,
                              VisionParams& io, int& io_input_is_nchw, std::string& err) const override {
        return resolve_classic_geometry(in_nSize, in_vShape, io, io_input_is_nchw, err);
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

    bool preprocessImage(axcv::Mat& img, const VisionParams& vp, ImagePreproc& out,
                         std::string& /*err*/) const override {
        std::vector<axcv::Mat> one{img};
        Smolvlm2ImageProcessor(one, out.pixel_blocks, vp.width, vp.height); // expected 5 blocks
        out.mode = ImagePreproc::PixelU8;
        out.collect_deepstack = false;
        return true;
    }

    bool preprocessVideo(std::vector<axcv::Mat>& frames, const VisionParams& vp, VideoPreproc& out,
                         std::string& /*err*/) const override {
        Smolvlm2VideoProcessor(frames, out.pixel_blocks, vp.width, vp.height);
        out.mode = ImagePreproc::PixelU8;
        out.collect_deepstack = false;
        out.num_media_for_tokenizer = (int)out.pixel_blocks.size();
        return true;
    }

    bool resolveInputGeometry(size_t in_nSize, const std::vector<unsigned int>& in_vShape, const std::string&,
                              VisionParams& io, int&, std::string& err) const override {
        const int old_w = io.width, old_h = io.height;
        int h = 0, w = 0, tmp_layout = -1;
        bool inferred = try_infer_hw_from_4d_shape_with_c3(in_vShape, h, w, &tmp_layout);
        if (!inferred && (in_nSize % 3u) == 0u) {
            const size_t hw = in_nSize / 3u;
            const int side = (int)(std::sqrt((double)hw) + 0.5);
            if ((size_t)side * (size_t)side == hw) { h = side; w = side; inferred = true; }
        }
        if (!inferred) {
            if (io.width <= 0 || io.height <= 0) { err = "failed to infer SmolVLM2 vision_width/vision_height from encoder input"; return false; }
            ALOGW("SmolVLM2 vision size inference failed; keep cfg=%dx%d", io.width, io.height);
        } else {
            if (old_w != w || old_h != h) ALOGW("SmolVLM2 vision size override: cfg=%dx%d -> model=%dx%d", old_w, old_h, w, h);
            io.width = w;
            io.height = h;
        }
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

    VideoPlanKind videoPlanKind() const override { return VideoPlanKind::Gemma4AutoReset; }
    const char* displayName() const override { return "Gemma4"; }
    AudioKind audioKind() const override { return AudioKind::Gemma4Audio; }

    bool resolveInputGeometry(size_t in_nSize, const std::vector<unsigned int>&, const std::string& encoder_path,
                              VisionParams& io, int&, std::string& err) const override {
        const int old_w = io.width, old_h = io.height;
        int parsed_h = 0, parsed_w = 0, parsed_tokens = 0;
        const bool parsed_profile = parse_gemma4_profile_from_path(encoder_path, parsed_h, parsed_w, parsed_tokens);
        const size_t eff_nsize = (in_nSize % sizeof(float) == 0) ? (in_nSize / sizeof(float)) : in_nSize;
        const int pixel_dim = std::max(1, io.patch_size) * std::max(1, io.patch_size) * 3;
        if (pixel_dim <= 0 || eff_nsize % (size_t)pixel_dim != 0) {
            err = "failed to infer Gemma4 vision patch layout from encoder input";
            return false;
        }
        const int patch_count = (int)(eff_nsize / (size_t)pixel_dim);
        if (parsed_profile) {
            io.height = parsed_h;
            io.width = parsed_w;
        } else if (io.width > 0 && io.height > 0) {
            const int expected_patch_count = (io.height / std::max(1, io.patch_size)) * (io.width / std::max(1, io.patch_size));
            if (expected_patch_count != patch_count) {
                ALOGW("Gemma4 input patch count mismatch: cfg=%dx%d -> %d patches, model=%d patches",
                      io.width, io.height, expected_patch_count, patch_count);
            }
        } else {
            err = "failed to infer Gemma4 vision_width/vision_height from encoder filename; please set config";
            return false;
        }
        if (old_w != io.width || old_h != io.height) {
            ALOGW("Gemma4 vision size override: cfg=%dx%d -> model=%dx%d", old_w, old_h, io.width, io.height);
        }
        return true;
    }
    bool resolveOutputTokens(size_t out_nSize, const std::string& encoder_path, int tokens_embed,
                             const VisionParams&, int& tpb, int& is_bf16) const override {
        int expected_h = 0, expected_w = 0, expected_tokens = 0;
        const bool parsed_profile = parse_gemma4_profile_from_path(encoder_path, expected_h, expected_w, expected_tokens);
        int b = -1, t = 0;
        if (pick_tokens_by_bytes(out_nSize, tokens_embed, 4, b, t) && (!parsed_profile || t == expected_tokens)) { is_bf16 = b; tpb = t; return true; }
        if (pick_tokens_by_bytes(out_nSize, tokens_embed, 2, b, t) && (!parsed_profile || t == expected_tokens)) { is_bf16 = b; tpb = t; return true; }
        return false;
    }

    bool preprocessImage(axcv::Mat& img, const VisionParams& vp, ImagePreproc& out,
                         std::string& /*err*/) const override {
        std::vector<unsigned char> pv;
        Gemma4ImageProcessor(img, pv, vp.height, vp.width, vp.patch_size);
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.0f;
        out.norm_std = 1.0f;
        out.pixel_blocks.push_back(std::move(pv));
        return true;
    }

    bool preprocessVideo(std::vector<axcv::Mat>& frames, const VisionParams& vp, VideoPreproc& out,
                         std::string& /*err*/) const override {
        out.pixel_blocks.reserve(frames.size());
        for (auto& frame : frames) {
            std::vector<unsigned char> pv;
            Gemma4ImageProcessor(frame, pv, vp.height, vp.width, vp.patch_size);
            out.pixel_blocks.push_back(std::move(pv));
        }
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.0f;
        out.norm_std = 1.0f;
        out.num_media_for_tokenizer = (int)frames.size();
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

    VideoPlanKind videoPlanKind() const override { return VideoPlanKind::SimpleBudgetFit; }
    const char* displayName() const override { return "MiniCPM-V-4.6"; }

    bool resolveOutputTokens(size_t out_nSize, const std::string&, int tokens_embed,
                             const VisionParams&, int& tpb, int& is_bf16) const override {
        int b = -1, t = 0;
        if (pick_tokens_by_bytes(out_nSize, tokens_embed, 4, b, t)) { is_bf16 = b; tpb = t; return true; }
        if (pick_tokens_by_bytes(out_nSize, tokens_embed, 2, b, t)) { is_bf16 = b; tpb = t; return true; }
        return false;
    }

    bool preprocessImage(axcv::Mat& img, const VisionParams& vp, ImagePreproc& out,
                         std::string& err) const override {
        std::vector<unsigned char> pv;
        if (MiniCPMV46ImageProcessor(img, pv, vp.height, vp.width, vp.patch_size) != 0) {
            err = "MiniCPM-V-4.6 image preprocessing failed";
            return false;
        }
        {
            unsigned char mn = 255, mx = 0;
            for (unsigned char b : pv) { if (b < mn) mn = b; if (b > mx) mx = b; }
            ALOGI("MiniCPM-V-4.6 pixel_values bytes=%zu min=%u max=%u (w=%d h=%d ps=%d)",
                  pv.size(), (unsigned)mn, (unsigned)mx, vp.width, vp.height, vp.patch_size);
        }
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.5f;
        out.norm_std = 0.5f;
        out.pixel_blocks.push_back(std::move(pv));
        return true;
    }

    bool preprocessVideo(std::vector<axcv::Mat>& frames, const VisionParams& vp, VideoPreproc& out,
                         std::string& err) const override {
        out.pixel_blocks.reserve(frames.size());
        for (auto& frame : frames) {
            std::vector<unsigned char> pv;
            if (MiniCPMV46ImageProcessor(frame, pv, vp.height, vp.width, vp.patch_size) != 0) {
                err = "MiniCPM-V-4.6 video frame preprocessing failed";
                return false;
            }
            out.pixel_blocks.push_back(std::move(pv));
        }
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.5f;
        out.norm_std = 0.5f;
        out.num_media_for_tokenizer = (int)frames.size();
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

    bool preprocessImage(axcv::Mat& img, const VisionParams& vp, ImagePreproc& out,
                         std::string& err) const override {
        // LocateAnything: [1600,3,14,14] uint8 patches -> normalize pixel/127.5-1 (mean/std 0.5)
        // -> image_encoder_mlp.axmodel -> 400x2048 tokens.
        std::vector<unsigned char> pv;
        if (LocateAnythingImageProcessor(img, pv, vp.height, vp.width, vp.patch_size) != 0) {
            err = "LocateAnything image preprocessing failed";
            return false;
        }
        out.mode = ImagePreproc::PixelNormalizedFloat;
        out.norm_mean = 0.5f;
        out.norm_std = 0.5f;
        out.pixel_blocks.push_back(std::move(pv));
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
