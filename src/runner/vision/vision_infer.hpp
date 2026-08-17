#pragma once

// Encoder-tensor geometry/profile inference helpers shared between VisionModule and the
// IVisionAdapter implementations (moved out of vision_module.cpp so adapters can reuse
// them for resolveInputGeometry / resolveOutputTokens). All pure / runner-free.

#include <cmath>
#include <cstddef>
#include <regex>
#include <string>

namespace vision {

// Infer Qwen-VL style H*W from patchified input byte count (input = H*W*temporal*channels).
inline bool try_infer_qwen_hw_from_input_bytes(size_t input_nbytes,
                                               int temporal_patch_size,
                                               int channels,
                                               int patch_size,
                                               int& io_h,
                                               int& io_w,
                                               std::string& note) {
    if (temporal_patch_size <= 0 || channels <= 0) return false;
    const size_t denom = (size_t)temporal_patch_size * (size_t)channels;
    if (denom == 0 || (input_nbytes % denom) != 0) return false;

    const size_t hw = input_nbytes / denom; // H*W
    if (hw == 0) return false;

    // Prefer square if possible.
    const int side = (int)(std::sqrt((double)hw) + 0.5);
    if ((size_t)side * (size_t)side == hw) {
        if (patch_size > 0 && (side % patch_size) != 0) {
            note = "perfect square but not divisible by patch_size";
            return false;
        }
        io_h = side;
        io_w = side;
        note = "square";
        return true;
    }

    // Try keep configured height and solve width.
    if (io_h > 0 && (hw % (size_t)io_h) == 0) {
        int w = (int)(hw / (size_t)io_h);
        if (patch_size <= 0 || ((io_h % patch_size) == 0 && (w % patch_size) == 0)) {
            io_w = w;
            note = "matched config height";
            return true;
        }
    }

    // Fallback: search a reasonable factor pair close to sqrt, honoring patch_size divisibility.
    size_t best_diff = (size_t)-1;
    int best_h = -1, best_w = -1;
    for (size_t h = 1; h * h <= hw; ++h) {
        if (hw % h) continue;
        size_t w = hw / h;
        if (patch_size > 0) {
            if (((int)h % patch_size) != 0) continue;
            if (((int)w % patch_size) != 0) continue;
        }
        size_t diff = (w > h) ? (w - h) : (h - w);
        if (diff < best_diff) {
            best_diff = diff;
            best_h = (int)h;
            best_w = (int)w;
        }
    }
    if (best_h > 0 && best_w > 0) {
        io_h = best_h;
        io_w = best_w;
        note = "factor-search";
        return true;
    }

    return false;
}

// Infer H/W (+ NCHW/NHWC) from a 4-D input shape with a channel==3 dimension.
template <typename ShapeVec>
inline bool try_infer_hw_from_4d_shape_with_c3(const ShapeVec& shape, int& out_h, int& out_w,
                                               int* out_is_nchw = nullptr) {
    if (shape.size() != 4) return false;
    if ((int)shape[1] == 3) { // NCHW
        out_h = (int)shape[2];
        out_w = (int)shape[3];
        if (out_is_nchw) *out_is_nchw = 1;
        return (out_h > 0 && out_w > 0);
    }
    if ((int)shape[3] == 3) { // NHWC
        out_h = (int)shape[1];
        out_w = (int)shape[2];
        if (out_is_nchw) *out_is_nchw = 0;
        return (out_h > 0 && out_w > 0);
    }
    return false;
}

// Parse a Gemma4 vision profile (h/w/tokens) from the encoder filename (..._h<H>_w<W>_t<T>...).
inline bool parse_gemma4_profile_from_path(const std::string& path, int& out_h, int& out_w, int& out_tokens) {
    std::smatch m;
    if (!std::regex_search(path, m, std::regex("_h(\\d+)_w(\\d+)_t(\\d+)"))) return false;
    out_h = std::stoi(m[1].str());
    out_w = std::stoi(m[2].str());
    out_tokens = std::stoi(m[3].str());
    return true;
}

// Pick tokens_per_block + output dtype from an output byte count, given bytes-per-element.
inline bool pick_tokens_by_bytes(size_t out_nSize, int tokens_embed_size, int bytes_per_elem,
                                 int& out_is_bf16, int& out_tokens_per_block) {
    if (bytes_per_elem <= 0) return false;
    if ((out_nSize % (size_t)bytes_per_elem) != 0) return false;
    const size_t elem = out_nSize / (size_t)bytes_per_elem;
    if (tokens_embed_size <= 0 || elem % (size_t)tokens_embed_size != 0) return false;
    out_tokens_per_block = (int)(elem / (size_t)tokens_embed_size);
    out_is_bf16 = (bytes_per_elem == 2) ? 1 : 0;
    return true;
}

} // namespace vision
