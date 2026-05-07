#include "image_processor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "files.hpp"

namespace {

// A minimal Pillow-compatible bicubic resizer for uint8 RGB images.
// Matches Pillow's `Image.resize(..., Image.Resampling.BICUBIC)` implementation (Pillow 12.x)
// based on src/libImaging/Resample.c.
//
// This exists because OpenCV's INTER_CUBIC differs from Pillow's bicubic kernel/rounding,
// and Qwen3-VL embedding alignment is sensitive to those small pixel-level differences.
struct PillowAxisCoeffs
{
    int ksize = 0;
    std::vector<int> bounds;         // [outSize * 2] => xmin, count
    std::vector<int32_t> coeffs_int; // [outSize * ksize], fixed-point normalized
};

constexpr int kPillowPrecisionBits = 22; // 32 - 8 - 2

static inline double pillow_bicubic_filter(double x)
{
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

static inline uint8_t pillow_clip8(int in)
{
    // Equivalent to Pillow's lookup-table clip8(in >> PRECISION_BITS).
    const int v = in >> kPillowPrecisionBits;
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static bool pillow_precompute_axis_coeffs(int inSize, int outSize, PillowAxisCoeffs& out)
{
    if (inSize <= 0 || outSize <= 0) return false;

    const double in0 = 0.0;
    const double in1 = (double)inSize;
    double filterscale = (in1 - in0) / (double)outSize;
    const double scale = filterscale;
    if (filterscale < 1.0) filterscale = 1.0;

    const double support = 2.0 * filterscale; // bicubic support is 2.0
    const int ksize = (int)std::ceil(support) * 2 + 1;
    if (ksize <= 0) return false;

    out.ksize = ksize;
    out.bounds.assign((size_t)outSize * 2, 0);
    out.coeffs_int.assign((size_t)outSize * (size_t)ksize, 0);

    const double ss = 1.0 / filterscale;
    const double fixed_scale = (double)(1u << kPillowPrecisionBits);

    std::vector<double> tmp((size_t)ksize, 0.0);

    for (int xx = 0; xx < outSize; ++xx)
    {
        const double center = in0 + (xx + 0.5) * scale;
        double ww = 0.0;

        // Round the value (matches Pillow's (int)(... + 0.5) cast behavior).
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;

        int xmax = (int)(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin; // convert to count
        if (xmax < 0) xmax = 0;
        if (xmax > ksize) xmax = ksize;

        std::fill(tmp.begin(), tmp.end(), 0.0);
        for (int x = 0; x < xmax; ++x)
        {
            const double w = pillow_bicubic_filter((x + xmin - center + 0.5) * ss);
            tmp[(size_t)x] = w;
            ww += w;
        }

        if (ww != 0.0)
        {
            for (int x = 0; x < xmax; ++x) tmp[(size_t)x] /= ww;
        }

        out.bounds[(size_t)xx * 2 + 0] = xmin;
        out.bounds[(size_t)xx * 2 + 1] = xmax;

        int32_t* dst = out.coeffs_int.data() + (size_t)xx * (size_t)ksize;
        for (int x = 0; x < ksize; ++x)
        {
            const double v = tmp[(size_t)x] * fixed_scale;
            if (tmp[(size_t)x] < 0.0) dst[x] = (int32_t)(-0.5 + v);
            else dst[x] = (int32_t)(0.5 + v);
        }
    }

    return true;
}

static bool pillow_resize_bgr_to_rgb_u8(const axcv::Mat& src_bgr, std::vector<unsigned char>& output_rgb, int dst_w, int dst_h)
{
    const int src_w = axcv::width(src_bgr);
    const int src_h = axcv::height(src_bgr);
    const int channels = axcv::channels(src_bgr);
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || channels <= 0 || axcv::empty(src_bgr)) {
        return false;
    }

    // Fast path: no resize needed, just BGR->RGB.
    if (src_w == dst_w && src_h == dst_h)
    {
        output_rgb.resize((size_t)dst_w * (size_t)dst_h * 3);
        for (int y = 0; y < src_h; ++y)
        {
            const uint8_t* sp = axcv::row_ptr(src_bgr, y);
            uint8_t* dp = output_rgb.data() + (size_t)y * (size_t)dst_w * 3;
            for (int x = 0; x < dst_w; ++x)
            {
                const uint8_t* px = sp + (size_t)x * (size_t)channels;
                dp[3 * x + 0] = px[2];
                dp[3 * x + 1] = px[1];
                dp[3 * x + 2] = px[0];
            }
        }
        return true;
    }

    // Convert to Pillow-like RGBX (4 bytes per pixel, X=0) for bit-exact resampling.
    const int src_step = src_w * 4;
    std::vector<uint8_t> src_rgbx((size_t)src_h * (size_t)src_step);
    for (int y = 0; y < src_h; ++y)
    {
        const uint8_t* sp = axcv::row_ptr(src_bgr, y);
        uint8_t* dp = src_rgbx.data() + (size_t)y * (size_t)src_step;
        for (int x = 0; x < src_w; ++x)
        {
            const uint8_t* px = sp + (size_t)x * (size_t)channels;
            dp[4 * x + 0] = px[2]; // R
            dp[4 * x + 1] = px[1]; // G
            dp[4 * x + 2] = px[0]; // B
            dp[4 * x + 3] = 0;     // X
        }
    }

    PillowAxisCoeffs xcoeff, ycoeff;
    if (!pillow_precompute_axis_coeffs(src_w, dst_w, xcoeff)) return false;
    if (!pillow_precompute_axis_coeffs(src_h, dst_h, ycoeff)) return false;

    // Horizontal pass: [src_h, dst_w, 4]
    const int tmp_step = dst_w * 4;
    std::vector<uint8_t> tmp_rgbx((size_t)src_h * (size_t)tmp_step, 0);
    for (int yy = 0; yy < src_h; ++yy)
    {
        const uint8_t* row = src_rgbx.data() + (size_t)yy * (size_t)src_step;
        uint8_t* out_row = tmp_rgbx.data() + (size_t)yy * (size_t)tmp_step;
        for (int xx = 0; xx < dst_w; ++xx)
        {
            const int xmin = xcoeff.bounds[(size_t)xx * 2 + 0];
            const int xmax = xcoeff.bounds[(size_t)xx * 2 + 1];
            const int32_t* k = xcoeff.coeffs_int.data() + (size_t)xx * (size_t)xcoeff.ksize;

            int ss0 = 1 << (kPillowPrecisionBits - 1);
            int ss1 = 1 << (kPillowPrecisionBits - 1);
            int ss2 = 1 << (kPillowPrecisionBits - 1);
            for (int x = 0; x < xmax; ++x)
            {
                const int sx = x + xmin;
                const uint8_t* px = row + (size_t)sx * 4;
                const int32_t kw = k[x];
                ss0 += (int)px[0] * kw;
                ss1 += (int)px[1] * kw;
                ss2 += (int)px[2] * kw;
            }
            out_row[4 * xx + 0] = pillow_clip8(ss0);
            out_row[4 * xx + 1] = pillow_clip8(ss1);
            out_row[4 * xx + 2] = pillow_clip8(ss2);
            out_row[4 * xx + 3] = 0;
        }
    }

    // Vertical pass -> RGB output [dst_h, dst_w, 3]
    output_rgb.resize((size_t)dst_h * (size_t)dst_w * 3);
    for (int yy = 0; yy < dst_h; ++yy)
    {
        const int ymin = ycoeff.bounds[(size_t)yy * 2 + 0];
        const int ymax = ycoeff.bounds[(size_t)yy * 2 + 1];
        const int32_t* k = ycoeff.coeffs_int.data() + (size_t)yy * (size_t)ycoeff.ksize;

        uint8_t* out_row = output_rgb.data() + (size_t)yy * (size_t)dst_w * 3;
        for (int xx = 0; xx < dst_w; ++xx)
        {
            int ss0 = 1 << (kPillowPrecisionBits - 1);
            int ss1 = 1 << (kPillowPrecisionBits - 1);
            int ss2 = 1 << (kPillowPrecisionBits - 1);
            for (int y = 0; y < ymax; ++y)
            {
                const uint8_t* row = tmp_rgbx.data() + (size_t)(y + ymin) * (size_t)tmp_step;
                const uint8_t* px = row + (size_t)xx * 4;
                const int32_t kw = k[y];
                ss0 += (int)px[0] * kw;
                ss1 += (int)px[1] * kw;
                ss2 += (int)px[2] * kw;
            }
            out_row[3 * xx + 0] = pillow_clip8(ss0);
            out_row[3 * xx + 1] = pillow_clip8(ss1);
            out_row[3 * xx + 2] = pillow_clip8(ss2);
        }
    }

    return true;
}

} // namespace

std::vector<axcv::Mat> ReadImages(const std::string& path) {
    std::vector<axcv::Mat> src;

    if (is_file(path)) {
        axcv::Mat img = axcv::imread(path, axcv::IMREAD_COLOR);
        if (axcv::empty(img)) {
            std::cerr << "failed to read image: " << path << "\n";
            return {};
        }
        src.push_back(img);
        return src;
    }

    if (is_directory(path)) {
        auto paths = list_files(path);
        for (auto& p : paths) {
            axcv::Mat img = axcv::imread(p, axcv::IMREAD_COLOR);
            if (axcv::empty(img)) {
                std::cerr << "failed to read image: " << p << "\n";
                continue;
            }
            src.push_back(img);
        }
        return src;
    }

    std::cerr << "invalid path: " << path << "\n";
    return {};
}

int Qwen2VideoProcessor(std::vector<axcv::Mat>& src,
                        std::vector<std::vector<unsigned char>>& output,
                        int tgt_h, int tgt_w,
                        int temporal_patch_size,
                        int merge_size,
                        int patch_size) {
    if (src.empty()) return 0;

    std::vector<std::vector<unsigned char>> imgs_resized;
    imgs_resized.reserve(src.size());

    for (auto& img : src) {
        std::vector<unsigned char> rgb;
        if (!pillow_resize_bgr_to_rgb_u8(img, rgb, tgt_w, tgt_h)) {
            return 0;
        }
        imgs_resized.push_back(std::move(rgb));
    }

    if (imgs_resized.empty()) return 0;

    // Qwen patchifier expects even number of frames for temporal_patch_size=2.
    if (temporal_patch_size > 1 && (imgs_resized.size() % (size_t)temporal_patch_size) != 0) {
        while ((imgs_resized.size() % (size_t)temporal_patch_size) != 0) {
            imgs_resized.push_back(imgs_resized.back());
        }
    }

    const int channel = 3;
    std::vector<unsigned char> patches;
    patches.resize(imgs_resized.size() * (size_t)tgt_w * (size_t)tgt_h * (size_t)channel);
    for (size_t i = 0; i < imgs_resized.size(); ++i) {
        const auto& rgb = imgs_resized[i];
        unsigned char* dst = patches.data() + i * (size_t)tgt_w * (size_t)tgt_h * (size_t)channel;
        std::memcpy(dst, rgb.data(), (size_t)tgt_w * (size_t)tgt_h * (size_t)channel);
    }

    const int grid_t = (int)imgs_resized.size() / temporal_patch_size;
    const int grid_h = tgt_h / patch_size;
    const int grid_w = tgt_w / patch_size;

    output.clear();
    output.reserve(grid_t);

    // Follow the reference reshape+transpose in the original branches.
    for (int d0 = 0; d0 < grid_t; d0++) {
        std::vector<unsigned char> out_t;
        out_t.reserve((size_t)(grid_h / merge_size) * (grid_w / merge_size) * merge_size * merge_size *
                      temporal_patch_size * patch_size * patch_size * channel);
        for (int d2 = 0; d2 < grid_h / merge_size; d2++) {
            for (int d5 = 0; d5 < grid_w / merge_size; d5++) {
                for (int d3 = 0; d3 < merge_size; d3++) {
                    for (int d6 = 0; d6 < merge_size; d6++) {
                        for (int d1 = 0; d1 < temporal_patch_size; d1++) {
                            for (int d4 = 0; d4 < patch_size; d4++) {
                                for (int d7 = 0; d7 < patch_size; d7++) {
                                    for (int d8 = 0; d8 < channel; d8++) {
                                        size_t idx = (size_t)d0 * temporal_patch_size * grid_h * patch_size * grid_w * patch_size * channel;
                                        idx += (size_t)d1 * grid_h * patch_size * grid_w * patch_size * channel;
                                        idx += (size_t)d2 * merge_size * patch_size * grid_w * patch_size * channel;
                                        idx += (size_t)d3 * patch_size * grid_w * patch_size * channel;
                                        idx += (size_t)d4 * grid_w * patch_size * channel;
                                        idx += (size_t)d5 * merge_size * patch_size * channel;
                                        idx += (size_t)d6 * patch_size * channel;
                                        idx += (size_t)d7 * channel;
                                        idx += (size_t)d8;

                                        out_t.push_back(patches[idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output.push_back(std::move(out_t));
    }

    return 0;
}

int PaddleOCRVLImageProcessor(axcv::Mat& src,
                              std::vector<unsigned char>& output,
                              int tgt_h, int tgt_w,
                              int patch_size) {
    // Resize to target size and convert BGR->RGB.
    axcv::Mat img_rs;
    if (axcv::width(src) != tgt_w || axcv::height(src) != tgt_h) {
        axcv::resize(src, img_rs, tgt_w, tgt_h);
    } else {
        img_rs = src;
    }
    axcv::Mat rgb;
    axcv::cvtColorBGR2RGB(img_rs, rgb);

    const int grid_h = tgt_h / patch_size;
    const int grid_w = tgt_w / patch_size;
    const int N = grid_h * grid_w;
    const int C = 3;

    // Output layout: [N, C, pH, pW] matching PaddleOCR-VL VIT input format (1, N, C, pH, pW).
    output.resize((size_t)N * C * patch_size * patch_size);

    size_t idx = 0;
    for (int n = 0; n < N; n++) {
        const int gh = n / grid_w;
        const int gw = n % grid_w;
        for (int c = 0; c < C; c++) {
            for (int ph = 0; ph < patch_size; ph++) {
                const int row = gh * patch_size + ph;
                const uint8_t* row_ptr = axcv::row_ptr(rgb, row);
                for (int pw = 0; pw < patch_size; pw++) {
                    const int col = gw * patch_size + pw;
                    output[idx++] = row_ptr[col * C + c];
                }
            }
        }
    }

    return 0;
}

int Gemma4ImageProcessor(axcv::Mat& src,
                         std::vector<unsigned char>& output,
                         int tgt_h, int tgt_w,
                         int patch_size) {
    axcv::Mat img_rs;
    if (axcv::width(src) != tgt_w || axcv::height(src) != tgt_h) {
        axcv::resize(src, img_rs, tgt_w, tgt_h);
    } else {
        img_rs = src;
    }

    axcv::Mat rgb;
    axcv::cvtColorBGR2RGB(img_rs, rgb);

    const int grid_h = tgt_h / patch_size;
    const int grid_w = tgt_w / patch_size;
    const int pixel_dim = patch_size * patch_size * 3;
    output.resize((size_t)grid_h * (size_t)grid_w * (size_t)pixel_dim);

    size_t idx = 0;
    for (int gh = 0; gh < grid_h; ++gh) {
        for (int gw = 0; gw < grid_w; ++gw) {
            for (int ph = 0; ph < patch_size; ++ph) {
                const int row = gh * patch_size + ph;
                const uint8_t* row_ptr = axcv::row_ptr(rgb, row);
                for (int pw = 0; pw < patch_size; ++pw) {
                    const int col = gw * patch_size + pw;
                    const uint8_t* pixel = row_ptr + col * 3;
                    output[idx++] = pixel[0];
                    output[idx++] = pixel[1];
                    output[idx++] = pixel[2];
                }
            }
        }
    }

    return 0;
}

static std::vector<axcv::Mat> splitImageSafe(axcv::Mat src, int rows, int cols, int tile_w, int tile_h) {
    std::vector<axcv::Mat> subImages;

    const int full_w = std::max(1, tile_w) * cols;
    const int full_h = std::max(1, tile_h) * rows;

    // SmolVLM2 reference: first resize to (2*tile_w)x(2*tile_h) then split to 2x2 tiles.
    if (axcv::width(src) != full_w || axcv::height(src) != full_h) {
        axcv::Mat img_rs;
        axcv::resize(src, img_rs, full_w, full_h);
        src = std::move(img_rs);
    }

    const int subHeight = axcv::height(src) / rows;
    const int subWidth = axcv::width(src) / cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int x = j * subWidth;
            int y = i * subHeight;
            int w = (j == cols - 1) ? axcv::width(src) - x : subWidth;
            int h = (i == rows - 1) ? axcv::height(src) - y : subHeight;

            if (x >= 0 && y >= 0 && x + w <= axcv::width(src) && y + h <= axcv::height(src)) {
                axcv::Mat subImage = axcv::crop_clone(src, x, y, w, h);
                if (axcv::width(subImage) != tile_w || axcv::height(subImage) != tile_h) {
                    axcv::Mat rs;
                    axcv::resize(subImage, rs, tile_w, tile_h);
                    subImage = std::move(rs);
                }
                subImages.push_back(std::move(subImage));
            }
        }
    }

    return subImages;
}

int Smolvlm2ImageProcessor(std::vector<axcv::Mat>& src,
                           std::vector<std::vector<unsigned char>>& output,
                           int tgt_w,
                           int tgt_h) {
    if (src.empty()) return 0;
    tgt_w = std::max(1, tgt_w);
    tgt_h = std::max(1, tgt_h);

    std::vector<axcv::Mat> resized;
    resized.reserve(src.size() * 5);

    for (auto& img : src) {
        auto splited = splitImageSafe(img, 2, 2, tgt_w, tgt_h);
        resized.insert(resized.end(), splited.begin(), splited.end());
        if (axcv::width(img) != tgt_w || axcv::height(img) != tgt_h) {
            axcv::Mat rs;
            axcv::resize(img, rs, tgt_w, tgt_h);
            img = std::move(rs);
        }
        resized.push_back(img);
    }

    output.clear();
    output.reserve(resized.size());
    for (auto& img : resized) {
        std::vector<unsigned char> imgdata;
        imgdata.resize((size_t)tgt_w * (size_t)tgt_h * 3);
        // Pack in row-major contiguous.
        for (int r = 0; r < tgt_h; ++r) {
            const uint8_t* sp = axcv::row_ptr(img, r);
            std::memcpy(imgdata.data() + (size_t)r * (size_t)tgt_w * 3, sp, (size_t)tgt_w * 3);
        }
        output.push_back(std::move(imgdata));
    }

    return 0;
}

int Smolvlm2VideoProcessor(std::vector<axcv::Mat>& src,
                           std::vector<std::vector<unsigned char>>& output,
                           int tgt_w,
                           int tgt_h) {
    if (src.empty()) return 0;
    tgt_w = std::max(1, tgt_w);
    tgt_h = std::max(1, tgt_h);

    output.clear();
    output.reserve(src.size());

    for (auto& img : src) {
        if (axcv::width(img) != tgt_w || axcv::height(img) != tgt_h) {
            axcv::Mat rs;
            axcv::resize(img, rs, tgt_w, tgt_h);
            img = std::move(rs);
        }
        std::vector<unsigned char> imgdata;
        imgdata.resize((size_t)tgt_w * (size_t)tgt_h * 3);
        for (int r = 0; r < tgt_h; ++r) {
            const uint8_t* sp = axcv::row_ptr(img, r);
            std::memcpy(imgdata.data() + (size_t)r * (size_t)tgt_w * 3, sp, (size_t)tgt_w * 3);
        }
        output.push_back(std::move(imgdata));
    }

    return 0;
}
