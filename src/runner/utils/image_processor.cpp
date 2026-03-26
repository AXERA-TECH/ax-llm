#include "image_processor.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

#include "files.hpp"

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

    std::vector<axcv::Mat> imgs_resized;
    imgs_resized.reserve(src.size());

    for (auto& img : src) {
        axcv::Mat img_rs;
        if (axcv::width(img) != tgt_w || axcv::height(img) != tgt_h) {
            axcv::resize(img, img_rs, tgt_w, tgt_h);
        } else {
            img_rs = img;
        }
        axcv::Mat rgb;
        axcv::cvtColorBGR2RGB(img_rs, rgb);
        img_rs = std::move(rgb);
        imgs_resized.push_back(img_rs);
    }

    if (imgs_resized.empty()) return 0;

    // Qwen patchifier expects even number of frames for temporal_patch_size=2.
    if (temporal_patch_size > 1 && (imgs_resized.size() % (size_t)temporal_patch_size) != 0) {
        while ((imgs_resized.size() % (size_t)temporal_patch_size) != 0) {
            imgs_resized.push_back(imgs_resized.back());
        }
    }

    const int channel = axcv::channels(imgs_resized[0]); // 3
    std::vector<unsigned char> patches;
    patches.resize(imgs_resized.size() * (size_t)tgt_w * (size_t)tgt_h * (size_t)channel);
    // Pack to contiguous [T, H, W, C] regardless of backend stride.
    for (size_t i = 0; i < imgs_resized.size(); ++i) {
        const auto& m = imgs_resized[i];
        unsigned char* dst = patches.data() + i * (size_t)tgt_w * (size_t)tgt_h * (size_t)channel;
        for (int r = 0; r < tgt_h; ++r) {
            const uint8_t* sp = axcv::row_ptr(m, r);
            std::memcpy(dst + (size_t)r * (size_t)tgt_w * (size_t)channel, sp, (size_t)tgt_w * (size_t)channel);
        }
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
