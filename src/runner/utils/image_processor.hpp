#pragma once

#include <string>
#include <vector>

#include "ax_cv.hpp"

// Read one image if `path` is file, or all regular files (sorted) if `path` is directory.
std::vector<axcv::Mat> ReadImages(const std::string& path);

// Qwen2.x/Qwen3-VL style patchifier for image/video frames.
// Output contains one entry per temporal grid segment.
int Qwen2VideoProcessor(std::vector<axcv::Mat>& src,
                        std::vector<std::vector<unsigned char>>& output,
                        int tgt_h, int tgt_w,
                        int temporal_patch_size = 2,
                        int merge_size = 2,
                        int patch_size = 14);

// PaddleOCR-VL image processor: produces patches in [N, C, pH, pW] format
// (channel-first per patch, row-major across patches, no spatial merge).
// This is the format expected by the PaddleOCR-VL VIT axmodel.
int PaddleOCRVLImageProcessor(axcv::Mat& src,
                              std::vector<unsigned char>& output,
                              int tgt_h, int tgt_w,
                              int patch_size = 14);

// Gemma4 image processor:
// Resize -> RGB -> patchify to [num_patches, patch_size * patch_size * 3] in HWC order.
// Values remain uint8 here; callers typically convert to float32 and rescale by 1/255.
int Gemma4ImageProcessor(axcv::Mat& src,
                         std::vector<unsigned char>& output,
                         int tgt_h, int tgt_w,
                         int patch_size = 16);

// SmolVLM2 image processor:
// For each input image, outputs 5 blocks: 2x2 tiles (4) + global resized image (1).
int Smolvlm2ImageProcessor(std::vector<axcv::Mat>& src,
                           std::vector<std::vector<unsigned char>>& output,
                           int tgt_w = 512,
                           int tgt_h = 512);

// SmolVLM2 video processor:
// For each frame, outputs 1 block (resized).
int Smolvlm2VideoProcessor(std::vector<axcv::Mat>& src,
                           std::vector<std::vector<unsigned char>>& output,
                           int tgt_w = 512,
                           int tgt_h = 512);
