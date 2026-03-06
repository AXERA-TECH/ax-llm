#pragma once

#include <cstdint>
#include <cstring>
#include <string>

// Minimal OpenCV-like facade used by vision preprocessing.
//
// Backend selection:
// - If OpenCV is found by CMake: compile with -DAXLLM_USE_OPENCV and use cv::Mat.
// - Otherwise: use SimpleCV (stb-based) as a fallback.
//
// Note: SimpleCV resize/cvtColor may have slight numeric differences vs OpenCV.

#if defined(AXLLM_USE_OPENCV)
#include <opencv2/opencv.hpp>

namespace axcv {

using Mat = cv::Mat;

static constexpr int IMREAD_COLOR = cv::IMREAD_COLOR;

inline bool empty(const Mat& m) { return m.empty(); }
inline int width(const Mat& m) { return m.cols; }
inline int height(const Mat& m) { return m.rows; }
inline int channels(const Mat& m) { return m.channels(); }
inline int step_bytes(const Mat& m) { return (int)m.step; }
inline const uint8_t* row_ptr(const Mat& m, int y) { return m.ptr<uint8_t>(y); }
inline uint8_t* row_ptr(Mat& m, int y) { return m.ptr<uint8_t>(y); }

inline Mat imread(const std::string& path, int flags = IMREAD_COLOR) { return cv::imread(path, flags); }

inline void resize(const Mat& src, Mat& dst, int dst_w, int dst_h)
{
    cv::resize(src, dst, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_CUBIC);
}

inline void cvtColorBGR2RGB(const Mat& src, Mat& dst) { cv::cvtColor(src, dst, cv::COLOR_BGR2RGB); }

inline Mat crop_clone(const Mat& src, int x, int y, int w, int h)
{
    return src(cv::Rect(x, y, w, h)).clone();
}

} // namespace axcv

#else

#include "SimpleCV.hpp"

namespace axcv {

using Mat = SimpleCV::Mat;

// Kept for call-site compatibility (ignored by SimpleCV wrapper).
static constexpr int IMREAD_COLOR = 1;

inline bool empty(const Mat& m) { return m.empty(); }
inline int width(const Mat& m) { return m.width; }
inline int height(const Mat& m) { return m.height; }
inline int channels(const Mat& m) { return m.channels; }
inline int step_bytes(const Mat& m) { return m.step; }
inline const uint8_t* row_ptr(const Mat& m, int y) { return (const uint8_t*)(m.data + (size_t)y * (size_t)m.step); }
inline uint8_t* row_ptr(Mat& m, int y) { return (uint8_t*)(m.data + (size_t)y * (size_t)m.step); }

inline Mat imread(const std::string& path, int /*flags*/ = IMREAD_COLOR)
{
    // Match OpenCV's cv::imread(IMREAD_COLOR) behavior (BGR 3-channel).
    return SimpleCV::imread(path, SimpleCV::ColorSpace::BGR);
}

inline void resize(const Mat& src, Mat& dst, int dst_w, int dst_h) { SimpleCV::resize(src, dst, dst_w, dst_h); }

inline void cvtColorBGR2RGB(const Mat& src, Mat& dst)
{
    SimpleCV::cvtColor(src, dst, SimpleCV::ColorSpace::RGB, SimpleCV::ColorSpace::BGR);
}

inline Mat crop_clone(const Mat& src, int x, int y, int w, int h)
{
    if (src.empty() || w <= 0 || h <= 0) return Mat();
    Mat out(h, w, src.channels);
    const int c = src.channels;
    for (int r = 0; r < h; ++r) {
        const uint8_t* sp = row_ptr(src, y + r) + (size_t)x * (size_t)c;
        uint8_t* dp = row_ptr(out, r);
        std::memcpy(dp, sp, (size_t)w * (size_t)c);
    }
    return out;
}

} // namespace axcv

#endif

