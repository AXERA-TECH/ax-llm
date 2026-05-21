#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bfloat16.hpp"
#include "sample_log.h"

#ifdef USE_AXCL
#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "utils/axcl_manager.h"
#else
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#endif

namespace qwen3_tts {

#ifdef USE_AXCL
using AxRunner = ax_runner_axcl;
#else
using AxRunner = ax_runner_ax650;
#endif

inline constexpr int kSampleRate = 24000;
inline constexpr int kSpeakerSamples = 72000;
inline constexpr int kSpeechEncoderSamples = 72000;
inline constexpr int kSpeechEncodeDownsample = 1920;
inline constexpr int kSpeechDecodeUpsample = 1920;
inline constexpr int kTalkerHidden = 1024;
inline constexpr int kTextHidden = 2048;
inline constexpr int kTalkerVocab = 3072;
inline constexpr int kCodecVocab = 2048;
inline constexpr int kCodeGroups = 16;
inline constexpr int kNumSubCodes = kCodeGroups - 1;
inline constexpr int kTalkerLayers = 28;
inline constexpr int kCodePredictorLayers = 5;
inline constexpr int kSpeechDecoderLayers = 8;

inline constexpr int kTtsPadTokenId = 151671;
inline constexpr int kTtsBosTokenId = 151672;
inline constexpr int kTtsEosTokenId = 151673;

inline constexpr int kCodecPadId = 2148;
inline constexpr int kCodecBosId = 2149;
inline constexpr int kCodecEosId = 2150;
inline constexpr int kCodecThinkId = 2154;
inline constexpr int kCodecNoThinkId = 2155;
inline constexpr int kCodecThinkBosId = 2156;
inline constexpr int kCodecThinkEosId = 2157;

inline std::string join_path(const std::string &a, const std::string &b)
{
    return (std::filesystem::path(a) / b).lexically_normal().string();
}

inline size_t shape_elems(const std::vector<unsigned int> &shape)
{
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    return n;
}

inline std::string shape_to_string(const std::vector<unsigned int> &shape)
{
    if (shape.empty()) return "[]";
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0) s += "x";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

inline std::vector<unsigned short> fp32_to_bf16_vec(const std::vector<float> &src)
{
    std::vector<unsigned short> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = fp32_to_bfloat16_rne(src[i]);
    return dst;
}

inline std::vector<float> bf16_to_fp32_vec(const unsigned short *src, size_t n)
{
    std::vector<float> dst(n);
    for (size_t i = 0; i < n; ++i) dst[i] = bfloat16(src[i]).fp32();
    return dst;
}

inline void add_bf16_to_float(const unsigned short *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] += bfloat16(src[i]).fp32();
}

inline std::vector<unsigned short> sum_bf16_embeddings(const std::vector<std::vector<unsigned short>> &parts)
{
    std::vector<float> acc((size_t)kTalkerHidden, 0.0f);
    for (const auto &p : parts)
    {
        if (p.size() != (size_t)kTalkerHidden) throw std::runtime_error("unexpected embedding size");
        add_bf16_to_float(p.data(), acc.data(), acc.size());
    }
    return fp32_to_bf16_vec(acc);
}

inline void append_bf16(std::vector<unsigned short> &dst, const std::vector<unsigned short> &src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

inline std::vector<unsigned short> slice_bf16_rows(const std::vector<unsigned short> &src,
                                                   int start,
                                                   int rows,
                                                   int width)
{
    if (rows <= 0) return {};
    const size_t off = (size_t)start * (size_t)width;
    const size_t cnt = (size_t)rows * (size_t)width;
    if (off + cnt > src.size()) throw std::runtime_error("slice_bf16_rows out of range");
    return std::vector<unsigned short>(src.begin() + (ptrdiff_t)off, src.begin() + (ptrdiff_t)(off + cnt));
}

inline void fit_audio(std::vector<float> &audio, size_t samples)
{
    if (audio.size() > samples) audio.resize(samples);
    else if (audio.size() < samples) audio.resize(samples, 0.0f);
}

#ifdef USE_AXCL
inline void tts_h2d(void *phy_dst, const void *src, size_t n, int devid)
{
    axcl_Memcpy(phy_dst, src, n, AXCL_MEMCPY_HOST_TO_DEVICE, devid);
}

inline void tts_d2h(void *dst, const void *phy_src, size_t n, int devid)
{
    axcl_Memcpy(dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_HOST, devid);
}

inline void *tensor_write_addr(const ax_runner_tensor_t &t) { return (void *)t.phyAddr; }
inline const void *tensor_read_addr(const ax_runner_tensor_t &t) { return (const void *)t.phyAddr; }
#else
inline void tts_h2d(void *vir_dst, const void *src, size_t n, int /*devid*/)
{
    std::memcpy(vir_dst, src, n);
}

inline void tts_d2h(void *dst, const void *vir_src, size_t n, int /*devid*/)
{
    std::memcpy(dst, vir_src, n);
}

inline void *tensor_write_addr(const ax_runner_tensor_t &t) { return t.pVirAddr; }
inline const void *tensor_read_addr(const ax_runner_tensor_t &t) { return (const void *)t.pVirAddr; }
#endif

inline const ax_runner_tensor_t &input_tensor(AxRunner &runner, int gid, const std::string &name)
{
    try
    {
        return runner.get_input(gid, name);
    }
    catch (...)
    {
        if (gid > 0)
        {
            try
            {
                return runner.get_input(gid, name + "_" + std::to_string(gid));
            }
            catch (...)
            {
            }
        }
    }
    throw std::runtime_error("missing input tensor: " + name + " gid=" + std::to_string(gid));
}

inline const ax_runner_tensor_t &output_tensor(AxRunner &runner, int gid, const std::string &name)
{
    try
    {
        return runner.get_output(gid, name);
    }
    catch (...)
    {
        if (gid > 0)
        {
            try
            {
                return runner.get_output(gid, name + "_" + std::to_string(gid));
            }
            catch (...)
            {
            }
        }
    }
    throw std::runtime_error("missing output tensor: " + name + " gid=" + std::to_string(gid));
}

inline void write_tensor(AxRunner &runner, const ax_runner_tensor_t &t, const void *data, size_t bytes)
{
    if (bytes > (size_t)t.nSize) throw std::runtime_error("input tensor overflow: " + t.sName);
    if ((size_t)t.nSize > 0 && tensor_write_addr(t) == nullptr)
        throw std::runtime_error("input tensor has null buffer: " + t.sName);
    if ((size_t)t.nSize > bytes)
    {
        std::vector<unsigned char> zeros((size_t)t.nSize, 0);
        if (bytes > 0) std::memcpy(zeros.data(), data, bytes);
        tts_h2d(tensor_write_addr(t), zeros.data(), zeros.size(), runner.get_devid());
    }
    else if (bytes > 0)
    {
        tts_h2d(tensor_write_addr(t), data, bytes, runner.get_devid());
    }
}

inline void read_tensor(AxRunner &runner, const ax_runner_tensor_t &t, void *dst, size_t bytes)
{
    if (bytes > (size_t)t.nSize) throw std::runtime_error("output tensor overflow: " + t.sName);
    if (bytes > 0 && tensor_read_addr(t) == nullptr)
        throw std::runtime_error("output tensor has null buffer: " + t.sName);
    tts_d2h(dst, tensor_read_addr(t), bytes, runner.get_devid());
}

class Runner
{
public:
    bool init(const std::string &path)
    {
        const int ret = runner_.init(path.c_str(), -1);
        if (ret != 0)
        {
            ALOGE("init axmodel failed: %s ret=%d", path.c_str(), ret);
            return false;
        }
        return true;
    }

    void deinit() { runner_.deinit(); }
    AxRunner &get() { return runner_; }

private:
    AxRunner runner_;
};

inline std::vector<float> tensor_to_float(AxRunner &runner, const ax_runner_tensor_t &t, size_t elems)
{
    if (elems == 0) elems = shape_elems(t.vShape);
    if (elems == 0) elems = (size_t)t.nSize / sizeof(float);

    if ((size_t)t.nSize >= elems * sizeof(float) && (size_t)t.nSize < elems * sizeof(float) + 16)
    {
        std::vector<float> out(elems);
        read_tensor(runner, t, out.data(), elems * sizeof(float));
        return out;
    }
    if ((size_t)t.nSize >= elems * sizeof(unsigned short))
    {
        std::vector<unsigned short> tmp(elems);
        read_tensor(runner, t, tmp.data(), elems * sizeof(unsigned short));
        return bf16_to_fp32_vec(tmp.data(), elems);
    }
    throw std::runtime_error("cannot convert tensor to float: " + t.sName);
}

inline std::vector<unsigned short> tensor_to_bf16(AxRunner &runner, const ax_runner_tensor_t &t, size_t elems)
{
    if (elems == 0) elems = shape_elems(t.vShape);
    if ((size_t)t.nSize >= elems * sizeof(unsigned short) && (size_t)t.nSize < elems * sizeof(float))
    {
        std::vector<unsigned short> out(elems);
        read_tensor(runner, t, out.data(), elems * sizeof(unsigned short));
        return out;
    }
    std::vector<float> f = tensor_to_float(runner, t, elems);
    return fp32_to_bf16_vec(f);
}

inline int select_from_logits(std::vector<float> logits,
                              bool do_sample,
                              int top_k,
                              float top_p,
                              float temperature,
                              std::mt19937 &rng)
{
    if (!do_sample)
    {
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    const float temp = std::max(temperature, 1e-5f);
    for (float &v : logits) v /= temp;

    if (top_k > 0 && top_k < (int)logits.size())
    {
        std::vector<float> sorted = logits;
        std::nth_element(sorted.begin(), sorted.end() - top_k, sorted.end());
        const float kth = sorted[sorted.size() - (size_t)top_k];
        for (float &v : logits)
            if (v < kth) v = -std::numeric_limits<float>::infinity();
    }

    const float max_v = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size(), 0.0f);
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        if (std::isfinite(logits[i]))
        {
            probs[i] = std::exp(logits[i] - max_v);
            sum += probs[i];
        }
    }
    if (sum <= 0.0) return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    for (float &p : probs) p = (float)((double)p / sum);

    if (top_p > 0.0f && top_p < 1.0f)
    {
        std::vector<int> order(probs.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return probs[(size_t)a] > probs[(size_t)b]; });
        double csum = 0.0;
        for (size_t rank = 0; rank < order.size(); ++rank)
        {
            csum += probs[(size_t)order[rank]];
            if (rank > 0 && csum > top_p) probs[(size_t)order[rank]] = 0.0f;
        }
        sum = 0.0;
        for (float p : probs) sum += p;
        if (sum > 0.0)
            for (float &p : probs) p = (float)((double)p / sum);
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

} // namespace qwen3_tts
