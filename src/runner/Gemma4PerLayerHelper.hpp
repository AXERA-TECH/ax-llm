#pragma once

#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define GEMMA4_HAVE_NEON 1
#else
#define GEMMA4_HAVE_NEON 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils/bfloat16.hpp"
#include "utils/memory_utils.hpp"
#include "utils/sample_log.h"

class Gemma4PerLayerHelper
{
    struct NpyView
    {
        std::string descr;
        std::vector<size_t> shape;
        size_t data_offset = 0;
        const void *data = nullptr;
        size_t data_size = 0;
    };

    enum class EmbedFormat
    {
        None,
        RawBF16,
        NpyF16,
        NpyBF16,
    };

    bool enabled_ = false;
    int vocab_size_ = 0;
    int hidden_size_ = 0;
    int num_hidden_layers_ = 0;
    int hidden_size_per_layer_input_ = 0;
    int pad_token_id_ = 0;
    float rms_norm_eps_ = 1e-6f;
    float embed_scale_ = 1.0f;
    float projection_scale_ = 1.0f;
    float merge_scale_ = 0.7071067811865475f;

    MMap embed_tokens_per_layer_map_;
    MMap projection_map_;
    MMap projection_norm_map_;
    EmbedFormat embed_format_ = EmbedFormat::None;
    size_t embed_row_stride_words_ = 0;
    NpyView embed_tokens_per_layer_view_;
    NpyView projection_view_;
    NpyView projection_norm_view_;

    static bool parse_npy_header(const void *data, size_t size, NpyView &view, std::string &err)
    {
        view = {};
        if (!data || size < 10)
        {
            err = "npy file too small";
            return false;
        }

        const auto *bytes = static_cast<const unsigned char *>(data);
        if (std::memcmp(bytes, "\x93NUMPY", 6) != 0)
        {
            err = "invalid npy magic";
            return false;
        }

        const unsigned char major = bytes[6];
        const unsigned char minor = bytes[7];
        (void)minor;

        size_t header_len = 0;
        size_t header_offset = 0;
        if (major == 1)
        {
            if (size < 10)
            {
                err = "npy v1 header truncated";
                return false;
            }
            header_len = (size_t)bytes[8] | ((size_t)bytes[9] << 8);
            header_offset = 10;
        }
        else if (major == 2 || major == 3)
        {
            if (size < 12)
            {
                err = "npy v2/v3 header truncated";
                return false;
            }
            header_len = (size_t)bytes[8] |
                         ((size_t)bytes[9] << 8) |
                         ((size_t)bytes[10] << 16) |
                         ((size_t)bytes[11] << 24);
            header_offset = 12;
        }
        else
        {
            err = "unsupported npy version";
            return false;
        }

        if (header_offset + header_len > size)
        {
            err = "npy header exceeds file size";
            return false;
        }

        const std::string header(reinterpret_cast<const char *>(bytes + header_offset), header_len);

        std::smatch match;
        if (!std::regex_search(header, match, std::regex("'descr'\\s*:\\s*'([^']+)'")))
        {
            err = "npy header missing descr";
            return false;
        }
        view.descr = match[1].str();

        if (std::regex_search(header, match, std::regex("'fortran_order'\\s*:\\s*(True|False)")))
        {
            if (match[1].str() != "False")
            {
                err = "fortran-order npy is not supported";
                return false;
            }
        }

        if (!std::regex_search(header, match, std::regex("'shape'\\s*:\\s*\\(([^\\)]*)\\)")))
        {
            err = "npy header missing shape";
            return false;
        }

        const std::string shape_text = match[1].str();
        std::stringstream ss(shape_text);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            std::stringstream item_stream(item);
            size_t dim = 0;
            if (!(item_stream >> dim))
            {
                continue;
            }
            view.shape.push_back(dim);
        }
        if (view.shape.empty())
        {
            err = "npy shape is empty";
            return false;
        }

        view.data_offset = header_offset + header_len;
        view.data = bytes + view.data_offset;
        view.data_size = size - view.data_offset;
        return true;
    }

    static size_t product(const std::vector<size_t> &shape)
    {
        size_t total = 1;
        for (size_t dim : shape) total *= dim;
        return total;
    }

    // Scalar fp32 dot product. Kept as a fallback when NEON isn't available.
    static inline float dot_f32_scalar(const float *a, const float *b, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

#if GEMMA4_HAVE_NEON
    // AArch64 NEON dot product. Unrolled 4x to expose ILP and avoid stalls on the
    // single FMA pipeline of Cortex-A55. `a` and `b` may be unaligned — vld1q is OK.
    static inline float dot_f32(const float *a, const float *b, int n)
    {
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        for (; i + 16 <= n; i += 16)
        {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),       vld1q_f32(b + i));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),   vld1q_f32(b + i + 4));
            acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),   vld1q_f32(b + i + 8));
            acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12),  vld1q_f32(b + i + 12));
        }
        float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        for (; i + 4 <= n; i += 4)
        {
            acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
        }
        float sum = vaddvq_f32(acc);
        for (; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }
#else
    static inline float dot_f32(const float *a, const float *b, int n)
    {
        return dot_f32_scalar(a, b, n);
    }
#endif

    const float *projection_ptr() const
    {
        return static_cast<const float *>(projection_view_.data);
    }

    const float *projection_norm_ptr() const
    {
        return static_cast<const float *>(projection_norm_view_.data);
    }

    static float fp16_to_fp32(uint16_t h)
    {
        const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
        const uint32_t exp = (h >> 10) & 0x1Fu;
        const uint32_t mant = h & 0x03FFu;

        uint32_t bits = 0;
        if (exp == 0)
        {
            if (mant == 0)
            {
                bits = sign;
            }
            else
            {
                uint32_t mantissa = mant;
                int32_t exp32 = 127 - 15 + 1;
                while ((mantissa & 0x0400u) == 0)
                {
                    mantissa <<= 1;
                    --exp32;
                }
                mantissa &= 0x03FFu;
                bits = sign | ((uint32_t)exp32 << 23) | (mantissa << 13);
            }
        }
        else if (exp == 0x1Fu)
        {
            bits = sign | 0x7F800000u | (mant << 13);
        }
        else
        {
            const uint32_t exp32 = exp + (127 - 15);
            bits = sign | (exp32 << 23) | (mant << 13);
        }

        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    const uint16_t *embed_row_ptr(int token_id) const
    {
        const void *base_ptr = nullptr;
        switch (embed_format_)
        {
        case EmbedFormat::RawBF16:
            base_ptr = embed_tokens_per_layer_map_.data();
            break;
        case EmbedFormat::NpyF16:
        case EmbedFormat::NpyBF16:
            base_ptr = embed_tokens_per_layer_view_.data;
            break;
        default:
            return nullptr;
        }
        const auto *base = static_cast<const uint16_t *>(base_ptr);
        return base + (size_t)token_id * embed_row_stride_words_;
    }

    float embed_value(const uint16_t *row, size_t offset) const
    {
        const uint16_t word = row[offset];
        switch (embed_format_)
        {
        case EmbedFormat::RawBF16:
        case EmbedFormat::NpyBF16:
            return bfloat16(word).fp32();
        case EmbedFormat::NpyF16:
            return fp16_to_fp32(word);
        default:
            return 0.0f;
        }
    }

    bool validate_paths(const std::string &embed_path,
                        const std::string &projection_path,
                        const std::string &projection_norm_path) const
    {
        return !embed_path.empty() && !projection_path.empty() && !projection_norm_path.empty();
    }

public:
    bool Init(int vocab_size,
              int hidden_size,
              int num_hidden_layers,
              int hidden_size_per_layer_input,
              int pad_token_id,
              float rms_norm_eps,
              const std::string &embed_path,
              const std::string &projection_path,
              const std::string &projection_norm_path)
    {
        enabled_ = false;
        vocab_size_ = vocab_size;
        hidden_size_ = hidden_size;
        num_hidden_layers_ = num_hidden_layers;
        hidden_size_per_layer_input_ = hidden_size_per_layer_input;
        pad_token_id_ = pad_token_id;
        rms_norm_eps_ = rms_norm_eps;
        embed_scale_ = std::sqrt((float)hidden_size_per_layer_input_);
        projection_scale_ = 1.0f / std::sqrt((float)hidden_size_);
        merge_scale_ = 1.0f / std::sqrt(2.0f);

        embed_tokens_per_layer_map_.close_file();
        projection_map_.close_file();
        projection_norm_map_.close_file();
        embed_format_ = EmbedFormat::None;
        embed_row_stride_words_ = 0;
        embed_tokens_per_layer_view_ = {};
        projection_view_ = {};
        projection_norm_view_ = {};

        if (hidden_size_per_layer_input_ <= 0)
        {
            return true;
        }

        if (!validate_paths(embed_path, projection_path, projection_norm_path))
        {
            ALOGE("Gemma4 per-layer paths are incomplete");
            return false;
        }

        if (!embed_tokens_per_layer_map_.open_file(embed_path.c_str()))
        {
            ALOGE("failed to mmap %s", embed_path.c_str());
            return false;
        }

        const size_t expect_embed_words =
            (size_t)vocab_size_ * (size_t)num_hidden_layers_ * (size_t)hidden_size_per_layer_input_;
        const size_t expect_embed_bytes = expect_embed_words * sizeof(uint16_t);
        std::string err;
        if (embed_tokens_per_layer_map_.size() >= 6 &&
            std::memcmp(embed_tokens_per_layer_map_.data(), "\x93NUMPY", 6) == 0)
        {
            if (!parse_npy_header(embed_tokens_per_layer_map_.data(), embed_tokens_per_layer_map_.size(), embed_tokens_per_layer_view_, err))
            {
                ALOGE("failed to parse %s: %s", embed_path.c_str(), err.c_str());
                return false;
            }
            if (embed_tokens_per_layer_view_.shape.size() != 2 ||
                (int)embed_tokens_per_layer_view_.shape[0] != vocab_size_ ||
                (int)embed_tokens_per_layer_view_.shape[1] != num_hidden_layers_ * hidden_size_per_layer_input_)
            {
                ALOGE("unexpected Gemma4 per-layer embed npy shape");
                return false;
            }
            if (embed_tokens_per_layer_view_.descr == "<f2" || embed_tokens_per_layer_view_.descr == "|f2" || embed_tokens_per_layer_view_.descr == "f2")
            {
                embed_format_ = EmbedFormat::NpyF16;
            }
            else if (embed_tokens_per_layer_view_.descr == "<u2" || embed_tokens_per_layer_view_.descr == "|u2" || embed_tokens_per_layer_view_.descr == "u2")
            {
                embed_format_ = EmbedFormat::NpyBF16;
            }
            else
            {
                ALOGE("unsupported Gemma4 per-layer embed dtype: %s", embed_tokens_per_layer_view_.descr.c_str());
                return false;
            }
            if (embed_tokens_per_layer_view_.data_size < expect_embed_bytes)
            {
                ALOGE("Gemma4 per-layer embed npy payload is truncated");
                return false;
            }
            embed_row_stride_words_ = embed_tokens_per_layer_view_.shape[1];
        }
        else
        {
            if (embed_tokens_per_layer_map_.size() != expect_embed_bytes)
            {
                ALOGE("unexpected Gemma4 per-layer embed size: got=%zu expect=%zu",
                      embed_tokens_per_layer_map_.size(),
                      expect_embed_bytes);
                return false;
            }
            embed_format_ = EmbedFormat::RawBF16;
            embed_row_stride_words_ = (size_t)num_hidden_layers_ * (size_t)hidden_size_per_layer_input_;
        }

        if (!projection_map_.open_file(projection_path.c_str()))
        {
            ALOGE("failed to mmap %s", projection_path.c_str());
            return false;
        }

        if (!parse_npy_header(projection_map_.data(), projection_map_.size(), projection_view_, err))
        {
            ALOGE("failed to parse %s: %s", projection_path.c_str(), err.c_str());
            return false;
        }
        if (projection_view_.descr != "<f4" && projection_view_.descr != "|f4" && projection_view_.descr != "f4")
        {
            ALOGE("unsupported Gemma4 projection dtype: %s", projection_view_.descr.c_str());
            return false;
        }
        if (!(
                (projection_view_.shape.size() == 3 &&
                 (int)projection_view_.shape[0] == num_hidden_layers_ &&
                 (int)projection_view_.shape[1] == hidden_size_per_layer_input_ &&
                 (int)projection_view_.shape[2] == hidden_size_) ||
                (projection_view_.shape.size() == 2 &&
                 (int)projection_view_.shape[0] == num_hidden_layers_ * hidden_size_per_layer_input_ &&
                 (int)projection_view_.shape[1] == hidden_size_)))
        {
            ALOGE("unexpected Gemma4 projection shape");
            return false;
        }
        if (projection_view_.data_size < product(projection_view_.shape) * sizeof(float))
        {
            ALOGE("Gemma4 projection npy payload is truncated");
            return false;
        }

        if (!projection_norm_map_.open_file(projection_norm_path.c_str()))
        {
            ALOGE("failed to mmap %s", projection_norm_path.c_str());
            return false;
        }
        if (!parse_npy_header(projection_norm_map_.data(), projection_norm_map_.size(), projection_norm_view_, err))
        {
            ALOGE("failed to parse %s: %s", projection_norm_path.c_str(), err.c_str());
            return false;
        }
        if (projection_norm_view_.descr != "<f4" && projection_norm_view_.descr != "|f4" && projection_norm_view_.descr != "f4")
        {
            ALOGE("unsupported Gemma4 projection norm dtype: %s", projection_norm_view_.descr.c_str());
            return false;
        }
        if (projection_norm_view_.shape.size() != 1 || (int)projection_norm_view_.shape[0] != hidden_size_per_layer_input_)
        {
            ALOGE("unexpected Gemma4 projection norm shape");
            return false;
        }
        if (projection_norm_view_.data_size < (size_t)hidden_size_per_layer_input_ * sizeof(float))
        {
            ALOGE("Gemma4 projection norm npy payload is truncated");
            return false;
        }

        enabled_ = true;
        ALOGI("Gemma4 per-layer helper enabled: vocab=%d hidden=%d layers=%d per_layer=%d pad=%d",
              vocab_size_,
              hidden_size_,
              num_hidden_layers_,
              hidden_size_per_layer_input_,
              pad_token_id_);
        return true;
    }

    void Deinit()
    {
        enabled_ = false;
        embed_tokens_per_layer_map_.close_file();
        projection_map_.close_file();
        projection_norm_map_.close_file();
        embed_format_ = EmbedFormat::None;
        embed_row_stride_words_ = 0;
        embed_tokens_per_layer_view_ = {};
        projection_view_ = {};
        projection_norm_view_ = {};
        decode_cache_.clear();
    }

    // Drop per-token decode cache (call on /reset so a new chat doesn't accumulate).
    void ClearDecodeCache() { decode_cache_.clear(); }

    bool enabled() const { return enabled_; }
    int hidden_size_per_layer_input() const { return hidden_size_per_layer_input_; }
    int pad_token_id() const { return pad_token_id_; }
    void reset_decode_stats(bool enable) const
    {
        decode_stats_enabled_.store(enable, std::memory_order_relaxed);
        decode_cache_hits_.store(0, std::memory_order_relaxed);
        decode_cache_misses_.store(0, std::memory_order_relaxed);
    }
    uint64_t decode_cache_hits() const { return decode_cache_hits_.load(std::memory_order_relaxed); }
    uint64_t decode_cache_misses() const { return decode_cache_misses_.load(std::memory_order_relaxed); }

    bool Compute(const std::vector<int> &token_ids,
                 const unsigned short *input_bf16,
                 int num_tokens,
                 int input_hidden_size,
                 std::vector<unsigned short> &out) const
    {
        if (!enabled_)
        {
            out.clear();
            return true;
        }
        if ((int)token_ids.size() != num_tokens)
        {
            ALOGE("Gemma4 per-layer token count mismatch: ids=%zu num_tokens=%d", token_ids.size(), num_tokens);
            return false;
        }
        if (!input_bf16 || input_hidden_size != hidden_size_)
        {
            ALOGE("Gemma4 per-layer hidden size mismatch");
            return false;
        }

        out.assign((size_t)num_tokens * (size_t)num_hidden_layers_ * (size_t)hidden_size_per_layer_input_, 0);
        const float *proj = projection_ptr();
        const float *norm = projection_norm_ptr();

        std::atomic<int> bad_token(-2);

        // Parallelize across tokens when we have more than one. For decode (num_tokens==1)
        // OpenMP has overhead with no benefit; let the runtime skip the team spawn.
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(num_tokens > 1)
#endif
        for (int token_idx = 0; token_idx < num_tokens; ++token_idx)
        {
            if (bad_token.load(std::memory_order_relaxed) != -2) continue;
            const int token_id = token_ids[(size_t)token_idx];
            if (token_id < 0 || token_id >= vocab_size_)
            {
                int expected = -2;
                bad_token.compare_exchange_strong(expected, token_id);
                continue;
            }

            std::vector<float> input_fp32((size_t)hidden_size_);
            std::vector<float> per_layer_proj((size_t)hidden_size_per_layer_input_);

            for (int i = 0; i < hidden_size_; ++i)
            {
                input_fp32[(size_t)i] = bfloat16(input_bf16[(size_t)token_idx * (size_t)hidden_size_ + (size_t)i]).fp32();
            }

            const unsigned short *token_embed = embed_row_ptr(token_id);
            for (int layer_idx = 0; layer_idx < num_hidden_layers_; ++layer_idx)
            {
                const float *layer_proj = proj + (size_t)layer_idx * (size_t)hidden_size_per_layer_input_ * (size_t)hidden_size_;
                float mean_square = 0.0f;

                for (int h = 0; h < hidden_size_per_layer_input_; ++h)
                {
                    const float *w = layer_proj + (size_t)h * (size_t)hidden_size_;
                    const float sum = dot_f32(input_fp32.data(), w, hidden_size_) * projection_scale_;
                    per_layer_proj[(size_t)h] = sum;
                    mean_square += sum * sum;
                }

                const float inv_rms = 1.0f / std::sqrt(mean_square / (float)hidden_size_per_layer_input_ + rms_norm_eps_);
                const size_t out_base = ((size_t)token_idx * (size_t)num_hidden_layers_ + (size_t)layer_idx) * (size_t)hidden_size_per_layer_input_;
                const size_t token_base = (size_t)layer_idx * (size_t)hidden_size_per_layer_input_;
                for (int h = 0; h < hidden_size_per_layer_input_; ++h)
                {
                    const float token_fp32 = embed_value(token_embed, token_base + (size_t)h) * embed_scale_;
                    const float proj_fp32 = per_layer_proj[(size_t)h] * inv_rms * norm[(size_t)h];
                    out[out_base + (size_t)h] = bfloat16((proj_fp32 + token_fp32) * merge_scale_).data;
                }
            }
        }

        const int bad = bad_token.load(std::memory_order_relaxed);
        if (bad != -2)
        {
            ALOGE("Gemma4 token id out of range: %d", bad);
            return false;
        }
        return true;
    }

    bool ComputeSingle(int token_id,
                       const unsigned short *input_bf16,
                       int input_hidden_size,
                       std::vector<unsigned short> &out) const
    {
        // Decode-path fast path: per-layer input is a pure function of token_id
        // because input_bf16 here is the scaled token embedding (deterministic per id).
        // Matches Python Gemma4PerLayerInputs.decode_input which caches by token_id.
        if (enabled_)
        {
            auto it = decode_cache_.find(token_id);
            if (it != decode_cache_.end())
            {
                if (decode_stats_enabled_.load(std::memory_order_relaxed))
                    decode_cache_hits_.fetch_add(1, std::memory_order_relaxed);
                out = it->second;
                return true;
            }
        }
        if (enabled_ && decode_stats_enabled_.load(std::memory_order_relaxed))
            decode_cache_misses_.fetch_add(1, std::memory_order_relaxed);
        std::vector<int> ids = {token_id};
        if (!Compute(ids, input_bf16, 1, input_hidden_size, out))
            return false;
        if (enabled_) cache_store(token_id, out.data());
        return true;
    }

    // Prefill counterpart of ComputeSingle. Building the per-layer input costs
    // 42 * 256 dot products of length 2560 per position -- about 27.5 M MAC -- and it
    // runs before the NPU sees anything, so on AX650 it is roughly half of TTFT. For a
    // text position the projection input is `text_scale * embed_tokens[token_id]`, a
    // uniform scalar times a table row, so the result is a pure function of the token
    // id, exactly as the decode path already assumes. Compute each distinct id once,
    // reuse it for every repeat, and share the cache with decode.
    //
    // `cacheable[i] == 0` marks a position whose embedding comes from the vision or
    // audio encoder. Those are not a function of any token id and must always be
    // computed; caching them would silently corrupt multimodal output.
    bool ComputeCached(const std::vector<int> &token_ids,
                       const unsigned short *input_bf16,
                       int num_tokens,
                       int input_hidden_size,
                       const std::vector<unsigned char> &cacheable,
                       std::vector<unsigned short> &out) const
    {
        if (!enabled_)
        {
            out.clear();
            return true;
        }
        if ((int)cacheable.size() != num_tokens || num_tokens <= 0)
            return Compute(token_ids, input_bf16, num_tokens, input_hidden_size, out);

        const size_t row = (size_t)num_hidden_layers_ * (size_t)hidden_size_per_layer_input_;
        const size_t hidden = (size_t)input_hidden_size;

        std::vector<int> batch_ids;
        std::vector<unsigned short> batch_embed;
        std::vector<int> batch_cache_id;                       // token id to memoise, else -1
        std::vector<int> plan((size_t)num_tokens, -1);         // index into the batch
        std::vector<const std::vector<unsigned short> *> hit((size_t)num_tokens, nullptr);
        std::unordered_map<int, int> first_in_batch;

        for (int i = 0; i < num_tokens; ++i)
        {
            const int id = token_ids[(size_t)i];
            if (cacheable[(size_t)i])
            {
                auto c = decode_cache_.find(id);
                if (c != decode_cache_.end() && c->second.size() == row)
                {
                    hit[(size_t)i] = &c->second;
                    continue;
                }
                auto f = first_in_batch.find(id);
                if (f != first_in_batch.end())
                {
                    plan[(size_t)i] = f->second;
                    continue;
                }
            }
            const int slot = (int)batch_ids.size();
            batch_ids.push_back(id);
            batch_embed.insert(batch_embed.end(),
                               input_bf16 + (size_t)i * hidden,
                               input_bf16 + ((size_t)i + 1) * hidden);
            batch_cache_id.push_back(cacheable[(size_t)i] ? id : -1);
            if (cacheable[(size_t)i]) first_in_batch.emplace(id, slot);
            plan[(size_t)i] = slot;
        }

        std::vector<unsigned short> batch_out;
        if (!batch_ids.empty() &&
            !Compute(batch_ids, batch_embed.data(), (int)batch_ids.size(), input_hidden_size, batch_out))
        {
            return false;
        }

        out.assign((size_t)num_tokens * row, 0);
        for (int i = 0; i < num_tokens; ++i)
        {
            const unsigned short *src = hit[(size_t)i]
                                            ? hit[(size_t)i]->data()
                                            : batch_out.data() + (size_t)plan[(size_t)i] * row;
            std::memcpy(out.data() + (size_t)i * row, src, row * sizeof(unsigned short));
        }

        // Insert after the copies above, so the pointers held in `hit` stay valid.
        for (size_t s = 0; s < batch_cache_id.size(); ++s)
        {
            if (batch_cache_id[s] >= 0)
                cache_store(batch_cache_id[s], batch_out.data() + s * row);
        }

        if (decode_stats_enabled_.load(std::memory_order_relaxed))
        {
            uint64_t hits = 0;
            for (int i = 0; i < num_tokens; ++i)
                if (hit[(size_t)i] || plan[(size_t)i] != i) ++hits;
            decode_cache_hits_.fetch_add(hits, std::memory_order_relaxed);
            decode_cache_misses_.fetch_add((uint64_t)batch_ids.size(), std::memory_order_relaxed);
        }
        if (std::getenv("AXLLM_PROFILE_PERLAYER"))
        {
            ALOGI("[perlayer] positions=%d computed=%zu (%.0f%% reused) cache=%zu/%d",
                  num_tokens, batch_ids.size(),
                  100.0 * (1.0 - (double)batch_ids.size() / (double)num_tokens),
                  decode_cache_.size(), cache_capacity());
        }
        return true;
    }

private:
    // Each entry is num_hidden_layers * hidden_size_per_layer_input bf16 values, so
    // 21.5 KiB for the 42x256 Gemma 4 E4B shape. Bounded so a long session cannot grow
    // without limit; frequent tokens land in it first, which is what matters.
    int cache_capacity() const
    {
        static const int cap = [] {
            if (const char *env = std::getenv("AXLLM_GEMMA4_PERLAYER_CACHE"))
            {
                const int v = std::atoi(env);
                if (v >= 0) return v;
            }
            return 2048;
        }();
        return cap;
    }

    void cache_store(int token_id, const unsigned short *row_data) const
    {
        const size_t row = (size_t)num_hidden_layers_ * (size_t)hidden_size_per_layer_input_;
        if ((int)decode_cache_.size() >= cache_capacity()) return;
        decode_cache_.emplace(token_id, std::vector<unsigned short>(row_data, row_data + row));
    }

    mutable std::unordered_map<int, std::vector<unsigned short>> decode_cache_;
    mutable std::atomic<bool> decode_stats_enabled_{false};
    mutable std::atomic<uint64_t> decode_cache_hits_{0};
    mutable std::atomic<uint64_t> decode_cache_misses_{0};
};
