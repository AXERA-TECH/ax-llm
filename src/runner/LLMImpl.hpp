// Auto-split from LLM.cpp: LLM::Impl class definition + file-local helpers.
// Method bodies for the embedding path live in LLM_embed.cpp.
#pragma once
#include "LLM.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#ifndef _WIN32
#include <sys/sysinfo.h>
#endif

#include "bfloat16.hpp"
#include "Gemma4PerLayerHelper.hpp"
#include "LLMEmbedSelector.hpp"
#include "LLMPostprocess.hpp"
#include "UTF8Filter.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "utils/memory_utils.hpp"
#include "sample_log.h"
#include "ChannelSectionFilter.hpp"

#include "vision/vision_module.hpp"

#include "ax_cmm_utils.hpp"  // memory queries + pre-load mem guard (both backends)
#include "KvSlotTypes.hpp"
#include "KvSlotSelect.hpp"
#include "MemGuard.hpp"
#include "KvSlotManager.hpp"

#include "LLMLayer.hpp"

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))



namespace {

template <typename RunnerT>
const ax_runner_tensor_t *try_get_group_input_tensor(RunnerT &runner, int grpid, const std::string &name)
{
    try
    {
        return &runner.get_input(grpid, name);
    }
    catch (...)
    {
    }

    if (grpid > 0)
    {
        try
        {
            return &runner.get_input(grpid, name + "_" + std::to_string(grpid));
        }
        catch (...)
        {
        }
    }

    return nullptr;
}

template <typename RunnerT>
const ax_runner_tensor_t *try_get_output_tensor(RunnerT &runner, const std::string &name)
{
    try
    {
        return &runner.get_output(name);
    }
    catch (...)
    {
    }
    return nullptr;
}

template <typename RunnerT>
const ax_runner_tensor_t *try_get_group_output_tensor(RunnerT &runner, int grpid, const std::string &name)
{
    try
    {
        return &runner.get_output(grpid, name);
    }
    catch (...)
    {
    }

    if (grpid > 0)
    {
        try
        {
            return &runner.get_output(grpid, name + "_" + std::to_string(grpid));
        }
        catch (...)
        {
        }
    }

    return nullptr;
}

static inline std::string safe_decode_token(const std::shared_ptr<BaseTokenizer> &tokenizer, int token_id)
{
    if (!tokenizer)
        return {};

    try
    {
        return tokenizer->decode(token_id);
    }
    catch (const std::exception &e)
    {
        ALOGW("tokenizer decode failed for token %d: %s", token_id, e.what());
    }
    catch (...)
    {
        ALOGW("tokenizer decode failed for token %d with unknown exception", token_id);
    }

    return {};
}

static inline std::string sanitize_utf8_text(const std::string &text)
{
    UTF8Filter filter;
    return filter.filter(text);
}

static inline uint64_t hash_u16_buffer(const unsigned short *data, int n)
{
    uint64_t h = 1469598103934665603ULL;
    const uint64_t prime = 1099511628211ULL;
    for (int i = 0; i < n; ++i)
    {
        h ^= (uint64_t)data[i];
        h *= prime;
    }
    return h;
}

static inline void summarize_bf16_buffer(const unsigned short *data,
                                         int n,
                                         float &min_v,
                                         float &max_v,
                                         float &mean_abs)
{
    min_v = std::numeric_limits<float>::infinity();
    max_v = -std::numeric_limits<float>::infinity();
    double sum_abs = 0.0;
    if (!data || n <= 0)
    {
        min_v = 0.0f;
        max_v = 0.0f;
        mean_abs = 0.0f;
        return;
    }
    for (int i = 0; i < n; ++i)
    {
        const float v = bfloat16(data[i]).fp32();
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum_abs += std::fabs(v);
    }
    mean_abs = (float)(sum_abs / (double)n);
}

static inline std::string shape_to_string(const std::vector<unsigned int> &shape)
{
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i) s.push_back('x');
        s += std::to_string(shape[i]);
    }
    if (s.empty()) s = "(none)";
    return s;
}

static inline std::string getenv_string(const char *name)
{
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

static inline bool is_history_prefix(const std::vector<Content> &prefix, const std::vector<Content> &full)
{
    if (prefix.size() > full.size())
        return false;

    for (size_t i = 0; i < prefix.size(); ++i)
    {
        if (!same_history_content(prefix[i], full[i]))
            return false;
    }

    return true;
}

static inline bool tokenizer_uses_hidden_channel_markup(const std::string &tokenizer_type)
{
    return tokenizer_type == "Gemma4" || tokenizer_type == "Gemma4VL";
}

static inline std::string strip_hidden_channel_sections(const std::string &text)
{
    ChannelSectionFilter filter;
    filter.reset();
    std::string out = filter.filter(text);
    out += filter.flush();
    return out;
}

} // namespace

struct LLM::Impl : public IKvSlotHost {
    UTF8Filter utf8_filter;
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;
    Gemma4PerLayerHelper gemma4_per_layer_helper;

    std::vector<Content> last_history_snapshot;
    std::vector<int> last_tokens_ids;
    std::vector<int> run_input_token_ids;
    std::vector<int> last_run_generated_token_ids;
    int last_run_prompt_token_num_ = 0;
    int get_last_prompt_token_num() const { return last_run_prompt_token_num_; }
    int get_last_completion_token_num() const { return (int)last_run_generated_token_ids.size(); }
    // Performance metrics for the most recent run. -1 means "not measured this
    // run"; consumers (OpenAI usage object) treat a negative/zero value as unset.
    float last_run_ttft_ms_ = -1.0f;
    float last_run_decode_tps_ = -1.0f;
    float last_run_prefill_tps_ = -1.0f;
    int last_run_prefill_tokens_ = -1;
    float get_last_ttft_ms() const { return last_run_ttft_ms_; }
    float get_last_decode_tps() const { return last_run_decode_tps_; }
    float get_last_prefill_tps() const { return last_run_prefill_tps_; }
    int get_last_prefill_tokens() const { return last_run_prefill_tokens_; }
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    std::vector<LinearStateSnapshot> linear_state_snapshots_;
    int precompute_len = 0;
    std::vector<int> prefill_history_kv_cache_num_grp;
    std::vector<int> prefill_symbolic_kv_cache_num_grp;

    // ---- Multi-slot prefix KV cache ----
    // Each slot mirrors the host-side context state; the device-side K/V lives in
    // a per-slot device buffer managed by the backend (zero-copy activate). The
    // working fields above (last_*, precompute_len, linear snapshots, full-cache
    // slot bookkeeping) always reflect the currently active slot.

    LLMAttrType _attr;
    MemGuard mem_guard_{_attr}; // CMM/DDR pre-load + running guard + teardown sentry (extracted)
    bool embedding_append_eos = false;
    int embedding_eos_token_id = -1;

    std::unique_ptr<vision::VisionModule> vision;
    vision::RunState vision_state;
    bool has_vision_state = false;
    int cached_mrope_next_pos = -1;
    int active_prefill_pos_start = -1;
    int active_token_pos_start = -1;

    std::vector<LLMLayer> llama_layers;

    // ---- Multi-slot prefix KV cache (extracted to KvSlotManager, stage 3b) ----
    KvSlotManager kv_mgr_{*this, llama_layers, _attr};
    int  slot_decode_gid_for_layer(int a, int b) const override { return decode_gid_for_layer(a, b); }
    bool slot_is_linear_layer(int i) const override { return is_linear_layer(i); }
    int  slot_kv_cache_size_for_layer(int i) const override { return kv_cache_size_for_layer(i); }
    int  slot_layer_devid_for(int i) const override { return layer_devid_for(i); }
    int  slot_cheap_prefill_capacity() const override { return cheap_prefill_capacity(); }
    int  slot_remaining_cmm_mb(int d) const override { return remaining_cmm_mb(d); }
    bool slot_dynamic_layer_load_enabled() const override { return dynamic_layer_load_enabled(); }
    int  slot_decode_grpid() const override { return decode_grpid; }
    int  slot_decode_grpids_back() const override { return decode_grpids_.empty() ? 0 : decode_grpids_.back(); }
    int  slot_precompute_len() const override { return precompute_len; }
    void slot_reset_kv_cache() override { ResetKVCache(); }
    void slot_capture_decode_state(KvCacheSlot &s) override {
        s.last_history_snapshot = last_history_snapshot;
        s.last_tokens_ids = last_tokens_ids;
        s.precompute_len = precompute_len;
        s.linear_state_snapshots = linear_state_snapshots_;
        s.cached_mrope_next_pos = cached_mrope_next_pos;
        s.full_cache_valid_slots = full_cache_valid_slots_;
        s.full_cache_has_sparse_slots = full_cache_has_sparse_slots_;
    }
    void slot_restore_decode_state(const KvCacheSlot &s) override {
        last_history_snapshot = s.last_history_snapshot;
        last_tokens_ids = s.last_tokens_ids;
        precompute_len = s.precompute_len;
        linear_state_snapshots_ = s.linear_state_snapshots;
        cached_mrope_next_pos = s.cached_mrope_next_pos;
        full_cache_valid_slots_ = s.full_cache_valid_slots;
        full_cache_has_sparse_slots_ = s.full_cache_has_sparse_slots;
    }
    bool deinited_ = false; // guards Deinit() against double-invocation (explicit call + destructor path)
    // Optional per-layer attention type (for models like Qwen3.5 that mix linear/full attention).
    std::vector<bool> layer_is_linear_attn;
    std::vector<int> layer_kv_cache_sizes;
    std::vector<int> shared_kv_source_layers;
    // Use a full-attention layer as reference for token-wise KV cache shapes.
    int cache_ref_full_layer_idx = 0;
    ax_runner_t llama_post;

    bool dynamic_layer_load_enabled_ = false;
    int dynamic_layer_pool_size_ = 0;
    std::vector<int> dynamic_layer_devids_;
    std::vector<unsigned char> dynamic_layer_loaded_;
    std::deque<int> dynamic_layer_lru_;

    int decode_grpid = 0;
    std::vector<int> decode_grpids_;             // sorted by decode capacity (ascending)
    std::vector<int> decode_max_token_len_grp_;  // same length as decode_grpids_
    std::vector<int> prefill_grpids_;            // sorted by prefill capacity (ascending), aligns with _attr.prefill_max_kv_cache_num_grp
    bool decode_only_prefill_mode_ = false;
    std::vector<std::vector<int>> layer_decode_grpids_;
    std::vector<std::vector<int>> layer_prefill_grpids_;
    std::vector<unsigned char> full_cache_valid_slots_;
    bool full_cache_has_sparse_slots_ = false;
    std::atomic<bool> b_stop{false};
    LLMPostprocess postprocess;
    std::string last_error_message;
    std::chrono::steady_clock::time_point request_start_time_;
    bool request_start_active_ = false;

    bool dynamic_layer_load_enabled() const
    {
        return dynamic_layer_load_enabled_ && dynamic_layer_pool_size_ > 0;
    }

    int layer_devid_for(int layer_idx) const
    {
#ifdef USE_AXCL
        if (layer_idx >= 0 && layer_idx < (int)dynamic_layer_devids_.size())
            return dynamic_layer_devids_[(size_t)layer_idx];
        if (!_attr.dev_ids.empty()) return _attr.dev_ids.front();
        return 0;
#else
        (void)layer_idx;
        return -1;
#endif
    }

    void dynamic_layer_touch(int layer_idx)
    {
        if (!dynamic_layer_load_enabled()) return;
        if (layer_idx < 0 || layer_idx >= (int)dynamic_layer_loaded_.size()) return;
        if (!dynamic_layer_loaded_[(size_t)layer_idx]) return;

        for (auto it = dynamic_layer_lru_.begin(); it != dynamic_layer_lru_.end(); ++it)
        {
            if (*it == layer_idx)
            {
                dynamic_layer_lru_.erase(it);
                break;
            }
        }
        dynamic_layer_lru_.push_back(layer_idx);
    }

    bool ensure_layer_loaded_no_prefetch(int layer_idx)
    {
        if (!dynamic_layer_load_enabled()) return true;
        if (layer_idx < 0 || layer_idx >= _attr.axmodel_num) return false;
        if ((int)dynamic_layer_loaded_.size() != _attr.axmodel_num) return false;

        if (dynamic_layer_loaded_[(size_t)layer_idx])
        {
            dynamic_layer_touch(layer_idx);
            return true;
        }

        while ((int)dynamic_layer_lru_.size() >= dynamic_layer_pool_size_)
        {
            const int evict_idx = dynamic_layer_lru_.front();
            dynamic_layer_lru_.pop_front();
            if (evict_idx < 0 || evict_idx >= _attr.axmodel_num) continue;
            if (!dynamic_layer_loaded_[(size_t)evict_idx]) continue;
            llama_layers[(size_t)evict_idx].layer.unload_handle_keep_io();
            dynamic_layer_loaded_[(size_t)evict_idx] = 0;
        }

        auto &lyr = llama_layers[(size_t)layer_idx];
        const int devid = layer_devid_for(layer_idx);
        const int ret = lyr.layer.load_handle_reuse_io(lyr.filename.c_str(), devid);
        if (ret != 0)
        {
            ALOGE("dynamic load layer %d failed: %s", layer_idx, lyr.filename.c_str());
            return false;
        }
        dynamic_layer_loaded_[(size_t)layer_idx] = 1;
        dynamic_layer_touch(layer_idx);
        return true;
    }

    bool ensure_layer_loaded(int layer_idx)
    {
        if (!dynamic_layer_load_enabled()) return true;

        if (!ensure_layer_loaded_no_prefetch(layer_idx))
            return false;

        // Implicit prefetch: keep the next layer's handle resident when pool_size>1.
        // This avoids a hard handle-load stall when we advance to the next layer.
        if (dynamic_layer_pool_size_ > 1)
        {
            const int next = layer_idx + 1;
            if (next >= 0 && next < _attr.axmodel_num)
            {
                if (!dynamic_layer_loaded_[(size_t)next])
                {
                    const bool ok = ensure_layer_loaded_no_prefetch(next);
                    if (!ok)
                    {
                        // Prefetch failures are non-fatal; we'll retry when the layer is actually needed.
                        ALOGW("dynamic prefetch next layer %d failed", next);
                    }
                }
            }
        }

        return true;
    }

    static std::string context_limit_user_message()
    {
        return "当前会话上下文已超过该模型的长度上限，请使用 /clean 清空历史并重新开始对话。";
    }

    static std::string video_quality_reset_notice()
    {
        return "提示：为保证视频解析质量，本次视频请求已按新会话处理，未使用此前对话上下文。\n\n";
    }

    void MarkRequestStart()
    {
        request_start_time_ = std::chrono::steady_clock::now();
        request_start_active_ = true;
    }

    void ClearRequestStart()
    {
        request_start_active_ = false;
    }

    float CurrentRequestElapsedMs() const
    {
        if (!request_start_active_)
            return -1.0f;
        const auto now = std::chrono::steady_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now - request_start_time_).count();
        return static_cast<float>(us) / 1000.0f;
    }

    static bool request_has_video_media(const std::vector<Content> &history,
                                        const std::vector<::MediaInputs> &media_inputs)
    {
        for (const auto &media : media_inputs)
        {
            if (media.content_index < history.size() && history[media.content_index].type == VIDEO)
                return true;
        }
        return false;
    }

    static bool normalize_away_video_history(std::vector<Content> &history,
                                             std::vector<::MediaInputs> &media_inputs)
    {
        if (history.empty()) return false;

        size_t current_user_index = history.size();
        for (size_t i = history.size(); i > 0; --i)
        {
            if (history[i - 1].role == USER)
            {
                current_user_index = i - 1;
                break;
            }
        }
        if (current_user_index >= history.size()) return false;

        std::vector<Content> normalized;
        normalized.reserve(history.size());
        for (size_t i = 0; i < current_user_index; ++i)
        {
            if (history[i].role == SYSTEM)
                normalized.push_back(history[i]);
        }

        const size_t new_user_index = normalized.size();
        normalized.push_back(history[current_user_index]);

        std::vector<::MediaInputs> normalized_media;
        if (history[current_user_index].type == VIDEO)
        {
            for (auto it = media_inputs.rbegin(); it != media_inputs.rend(); ++it)
            {
                if (it->content_index == current_user_index && !it->uris.empty())
                {
                    normalized_media.push_back({new_user_index, it->uris});
                    break;
                }
            }
        }
        media_inputs = std::move(normalized_media);
        history = std::move(normalized);
        return true;
    }

    void clear_last_error()
    {
        last_error_message.clear();
    }

    void set_last_error(std::string message)
    {
        last_error_message = std::move(message);
    }

    void set_context_limit_error()
    {
        last_error_message = context_limit_user_message();
    }

    std::string get_last_error() const
    {
        return last_error_message;
    }

    // ---- small helpers ----
    static int post_process(LLMPostprocess &postprocess, unsigned short *p, int n, std::vector<int> &history, float *val = 0)
    {
        (void)val;
        return postprocess.apply_bf16(p, n, history);
    }

    static inline void fill_indices(unsigned int *dst, int start, int count)
    { for (int i = 0; i < count; ++i) dst[i] = (unsigned int)(start + i); }

    static inline void build_prefill_mask(std::vector<unsigned short> &mask_tmp,
                                          int kv_cache_num,
                                          int token_rows,
                                          int history_len,
                                          int valid_rows,
                                          bool sliding_attention = false,
                                          int sliding_window = 0)
    {
        bfloat16 bf16 = -65536.f;
        std::fill(mask_tmp.begin(), mask_tmp.end(), bf16.data);
        const int rows = std::max(0, std::min(token_rows, valid_rows));
        for (int r = 0; r < rows; ++r) {
            auto row = mask_tmp.data() + r * (kv_cache_num + token_rows);
            int history_start = 0;
            int current_start = 0;
            if (sliding_attention && sliding_window > 0)
            {
                const int q_pos = history_len + r;
                history_start = std::max(0, q_pos - sliding_window + 1);
                current_start = std::max(0, r - sliding_window + 1);
            }
            for (int j = history_start; j < history_len; ++j) row[j] = 0;
            int cur = kv_cache_num; for (int j = cur + current_start; j < cur + r + 1; ++j) row[j] = 0;
        }
    }

    void build_layer_prefill_mask(std::vector<unsigned short> &mask_tmp,
                                  int kv_cache_num,
                                  int token_rows,
                                  int history_len,
                                  int valid_rows,
                                  int layer_idx) const
    {
        build_prefill_mask(mask_tmp,
                           kv_cache_num,
                           token_rows,
                           history_len,
                           valid_rows,
                           is_sliding_attention_layer(layer_idx),
                           _attr.sliding_window);
    }

    static inline void build_decode_mask(std::vector<unsigned short> &mask_tmp,
                                         int mask_elems,
                                         int visible_past_tokens,
                                         bool sliding_attention,
                                         int sliding_window)
    {
        bfloat16 bf16 = -65536.f;
        if (mask_elems <= 0) return;
        const int elems = std::min(mask_elems, (int)mask_tmp.size());
        if (elems <= 0) return;
        std::fill(mask_tmp.begin(), mask_tmp.begin() + elems, bf16.data);

        const int cache_len = elems - 1;
        const int end = std::min(std::max(0, visible_past_tokens), cache_len);
        int start = 0;
        if (sliding_attention && sliding_window > 0)
        {
            start = std::max(0, end - sliding_window + 1);
        }
        for (int i = start; i < end; ++i) mask_tmp[(size_t)i] = 0;
        // Each decode shape-group has its own mask length. When sharing a larger
        // `decode_mask` buffer across groups, make sure we unmask the tail
        // element within the active prefix (not `mask_tmp.back()`).
        mask_tmp[(size_t)(elems - 1)] = 0;
    }

    void build_layer_decode_mask(std::vector<unsigned short> &mask_tmp,
                                 int mask_elems,
                                 int visible_past_tokens,
                                 int layer_idx) const
    {
        build_decode_mask(mask_tmp,
                          mask_elems,
                          visible_past_tokens,
                          is_sliding_attention_layer(layer_idx),
                          _attr.sliding_window);
    }

    void reset_full_cache_slot_state()
    {
        full_cache_valid_slots_.clear();
        full_cache_has_sparse_slots_ = false;
    }

    void ensure_full_cache_slot_state(int min_slots)
    {
        const int model_slots = std::max(_attr.max_token_len + 1, _attr.kv_cache_num + 1);
        const int slots = std::max(model_slots, min_slots);
        if (slots > 0 && full_cache_valid_slots_.size() < (size_t)slots)
            full_cache_valid_slots_.resize((size_t)slots, 0);
    }

    void mark_full_cache_slots(int start, int count)
    {
        if (count <= 0) return;
        const int end = start + count;
        if (end <= 0) return;
        ensure_full_cache_slot_state(end);
        const int begin = std::max(0, start);
        for (int i = begin; i < end; ++i)
            full_cache_valid_slots_[(size_t)i] = 1;
    }

    void mark_full_cache_slot(int pos)
    {
        mark_full_cache_slots(pos, 1);
    }

    bool use_sparse_full_cache_mask() const
    {
        return full_cache_has_sparse_slots_ && !full_cache_valid_slots_.empty();
    }

    void build_sparse_layer_decode_mask(std::vector<unsigned short> &mask_tmp,
                                        int mask_elems,
                                        int current_pos,
                                        int layer_idx) const
    {
        bfloat16 bf16 = -65536.f;
        if (mask_elems <= 0) return;
        const int elems = std::min(mask_elems, (int)mask_tmp.size());
        if (elems <= 0) return;
        std::fill(mask_tmp.begin(), mask_tmp.begin() + elems, bf16.data);

        const int cache_len = elems - 1;
        int start = 0;
        if (is_sliding_attention_layer(layer_idx) && _attr.sliding_window > 0)
            start = std::max(0, current_pos - _attr.sliding_window + 1);
        const int limit = std::min(cache_len, (int)full_cache_valid_slots_.size());
        for (int i = start; i < limit; ++i)
        {
            if (full_cache_valid_slots_[(size_t)i])
                mask_tmp[(size_t)i] = 0;
        }
        mask_tmp[(size_t)(elems - 1)] = 0;
    }

    void build_sparse_layer_prefill_mask(std::vector<unsigned short> &mask_tmp,
                                         int kv_cache_num,
                                         int token_rows,
                                         int history_len,
                                         int valid_rows,
                                         int layer_idx) const
    {
        bfloat16 bf16 = -65536.f;
        std::fill(mask_tmp.begin(), mask_tmp.end(), bf16.data);
        const int rows = std::max(0, std::min(token_rows, valid_rows));
        for (int r = 0; r < rows; ++r)
        {
            auto row = mask_tmp.data() + r * (kv_cache_num + token_rows);
            int history_start = 0;
            int current_start = 0;
            if (is_sliding_attention_layer(layer_idx) && _attr.sliding_window > 0)
            {
                const int q_pos = history_len + r;
                history_start = std::max(0, q_pos - _attr.sliding_window + 1);
                current_start = std::max(0, r - _attr.sliding_window + 1);
            }
            const int history_end = std::min({history_len, kv_cache_num, (int)full_cache_valid_slots_.size()});
            for (int j = history_start; j < history_end; ++j)
            {
                if (full_cache_valid_slots_[(size_t)j])
                    row[j] = 0;
            }
            const int cur = kv_cache_num;
            for (int j = cur + current_start; j < cur + r + 1; ++j)
                row[j] = 0;
        }
    }

    void copy_shared_prefill_cache(const ax_runner_tensor_t &dst,
                                   const ax_runner_tensor_t &src,
                                   int layer_kv,
                                   int history_len,
                                   int current_dst_start,
                                   int current_src_start,
                                   int current_tokens,
                                   int devid)
    {
        if (layer_kv <= 0) return;
        const size_t bytes_per_token = (size_t)layer_kv * sizeof(unsigned short);
        if (bytes_per_token == 0) return;

        const size_t dst_tokens = (size_t)dst.nSize / bytes_per_token;
        const size_t src_tokens = (size_t)src.nSize / bytes_per_token;
        const size_t history_tokens = std::min({(size_t)std::max(0, history_len), dst_tokens, src_tokens});

        llm_memset(LLM_WADDR(dst), 0, dst.nSize, devid);
        if (history_tokens > 0)
        {
            llm_d2d(LLM_WADDR(dst),
                    LLM_RADDR(src),
                    history_tokens * bytes_per_token,
                    devid);
        }

        const size_t dst_start = (size_t)std::max(0, current_dst_start);
        const size_t src_start = (size_t)std::max(0, current_src_start);
        if (current_tokens <= 0 || dst_start >= dst_tokens || src_start >= src_tokens)
            return;

        const size_t copy_tokens = std::min({(size_t)current_tokens, dst_tokens - dst_start, src_tokens - src_start});
        if (copy_tokens == 0)
            return;

        auto *dst_base = (unsigned char *)LLM_WADDR(dst);
        const auto *src_base = (const unsigned char *)LLM_RADDR(src);
        llm_d2d(dst_base + dst_start * bytes_per_token,
                src_base + src_start * bytes_per_token,
                copy_tokens * bytes_per_token,
                devid);
    }

    void clear_all_group_kv_cache_tensors()
    {
        for (int i = 0; i < _attr.axmodel_num; ++i)
        {
            auto &lyr = llama_layers[i];
            const int devid = LLM_DEVID(lyr);
            const int ng = lyr.layer.get_num_input_groups();
            for (int gid = 0; gid < ng; ++gid)
            {
                // Some models/backends may not map KV cache buffers into CPU virtual address
                // space for every group (or may use group-suffixed tensor names). Clearing KV
                // is best-effort: skip tensors that are missing or not writable from host.
                const auto *k = try_get_group_input_tensor(lyr.layer, gid, "K_cache");
                const auto *v = try_get_group_input_tensor(lyr.layer, gid, "V_cache");
                if (k)
                {
                    void *kaddr = LLM_WADDR(*k);
                    if (kaddr && k->nSize) llm_memset(kaddr, 0, k->nSize, devid);
                }
                if (v)
                {
                    void *vaddr = LLM_WADDR(*v);
                    if (vaddr && v->nSize) llm_memset(vaddr, 0, v->nSize, devid);
                }
            }
        }
    }

    void drop_linear_state_snapshots_after(int token_len)
    {
        linear_state_snapshots_.erase(
            std::remove_if(linear_state_snapshots_.begin(),
                           linear_state_snapshots_.end(),
                           [token_len](const LinearStateSnapshot &snapshot) {
                               return snapshot.token_len > token_len;
                           }),
            linear_state_snapshots_.end());
    }

    int best_linear_state_snapshot_len(int token_len) const
    {
        int best = -1;
        for (const auto &snapshot : linear_state_snapshots_)
        {
            if (snapshot.token_len <= token_len && snapshot.token_len > best)
                best = snapshot.token_len;
        }
        return best;
    }

    void capture_linear_state_snapshot(int token_len)
    {
        if (token_len <= 0 || !has_linear_attention_layers()) return;

        LinearStateSnapshot snapshot;
        snapshot.token_len = token_len;
        snapshot.k.resize((size_t)_attr.axmodel_num);
        snapshot.v.resize((size_t)_attr.axmodel_num);

        for (int i = 0; i < _attr.axmodel_num; ++i)
        {
            if (!is_linear_layer(i)) continue;

            auto &lyr = llama_layers[(size_t)i];
            const int layer_decode_grpid = decode_gid_for_layer(i, decode_grpid);
            auto &t_k = lyr.layer.get_input(layer_decode_grpid, "K_cache");
            auto &t_v = lyr.layer.get_input(layer_decode_grpid, "V_cache");
            snapshot.k[(size_t)i].resize((size_t)t_k.nSize / sizeof(unsigned short));
            snapshot.v[(size_t)i].resize((size_t)t_v.nSize / sizeof(unsigned short));
            llm_d2h(snapshot.k[(size_t)i].data(), LLM_RADDR(t_k), t_k.nSize, LLM_DEVID(lyr));
            llm_d2h(snapshot.v[(size_t)i].data(), LLM_RADDR(t_v), t_v.nSize, LLM_DEVID(lyr));
        }

        auto it = std::find_if(linear_state_snapshots_.begin(),
                               linear_state_snapshots_.end(),
                               [token_len](const LinearStateSnapshot &existing) {
                                   return existing.token_len == token_len;
                               });
        if (it != linear_state_snapshots_.end())
            *it = std::move(snapshot);
        else
            linear_state_snapshots_.push_back(std::move(snapshot));
    }

    bool restore_linear_state_snapshot_to_host_cache(int token_len)
    {
        if (!has_linear_attention_layers() || token_len <= 0) return true;

        auto it = std::find_if(linear_state_snapshots_.begin(),
                               linear_state_snapshots_.end(),
                               [token_len](const LinearStateSnapshot &snapshot) {
                                   return snapshot.token_len == token_len;
                               });
        if (it == linear_state_snapshots_.end())
            return false;

        if ((int)k_caches.size() != _attr.axmodel_num) k_caches.resize((size_t)_attr.axmodel_num);
        if ((int)v_caches.size() != _attr.axmodel_num) v_caches.resize((size_t)_attr.axmodel_num);

        for (int i = 0; i < _attr.axmodel_num; ++i)
        {
            if (!is_linear_layer(i)) continue;
            k_caches[(size_t)i] = it->k[(size_t)i];
            v_caches[(size_t)i] = it->v[(size_t)i];
        }
        return true;
    }

    static inline std::vector<float> l2norm(std::vector<float> embedding)
    {
        float norm2 = 0.0f;
        for (const float v : embedding) norm2 += v * v;
        const float norm = std::sqrt(norm2);
        if (norm > 1e-12f)
        {
            for (float &v : embedding) v /= norm;
        }
        return embedding;
    }

    bool use_gemma4_scaled_text_input() const
    {
        return gemma4_per_layer_helper.enabled();
    }

    float gemma4_text_scale() const
    {
        return std::sqrt((float)_attr.tokens_embed_size);
    }

    static inline void scale_bf16_buffer_inplace(unsigned short *data, size_t elem_count, float scale)
    {
        if (!data || elem_count == 0) return;
        if (std::fabs(scale - 1.0f) < 1e-6f) return;
        for (size_t i = 0; i < elem_count; ++i)
        {
            data[i] = bfloat16(bfloat16(data[i]).fp32() * scale).data;
        }
    }

    bool is_vision_token_position(int abs_pos) const
    {
        return has_vision_state &&
               abs_pos >= 0 &&
               (size_t)abs_pos < vision_state.pos2vision.size() &&
               vision_state.pos2vision[(size_t)abs_pos] >= 0;
    }

    void scale_prefill_text_embeds_inplace(unsigned short *data, int num_tokens, int abs_start_pos) const
    {
        if (!use_gemma4_scaled_text_input() || !data || num_tokens <= 0) return;

        const float scale = gemma4_text_scale();
        for (int i = 0; i < num_tokens; ++i)
        {
            if (is_vision_token_position(abs_start_pos + i)) continue;
            scale_bf16_buffer_inplace(data + (size_t)i * (size_t)_attr.tokens_embed_size,
                                      (size_t)_attr.tokens_embed_size,
                                      scale);
        }
    }

    void scale_all_embeds_inplace(unsigned short *data, int num_tokens) const
    {
        if (!use_gemma4_scaled_text_input() || !data || num_tokens <= 0) return;
        scale_bf16_buffer_inplace(data,
                                  (size_t)num_tokens * (size_t)_attr.tokens_embed_size,
                                  gemma4_text_scale());
    }

    static inline int tolower_uc(int c) { return std::tolower((unsigned char)c); }

    static inline std::string key_of(const std::string &s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c : s)
        {
            const unsigned char uc = (unsigned char)c;
            if (std::isalnum(uc)) out.push_back((char)tolower_uc(uc));
        }
        return out;
    }

    static inline void embedding_profile_for_tokenizer(const std::string &tokenizer_type, bool &append_eos, int &eos_token_id)
    {
        const std::string key = key_of(tokenizer_type);
        if (key == "qwen3" || key == "qwen3vl")
        {
            // Align with /home/axera/libembeding.axera (Qwen3-Embedding-0.6B)
            append_eos = true;
            eos_token_id = 151643;
            return;
        }

        append_eos = false;
        eos_token_id = -1;
    }

    bool is_minicpmv46_tokenizer() const
    {
        const std::string key = key_of(_attr.tokenizer_type);
        return key == "minicpmv46" || key == "minicpmv46vl";
    }

    int group_index_by_gid(const std::vector<int> &grpids, int gid) const
    {
        for (size_t i = 0; i < grpids.size(); ++i)
        {
            if (grpids[i] == gid) return (int)i;
        }
        return -1;
    }

    int prefill_capacity_by_gid(int gid) const
    {
        const int idx = group_index_by_gid(prefill_grpids_, gid);
        if (idx < 0 || idx >= (int)_attr.prefill_max_kv_cache_num_grp.size()) return -1;
        return _attr.prefill_max_kv_cache_num_grp[(size_t)idx];
    }

    // Largest single-chunk prefill that needs no cached history (history_cap == 0).
    // A request within this size is cheaper to prefill fresh than to reuse via the
    // model's wider history-bearing prefill group, so multi-slot skips reuse below it.
    int cheap_prefill_capacity() const
    {
        int cap = 0;
        const size_t n = std::min(prefill_history_kv_cache_num_grp.size(), _attr.prefill_max_kv_cache_num_grp.size());
        for (size_t i = 0; i < n; ++i)
            if (prefill_history_kv_cache_num_grp[i] == 0)
                cap = std::max(cap, _attr.prefill_max_kv_cache_num_grp[i]);
        return cap;
    }

    int decode_capacity_by_gid(int gid) const
    {
        const int idx = group_index_by_gid(decode_grpids_, gid);
        if (idx < 0 || idx >= (int)decode_max_token_len_grp_.size()) return -1;
        return decode_max_token_len_grp_[(size_t)idx];
    }

    int prefill_history_capacity_by_gid(int gid) const
    {
        const int idx = group_index_by_gid(prefill_grpids_, gid);
        if (idx < 0 || idx >= (int)prefill_history_kv_cache_num_grp.size()) return -1;
        return prefill_history_kv_cache_num_grp[(size_t)idx];
    }

    // Largest history (already-cached KV prefix) a single prefill can attend to. Prefill
    // shape-groups are sized for chunked prefill, so their max history cap is smaller than
    // the model's full context. A reused KV prefix longer than this cannot be set up in one
    // SetKVCache, so the prefix-reuse path must cap to it (and recompute the extra tokens).
    int max_prefill_history_cap() const
    {
        int m = 0;
        for (int gid : prefill_grpids_) { const int c = prefill_history_capacity_by_gid(gid); if (c > m) m = c; }
        return m;
    }

    int prefill_symbolic_capacity_by_gid(int gid) const
    {
        const int idx = group_index_by_gid(prefill_grpids_, gid);
        if (idx < 0 || idx >= (int)prefill_symbolic_kv_cache_num_grp.size()) return -1;
        return prefill_symbolic_kv_cache_num_grp[(size_t)idx];
    }

    int prefill_history_capacity_by_mask(int prefill_grpid)
    {
        if (_attr.prefill_token_num <= 0) return -1;
        if (cache_ref_full_layer_idx < 0 || cache_ref_full_layer_idx >= _attr.axmodel_num) return -1;
        try
        {
            const auto &mask_t = llama_layers[(size_t)cache_ref_full_layer_idx].layer.get_input(prefill_grpid, "mask");
            const int mask_elems = (int)((size_t)mask_t.nSize / sizeof(unsigned short));
            if (mask_elems <= 0) return -1;
            if ((mask_elems % _attr.prefill_token_num) != 0) return -1;
            const int cols = mask_elems / _attr.prefill_token_num;
            const int kv_from_mask = cols - _attr.prefill_token_num;
            if (kv_from_mask < 0) return -1;
            return kv_from_mask;
        }
        catch (...)
        {
        }
        return -1;
    }

    int prefill_history_capacity_for_layer_group(int layer_idx, int prefill_grpid)
    {
        if (_attr.prefill_token_num <= 0) return -1;
        if (layer_idx < 0 || layer_idx >= _attr.axmodel_num) return -1;
        try
        {
            const auto &mask_t = llama_layers[(size_t)layer_idx].layer.get_input(prefill_grpid, "mask");
            const int mask_elems = (int)((size_t)mask_t.nSize / sizeof(unsigned short));
            if (mask_elems > 0 && (mask_elems % _attr.prefill_token_num) == 0)
            {
                const int cols = mask_elems / _attr.prefill_token_num;
                const int kv_from_mask = cols - _attr.prefill_token_num;
                if (kv_from_mask >= 0) return kv_from_mask;
            }
        }
        catch (const std::exception &)
        {
        }

        return -1;
    }

    int choose_prefill_gid(int needed_tokens) const
    {
        if (prefill_grpids_.empty() || _attr.prefill_max_kv_cache_num_grp.empty()) return 1;
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size() && i < prefill_grpids_.size(); ++i)
        {
            if (needed_tokens <= _attr.prefill_max_kv_cache_num_grp[i]) return prefill_grpids_[i];
        }
        return prefill_grpids_.back();
    }

    int choose_decode_gid(int needed_tokens) const
    {
        if (decode_grpids_.empty() || decode_max_token_len_grp_.empty()) return 0;
        for (size_t i = 0; i < decode_max_token_len_grp_.size() && i < decode_grpids_.size(); ++i)
        {
            if (needed_tokens <= decode_max_token_len_grp_[i]) return decode_grpids_[i];
        }
        return decode_grpids_.back();
    }

    int select_prefill_group(int history_len, int chunk_tokens, bool prefer_symbolic_group = false) const
    {
        if (prefill_grpids_.empty()) return -1;

        const int safe_history_len = std::max(0, history_len);
        const int safe_chunk_tokens = std::max(0, chunk_tokens);
        const int total_tokens = safe_history_len + safe_chunk_tokens;

        if (prefer_symbolic_group && !prefill_symbolic_kv_cache_num_grp.empty())
        {
            for (size_t i = 0; i < prefill_grpids_.size() && i < prefill_symbolic_kv_cache_num_grp.size(); ++i)
            {
                if (total_tokens <= prefill_symbolic_kv_cache_num_grp[i])
                    return prefill_grpids_[i];
            }
        }

        for (size_t i = 0; i < prefill_grpids_.size() &&
                           i < prefill_history_kv_cache_num_grp.size() &&
                           i < _attr.prefill_max_kv_cache_num_grp.size();
             ++i)
        {
            const int history_cap = prefill_history_kv_cache_num_grp[i];
            const int total_cap = _attr.prefill_max_kv_cache_num_grp[i];
            if (safe_history_len <= history_cap && total_tokens <= total_cap)
                return prefill_grpids_[i];
        }
        return -1;
    }

    int select_stateless_prefill_group(int input_tokens) const
    {
        if (prefill_grpids_.empty()) return -1;

        const int safe_input_tokens = std::max(0, input_tokens);
        for (size_t i = 0; i < prefill_grpids_.size() && i < _attr.prefill_max_kv_cache_num_grp.size(); ++i)
        {
            if (safe_input_tokens <= _attr.prefill_max_kv_cache_num_grp[i])
                return prefill_grpids_[i];
        }
        return -1;
    }

    std::vector<int> detect_prefill_grpids(ax_runner_t &layer) const
    {
        std::vector<int> gids;
        const int group_count = layer.get_num_input_groups();
        gids.reserve((size_t)group_count);
        for (int gid = 0; gid < group_count; ++gid)
        {
            try
            {
                const auto &t_idx = layer.get_input(gid, "indices");
                const int idx_elems = (int)((size_t)t_idx.nSize / sizeof(unsigned int));
                if (idx_elems > 1) gids.push_back(gid);
            }
            catch (const std::exception &)
            {
            }
        }
        return gids;
    }

    std::vector<int> detect_decode_grpids(ax_runner_t &layer) const
    {
        std::vector<int> gids;
        const int group_count = layer.get_num_input_groups();
        gids.reserve((size_t)group_count);
        for (int gid = 0; gid < group_count; ++gid)
        {
            try
            {
                const auto &t_idx = layer.get_input(gid, "indices");
                const int idx_elems = (int)((size_t)t_idx.nSize / sizeof(unsigned int));
                if (idx_elems == 1) gids.push_back(gid);
            }
            catch (const std::exception &)
            {
            }
        }
        return gids;
    }

    void init_layer_groups();

    int decode_gid_for_layer(int layer_idx, int requested_gid) const
    {
        if (requested_gid < 0) return requested_gid;
        if (layer_idx < 0 || layer_idx >= (int)layer_decode_grpids_.size()) return requested_gid;

        const auto &gids = layer_decode_grpids_[(size_t)layer_idx];
        if (gids.empty()) return requested_gid;

        const int requested_idx = group_index_by_gid(decode_grpids_, requested_gid);
        if (requested_idx < 0)
        {
            if (std::find(gids.begin(), gids.end(), requested_gid) != gids.end()) return requested_gid;
            return gids.back();
        }

        const size_t mapped_idx = std::min((size_t)requested_idx, gids.size() - 1);
        return gids[mapped_idx];
    }

    int prefill_gid_for_layer(int layer_idx, int requested_gid) const
    {
        if (requested_gid < 0) return requested_gid;
        if (layer_idx < 0 || layer_idx >= (int)layer_prefill_grpids_.size()) return requested_gid;

        const auto &gids = layer_prefill_grpids_[(size_t)layer_idx];
        if (gids.empty()) return requested_gid;

        const int requested_idx = group_index_by_gid(prefill_grpids_, requested_gid);
        if (requested_idx < 0)
        {
            if (std::find(gids.begin(), gids.end(), requested_gid) != gids.end()) return requested_gid;
            return gids.back();
        }

        const size_t mapped_idx = std::min((size_t)requested_idx, gids.size() - 1);
        return gids[mapped_idx];
    }

    void copy_linear_state_to_group(ax_runner_t &layer,
                                    int gid,
                                    const ax_runner_tensor_t &out_k,
                                    const ax_runner_tensor_t &out_v,
                                    int devid) const
    {
        try
        {
            auto &dst_k = layer.get_input(gid, "K_cache");
            auto &dst_v = layer.get_input(gid, "V_cache");
            llm_d2d(LLM_WADDR(dst_k), LLM_RADDR(out_k), std::min((size_t)dst_k.nSize, (size_t)out_k.nSize), devid);
            llm_d2d(LLM_WADDR(dst_v), LLM_RADDR(out_v), std::min((size_t)dst_v.nSize, (size_t)out_v.nSize), devid);
        }
        catch (const std::exception &e)
        {
            ALOGW("skip linear state sync for gid=%d: %s", gid, e.what());
        }
    }

    void sync_linear_state_to_decode_groups(int layer_idx,
                                            ax_runner_t &layer,
                                            const ax_runner_tensor_t &out_k,
                                            const ax_runner_tensor_t &out_v,
                                            int devid) const
    {
        if (layer_idx >= 0 && layer_idx < (int)layer_decode_grpids_.size())
        {
            for (const int gid : layer_decode_grpids_[(size_t)layer_idx])
                copy_linear_state_to_group(layer, gid, out_k, out_v, devid);
        }
        else
        {
            for (const int gid : decode_grpids_)
                copy_linear_state_to_group(layer, gid, out_k, out_v, devid);
        }
    }

    void sync_linear_state_to_prefill_groups(int layer_idx,
                                             ax_runner_t &layer,
                                             const ax_runner_tensor_t &out_k,
                                             const ax_runner_tensor_t &out_v,
                                             int devid,
                                             bool skip_cold_prefill_group) const
    {
        if (layer_idx < 0 || layer_idx >= (int)layer_prefill_grpids_.size()) return;
        const auto &gids = layer_prefill_grpids_[(size_t)layer_idx];
        const size_t start = (skip_cold_prefill_group && gids.size() > 1) ? 1 : 0;
        for (size_t i = start; i < gids.size(); ++i)
            copy_linear_state_to_group(layer, gids[i], out_k, out_v, devid);
    }

    void sync_linear_state_after_prefill(int layer_idx,
                                         ax_runner_t &layer,
                                         const ax_runner_tensor_t &out_k,
                                         const ax_runner_tensor_t &out_v,
                                         int devid) const
    {
        sync_linear_state_to_decode_groups(layer_idx, layer, out_k, out_v, devid);
        sync_linear_state_to_prefill_groups(layer_idx, layer, out_k, out_v, devid, true);
    }

    void sync_linear_input_state_from_group(int layer_idx,
                                            ax_runner_t &layer,
                                            int src_gid,
                                            int devid,
                                            bool sync_prefill_groups) const
    {
        const ax_runner_tensor_t *src_out_k = try_get_group_output_tensor(layer, src_gid, "K_cache_out");
        const ax_runner_tensor_t *src_out_v = try_get_group_output_tensor(layer, src_gid, "V_cache_out");
        const bool prefer_outputs = src_out_k && src_out_v &&
                                    LLM_RADDR(*src_out_k) != nullptr &&
                                    LLM_RADDR(*src_out_v) != nullptr;

        if (layer_idx >= 0 && layer_idx < (int)layer_decode_grpids_.size())
        {
            for (const int gid : layer_decode_grpids_[(size_t)layer_idx])
                copy_linear_input_state_to_group(layer, gid, src_gid, devid, src_out_k, src_out_v, prefer_outputs);
        }
        else
        {
            for (const int gid : decode_grpids_)
                copy_linear_input_state_to_group(layer, gid, src_gid, devid, src_out_k, src_out_v, prefer_outputs);
        }

        if (!sync_prefill_groups) return;
        if (layer_idx < 0 || layer_idx >= (int)layer_prefill_grpids_.size()) return;
        for (const int gid : layer_prefill_grpids_[(size_t)layer_idx])
            copy_linear_input_state_to_group(layer, gid, src_gid, devid, src_out_k, src_out_v, prefer_outputs);
    }

    void copy_linear_input_state_to_group(ax_runner_t &layer,
                                          int dst_gid,
                                          int src_gid,
                                          int devid,
                                          const ax_runner_tensor_t *src_out_k = nullptr,
                                          const ax_runner_tensor_t *src_out_v = nullptr,
                                          bool prefer_outputs = false) const
    {
        if (dst_gid == src_gid && !prefer_outputs) return;
        try
        {
            auto &dst_k = layer.get_input(dst_gid, "K_cache");
            auto &dst_v = layer.get_input(dst_gid, "V_cache");

            const ax_runner_tensor_t *src_k = src_out_k;
            const ax_runner_tensor_t *src_v = src_out_v;
            if (!prefer_outputs || !src_k || !src_v)
            {
                src_k = &layer.get_input(src_gid, "K_cache");
                src_v = &layer.get_input(src_gid, "V_cache");
            }

            const size_t copy_k_bytes = std::min((size_t)dst_k.nSize, (size_t)src_k->nSize);
            const size_t copy_v_bytes = std::min((size_t)dst_v.nSize, (size_t)src_v->nSize);
            const bool same_k_buffer = LLM_WADDR(dst_k) == LLM_RADDR(*src_k);
            const bool same_v_buffer = LLM_WADDR(dst_v) == LLM_RADDR(*src_v);
            if (std::getenv("AXLLM_DEBUG_LINEAR_STATE"))
            {
                ALOGI("linear state copy: src_gid=%d dst_gid=%d prefer_outputs=%d src_k=%p(%u) dst_k=%p(%u) src_v=%p(%u) dst_v=%p(%u)",
                      src_gid,
                      dst_gid,
                      prefer_outputs ? 1 : 0,
                      LLM_RADDR(*src_k),
                      src_k->nSize,
                      LLM_WADDR(dst_k),
                      dst_k.nSize,
                      LLM_RADDR(*src_v),
                      src_v->nSize,
                      LLM_WADDR(dst_v),
                      dst_v.nSize);
            }
            if (!same_k_buffer && copy_k_bytes > 0)
            {
                if (!LLM_WADDR(dst_k) || !LLM_RADDR(*src_k))
                {
                    ALOGW("skip linear K copy due to null buffer: src_gid=%d dst_gid=%d bytes=%zu", src_gid, dst_gid, copy_k_bytes);
                }
                else
                {
                    llm_d2d(LLM_WADDR(dst_k), LLM_RADDR(*src_k), copy_k_bytes, devid);
                }
            }
            if (!same_v_buffer && copy_v_bytes > 0)
            {
                if (!LLM_WADDR(dst_v) || !LLM_RADDR(*src_v))
                {
                    ALOGW("skip linear V copy due to null buffer: src_gid=%d dst_gid=%d bytes=%zu", src_gid, dst_gid, copy_v_bytes);
                }
                else
                {
                    llm_d2d(LLM_WADDR(dst_v), LLM_RADDR(*src_v), copy_v_bytes, devid);
                }
            }
        }
        catch (const std::exception &e)
        {
            ALOGW("skip linear input state copy src_gid=%d dst_gid=%d: %s", src_gid, dst_gid, e.what());
        }
    }

    void dump_group_tensor_layout(ax_runner_t &layer, int gid, const char *tag) const
    {
        if (!std::getenv("AXLLM_DEBUG_LAYER0_IO") || !tag)
            return;

        const int num_inputs = layer.get_num_inputs(gid);
        const int num_outputs = layer.get_num_outputs(gid);
        ALOGI("DBGIO %s gid=%d num_inputs=%d num_outputs=%d", tag, gid, num_inputs, num_outputs);
        for (int i = 0; i < num_inputs; ++i)
        {
            const auto &t = layer.get_input(gid, i);
            ALOGI("DBGIO %s input[%d] name=%s shape=%s bytes=%u phy=%p vir=%p",
                  tag,
                  i,
                  t.sName.c_str(),
                  shape_to_string(t.vShape).c_str(),
                  t.nSize,
                  (void *)t.phyAddr,
                  t.pVirAddr);
        }
        for (int i = 0; i < num_outputs; ++i)
        {
            const auto &t = layer.get_output(gid, i);
            ALOGI("DBGIO %s output[%d] name=%s shape=%s bytes=%u phy=%p vir=%p",
                  tag,
                  i,
                  t.sName.c_str(),
                  shape_to_string(t.vShape).c_str(),
                  t.nSize,
                  (void *)t.phyAddr,
                  t.pVirAddr);
        }
    }

    void dump_selected_prefill_tensors(ax_runner_t &layer, int gid, int step) const
    {
        if (!std::getenv("AXLLM_DEBUG_LAYER0_IO"))
            return;

        auto dump_bf16_tensor = [&](const char *name) {
            try
            {
                const auto &t = layer.get_input(gid, name);
                const int n = (int)(t.nSize / sizeof(unsigned short));
                if (n <= 0 || !t.pVirAddr) return;
                float min_v = 0.0f, max_v = 0.0f, mean_abs = 0.0f;
                summarize_bf16_buffer((const unsigned short *)t.pVirAddr, n, min_v, max_v, mean_abs);
                ALOGI("DBGIO step=%d gid=%d %s hash=0x%llx min=%.6f max=%.6f mean_abs=%.6f",
                      step,
                      gid,
                      name,
                      (unsigned long long)hash_u16_buffer((const unsigned short *)t.pVirAddr, n),
                      min_v,
                      max_v,
                      mean_abs);
            }
            catch (const std::exception &)
            {
            }
        };

        auto dump_u32_tensor = [&](const char *name, int limit) {
            try
            {
                const auto &t = layer.get_input(gid, name);
                const int n = (int)(t.nSize / sizeof(unsigned int));
                if (n <= 0 || !t.pVirAddr) return;
                std::string vals;
                const unsigned int *ptr = (const unsigned int *)t.pVirAddr;
                const int take = std::min(n, limit);
                for (int i = 0; i < take; ++i)
                {
                    if (i) vals.push_back(',');
                    vals += std::to_string(ptr[i]);
                }
                ALOGI("DBGIO step=%d gid=%d %s first=%s", step, gid, name, vals.c_str());
            }
            catch (const std::exception &)
            {
            }
        };

        dump_u32_tensor("indices", 32);
        dump_bf16_tensor("mask");
        dump_bf16_tensor("input");
        dump_bf16_tensor("K_cache");
        dump_bf16_tensor("V_cache");
    }

    void fill_linear_prefill_mask(std::vector<unsigned short> &linear_mask_tmp,
                                  size_t elems,
                                  int input_num_token) const
    {
        const unsigned short one = bfloat16(1.0f).data;
        linear_mask_tmp.assign(elems, 0);
        const int n = std::min((int)elems, input_num_token);
        const std::string mode = getenv_string("AXLLM_LINEAR_PREFILL_MASK_MODE");
        if (mode == "all_one")
        {
            std::fill(linear_mask_tmp.begin(), linear_mask_tmp.end(), one);
            return;
        }
        if (mode == "all_zero")
        {
            return;
        }
        if (mode == "prefix_zero_rest_one")
        {
            std::fill(linear_mask_tmp.begin(), linear_mask_tmp.end(), one);
            for (int i = 0; i < n; ++i) linear_mask_tmp[(size_t)i] = 0;
            return;
        }
        for (int i = 0; i < n; ++i) linear_mask_tmp[(size_t)i] = one;
    }

    void copy_full_cache_prefix_to_group(ax_runner_t &layer,
                                         int dst_gid,
                                         int src_gid,
                                         int layer_kv,
                                         int valid_tokens,
                                         int devid,
                                         bool clear_dst) const
    {
        if (layer_kv <= 0 || valid_tokens <= 0) return;
        auto &src_k = layer.get_input(src_gid, "K_cache");
        auto &src_v = layer.get_input(src_gid, "V_cache");
        auto &dst_k = layer.get_input(dst_gid, "K_cache");
        auto &dst_v = layer.get_input(dst_gid, "V_cache");
        const size_t bytes_per_token = (size_t)layer_kv * sizeof(unsigned short);
        if (bytes_per_token == 0) return;

        const size_t src_tokens_k = (size_t)src_k.nSize / bytes_per_token;
        const size_t src_tokens_v = (size_t)src_v.nSize / bytes_per_token;
        const size_t dst_tokens_k = (size_t)dst_k.nSize / bytes_per_token;
        const size_t dst_tokens_v = (size_t)dst_v.nSize / bytes_per_token;
        const size_t copy_tokens_k = std::min({(size_t)valid_tokens, src_tokens_k, dst_tokens_k});
        const size_t copy_tokens_v = std::min({(size_t)valid_tokens, src_tokens_v, dst_tokens_v});
        const bool same_k_buffer = LLM_WADDR(dst_k) == LLM_RADDR(src_k);
        const bool same_v_buffer = LLM_WADDR(dst_v) == LLM_RADDR(src_v);

        if (clear_dst && dst_gid != src_gid && !same_k_buffer)
            llm_memset(LLM_WADDR(dst_k), 0, dst_k.nSize, devid);
        if (clear_dst && dst_gid != src_gid && !same_v_buffer)
            llm_memset(LLM_WADDR(dst_v), 0, dst_v.nSize, devid);

        if (copy_tokens_k > 0 && !same_k_buffer)
            llm_d2d(LLM_WADDR(dst_k), LLM_RADDR(src_k), copy_tokens_k * bytes_per_token, devid);
        if (copy_tokens_v > 0 && !same_v_buffer)
            llm_d2d(LLM_WADDR(dst_v), LLM_RADDR(src_v), copy_tokens_v * bytes_per_token, devid);
    }

    void sync_device_kv_cache_from_decode(int src_decode_grpid,
                                          int dst_decode_grpid,
                                          int valid_tokens,
                                          bool sync_prefill_groups)
    {
        if (valid_tokens <= 0) return;
        // ALOGI("sync KV cache from decode: src_gid=%d dst_gid=%d valid_tokens=%d sync_prefill=%d",
        //       src_decode_grpid,
        //       dst_decode_grpid,
        //       valid_tokens,
        //       sync_prefill_groups ? 1 : 0);
        for (int m = 0; m < _attr.axmodel_num; ++m)
        {
            auto &lyr = llama_layers[(size_t)m];
            const int devid = LLM_DEVID(lyr);
            const int src_layer_decode_gid = decode_gid_for_layer(m, src_decode_grpid);
            const int dst_layer_decode_gid = decode_gid_for_layer(m, dst_decode_grpid);

            if (is_linear_layer(m))
            {
                copy_linear_input_state_to_group(lyr.layer, dst_layer_decode_gid, src_layer_decode_gid, devid);
                if (sync_prefill_groups && m >= 0 && m < (int)layer_prefill_grpids_.size())
                {
                    for (const int gid : layer_prefill_grpids_[(size_t)m])
                        copy_linear_input_state_to_group(lyr.layer, gid, src_layer_decode_gid, devid);
                }
                continue;
            }

            const int layer_kv = kv_cache_size_for_layer(m);
            if (dst_layer_decode_gid != src_layer_decode_gid)
                copy_full_cache_prefix_to_group(lyr.layer, dst_layer_decode_gid, src_layer_decode_gid, layer_kv, valid_tokens, devid, true);

            if (sync_prefill_groups && m >= 0 && m < (int)layer_prefill_grpids_.size())
            {
                for (const int gid : layer_prefill_grpids_[(size_t)m])
                    copy_full_cache_prefix_to_group(lyr.layer, gid, src_layer_decode_gid, layer_kv, valid_tokens, devid, true);
            }
        }
    }

    void init_shared_kv_source_layers();

    int kv_cache_size_for_layer(int layer_idx) const
    {
        if (layer_idx >= 0 && layer_idx < (int)layer_kv_cache_sizes.size() && layer_kv_cache_sizes[(size_t)layer_idx] > 0)
            return layer_kv_cache_sizes[(size_t)layer_idx];
        return _attr.kv_cache_size;
    }

    int shared_kv_source_for_layer(int layer_idx) const
    {
        if (layer_idx >= 0 && layer_idx < (int)shared_kv_source_layers.size())
            return shared_kv_source_layers[(size_t)layer_idx];
        return -1;
    }

    bool init_groups_from_model(ax_runner_t &ref_layer);

    void init_layer_types();

    bool is_linear_layer(int layer_idx) const
    {
        return layer_idx >= 0 &&
               layer_idx < (int)layer_is_linear_attn.size() &&
               layer_is_linear_attn[(size_t)layer_idx];
    }

    bool has_linear_attention_layers() const
    {
        for (bool is_linear : layer_is_linear_attn)
        {
            if (is_linear) return true;
        }
        return false;
    }

    bool is_sliding_attention_layer(int layer_idx) const
    {
        return _attr.sliding_window > 0 &&
               layer_idx >= 0 &&
               layer_idx < (int)_attr.layer_types.size() &&
               _attr.layer_types[(size_t)layer_idx] == "sliding_attention";
    }

    int first_full_layer_idx() const
    {
        for (int i = 0; i < (int)layer_is_linear_attn.size(); ++i)
        {
            if (!layer_is_linear_attn[(size_t)i]) return i;
        }
        return -1;
    }

#ifdef USE_AXCL
    std::vector<int> distributeModels(int cardCount, int modelCount);
#endif

    std::vector<int> diff_token_ids(const std::vector<int> &ids1, const std::vector<int> &ids2, int &offset) const
    {
        int min_len = (int)std::min(ids1.size(), ids2.size());
        offset = 0;
        for (int i = 0; i < min_len; i++) { if (ids1[i] == ids2[i]) offset++; else break; }
        if (offset >= (int)ids2.size()) return {};
        return std::vector<int>(ids2.begin() + offset, ids2.end());
    }


    bool is_qwen_chat_tokenizer() const
    {
        const std::string key = key_of(_attr.tokenizer_type);
        return key.find("qwen") != std::string::npos;
    }

    bool supports_cached_im_chat_turn_tokens() const
    {
        const std::string key = key_of(_attr.tokenizer_type);
        return key.find("qwen") != std::string::npos ||
               key.find("minicpmv46") != std::string::npos;
    }

    std::string cached_im_chat_generation_prompt() const
    {
        const std::string key = key_of(_attr.tokenizer_type);
        if (key.find("minicpmv46") != std::string::npos)
            return "<think>\n\n</think>\n\n";
        return {};
    }

    bool has_new_media_input(const std::vector<::MediaInputs> &media_inputs, size_t append_start) const
    {
        for (const auto &media : media_inputs)
        {
            if (media.content_index >= append_start)
                return true;
        }
        return false;
    }

    bool appended_history_is_text_only_user_turn(const std::vector<Content> &history, size_t append_start) const
    {
        if (append_start >= history.size()) return false;
        for (size_t i = append_start; i < history.size(); ++i)
        {
            const auto &content = history[i];
            if (content.role != USER || content.type != TEXT)
                return false;
        }
        return history.back().role == USER;
    }

    bool appended_history_is_user_turn_with_media(const std::vector<Content> &history, size_t append_start) const
    {
        if (append_start >= history.size()) return false;
        bool has_media = false;
        for (size_t i = append_start; i < history.size(); ++i)
        {
            const auto &content = history[i];
            if (content.role != USER)
                return false;
            if (content.type == IMAGE || content.type == VIDEO || content.type == AUDIO)
                has_media = true;
            else if (content.type != TEXT)
                return false;
        }
        return has_media && history.back().role == USER;
    }

    void strip_empty_prompt_prefix(std::vector<int> &ids) const
    {
        const std::vector<int> prefix = tokenizer->encode(std::string());
        if (!prefix.empty() && ids.size() >= prefix.size() &&
            std::equal(prefix.begin(), prefix.end(), ids.begin()))
        {
            ids.erase(ids.begin(), ids.begin() + static_cast<std::vector<int>::difference_type>(prefix.size()));
        }
    }

    std::vector<int> encode_text_without_tokenizer_prefix(const std::string &text) const
    {
        std::vector<int> ids = tokenizer->encode(text);
        const std::vector<int> prefix = tokenizer->encode(std::string());
        if (!prefix.empty() && ids.size() >= prefix.size() &&
            std::equal(prefix.begin(), prefix.end(), ids.begin()))
        {
            ids.erase(ids.begin(), ids.begin() + static_cast<std::vector<int>::difference_type>(prefix.size()));
        }
        return ids;
    }

    std::string describe_token_prefix(const std::vector<int> &ids, size_t max_count) const
    {
        std::string out;
        const size_t n = std::min(max_count, ids.size());
        for (size_t i = 0; i < n; ++i)
        {
            if (i) out += ", ";
            out += std::to_string(ids[i]);
            out += "='";
            out += sanitize_utf8_text(safe_decode_token(tokenizer, ids[i]));
            out += "'";
        }
        if (ids.size() > n) out += ", ...";
        return out;
    }

    std::string describe_token_window(const std::vector<int> &ids, int center, int radius) const
    {
        if (ids.empty()) return {};
        const int begin = std::max(0, center - radius);
        const int end = std::min((int)ids.size(), center + radius + 1);
        std::string out;
        for (int i = begin; i < end; ++i)
        {
            if (!out.empty()) out += ", ";
            if (i == center) out += "*";
            out += std::to_string(i);
            out += ":";
            out += std::to_string(ids[(size_t)i]);
            out += "='";
            out += sanitize_utf8_text(safe_decode_token(tokenizer, ids[(size_t)i]));
            out += "'";
        }
        return out;
    }

    bool build_cached_im_chat_text_turn_tokens(const std::vector<Content> &history,
                                               size_t append_start,
                                               std::vector<int> &new_tokens) const
    {
        if (!tokenizer || !supports_cached_im_chat_turn_tokens()) return false;
        if (last_history_snapshot.empty() || append_start != last_history_snapshot.size()) return false;
        if (last_history_snapshot.back().role != ASSISTANT) return false;
        if (!appended_history_is_text_only_user_turn(history, append_start)) return false;
        if ((int)last_tokens_ids.size() != precompute_len) return false;

        // Prefer the tokenizer's real chat template. This keeps text-only follow-up
        // turns aligned with Qwen's prompt formatting and any tokenizer-side policy
        // (for example thinking retention) instead of relying on a hand-written suffix.
        std::vector<int> full_tokens = tokenizer->encode(history);
        int full_offset = 0;
        auto full_diff = diff_token_ids(last_tokens_ids, full_tokens, full_offset);
        if (full_offset == (int)last_tokens_ids.size() &&
            full_tokens.size() >= last_tokens_ids.size() &&
            !full_diff.empty())
        {
            ALOGI("cached IM chat text turn uses full template diff: input_tokens=%zu head=[%s]",
                  full_diff.size(),
                  describe_token_prefix(full_diff, 8).c_str());
            new_tokens = std::move(full_tokens);
            return true;
        }

        ALOGI("cached IM chat text turn full-template prefix mismatch: offset=%d cached=%zu full=%zu, fallback to suffix",
              full_offset,
              last_tokens_ids.size(),
              full_tokens.size());
        if (std::getenv("AXLLM_DEBUG_TOKEN_PREFIX"))
        {
            ALOGI("cached token window near mismatch: [%s]",
                  describe_token_window(last_tokens_ids, full_offset, 4).c_str());
            ALOGI("full-template token window near mismatch: [%s]",
                  describe_token_window(full_tokens, full_offset, 4).c_str());
        }

        std::string suffix;
        suffix.reserve(256);
        suffix += "<|im_end|>\n";
        for (size_t i = append_start; i < history.size(); ++i)
        {
            suffix += "<|im_start|>user\n";
            suffix += history[i].data;
            suffix += "<|im_end|>\n";
        }
        suffix += "<|im_start|>assistant\n";
        suffix += cached_im_chat_generation_prompt();

        std::vector<int> suffix_tokens = encode_text_without_tokenizer_prefix(suffix);
        if (suffix_tokens.empty()) return false;

        ALOGI("cached IM chat text turn uses suffix fallback: input_tokens=%zu head=[%s]",
              suffix_tokens.size(),
              describe_token_prefix(suffix_tokens, 8).c_str());

        new_tokens = last_tokens_ids;
        new_tokens.insert(new_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
        return true;
    }

    bool build_cached_im_chat_media_turn_tokens(const std::vector<Content> &history,
                                                const std::vector<::MediaInputs> &media_inputs,
                                                size_t append_start,
                                                std::vector<int> &new_tokens,
                                                vision::RunState &state_out,
                                                std::string &err) const
    {
        if (!tokenizer || !vision || !vision->enabled() || !supports_cached_im_chat_turn_tokens())
            return false;
        if (last_history_snapshot.empty() || append_start != last_history_snapshot.size()) return false;
        if (last_history_snapshot.back().role != ASSISTANT) return false;
        if (!appended_history_is_user_turn_with_media(history, append_start)) return false;
        if ((int)last_tokens_ids.size() != precompute_len) return false;

        std::vector<Content> tail_history(history.begin() + static_cast<std::vector<Content>::difference_type>(append_start),
                                          history.end());
        std::vector<::MediaInputs> tail_media_inputs;
        tail_media_inputs.reserve(media_inputs.size());
        for (const auto &media : media_inputs)
        {
            if (media.content_index < append_start)
                continue;
            tail_media_inputs.push_back({media.content_index - append_start, media.uris});
        }
        if (tail_media_inputs.empty()) return false;

        std::vector<vision::MediaInputs> vmins;
        vmins.reserve(tail_media_inputs.size());
        for (const auto &m : tail_media_inputs) vmins.push_back({m.content_index, m.uris});

        std::vector<int> close_tokens = encode_text_without_tokenizer_prefix("<|im_end|>\n");
        if (close_tokens.empty()) return false;

        vision::PromptBudget budget;
        const int max_cap = !_attr.prefill_max_kv_cache_num_grp.empty()
                                ? _attr.prefill_max_kv_cache_num_grp.back()
                                : _attr.prefill_max_token_num;
        int remaining = std::max(0, max_cap - precompute_len - (int)close_tokens.size());
        if (_attr.prefill_token_num > 0)
        {
            remaining = ALIGN_DOWN(remaining, _attr.prefill_token_num);
        }
        budget.prefill_token_num = _attr.prefill_token_num;
        budget.max_total_tokens = remaining;
        budget.max_tail_tokens = remaining;

        std::vector<Content> prepared_tail;
        std::vector<int> tail_ids;
        vision::RunState tail_state;
        if (!vision->Prepare(tail_history, vmins, &budget, prepared_tail, tail_ids, tail_state, err, true))
            return false;
        strip_empty_prompt_prefix(tail_ids);

        if (tail_ids.empty())
            return false;

        new_tokens = last_tokens_ids;
        const size_t close_start = new_tokens.size();
        new_tokens.insert(new_tokens.end(), close_tokens.begin(), close_tokens.end());
        const size_t tail_start = new_tokens.size();
        new_tokens.insert(new_tokens.end(), tail_ids.begin(), tail_ids.end());

        state_out = {};
        state_out.pos2vision.assign(new_tokens.size(), -1);
        for (size_t i = 0; i < tail_state.pos2vision.size(); ++i)
        {
            const size_t dst = tail_start + i;
            if (dst >= state_out.pos2vision.size()) break;
            state_out.pos2vision[dst] = tail_state.pos2vision[i];
        }
        state_out.vision_embed = std::move(tail_state.vision_embed);
        state_out.deepstack_features = std::move(tail_state.deepstack_features);
        state_out.position_ids = std::move(tail_state.position_ids);
        if (tail_state.decode_start >= 0)
            state_out.decode_start = (int)tail_start + tail_state.decode_start;

        ALOGI("cached IM chat media turn uses tail Prepare: close_tokens=%zu tail_tokens=%zu cached_tokens=%zu media_inputs=%zu",
              close_tokens.size(),
              tail_ids.size(),
              close_start,
              tail_media_inputs.size());
        return true;
    }

    // Guard a device (CMM) load. `est_mb` defaults to the model file size.
    bool guard_device_load(const std::string &file, int devid, int est_mb = -1)
    {
        if (!_attr.mem_guard_enable) return true;
        if (est_mb < 0) est_mb = estimate_model_mb(file);
        const int remain = mem_guard_.device_remaining_cmm_mb(devid);
        if (mem_guard_allow_load(_attr.mem_guard_enable, _attr.mem_guard_floor_mb,
                                 _attr.mem_guard_on_unsafe, file, est_mb, remain))
            return true;
        set_last_error("CMM 不足，已中止加载: " + file + "（可调小模型/释放显存，或在 config.json 设 mem_guard_enable=false）");
        ALOGE("mem-guard aborted device load: %s (est %d MB, remain %d MB)", file.c_str(), est_mb, remain);
        return false;
    }

    // Guard a host (DDR) load.
    bool guard_host_load(const std::string &file, int est_mb = -1)
    {
        if (!_attr.mem_guard_enable) return true;
        if (est_mb < 0) est_mb = estimate_model_mb(file);
        const int remain = get_remaining_ddr_size();
        if (mem_guard_allow_load(_attr.mem_guard_enable, _attr.mem_guard_floor_mb,
                                 _attr.mem_guard_on_unsafe, file, est_mb, remain))
            return true;
        set_last_error("主机内存(DDR)不足，已中止加载: " + file + "（或在 config.json 设 mem_guard_enable=false）");
        ALOGE("mem-guard aborted host load: %s (est %d MB, remain %d MB)", file.c_str(), est_mb, remain);
        return false;
    }

    // Pre-flight memory budget check before loading any model piece. Sums the
    // estimated footprint (~file sizes) and checks it against remaining CMM (per
    // device) and host DDR. `dev_of_layer` gives each layer's device id for
    // multi-device (AXCL); empty => single on-chip device (AX650, devid -1).
    bool mem_preflight(const std::vector<int> &dev_of_layer)
    {
        if (!_attr.mem_guard_enable) return true;

        // Host DDR: token embedding (only when read fully into RAM, not mmap) +
        // optional Gemma per-layer projection weights.
        int host_mb = 0;
        if (!_attr.b_use_mmap_load_embed)
            host_mb += estimate_model_mb(_attr.filename_tokens_embed);
        if (_attr.hidden_size_per_layer_input > 0)
        {
            host_mb += estimate_model_mb(_attr.filename_tokens_embed_per_layer);
            host_mb += estimate_model_mb(_attr.filename_per_layer_model_projection);
            host_mb += estimate_model_mb(_attr.filename_per_layer_projection_norm);
        }
        if (host_mb > 0 && !guard_host_load("host: token-embedding / per-layer weights", host_mb))
            return false;

        // CMM: vision/audio encoders (always fully resident) + post + layers.
        int enc_mb = 0;
        if (_attr.vlm_type != VLMType::None)
        {
            enc_mb += estimate_model_mb(_attr.filename_image_encoder_axmodel);
            enc_mb += estimate_model_mb(_attr.filename_audio_encoder_axmodel_5s);
            enc_mb += estimate_model_mb(_attr.filename_audio_encoder_axmodel_30s);
        }
        // Dynamic load frees layer weights after init, so don't count them; their
        // residency is bounded by the pool, and dynamic load is the save-CMM mode.
        const bool count_layers = !dynamic_layer_load_enabled();

        if (dev_of_layer.empty())
        {
            int cmm_mb = enc_mb + estimate_model_mb(_attr.filename_post_axmodel);
            if (count_layers)
                for (auto &l : llama_layers) cmm_mb += estimate_model_mb(l.filename);
            if (!guard_device_load("device CMM (encoders + post" + std::string(count_layers ? " + all layers)" : ")"), -1, cmm_mb))
                return false;
        }
        else
        {
            std::map<int, int> need;
            if (count_layers)
                for (size_t i = 0; i < llama_layers.size() && i < dev_of_layer.size(); ++i)
                    need[dev_of_layer[i]] += estimate_model_mb(llama_layers[i].filename);
            const int post_dev = dev_of_layer.back();
            need[post_dev] += estimate_model_mb(_attr.filename_post_axmodel);
            need[dev_of_layer.front()] += enc_mb;
            for (const auto &kv : need)
                if (!guard_device_load("device " + std::to_string(kv.first) + " CMM", kv.first, kv.second))
                    return false;
        }
        return true;
    }

    // ---- Running (measurement-based) load guard ----
    // The measurement/extrapolation engine + its state now live in MemGuard
    // (mem_guard_). Impl keeps the topology-aware entry point below.

    // Build each layer's device id (via layer_devid_for) and hand it to the guard.
    void running_guard_init()
    {
        std::vector<int> devid_of_layer;
        if (_attr.mem_guard_enable && _attr.axmodel_num > 0)
        {
            devid_of_layer.reserve((size_t)_attr.axmodel_num);
            for (int i = 0; i < _attr.axmodel_num; ++i) devid_of_layer.push_back(layer_devid_for(i));
        }
        mem_guard_.running_guard_init(devid_of_layer);
    }
    // Sequential-path wrapper: evaluate, log, and on abort set last_error. Returns
    // false to abort. Safe only on the main thread (logs / writes last_error).
    bool running_guard_check(int devid, int loaded_on_dev)
    {
        if (mem_guard_.is_guard_settled(devid)) return true; // already confidently safe -> skip the CMM query
        const MemGuard::GuardVerdict v = mem_guard_.running_guard_eval(devid, loaded_on_dev);
        if (v.confident) mem_guard_.mark_guard_settled(devid);
        if (v.warned)
            ALOGW("[mem-guard] WARN(measured): %s (projected +%d MB, remain %d MB) -> continuing",
                  v.what.c_str(), v.projected, v.remain);
        if (v.ok) return true;
        ALOGE("mem-guard aborted mid-load: %s (projected +%d MB, remain %d MB)", v.what.c_str(), v.projected, v.remain);
        set_last_error("CMM 不足，已中止加载(实测外推): " + v.what +
                       "（可调小模型/释放显存，或在 config.json 设 mem_guard_enable=false）");
        return false;
    }

    bool Init(LLMAttrType attr);

    void Deinit();

    void Stop() { b_stop.store(true, std::memory_order_relaxed); }

    bool EmbedTokens(const std::vector<int> &token_ids, std::vector<float> &out_embedding);

    bool EmbedHistory(const std::vector<Content> &history_in,
                      const std::vector<::MediaInputs> &media_inputs,
                      std::vector<float> &out_embedding);

    bool EmbedText(const std::string &text, std::vector<float> &out_embedding);

    bool EmbedBatch(const std::vector<std::string> &inputs, std::vector<std::vector<float>> &out_embeddings);

    int GenerateKVCachePrefill(std::vector<int> &_token_ids,
                               std::vector<std::vector<unsigned short>> &k_caches,
                               std::vector<std::vector<unsigned short>> &v_caches,
                               int &prefill_precompute_len);

    int GetKVCache(std::vector<std::vector<unsigned short>> &kv_k, std::vector<std::vector<unsigned short>> &kv_v, int &kv_precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        int inferred_precompute_len = 0;
        const int ref_decode_grpid = decode_gid_for_layer(cache_ref_full_layer_idx, decode_grpid);
        auto &t_mask = llama_layers[(size_t)cache_ref_full_layer_idx].layer.get_input(ref_decode_grpid, "mask");
        std::vector<unsigned short> mask(t_mask.nSize / sizeof(unsigned short), bf16.data);
        llm_d2h(mask.data(), LLM_RADDR(t_mask), t_mask.nSize, LLM_DEVID(llama_layers[(size_t)cache_ref_full_layer_idx]));
        for (size_t i = 0; i < mask.size(); i++) { if (mask[i] == bf16.data) { inferred_precompute_len = (int)i + 1; break; } }
        kv_precompute_len = precompute_len > 0 ? precompute_len : inferred_precompute_len;
        ALOGI("precompute_len:%d, remaining:%d%s",
              kv_precompute_len,
              _attr.prefill_max_kv_cache_num_grp.back() - kv_precompute_len,
              precompute_len > 0 ? " (tracked)" : " (mask inferred)");
        (void)kv_k;
        (void)kv_v;
        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();
        return 0;
    }

    int SetKVCache(std::vector<std::vector<unsigned short>> &kv_k,
                   std::vector<std::vector<unsigned short>> &kv_v,
                   int _precompute_len, int input_num_token)
    {
        const int needed = _precompute_len + input_num_token;
        const int first_chunk_tokens = std::min(input_num_token, _attr.prefill_token_num);
        const bool prefer_symbolic_group = has_vision_state && _precompute_len > 0;
        const int prev_decode_grpid = decode_grpid;
        decode_grpid = choose_decode_gid(std::max(1, needed));
        _attr.prefill_grpid = select_prefill_group(_precompute_len, first_chunk_tokens, prefer_symbolic_group);
        int kv_cache_num = prefill_capacity_by_gid(_attr.prefill_grpid);
        const int history_cap = prefill_history_capacity_by_gid(_attr.prefill_grpid);
        const int symbolic_cap = prefill_symbolic_capacity_by_gid(_attr.prefill_grpid);
        ALOGI("decode_grpid:%d prefill_grpid:%d history_cap:%d total_cap:%d symbolic_cap:%d precompute_len:%d input_num_token:%d prefer_symbolic_group:%d",
              decode_grpid,
              _attr.prefill_grpid,
              history_cap,
              kv_cache_num,
              symbolic_cap,
              _precompute_len,
              input_num_token,
              prefer_symbolic_group ? 1 : 0);
        if (_attr.prefill_grpid < 0 || kv_cache_num <= 0)
        {
            set_context_limit_error();
            ALOGE("invalid prefill_grpid=%d", _attr.prefill_grpid);
            return -1;
        }
        // Remaining prefill budget should be derived from the model capacity, not accumulated across calls.
        // Otherwise, a failed prefill (e.g. context overflow) can make it negative and break `/reset`.
        const int max_cap = _attr.prefill_max_kv_cache_num_grp.empty() ? 0 : _attr.prefill_max_kv_cache_num_grp.back();
        int remaining = max_cap - _precompute_len;
        if (remaining < 0) remaining = 0;
        remaining = ALIGN_DOWN(remaining, _attr.prefill_token_num);
        _attr.prefill_max_token_num = remaining;
        ALOGI("current prefill_max_token_num:%d", remaining);
        if (_precompute_len > history_cap) {
            set_context_limit_error();
            ALOGE("precompute_len(%d) > history_cap(%d)", _precompute_len, history_cap);
            return -1;
        }
        if (_precompute_len + first_chunk_tokens > kv_cache_num) {
            set_context_limit_error();
            ALOGE("precompute_len(%d) + first_chunk_tokens(%d) > kv_cache_num(%d)", _precompute_len, first_chunk_tokens, kv_cache_num);
            return -1;
        }
        if (input_num_token > remaining) {
            set_context_limit_error();
            ALOGE("input_num_token(%d) > prefill_max_token_num(%d)", input_num_token, remaining);
            return -1;
        }
        if (_precompute_len == 0) { clear_all_group_kv_cache_tensors(); reset_full_cache_slot_state(); ALOGI("first run"); return 0; }
        if (full_cache_valid_slots_.empty()) mark_full_cache_slots(0, _precompute_len);
        // KV always lives on the device (single context, or the active prefix-cache
        // slot's own device buffer). Just sync the cached prefix across shape groups.
        (void)kv_k;
        (void)kv_v;
        sync_device_kv_cache_from_decode(prev_decode_grpid, decode_grpid, _precompute_len, true);
        return 0;
    }

    void ResetKVCache()
    {
        last_tokens_ids.clear(); last_history_snapshot.clear(); run_input_token_ids.clear(); last_run_generated_token_ids.clear(); k_caches.clear(); v_caches.clear(); linear_state_snapshots_.clear(); precompute_len = 0; cached_mrope_next_pos = -1; active_prefill_pos_start = -1; active_token_pos_start = -1; reset_full_cache_slot_state();
        decode_grpid = decode_grpids_.empty() ? 0 : decode_grpids_.back();
        _attr.prefill_grpid = prefill_grpids_.empty() ? 1 : prefill_grpids_.back();
        if (!_attr.prefill_max_kv_cache_num_grp.empty())
        {
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();
        }
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr  = llama_layers[i]; int devid = LLM_DEVID(lyr);
            const int ng = lyr.layer.get_num_input_groups();
            for (int gid = 0; gid < ng; ++gid)
            {
                auto &k = lyr.layer.get_input(gid, "K_cache");
                auto &v = lyr.layer.get_input(gid, "V_cache");
                llm_memset(LLM_WADDR(k), 0, k.nSize, devid);
                llm_memset(LLM_WADDR(v), 0, v.nSize, devid);
            }
        }
    }

    // Remaining device CMM in MB for a layer's device (AX650: single chip).
    int remaining_cmm_mb(int devid) const
    {
#ifdef USE_AXCL
        return axcl_GetCMMRemain(devid);
#else
        (void)devid;
        return get_remaining_cmm_size();
#endif
    }


    // Diagnostic: FNV-1a over the ACTIVE decode-group KV content (same tensors +
    // element counts host_dump_active_kv reads). Lets golden tests detect KV-content
    // corruption from KV/slot refactors; GetKVCache only exposes precompute_len.
    uint64_t hash_active_kv()
    {
        uint64_t h = 1469598103934665603ULL;
        auto fold = [&h](const void *p, size_t n) {
            const unsigned char *b = (const unsigned char *)p;
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
        };
        fold(&precompute_len, sizeof(precompute_len));
        std::vector<unsigned short> tmp;
        for (int m = 0; m < _attr.axmodel_num; ++m)
        {
            auto &lyr = llama_layers[(size_t)m];
            const int devid = LLM_DEVID(lyr);
            const int gid = decode_gid_for_layer(m, decode_grpid);
            auto &t_k = lyr.layer.get_input(gid, "K_cache");
            auto &t_v = lyr.layer.get_input(gid, "V_cache");
            size_t k_elems, v_elems;
            if (is_linear_layer(m))
            {
                k_elems = (size_t)t_k.nSize / sizeof(unsigned short);
                v_elems = (size_t)t_v.nSize / sizeof(unsigned short);
            }
            else
            {
                const size_t layer_kv = (size_t)kv_cache_size_for_layer(m);
                k_elems = v_elems = (size_t)std::max(0, precompute_len) * layer_kv;
            }
            if (k_elems)
            {
                tmp.resize(k_elems);
                llm_d2h(tmp.data(), LLM_RADDR(t_k), std::min(k_elems * sizeof(unsigned short), (size_t)t_k.nSize), devid);
                fold(tmp.data(), k_elems * sizeof(unsigned short));
            }
            if (v_elems)
            {
                tmp.resize(v_elems);
                llm_d2h(tmp.data(), LLM_RADDR(t_v), std::min(v_elems * sizeof(unsigned short), (size_t)t_v.nSize), devid);
                fold(tmp.data(), v_elems * sizeof(unsigned short));
            }
        }
        return h;
    }

    // Diagnostic: FNV-1a over the last Prepare()'s vision encoder output
    // (vision_state.vision_embed). Golden fingerprint for VLM image preprocessing.
    uint64_t hash_last_vision_embed()
    {
        if (!has_vision_state || vision_state.vision_embed.empty()) return 0;
        const auto &ve = vision_state.vision_embed;
        uint64_t h = 1469598103934665603ULL;
        const unsigned char *b = (const unsigned char *)ve.data();
        const size_t n = ve.size() * sizeof(unsigned short);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
        return h;
    }


    std::string Run(std::vector<unsigned short> &test_embed, int output_max_token = -1);

    std::vector<Content> Run(std::vector<Content> history, int output_max_token = -1);

    std::vector<Content> Run(std::vector<Content> history, const std::vector<::MediaInputs> &media_inputs, int output_max_token = -1);
};


