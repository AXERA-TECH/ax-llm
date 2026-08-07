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
#include "MemGuard.hpp"

#ifdef USE_AXCL
#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "utils/axcl_manager.h"
using ax_runner_t = ax_runner_axcl;
#else
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include <ax_sys_api.h>
using ax_runner_t = ax_runner_ax650;
#endif

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))

#ifdef USE_AXCL
static inline void llm_memset(void *phy, int val, size_t n, int devid) { axcl_Memset(phy, (uint8_t)val, n, devid); }
static inline void llm_h2d(void *phy_dst, const void *src, size_t n, int devid) { axcl_Memcpy(phy_dst, src, n, AXCL_MEMCPY_HOST_TO_DEVICE, devid); }
static inline void llm_d2h(void *dst, const void *phy_src, size_t n, int devid) { axcl_Memcpy(dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_HOST, devid); }
static inline void llm_d2d(void *phy_dst, const void *phy_src, size_t n, int devid) { axcl_Memcpy(phy_dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_DEVICE, devid); }
#define LLM_WADDR(t)      ((void *)(t).phyAddr)
#define LLM_RADDR(t)      ((const void *)(t).phyAddr)
#define LLM_DEVID(layer_obj) ((layer_obj).layer.get_devid())
#else
static inline void llm_memset(void *vir, int val, size_t n, int /*devid*/) { memset(vir, val, n); }
static inline void llm_h2d(void *vir_dst, const void *src, size_t n, int /*devid*/) { memcpy(vir_dst, src, n); }
static inline void llm_d2h(void *dst, const void *vir_src, size_t n, int /*devid*/) { memcpy(dst, vir_src, n); }
static inline void llm_d2d(void *vir_dst, const void *vir_src, size_t n, int /*devid*/) { memcpy(vir_dst, vir_src, n); }
#define LLM_WADDR(t)      ((t).pVirAddr)
#define LLM_RADDR(t)      ((const void *)(t).pVirAddr)
#define LLM_DEVID(layer_obj) (0)
#endif

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

static inline bool same_history_content(const Content &lhs, const Content &rhs)
{
    return lhs.role == rhs.role &&
           lhs.type == rhs.type &&
           lhs.data == rhs.data;
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

struct LLM::Impl {
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
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    struct LinearStateSnapshot {
        int token_len = 0;
        std::vector<std::vector<unsigned short>> k;
        std::vector<std::vector<unsigned short>> v;
    };
    std::vector<LinearStateSnapshot> linear_state_snapshots_;
    int precompute_len = 0;
    std::vector<int> prefill_history_kv_cache_num_grp;
    std::vector<int> prefill_symbolic_kv_cache_num_grp;

    // ---- Multi-slot prefix KV cache ----
    // Each slot mirrors the host-side context state; the device-side K/V lives in
    // a per-slot device buffer managed by the backend (zero-copy activate). The
    // working fields above (last_*, precompute_len, linear snapshots, full-cache
    // slot bookkeeping) always reflect the currently active slot.
    enum class KvSlotLocation { Device, Host };
    struct KvCacheSlot {
        bool used = false;
        std::vector<Content> last_history_snapshot;
        std::vector<int> last_tokens_ids;
        int precompute_len = 0;
        std::vector<LinearStateSnapshot> linear_state_snapshots;
        int cached_mrope_next_pos = -1;
        std::vector<unsigned char> full_cache_valid_slots;
        bool full_cache_has_sparse_slots = false;
        uint64_t lru = 0;
        // Host-mode only: per-layer host copy of this slot's device K/V. Swapped
        // in/out of the single engine KV buffer on activation.
        std::vector<std::vector<unsigned short>> host_k, host_v;
    };
    std::vector<KvCacheSlot> kv_slots_;
    int kv_active_slot_idx_ = 0;
    bool multi_slot_enabled_ = false;
    bool multi_slot_active_request_ = false; // this request is served from a slot
    KvSlotLocation kv_slot_location_ = KvSlotLocation::Device;
    uint64_t kv_slot_lru_counter_ = 0;
    // A used slot is reused when it shares at least this many leading tokens with
    // the request (covers same-system-prompt requests; avoids clobbering a slot
    // for only the few chat-template header tokens every request trivially shares).
    static constexpr int kSlotReuseMinPrefix = 8;

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

    struct LLMLayer {
        ax_runner_t layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
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

    void init_layer_groups()
    {
        layer_decode_grpids_.assign((size_t)_attr.axmodel_num, {});
        layer_prefill_grpids_.assign((size_t)_attr.axmodel_num, {});
        for (int i = 0; i < _attr.axmodel_num; ++i)
        {
            layer_decode_grpids_[(size_t)i] = detect_decode_grpids(llama_layers[(size_t)i].layer);
            layer_prefill_grpids_[(size_t)i] = detect_prefill_grpids(llama_layers[(size_t)i].layer);
            const auto &decode_gids = layer_decode_grpids_[(size_t)i];
            const auto &gids = layer_prefill_grpids_[(size_t)i];
            if (decode_gids.empty())
            {
                ALOGW("layer %d has no decode groups detected", i);
            }
            else if (decode_gids.size() < decode_grpids_.size())
            {
                ALOGI("layer %d decode groups=%zu ref=%zu, reuse gid %d for later decode shapes",
                      i,
                      decode_gids.size(),
                      decode_grpids_.size(),
                      decode_gids.back());
            }
            if (gids.empty())
            {
                ALOGW("layer %d has no prefill groups detected", i);
                continue;
            }
            if (gids.size() < prefill_grpids_.size())
            {
                ALOGI("layer %d prefill groups=%zu ref=%zu, reuse gid %d for later prefill chunks",
                      i,
                      gids.size(),
                      prefill_grpids_.size(),
                      gids.back());
            }
        }
    }

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

    void init_shared_kv_source_layers()
    {
        shared_kv_source_layers.assign(_attr.axmodel_num, -1);

        if (_attr.layer_types.empty() || _attr.num_kv_shared_layers <= 0)
            return;

        const int num_layers = std::min(_attr.axmodel_num, (int)_attr.layer_types.size());
        const int first_shared_layer = num_layers - _attr.num_kv_shared_layers;
        if (first_shared_layer <= 0)
            return;

        std::vector<std::string> prev_layers(_attr.layer_types.begin(), _attr.layer_types.begin() + first_shared_layer);
        for (int layer_idx = first_shared_layer; layer_idx < num_layers; ++layer_idx)
        {
            const std::string &layer_type = _attr.layer_types[(size_t)layer_idx];
            for (int prev_idx = (int)prev_layers.size() - 1; prev_idx >= 0; --prev_idx)
            {
                if (prev_layers[(size_t)prev_idx] == layer_type)
                {
                    shared_kv_source_layers[(size_t)layer_idx] = prev_idx;
                    break;
                }
            }
        }
    }

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

    bool init_groups_from_model(ax_runner_t &ref_layer)
    {
        const int group_count = ref_layer.get_num_input_groups();
        if (group_count <= 0)
        {
            ALOGE("invalid group_count=%d", group_count);
            return false;
        }

        std::vector<int> decode_gids;
        std::vector<int> prefill_gids;
        decode_gids.reserve((size_t)group_count);
        prefill_gids.reserve((size_t)group_count);

        for (int gid = 0; gid < group_count; ++gid)
        {
            try
            {
                const auto &t_idx = ref_layer.get_input(gid, "indices");
                const int idx_elems = (int)((size_t)t_idx.nSize / sizeof(unsigned int));
                if (idx_elems == 1) decode_gids.push_back(gid);
                else prefill_gids.push_back(gid);
            }
            catch (const std::exception &)
            {
                // Skip groups without indices tensor.
            }
        }

        if (decode_gids.empty())
        {
            ALOGE("no decode groups detected");
            return false;
        }
        decode_only_prefill_mode_ = false;
        if (prefill_gids.empty())
        {
            decode_only_prefill_mode_ = true;
            ALOGW("no native prefill groups detected, fallback to decode-only prefill mode");
        }

        // Detect prefill_token_num from the first prefill group.
        // Prefer `indices` last-dim (most consistent across models/backends). Fall back to `mask` shape.
        int detected_prefill_token_num = 0;
        if (decode_only_prefill_mode_)
        {
            detected_prefill_token_num = 1;
        }
        else
        {
            const int gid = prefill_gids.front();
            try
            {
                const auto &t_idx = ref_layer.get_input(gid, "indices");
                if (!t_idx.vShape.empty()) detected_prefill_token_num = (int)t_idx.vShape.back();
            }
            catch (const std::exception &)
            {
            }
            // Guard against picking the batch dimension (often 1).
            if (detected_prefill_token_num <= 1)
            {
                try
                {
                    const auto &t_mask = ref_layer.get_input(gid, "mask");
                    if (!t_mask.vShape.empty())
                    {
                        if (t_mask.vShape.size() >= 2 && t_mask.vShape[0] == 1)
                            detected_prefill_token_num = (int)t_mask.vShape[1];
                        else
                            detected_prefill_token_num = (int)t_mask.vShape[0];
                    }
                }
                catch (const std::exception &)
                {
                }
            }
            if (detected_prefill_token_num <= 1) detected_prefill_token_num = _attr.prefill_token_num;
        }

        _attr.prefill_token_num = detected_prefill_token_num;
        ALOGI("prefill_token_num : %d", _attr.prefill_token_num);

        // Build decode groups: capacity from mask length (max_token_len), fallback to K_cache tokens.
        std::vector<std::pair<int, int>> decode_pairs; // (cap, gid)
        for (const int gid : decode_gids)
        {
            int cap = -1;
            try
            {
                const auto &t_mask = ref_layer.get_input(gid, "mask");
                const int elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
                if (elems > 0) cap = elems - 1;
            }
            catch (const std::exception &)
            {
            }
            if (cap < 0)
            {
                try
                {
                    const auto &t_k = ref_layer.get_input(gid, "K_cache");
                    const size_t denom = (size_t)_attr.kv_cache_size * sizeof(unsigned short);
                    if (denom > 0) cap = (int)((size_t)t_k.nSize / denom);
                }
                catch (const std::exception &)
                {
                }
            }
            if (cap > 0) decode_pairs.push_back({cap, gid});
        }

        // Build prefill groups:
        // - total capacity: number of tokens handled by the prefill group (`history + chunk`)
        // - history capacity: visible cached tokens before the current chunk
        // - symbolic capacity: raw `K_cache` shape, useful for some multimodal models
        struct PrefillGroupInfo {
            int total_cap = -1;
            int history_cap = -1;
            int symbolic_cap = -1;
            int gid = -1;
        };
        std::vector<PrefillGroupInfo> prefill_infos;
        const auto &prefill_scan_gids = decode_only_prefill_mode_ ? decode_gids : prefill_gids;
        for (const int gid : prefill_scan_gids)
        {
            int total_cap = -1;
            int history_cap = -1;
            int symbolic_cap = -1;
            if (decode_only_prefill_mode_)
            {
                int decode_cap = -1;
                try
                {
                    const auto &t_mask = ref_layer.get_input(gid, "mask");
                    const int elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
                    if (elems > 0) decode_cap = elems - 1;
                }
                catch (const std::exception &)
                {
                }
                if (decode_cap < 0)
                {
                    try
                    {
                        const auto &t_k = ref_layer.get_input(gid, "K_cache");
                        const size_t denom = (size_t)_attr.kv_cache_size * sizeof(unsigned short);
                        if (denom > 0) decode_cap = (int)((size_t)t_k.nSize / denom);
                    }
                    catch (const std::exception &)
                    {
                    }
                }
                if (decode_cap >= 0)
                {
                    history_cap = decode_cap;
                    total_cap = decode_cap + 1;
                    symbolic_cap = decode_cap;
                    prefill_infos.push_back({total_cap, history_cap, symbolic_cap, gid});
                }
                continue;
            }
            try
            {
                const auto &t_mask = ref_layer.get_input(gid, "mask");
                const int elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
                if (_attr.prefill_token_num > 0 && elems > 0 && (elems % _attr.prefill_token_num) == 0)
                {
                    const int cols = elems / _attr.prefill_token_num;
                    if (cols >= _attr.prefill_token_num)
                    {
                        total_cap = cols;
                        history_cap = cols - _attr.prefill_token_num;
                    }
                }
            }
            catch (const std::exception &)
            {
            }
            try
            {
                const auto &t_k = ref_layer.get_input(gid, "K_cache");
                if (t_k.vShape.size() >= 2)
                {
                    symbolic_cap = (int)t_k.vShape[1];
                }
                if (symbolic_cap <= 0)
                {
                    const size_t denom = (size_t)_attr.kv_cache_size * sizeof(unsigned short);
                    if (denom > 0) symbolic_cap = (int)((size_t)t_k.nSize / denom);
                }
            }
            catch (const std::exception &)
            {
            }
            if (total_cap < 0 && symbolic_cap >= 0)
            {
                total_cap = std::max(_attr.prefill_token_num, symbolic_cap + _attr.prefill_token_num);
                history_cap = std::max(0, total_cap - _attr.prefill_token_num);
            }
            if (total_cap >= 0)
            {
                prefill_infos.push_back({total_cap, history_cap, symbolic_cap, gid});
            }
        }

        if (decode_pairs.empty() || prefill_infos.empty())
        {
            ALOGE("failed to parse groups (decode=%zu prefill=%zu)", decode_pairs.size(), prefill_infos.size());
            return false;
        }

        auto dedup_sorted = [](std::vector<std::pair<int, int>> &pairs) {
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end(),
                                    [](const auto &a, const auto &b) { return a.first == b.first; }),
                        pairs.end());
        };
        dedup_sorted(decode_pairs);
        std::sort(prefill_infos.begin(), prefill_infos.end(), [](const PrefillGroupInfo &a, const PrefillGroupInfo &b) {
            if (a.total_cap != b.total_cap) return a.total_cap < b.total_cap;
            return a.gid < b.gid;
        });
        prefill_infos.erase(std::unique(prefill_infos.begin(), prefill_infos.end(),
                                        [](const PrefillGroupInfo &a, const PrefillGroupInfo &b) {
                                            return a.total_cap == b.total_cap;
                                        }),
                            prefill_infos.end());

        auto shape_to_str = [](const std::vector<unsigned int> &shape) -> std::string {
            std::string s;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (i) s.push_back('x');
                s += std::to_string(shape[i]);
            }
            if (s.empty()) s = "(none)";
            return s;
        };

        // Extra debug: print per-group KV cache tensor sizing for sanity checks.
        for (const auto &p : decode_pairs)
        {
            const int cap = p.first;
            const int gid = p.second;
            try
            {
                const auto &t_mask = ref_layer.get_input(gid, "mask");
                const auto &t_k = ref_layer.get_input(gid, "K_cache");
                const auto &t_v = ref_layer.get_input(gid, "V_cache");
                ALOGD("decode gid=%d cap=%d mask=%s(%zuB) K_cache=%s(%zuB) V_cache=%s(%zuB)",
                      gid,
                      cap,
                      shape_to_str(t_mask.vShape).c_str(),
                      (size_t)t_mask.nSize,
                      shape_to_str(t_k.vShape).c_str(),
                      (size_t)t_k.nSize,
                      shape_to_str(t_v.vShape).c_str(),
                      (size_t)t_v.nSize);
            }
            catch (const std::exception &)
            {
            }
        }
        for (const auto &info : prefill_infos)
        {
            const int gid = info.gid;
            try
            {
                const auto &t_mask = ref_layer.get_input(gid, "mask");
                const auto &t_k = ref_layer.get_input(gid, "K_cache");
                const auto &t_v = ref_layer.get_input(gid, "V_cache");
                ALOGD("prefill gid=%d total_cap=%d history_cap=%d symbolic_cap=%d mask=%s(%zuB) K_cache=%s(%zuB) V_cache=%s(%zuB)",
                      gid,
                      info.total_cap,
                      info.history_cap,
                      info.symbolic_cap,
                      shape_to_str(t_mask.vShape).c_str(),
                      (size_t)t_mask.nSize,
                      shape_to_str(t_k.vShape).c_str(),
                      (size_t)t_k.nSize,
                      shape_to_str(t_v.vShape).c_str(),
                      (size_t)t_v.nSize);
            }
            catch (const std::exception &)
            {
            }
        }

        decode_grpids_.clear();
        decode_max_token_len_grp_.clear();
        for (const auto &p : decode_pairs)
        {
            decode_max_token_len_grp_.push_back(p.first);
            decode_grpids_.push_back(p.second);
        }

        prefill_grpids_.clear();
        _attr.prefill_max_kv_cache_num_grp.clear();
        prefill_history_kv_cache_num_grp.clear();
        prefill_symbolic_kv_cache_num_grp.clear();
        for (const auto &info : prefill_infos)
        {
            _attr.prefill_max_kv_cache_num_grp.push_back(info.total_cap);
            prefill_history_kv_cache_num_grp.push_back(std::max(0, info.history_cap));
            prefill_symbolic_kv_cache_num_grp.push_back(std::max(0, info.symbolic_cap));
            prefill_grpids_.push_back(info.gid);
        }

        // Default to the largest groups.
        decode_grpid = decode_grpids_.back();
        _attr.prefill_grpid = prefill_grpids_.back();
        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();

        // Canonical context capacity should follow the largest decode/prefill group.
        // Some models provide multiple decode groups (2k/4k/8k/16k), and group 0 may
        // not be the largest. Use discovered capacities to drive global limits.
        const int max_decode = decode_max_token_len_grp_.back();
        const int max_prefill = _attr.prefill_max_kv_cache_num_grp.back();
        const int canonical_cap = std::max(max_decode, max_prefill);
        if (canonical_cap > 0)
        {
            _attr.max_token_len = canonical_cap;
            _attr.kv_cache_num = canonical_cap;
        }

        // Print group summary for debugging.
        for (size_t i = 0; i < decode_grpids_.size(); ++i)
        {
            ALOGI("decode grp: %zu, gid: %d, max_token_len : %d", i, decode_grpids_[i], decode_max_token_len_grp_[i]);
        }
        for (size_t i = 0; i < prefill_grpids_.size(); ++i)
        {
            ALOGI("prefill grp: %zu, gid: %d, history_cap: %d, total_cap: %d, symbolic_cap: %d",
                  i,
                  prefill_grpids_[i],
                  prefill_history_kv_cache_num_grp[i],
                  _attr.prefill_max_kv_cache_num_grp[i],
                  prefill_symbolic_kv_cache_num_grp[i]);
        }
        ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        return true;
    }

    void init_layer_types()
    {
        layer_is_linear_attn.assign(_attr.axmodel_num, false);
        if (!_attr.layer_types.empty())
        {
            const int num_layers = std::min(_attr.axmodel_num, (int)_attr.layer_types.size());
            for (int i = 0; i < num_layers; ++i)
            {
                const std::string &layer_type = _attr.layer_types[(size_t)i];
                if (layer_type == "linear_attention")
                {
                    layer_is_linear_attn[(size_t)i] = true;
                }
                else if (layer_type == "full_attention" || layer_type == "sliding_attention")
                {
                    // Gemma4 uses `sliding_attention` to distinguish local/global attention style,
                    // not the legacy runtime's "linear attention" cache/mask path.
                    layer_is_linear_attn[(size_t)i] = false;
                }
                else
                {
                    ALOGW("unknown layer_type[%d]=%s, fallback to full-attention", i, layer_type.c_str());
                    layer_is_linear_attn[(size_t)i] = false;
                }
            }
            return;
        }

        const int interval = _attr.full_attention_interval;
        if (interval <= 0) return;
        for (int i = 0; i < _attr.axmodel_num; ++i)
        {
            const bool is_full = (((i + 1) % interval) == 0);
            layer_is_linear_attn[(size_t)i] = !is_full;
        }
    }

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
    std::vector<int> distributeModels(int cardCount, int modelCount)
    {
        std::vector<int> assign(modelCount, 0);
        if (cardCount <= 0 || modelCount <= 0) return assign;
        int base = modelCount / cardCount;
        int rem  = modelCount % cardCount;
        int idx  = 0;
        for (int c = 0; c < cardCount; c++) {
            int cnt = base + (c < rem ? 1 : 0);
            for (int i = 0; i < cnt; i++) assign[idx++] = c;
        }
        return assign;
    }
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

    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        this->_attr = attr;
        deinited_ = false;
        mem_guard_.capture_sentry_baseline();
        dynamic_layer_load_enabled_ = _attr.dynamic_load_enable;
        dynamic_layer_pool_size_ = _attr.dynamic_load_pool_size;
        if (dynamic_layer_load_enabled_ && dynamic_layer_pool_size_ <= 0)
            dynamic_layer_pool_size_ = 2;
        if (dynamic_layer_pool_size_ > 0)
            dynamic_layer_pool_size_ = std::min(dynamic_layer_pool_size_, std::max(1, _attr.axmodel_num));
        dynamic_layer_loaded_.assign((size_t)std::max(0, _attr.axmodel_num), 0);
        dynamic_layer_lru_.clear();
        dynamic_layer_devids_.clear();
        embedding_profile_for_tokenizer(_attr.tokenizer_type, embedding_append_eos, embedding_eos_token_id);
        init_layer_types();
        init_shared_kv_source_layers();
        cache_ref_full_layer_idx = first_full_layer_idx();
        if (cache_ref_full_layer_idx < 0) cache_ref_full_layer_idx = 0;
        if (_attr.full_attention_interval > 0)
        {
            ALOGI("mixed attention enabled: full_attention_interval=%d ref_full_layer_idx=%d",
                  _attr.full_attention_interval,
                  cache_ref_full_layer_idx);
        }
        if (!_attr.layer_types.empty() && _attr.num_kv_shared_layers > 0)
        {
            ALOGI("shared kv enabled: num_kv_shared_layers=%d", _attr.num_kv_shared_layers);
        }
        if (!_attr.layer_types.empty())
        {
            int sliding_count = 0;
            int full_count = 0;
            int linear_count = 0;
            for (const auto &layer_type : _attr.layer_types)
            {
                if (layer_type == "sliding_attention")
                    sliding_count++;
                else if (layer_type == "full_attention")
                    full_count++;
                else if (layer_type == "linear_attention")
                    linear_count++;
            }
            ALOGI("attention config: layers=%zu sliding=%d full=%d linear=%d sliding_window=%d ref_full_layer_idx=%d",
                  _attr.layer_types.size(),
                  sliding_count,
                  full_count,
                  linear_count,
                  _attr.sliding_window,
                  cache_ref_full_layer_idx);
        }

#ifdef USE_AXCL
        // AXCL init may spawn worker threads that print logs. Do it before the progress bar starts.
        for (auto &devid : _attr.dev_ids)
        {
            if (axcl_Init(devid) != 0)
            {
                ALOGE("axcl_Init(%d) failed", devid);
                return false;
            }
        }
#endif

        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        tokenizer = create_tokenizer(this->_attr.tokenizer_type);
        if (!tokenizer) { ALOGE("create_tokenizer(%s) failed", this->_attr.tokenizer_type.c_str()); return false; }
        if (!tokenizer->load(attr.url_tokenizer_model)) { ALOGE("tokenizer.init(%s) failed", attr.url_tokenizer_model.c_str()); return false; }
        tokenizer->set_think_in_prompt(!tokenizer_uses_hidden_channel_markup(this->_attr.tokenizer_type));
        tokenizer->set_generation_thinking_mode(this->_attr.generation_thinking_mode);
        if (this->_attr.generation_thinking_mode != ThinkingMode::Unspecified && !tokenizer->supports_thinking_toggle())
            ALOGW("[thinking] tokenizer_type='%s' does not honor thinking_mode/enable_thinking; the setting is ignored "
                  "(currently supported: Qwen3 family, MiniCPM5)", this->_attr.tokenizer_type.c_str());
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");

#ifdef USE_AXCL
        llama_layers.resize(attr.axmodel_num);
        auto dev_assign = distributeModels((int)_attr.dev_ids.size(), attr.axmodel_num);
        std::vector<int> rets(dynamic_layer_load_enabled() ? 0 : attr.axmodel_num, 0);
        dynamic_layer_devids_.assign((size_t)attr.axmodel_num, _attr.dev_ids.empty() ? 0 : _attr.dev_ids.front());

        // Prepare filenames first (thread-safe, no I/O).
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            char path[1024];
            std::snprintf(path, sizeof(path), attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = path;
            const int dev_idx = (dev_assign.empty() ? 0 : dev_assign[i]);
            if (dev_idx >= 0 && (size_t)dev_idx < _attr.dev_ids.size())
                dynamic_layer_devids_[(size_t)i] = _attr.dev_ids[(size_t)dev_idx];
        }

        if (!mem_preflight(dynamic_layer_devids_)) return false;
        running_guard_init();

        if (dynamic_layer_load_enabled())
        {
            if (_attr.dev_ids.empty())
            {
                ALOGE("dynamic layer loading requires at least one AXCL device");
                return false;
            }
            if (_attr.dev_ids.size() != 1)
            {
                ALOGE("dynamic layer loading only supports single AXCL device for now (dev_ids=%zu)", _attr.dev_ids.size());
                return false;
            }

            const int devid = _attr.dev_ids.front();
            for (int i = 0; i < attr.axmodel_num; i++)
            {
                if (!running_guard_check(devid, i)) return false;
                const int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), devid);
                if (ret != 0)
                {
                    ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                    return false;
                }
                // Keep IO (KV etc), but unload weights/handle to save CMM.
                llama_layers[i].layer.unload_handle_keep_io();

                char path[256];
                std::snprintf(path, sizeof(path), "init %d axmodel io ok,remain_cmm(%d MB)", i, axcl_GetCMMRemain(devid));
                update_cqdm(&cqdm, i + 1, "count", path);
            }

            // Start with an empty residency pool (handles will be loaded on-demand).
            std::fill(dynamic_layer_loaded_.begin(), dynamic_layer_loaded_.end(), 0);
            dynamic_layer_lru_.clear();

            int ret = llama_post.init(attr.filename_post_axmodel.c_str(), devid);
            if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
            char path[1024];
            sprintf(path, "init post axmodel ok,remain_cmm(%d MB)", axcl_GetCMMRemain(devid));
            update_cqdm(&cqdm, attr.axmodel_num + 1, "count", path);
        }
        else
        {
        // Load models in parallel across devices (per-device sequential), while the main thread updates progress.
        struct LoadResult {
            int idx = -1;
            int ret = -1;
            int devid = -1;
            int remain_mb = -1;
            bool skipped = false;   // not loaded (a prior guard abort stopped the run)
            int guard = 0;          // 0=ok, 1=warn(loaded anyway), 2=mem-guard abort(not loaded)
            std::string guard_what;
            std::string msg;
        };

        std::vector<std::vector<int>> models_per_dev(_attr.dev_ids.size());
        for (int i = 0; i < attr.axmodel_num; ++i)
        {
            const int dev_idx = (dev_assign.empty() ? 0 : dev_assign[i]);
            if (dev_idx >= 0 && (size_t)dev_idx < models_per_dev.size()) models_per_dev[(size_t)dev_idx].push_back(i);
        }

        std::mutex q_mu;
        std::condition_variable q_cv;
        std::queue<LoadResult> q;
        std::atomic<bool> load_abort{false};  // measured guard refused -> stop all device threads
        std::vector<std::thread> loaders;
        loaders.reserve(_attr.dev_ids.size());

        for (size_t dev_idx = 0; dev_idx < _attr.dev_ids.size(); ++dev_idx)
        {
            const int devid = _attr.dev_ids[dev_idx];
            loaders.emplace_back([&, dev_idx, devid]() {
                int loaded_on_dev = 0;
                for (const int i : models_per_dev[dev_idx])
                {
                    LoadResult r;
                    r.idx = i;
                    r.devid = devid;
                    if (load_abort.load(std::memory_order_relaxed))
                    {
                        r.skipped = true; // another layer/device already breached the budget
                    }
                    else
                    {
                        // Measured-extrapolation guard (pure -> thread-safe). On abort,
                        // stop this and the other device threads before over-allocating.
                        const auto v = mem_guard_.running_guard_eval(devid, loaded_on_dev);
                        if (!v.ok)
                        {
                            load_abort.store(true, std::memory_order_relaxed);
                            r.guard = 2;
                            r.guard_what = v.what;
                            r.remain_mb = v.remain;
                            r.skipped = true;
                        }
                        else
                        {
                            const int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), devid);
                            const int remain = axcl_GetCMMRemain(devid);
                            ++loaded_on_dev;
                            char buf[256];
                            std::snprintf(buf, sizeof(buf), "init %d axmodel ok,devid(%d) remain_cmm(%d MB)", i, devid, remain);
                            r.ret = ret;
                            r.remain_mb = remain;
                            r.msg = buf;
                            if (v.warned) { r.guard = 1; r.guard_what = v.what; }
                        }
                    }

                    {
                        std::lock_guard<std::mutex> lk(q_mu);
                        q.push(std::move(r));
                    }
                    q_cv.notify_one();
                }
            });
        }

        int progress_step = 1;
        int finished = 0;
        bool guard_aborted = false;
        while (finished < attr.axmodel_num)
        {
            LoadResult r;
            {
                std::unique_lock<std::mutex> lk(q_mu);
                q_cv.wait(lk, [&]() { return !q.empty(); });
                r = std::move(q.front());
                q.pop();
            }
            finished++;
            if (r.guard == 2)
            {
                guard_aborted = true;
                ALOGE("mem-guard aborted mid-load: %s (remain %d MB)", r.guard_what.c_str(), r.remain_mb);
                set_last_error("CMM 不足，已中止加载(实测外推): " + r.guard_what +
                               "（可调小模型/释放显存，或在 config.json 设 mem_guard_enable=false）");
                continue;
            }
            if (r.skipped) continue; // a prior abort stopped this layer
            if (r.guard == 1)
                ALOGW("[mem-guard] WARN(measured): %s (remain %d MB) -> continuing", r.guard_what.c_str(), r.remain_mb);
            if (r.idx >= 0 && r.idx < attr.axmodel_num) rets[r.idx] = r.ret;
            update_cqdm(&cqdm, progress_step++, "count", r.msg.c_str());
        }

        for (auto &t : loaders)
        {
            if (t.joinable()) t.join();
        }

        if (guard_aborted) return false;

        for (int i = 0; i < attr.axmodel_num; i++) { if (rets[i] != 0) { ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str()); return false; } }
        {
            int post_devid = llama_layers.back().layer.get_devid();
            int ret = llama_post.init(attr.filename_post_axmodel.c_str(), post_devid);
            if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
            char path[1024];
            sprintf(path, "init post axmodel ok,remain_cmm(%d MB)", axcl_GetCMMRemain(post_devid));
            update_cqdm(&cqdm, attr.axmodel_num + 1, "count", path);
        }
        }
#else
        llama_layers.resize(attr.axmodel_num);
        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;
        }
        if (!mem_preflight({})) return false;
        running_guard_init();
        if (dynamic_layer_load_enabled())
        {
            for (int i = 0; i < attr.axmodel_num; i++)
            {
                if (!running_guard_check(-1, i)) return false;
                int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), -1);
                if (ret != 0) { ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str()); return false; }
                llama_layers[i].layer.set_auto_sync_before_inference(true);
                llama_layers[i].layer.set_auto_sync_after_inference(true);
                llama_layers[i].layer.unload_handle_keep_io();
                int remain_cmm = get_remaining_cmm_size();
                sprintf(axmodel_path, "init %d axmodel io ok,remain_cmm(%d MB)", i, remain_cmm);
                update_cqdm(&cqdm, i + 1, "count", axmodel_path);
            }

            // Start with an empty residency pool (handles will be loaded on-demand).
            std::fill(dynamic_layer_loaded_.begin(), dynamic_layer_loaded_.end(), 0);
            dynamic_layer_lru_.clear();

            int ret = llama_post.init(attr.filename_post_axmodel.c_str(), -1);
            if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
            llama_post.set_auto_sync_before_inference(true);
            llama_post.set_auto_sync_after_inference(true);
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
            update_cqdm(&cqdm, attr.axmodel_num + 1, "count", axmodel_path);
        }
        else
        {
            for (int i = 0; i < attr.axmodel_num; i++)
            {
                if (!running_guard_check(-1, i)) return false;
                int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), -1);
                if (ret != 0) { ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str()); return false; }
                llama_layers[i].layer.set_auto_sync_before_inference(true);
                llama_layers[i].layer.set_auto_sync_after_inference(true);
                int remain_cmm = get_remaining_cmm_size();
                sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
                update_cqdm(&cqdm, i + 1, "count", axmodel_path);
            }
            {
                int ret = llama_post.init(attr.filename_post_axmodel.c_str(), -1);
                if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
                llama_post.set_auto_sync_before_inference(true);
                llama_post.set_auto_sync_after_inference(true);
                int remain_cmm = get_remaining_cmm_size();
                sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
                update_cqdm(&cqdm, attr.axmodel_num + 1, "count", axmodel_path);
            }
        }
#endif
        axllm::Logger::finish_inplace_line();
        {
            auto &ref_layer = llama_layers[(size_t)cache_ref_full_layer_idx].layer;
            _attr.max_token_len = ref_layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            ALOGI("max_token_len : %d", _attr.max_token_len);
            _attr.kv_cache_size = ref_layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num  = ref_layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num) { ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num); return false; }
            if (!init_groups_from_model(ref_layer)) return false;
            init_layer_groups();
            if (std::getenv("AXLLM_DEBUG_LAYER0_IO"))
            {
                const int layer0_decode_grpid = decode_gid_for_layer(0, decode_grpid);
                dump_group_tensor_layout(llama_layers[0].layer, layer0_decode_grpid, "layer0_decode");
                if (!prefill_grpids_.empty())
                {
                    const int layer0_prefill_grpid = prefill_gid_for_layer(0, prefill_grpids_[0]);
                    dump_group_tensor_layout(llama_layers[0].layer, layer0_prefill_grpid, "layer0_prefill");
                }
            }
            layer_kv_cache_sizes.assign(_attr.axmodel_num, _attr.kv_cache_size);
            for (int i = 0; i < _attr.axmodel_num; ++i)
            {
                const int layer_decode_grpid = decode_gid_for_layer(i, decode_grpid);
                auto &layer_k = llama_layers[(size_t)i].layer.get_input(layer_decode_grpid, "K_cache");
                const int layer_kv_size = (int)(layer_k.nSize / sizeof(unsigned short) / std::max(1, _attr.kv_cache_num));
                if (layer_kv_size > 0)
                    layer_kv_cache_sizes[(size_t)i] = layer_kv_size;
            }
        }

        // embed file check → then init
        {
            const int layer0_decode_grpid = decode_gid_for_layer(0, decode_grpid);
            auto &t_in0 = llama_layers[0].layer.get_input(layer0_decode_grpid, "input");
            int model_embed_sz = t_in0.nSize / (int)sizeof(unsigned short);
            if (model_embed_sz != _attr.tokens_embed_size)
            {
                ALOGE("tokens_embed_size mismatch: config(%d) != model(%d). Please fix config or embed file.", _attr.tokens_embed_size, model_embed_sz);
                return false;
            }
            if (!embed_selector.Init(_attr.filename_tokens_embed, _attr.tokens_embed_num, _attr.tokens_embed_size, _attr.b_use_mmap_load_embed))
            {
                ALOGE("embed_selector.Init(%s, %d, %d) failed", _attr.filename_tokens_embed.c_str(), _attr.tokens_embed_num, _attr.tokens_embed_size);
                return false;
            }
            update_cqdm(&cqdm, attr.axmodel_num + 2, "count", "embed_selector init ok");
        }
        axllm::Logger::finish_inplace_line();

        if (_attr.hidden_size_per_layer_input > 0)
        {
            if (!gemma4_per_layer_helper.Init(_attr.tokens_embed_num,
                                              _attr.tokens_embed_size,
                                              _attr.axmodel_num,
                                              _attr.hidden_size_per_layer_input,
                                              _attr.pad_token_id,
                                              _attr.rms_norm_eps,
                                              _attr.filename_tokens_embed_per_layer,
                                              _attr.filename_per_layer_model_projection,
                                              _attr.filename_per_layer_projection_norm))
            {
                ALOGE("Gemma4 per-layer helper init failed");
                return false;
            }
        }

        // Optional VLM vision encoder (runtime controlled by attr.vlm_type).
        has_vision_state = false;
        if (_attr.vlm_type != VLMType::None)
        {
            vision.reset(new vision::VisionModule());
            std::string verr;
            int vdevid =
#ifdef USE_AXCL
                llama_layers[0].layer.get_devid();
#else
                -1;
#endif

            if (!vision->Init(_attr.vlm_type,
                              _attr.filename_image_encoder_axmodel,
                              _attr.vision_cache_dir,
                              _attr.tokens_embed_size,
                              vdevid,
                              tokenizer,
                              _attr.vision_width,
                              _attr.vision_height,
                              _attr.vision_temporal_patch_size,
                              _attr.vision_spatial_merge_size,
                              _attr.vision_patch_size,
                              _attr.vision_fps,
                              _attr.vision_tokens_per_second,
                              _attr.vision_num_frames,
                              _attr.vision_do_sample_frames,
                              _attr.filename_audio_encoder_axmodel_5s,
                              _attr.filename_audio_encoder_axmodel_30s,
                              verr))
            {
                ALOGE("vision.Init(vlm_type=%s/%d) failed: %s",
                      std::string(VLMTypeName(_attr.vlm_type)).c_str(),
                      (int)_attr.vlm_type,
                      verr.c_str());
                return false;
            }
        }

        if (!this->_attr.post_config_path.empty())
        {
            if (!postprocess.load_config(this->_attr.post_config_path))
            {
                ALOGW("load postprocess config(%s) failed", this->_attr.post_config_path.c_str());
            }
        }
        postprocess.set_pad_token_id(_attr.pad_token_id);

        init_kv_slots();

        ALOGI("LLM init ok");
        return true;
    }

    void Deinit()
    {
        if (deinited_) return;
        deinited_ = true;
        for (size_t i = 0; i < llama_layers.size(); i++) llama_layers[i].layer.deinit();
        llama_post.deinit();
        embed_selector.Deinit();
        gemma4_per_layer_helper.Deinit();
        if (vision) vision->Deinit();
        mem_guard_.CheckCmmBalance("Deinit"); // automated teardown-balance self-check
#ifdef USE_AXCL
        for (auto &devid : _attr.dev_ids) axcl_Exit(devid);
#endif
    }

    void Stop() { b_stop.store(true, std::memory_order_relaxed); }

    bool EmbedTokens(const std::vector<int> &token_ids, std::vector<float> &out_embedding)
    {
        b_stop.store(false, std::memory_order_relaxed);

        if (token_ids.empty())
        {
            out_embedding.clear();
            return true;
        }

        // Debug: print token ids for alignment checks (opt-in via env var).
        // Useful when aligning embedding outputs against Python reference scripts.
        if (std::getenv("AXLLM_DEBUG_EMBED_TOKEN_IDS"))
        {
            std::string s;
            s.reserve(token_ids.size() * 8);
            for (size_t i = 0; i < token_ids.size(); ++i)
            {
                if (i) s.push_back(',');
                s += std::to_string(token_ids[i]);
            }
            ALOGI("EmbedTokens token_ids(len=%zu): %s", token_ids.size(), s.c_str());
        }

        const int input_embed_num = (int)token_ids.size();
        if (_attr.prefill_token_num <= 0 || _attr.tokens_embed_size <= 0 || _attr.kv_cache_size <= 0)
        {
            ALOGE("LLM embedding not initialized correctly (prefill_token_num/embed_size/kv_cache_size)");
            return false;
        }

        const int prefill_split_num = (int)std::ceil((double)input_embed_num / (double)_attr.prefill_token_num);

        // Stateless embedding: clear all KV caches once to avoid group-specific assumptions (e.g. gid=1 may have history_cap=0).
        clear_all_group_kv_cache_tensors();

        std::vector<int> prefill_grp_list;
        prefill_grp_list.resize(prefill_split_num, -1);
        for (int p = 0; p < prefill_split_num; ++p)
        {
            const int history_len = p * _attr.prefill_token_num;
            const int chunk_tokens = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;
            const bool prefer_symbolic_group = has_vision_state && history_len > 0;
            const int gid = select_prefill_group(history_len, chunk_tokens, prefer_symbolic_group);
            if (gid < 0)
            {
                ALOGE("failed to select prefill group for embedding: history_len=%d chunk_tokens=%d", history_len, chunk_tokens);
                return false;
            }
            prefill_grp_list[p] = gid;
        }

        std::vector<unsigned short> all_embed((size_t)input_embed_num * (size_t)_attr.tokens_embed_size);
        for (int i = 0; i < input_embed_num; i++)
        {
            if (has_vision_state &&
                (size_t)i < vision_state.pos2vision.size() &&
                vision_state.pos2vision[(size_t)i] >= 0 &&
                !vision_state.vision_embed.empty())
            {
                const int vidx = vision_state.pos2vision[(size_t)i];
                const size_t src_off = (size_t)vidx * (size_t)_attr.tokens_embed_size;
                if (src_off + (size_t)_attr.tokens_embed_size <= vision_state.vision_embed.size())
                {
                    std::memcpy(all_embed.data() + (size_t)i * (size_t)_attr.tokens_embed_size,
                                vision_state.vision_embed.data() + src_off,
                                (size_t)_attr.tokens_embed_size * sizeof(unsigned short));
                    continue;
                }
            }
            embed_selector.getByIndex((unsigned int)token_ids[(size_t)i], all_embed.data() + (size_t)i * (size_t)_attr.tokens_embed_size);
        }

        std::vector<unsigned short> last_hidden((size_t)_attr.tokens_embed_size, 0);
        std::vector<unsigned short> embed_tmp((size_t)_attr.prefill_token_num * (size_t)_attr.tokens_embed_size, 0);
        std::vector<unsigned short> mask_tmp;
        std::vector<unsigned short> linear_mask_tmp;
        bfloat16 bf16_one = 1.0f;

        for (int p = 0; p < prefill_split_num; p++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;

            const int history_len = p * _attr.prefill_token_num;
            const int input_num_token = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;

            const int prefill_grpid = prefill_grp_list[p];
            int kv_cache_num = prefill_history_capacity_by_gid(prefill_grpid);
            const int kv_from_mask = prefill_history_capacity_by_mask(prefill_grpid);
            if (kv_from_mask >= 0) kv_cache_num = kv_from_mask;
            if (kv_cache_num < 0)
            {
                ALOGE("invalid kv_cache_num=%d for prefill_grpid=%d", kv_cache_num, prefill_grpid);
                return false;
            }

            mask_tmp.assign((size_t)_attr.prefill_token_num * (size_t)(kv_cache_num + _attr.prefill_token_num), bfloat16(-65536.f).data);
            build_prefill_mask(mask_tmp, kv_cache_num, _attr.prefill_token_num, history_len, input_num_token);

            std::fill(embed_tmp.begin(), embed_tmp.end(), 0);
            const size_t copy_tokens = (size_t)input_num_token;
            std::memcpy(embed_tmp.data(),
                        all_embed.data() + (size_t)history_len * (size_t)_attr.tokens_embed_size,
                        copy_tokens * (size_t)_attr.tokens_embed_size * sizeof(unsigned short));

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break;
                auto &lyr = llama_layers[m];
                const int devid = LLM_DEVID(lyr);
                const int layer_prefill_grpid = prefill_gid_for_layer(m, prefill_grpid);

                // indices
                const auto &t_idx = lyr.layer.get_input(layer_prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr;
                std::memset(idx_ptr, 0, (size_t)t_idx.nSize);
                {
                    const int start_pos = history_len;
                    const int idx_elems = (int)(t_idx.nSize / (int)sizeof(unsigned int));
                    int idx_rows = _attr.prefill_token_num > 0 ? (idx_elems / _attr.prefill_token_num) : 1;
                    if (idx_rows <= 0) idx_rows = 1;

                    const bool use_pos_ids = has_vision_state &&
                                             idx_rows >= 3 &&
                                             vision_state.position_ids.size() >= 3 &&
                                             !vision_state.position_ids.empty();

                    // Some models (e.g. Qwen-VL mRoPE) use multi-row indices. For multimodal embedding,
                    // use vision_state.position_ids when available; otherwise fill sequential positions.
                    for (int r = 0; r < idx_rows; ++r)
                    {
                        for (int j = 0; j < input_num_token; ++j)
                        {
                            unsigned int v = (unsigned int)(start_pos + j);
                            if (use_pos_ids)
                            {
                                if ((size_t)r < vision_state.position_ids.size())
                                {
                                    const auto &row = vision_state.position_ids[(size_t)r];
                                    const int abs_pos = start_pos + j;
                                    if ((size_t)abs_pos < row.size())
                                    {
                                        v = (unsigned int)row[(size_t)abs_pos];
                                    }
                                }
                            }
                            idx_ptr[(size_t)r * (size_t)_attr.prefill_token_num + (size_t)j] = v;
                        }
                    }
                }
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, (size_t)t_idx.nSize, devid);

                // mask
                const auto &t_mask = lyr.layer.get_input(layer_prefill_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    fill_linear_prefill_mask(linear_mask_tmp, elems, input_num_token);
                    llm_h2d(LLM_WADDR(t_mask), linear_mask_tmp.data(), linear_mask_tmp.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
                }

                // input
                const auto &t_in = lyr.layer.get_input(layer_prefill_grpid, "input");
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);

                // inference
                if (!ensure_layer_loaded(m)) return false;
                lyr.layer.inference(layer_prefill_grpid);

                // KV cache update
                const auto &in_k  = lyr.layer.get_input(layer_prefill_grpid, "K_cache");
                const auto &in_v  = lyr.layer.get_input(layer_prefill_grpid, "V_cache");
                if (is_linear_layer(m))
                {
                    (void)in_k;
                    (void)in_v;
                    sync_linear_input_state_from_group(m, lyr.layer, layer_prefill_grpid, devid, true);
                }
                else
                {
                    const auto &out_k = lyr.layer.get_output(layer_prefill_grpid, "K_cache_out");
                    const auto &out_v = lyr.layer.get_output(layer_prefill_grpid, "V_cache_out");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    const size_t layer_kv_off = (size_t)history_len * (size_t)layer_kv;
                    const size_t layer_kv_sz = (size_t)input_num_token * (size_t)layer_kv * sizeof(unsigned short);
                    llm_d2d((unsigned short *)LLM_WADDR(in_k) + layer_kv_off, LLM_RADDR(out_k), layer_kv_sz, devid);
                    llm_d2d((unsigned short *)LLM_WADDR(in_v) + layer_kv_off, LLM_RADDR(out_v), layer_kv_sz, devid);
                }

                // output -> embed_tmp for next layer
                const auto &t_out = lyr.layer.get_output(layer_prefill_grpid, "output");
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), std::min((size_t)t_out.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);

                // Qwen3-VL deepstack injection (vision tokens only).
                if (has_vision_state &&
                    !vision_state.deepstack_features.empty() &&
                    (size_t)m < vision_state.deepstack_features.size() &&
                    !vision_state.deepstack_features[m].empty())
                {
                    const int start_pos = history_len;
                    const int emb_sz = _attr.tokens_embed_size;
                    const auto &feat = vision_state.deepstack_features[m];
                    for (int j = 0; j < input_num_token; ++j)
                    {
                        const int abs_pos = start_pos + j;
                        if ((size_t)abs_pos >= vision_state.pos2vision.size()) continue;
                        const int vidx = vision_state.pos2vision[(size_t)abs_pos];
                        if (vidx < 0) continue;

                        const float *fv = feat.data() + (size_t)vidx * (size_t)emb_sz;
                        unsigned short *ev = embed_tmp.data() + (size_t)j * (size_t)emb_sz;
                        for (int di = 0; di < emb_sz; ++di)
                        {
                            unsigned int tmp_bf16 = ((unsigned int)ev[di]) << 16;
                            float fp32 = *reinterpret_cast<float *>(&tmp_bf16);
                            ev[di] = fp32_to_bfloat16_rne(fp32 + fv[di]);
                        }
                    }
                }
            }

            if (p == prefill_split_num - 1)
            {
                const int local_last = input_embed_num - history_len - 1;
                if (local_last >= 0 && local_last < _attr.prefill_token_num)
                {
                    std::memcpy(last_hidden.data(),
                                embed_tmp.data() + (size_t)local_last * (size_t)_attr.tokens_embed_size,
                                (size_t)_attr.tokens_embed_size * sizeof(unsigned short));
                }
            }
        }

        // Prefer post-process normalized output when available (e.g. Qwen3-VL-Embedding post exposes `output_norm`).
        // Otherwise fall back to the last hidden state directly.
        std::vector<unsigned short> embed_bf16 = last_hidden;

        const ax_runner_tensor_t *t_post_out = nullptr;
        const ax_runner_tensor_t *t_out_norm = try_get_output_tensor(llama_post, "output_norm");
        if (t_out_norm && (size_t)t_out_norm->nSize == (size_t)_attr.tokens_embed_size * sizeof(unsigned short))
        {
            t_post_out = t_out_norm;
        }
        else
        {
            const ax_runner_tensor_t *t_out = try_get_output_tensor(llama_post, "output");
            if (t_out && (size_t)t_out->nSize == (size_t)_attr.tokens_embed_size * sizeof(unsigned short))
            {
                t_post_out = t_out;
            }
        }

        if (t_post_out)
        {
            const auto &t_in = llama_post.get_input("input");
            llm_h2d(LLM_WADDR(t_in),
                    last_hidden.data(),
                    std::min((size_t)t_in.nSize, last_hidden.size() * sizeof(unsigned short)),
                    llama_post.get_devid());
            llama_post.inference();

            std::vector<unsigned short> post_out((size_t)_attr.tokens_embed_size);
            llm_d2h(post_out.data(),
                    LLM_RADDR(*t_post_out),
                    std::min((size_t)t_post_out->nSize, post_out.size() * sizeof(unsigned short)),
                    llama_post.get_devid());
            embed_bf16 = std::move(post_out);
        }

        out_embedding.resize((size_t)_attr.tokens_embed_size);
        for (int i = 0; i < _attr.tokens_embed_size; i++)
        {
            out_embedding[(size_t)i] = bfloat16(embed_bf16[(size_t)i]).fp32();
        }
        out_embedding = l2norm(std::move(out_embedding));
        return true;
    }

    bool EmbedHistory(const std::vector<Content> &history_in,
                      const std::vector<::MediaInputs> &media_inputs,
                      std::vector<float> &out_embedding)
    {
        if (!tokenizer)
        {
            ALOGE("LLM not initialized");
            return false;
        }

        has_vision_state = false;
        vision_state = {};

        std::vector<int> token_ids;
        if (vision && vision->enabled())
        {
            if (!media_inputs.empty())
            {
                std::vector<vision::MediaInputs> vmins;
                vmins.reserve(media_inputs.size());
                for (const auto &m : media_inputs) vmins.push_back({m.content_index, m.uris});

                std::vector<Content> prepared_history;
                vision::RunState st;
                std::string verr;
                if (!vision->Prepare(history_in, vmins, nullptr, prepared_history, token_ids, st, verr, false))
                {
                    ALOGE("vision.Prepare failed: %s", verr.c_str());
                    return false;
                }
                (void)prepared_history;
                vision_state = std::move(st);
                has_vision_state = true;
            }
            else
            {
                token_ids = tokenizer->encode(history_in);
            }
        }
        else
        {
            token_ids = tokenizer->encode(history_in);
        }

        // Align with Python reference implementations for Qwen3/Qwen3-VL embedding models:
        // they include the final pad/end-of-text token (151643) as the pooled position.
        if (embedding_append_eos && embedding_eos_token_id >= 0)
        {
            token_ids.push_back(embedding_eos_token_id);
        }

        const bool ok = EmbedTokens(token_ids, out_embedding);

        has_vision_state = false;
        vision_state = {};

        return ok;
    }

    bool EmbedText(const std::string &text, std::vector<float> &out_embedding)
    {
        if (!tokenizer)
        {
            ALOGE("LLM not initialized");
            return false;
        }
        std::vector<int> token_ids = tokenizer->encode(text);
        if (embedding_append_eos && embedding_eos_token_id >= 0) token_ids.push_back(embedding_eos_token_id);

        if (_attr.max_token_len > 0 && (int)token_ids.size() > _attr.max_token_len)
        {
            token_ids.resize((size_t)_attr.max_token_len);
            if (embedding_append_eos && !token_ids.empty()) token_ids.back() = embedding_eos_token_id;
        }

        return EmbedTokens(token_ids, out_embedding);
    }

    bool EmbedBatch(const std::vector<std::string> &inputs, std::vector<std::vector<float>> &out_embeddings)
    {
        out_embeddings.clear();
        out_embeddings.reserve(inputs.size());
        for (const auto &s : inputs)
        {
            std::vector<float> e;
            if (!EmbedText(s, e)) return false;
            out_embeddings.push_back(std::move(e));
        }
        return true;
    }

    int GenerateKVCachePrefill(std::vector<int> &_token_ids,
                               std::vector<std::vector<unsigned short>> &k_caches,
                               std::vector<std::vector<unsigned short>> &v_caches,
                               int &prefill_precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        int input_embed_num = (int)_token_ids.size();
        prefill_precompute_len = input_embed_num;
        k_caches.resize(_attr.axmodel_num);
        v_caches.resize(_attr.axmodel_num);

        int prefill_split_num = (int)ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);

        if (input_embed_num == 0)
        {
            for (int i = 0; i < _attr.axmodel_num; i++)
            {
                const int layer_kv = kv_cache_size_for_layer(i);
                k_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
                v_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
            }
            ALOGI("input token num is 0, skip");
            return 0;
        }

        std::vector<int> prefill_grp_list;
        prefill_grp_list.resize(prefill_split_num, -1);
        int max_prefill_gid = -1;
        for (int p = 0; p < prefill_split_num; ++p)
        {
            const int history_len = p * _attr.prefill_token_num;
            const int chunk_tokens = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;
            const int gid = select_prefill_group(history_len, chunk_tokens, false);
            if (gid < 0)
            {
                ALOGE("failed to select prefill group for KV prefill: history_len=%d chunk_tokens=%d", history_len, chunk_tokens);
                return -1;
            }
            prefill_grp_list[p] = gid;
            if (gid > max_prefill_gid) max_prefill_gid = gid;
        }

        const int prefill_decode_grpid = choose_decode_gid(std::max(1, input_embed_num));
        decode_grpid = prefill_decode_grpid;
        clear_all_group_kv_cache_tensors();

        std::vector<unsigned short> test_embed(_token_ids.size() * _attr.tokens_embed_size);
        for (size_t i = 0; i < _token_ids.size(); i++) embed_selector.getByIndex(_token_ids[i], test_embed.data() + i * _attr.tokens_embed_size);
        std::vector<unsigned short> linear_mask_tmp;
        bfloat16 bf16_one = 1.0f;

        for (int p = 0; p < prefill_split_num; p++)
        {
            int input_num_token = (p == prefill_split_num - 1) ? input_embed_num - p * _attr.prefill_token_num : _attr.prefill_token_num;
            const int history_len = p * _attr.prefill_token_num;
            const int prefill_grpid = prefill_grp_list[p];
            int kv_cache_num = prefill_history_capacity_by_gid(prefill_grpid);
            const int kv_from_mask = prefill_history_capacity_by_mask(prefill_grpid);
            if (kv_from_mask >= 0) kv_cache_num = kv_from_mask;
            if (kv_cache_num < 0)
            {
                ALOGE("invalid kv_cache_num=%d for prefill_grpid=%d", kv_cache_num, prefill_grpid);
                return -1;
            }
            std::vector<unsigned short> mask_tmp(_attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            size_t copy_tokens = (p == prefill_split_num - 1) ? (size_t)(input_embed_num - p * _attr.prefill_token_num) : (size_t)_attr.prefill_token_num;
            memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                auto &lyr = llama_layers[m]; int devid = LLM_DEVID(lyr);
                const int layer_prefill_grpid = prefill_gid_for_layer(m, prefill_grpid);
                auto &t_idx = lyr.layer.get_input(layer_prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr; memset(idx_ptr, 0, t_idx.nSize);
                int idx_i = 0; for (int i = 0; i < input_num_token; ++i) idx_ptr[idx_i++] = (unsigned int)(p * _attr.prefill_token_num + i);
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);
                auto &t_mask = lyr.layer.get_input(layer_prefill_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    fill_linear_prefill_mask(linear_mask_tmp, elems, input_num_token);
                    llm_h2d(LLM_WADDR(t_mask), linear_mask_tmp.data(), linear_mask_tmp.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    build_layer_prefill_mask(mask_tmp, kv_cache_num, _attr.prefill_token_num, history_len, input_num_token, m);
                    llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
                }
                auto &t_in = lyr.layer.get_input(layer_prefill_grpid, "input"); llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);
                if (!ensure_layer_loaded(m)) return -1;
                lyr.layer.inference(layer_prefill_grpid);
                const int layer_decode_grpid = decode_gid_for_layer(m, prefill_decode_grpid);
                auto &dec_k  = lyr.layer.get_input(layer_decode_grpid, "K_cache");
                auto &dec_v  = lyr.layer.get_input(layer_decode_grpid, "V_cache");
                const int layer_kv = kv_cache_size_for_layer(m);
                if (is_linear_layer(m))
                {
                    (void)dec_k;
                    (void)dec_v;
                    sync_linear_input_state_from_group(m, lyr.layer, layer_prefill_grpid, devid, true);
                }
                else
                {
                    auto &out_k  = lyr.layer.get_output(layer_prefill_grpid, "K_cache_out");
                    auto &out_v  = lyr.layer.get_output(layer_prefill_grpid, "V_cache_out");
                    const int kv_off = history_len * layer_kv;
                    const size_t kv_sz = (size_t)input_num_token * (size_t)layer_kv * sizeof(unsigned short);
                    llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                    llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);

                    const int ng = (int)lyr.layer.get_num_input_groups();
                    const int max_gid = std::min(max_prefill_gid, ng - 1);
                    for (int gid = layer_prefill_grpid + 1; gid <= max_gid; ++gid)
                    {
                        auto &gk = lyr.layer.get_input(gid, "K_cache");
                        auto &gv = lyr.layer.get_input(gid, "V_cache");
                        const int cap_tokens_k = (int)(gk.nSize / (size_t)(layer_kv * (int)sizeof(unsigned short)));
                        const int cap_tokens_v = (int)(gv.nSize / (size_t)(layer_kv * (int)sizeof(unsigned short)));
                        if (history_len + input_num_token <= cap_tokens_k)
                            llm_d2d((unsigned short *)LLM_WADDR(gk) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                        if (history_len + input_num_token <= cap_tokens_v)
                            llm_d2d((unsigned short *)LLM_WADDR(gv) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                    }
                }
                auto &t_out = lyr.layer.get_output(layer_prefill_grpid, "output"); llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), std::min((size_t)t_out.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);
            }
        }

        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i]; int devid = LLM_DEVID(lyr);
            const int layer_decode_grpid = decode_gid_for_layer(i, prefill_decode_grpid);
            auto &t_k = lyr.layer.get_input(layer_decode_grpid, "K_cache");
            auto &t_v = lyr.layer.get_input(layer_decode_grpid, "V_cache");
            if (is_linear_layer(i))
            {
                k_caches[i].resize((size_t)t_k.nSize / sizeof(unsigned short));
                v_caches[i].resize((size_t)t_v.nSize / sizeof(unsigned short));
                llm_d2h(k_caches[i].data(), LLM_RADDR(t_k), t_k.nSize, devid);
                llm_d2h(v_caches[i].data(), LLM_RADDR(t_v), t_v.nSize, devid);
            }
            else
            {
                const int layer_kv = kv_cache_size_for_layer(i);
                k_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
                v_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
                const size_t kv_bytes = (size_t)prefill_precompute_len * (size_t)layer_kv * sizeof(unsigned short);
                llm_d2h(k_caches[i].data(), LLM_RADDR(t_k), std::min(kv_bytes, (size_t)t_k.nSize), devid);
                llm_d2h(v_caches[i].data(), LLM_RADDR(t_v), std::min(kv_bytes, (size_t)t_v.nSize), devid);
            }
        }
        return 0;
    }

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

    // ---- Multi-slot prefix KV cache management ----
    // Called from Init() after layers/groups are ready. Estimates how many slots
    // actually fit before allocating, reduces to what fits, and warns when the
    // configured count cannot be satisfied. Safe no-op when slots<=1.
    void init_kv_slots()
    {
        multi_slot_enabled_ = false;
        multi_slot_active_request_ = false;
        kv_slots_.clear();
        kv_active_slot_idx_ = 0;
        kv_slot_lru_counter_ = 0;

        const int want = _attr.kv_cache_slots;
        if (want <= 1) return;

        if (dynamic_layer_load_enabled())
        {
            ALOGW("kv_cache_slots=%d ignored: not supported together with dynamic_load_enable yet", want);
            return;
        }

        std::string loc = _attr.kv_cache_slot_location;
        std::transform(loc.begin(), loc.end(), loc.begin(), ::tolower);
        kv_slot_location_ = (loc == "host" || loc == "ddr") ? KvSlotLocation::Host : KvSlotLocation::Device;
        if (loc != "host" && loc != "ddr" && loc != "device")
            ALOGW("kv_cache_slot_location='%s' unknown; defaulting to 'device'", _attr.kv_cache_slot_location.c_str());

        const std::string vlm_tag = _attr.vlm_type == VLMType::None ? std::string("no") : std::string(VLMTypeName(_attr.vlm_type));
        int granted = want;

        if (kv_slot_location_ == KvSlotLocation::Device)
        {
            // 1) Build per-layer slot metadata and sum per-slot device bytes by device.
            std::map<int, size_t> per_dev_bytes;
            for (int i = 0; i < _attr.axmodel_num; ++i)
            {
                const size_t b = llama_layers[i].layer.kv_cache_slots_prepare();
                if (b == 0)
                {
                    ALOGE("layer %d has no slottable KV tensors; disabling multi-slot", i);
                    for (int j = 0; j <= i; ++j) llama_layers[j].layer.kv_cache_slots_set_count(1);
                    return;
                }
                per_dev_bytes[layer_devid_for(i)] += b;
            }

            // 2) Budget: keep a safety margin of free CMM for runtime allocations.
            //    Honor mem_guard_floor_mb when it asks for a larger reserve (never smaller).
            const long long floor_b = _attr.mem_guard_enable ? (long long)_attr.mem_guard_floor_mb * 1024 * 1024 : 0;
            const long long margin = std::max(256LL * 1024 * 1024, floor_b);
            for (const auto &kv : per_dev_bytes)
            {
                const int dev = kv.first;
                const long long per_slot = (long long)kv.second;
                const long long free_b = (long long)remaining_cmm_mb(dev) * 1024LL * 1024LL;
                const long long usable = free_b - margin;
                int fit = 1;
                if (per_slot > 0 && usable > 0) fit = 1 + (int)(usable / per_slot);
                ALOGI("kv slot budget: dev=%d per_slot=%lldMB free=%lldMB margin=256MB -> max_slots=%d",
                      dev, per_slot >> 20, free_b >> 20, fit);
                granted = std::min(granted, fit);
            }
            if (granted < 1) granted = 1;
            if (granted < want)
                ALOGW("⚠ kv_cache_slots=%d requested but device CMM only fits %d; reducing to %d", want, granted, granted);
            if (granted <= 1)
            {
                ALOGW("⚠ kv_cache_slots disabled: device CMM cannot fit even one extra slot");
                for (int i = 0; i < _attr.axmodel_num; ++i) llama_layers[i].layer.kv_cache_slots_set_count(1);
                return;
            }

            // 3) Allocate the budgeted count; take the min actually achieved
            //    (fragmentation may yield less than the estimate), then trim all
            //    layers to that common count.
            int achieved = granted;
            for (int i = 0; i < _attr.axmodel_num; ++i)
                achieved = std::min(achieved, llama_layers[i].layer.kv_cache_slots_alloc(granted));
            if (achieved < granted)
                ALOGW("⚠ kv_cache_slots: allocated %d of estimated %d (CMM fragmentation); using %d", achieved, granted, achieved);
            for (int i = 0; i < _attr.axmodel_num; ++i) llama_layers[i].layer.kv_cache_slots_set_count(achieved);
            granted = achieved;
            if (granted <= 1)
            {
                ALOGW("⚠ kv_cache_slots disabled after allocation (only 1 slot available)");
                return;
            }
        }
        else // Host mode: one device buffer + N host copies; bound by host RAM.
        {
            size_t per_slot = 0;
            const int gid0 = decode_grpids_.empty() ? 0 : decode_grpids_.back();
            for (int i = 0; i < _attr.axmodel_num; ++i)
            {
                const int gid = decode_gid_for_layer(i, gid0);
                per_slot += (size_t)llama_layers[i].layer.get_input(gid, "K_cache").nSize;
                per_slot += (size_t)llama_layers[i].layer.get_input(gid, "V_cache").nSize;
            }
            long long free_ram = 0;
#ifndef _WIN32
            struct sysinfo si;
            if (sysinfo(&si) == 0) free_ram = (long long)si.freeram * (long long)si.mem_unit;
#endif
            const long long host_floor_b = _attr.mem_guard_enable ? (long long)_attr.mem_guard_floor_mb * 1024 * 1024 : 0;
            const long long margin = std::max(512LL * 1024 * 1024, host_floor_b);
            int fit = want;
            if (free_ram > 0 && per_slot > 0)
            {
                const long long usable = free_ram - margin;
                fit = 1;
                if (usable > 0) fit = 1 + (int)(usable / (long long)per_slot);
            }
            ALOGI("kv slot budget (host): per_slot<=%zuMB free_ram=%lldMB -> max_slots=%d",
                  per_slot >> 20, free_ram >> 20, fit);
            granted = std::min(want, fit);
            if (granted < 1) granted = 1;
            if (granted < want)
                ALOGW("⚠ kv_cache_slots=%d requested but host RAM only fits ~%d; reducing to %d", want, granted, granted);
            if (granted <= 1)
            {
                ALOGW("⚠ kv_cache_slots disabled: host RAM cannot fit even one extra slot");
                return;
            }
        }

        kv_slots_.resize(granted);
        kv_active_slot_idx_ = 0;
        multi_slot_enabled_ = true;
        if (granted < want)
            ALOGW("multi-slot prefix KV cache: %d/%d slots enabled (location=%s vlm=%s)", granted, want,
                  kv_slot_location_ == KvSlotLocation::Host ? "host" : "device", vlm_tag.c_str());
        else
            ALOGI("multi-slot prefix KV cache enabled: slots=%d location=%s vlm=%s", granted,
                  kv_slot_location_ == KvSlotLocation::Host ? "host" : "device", vlm_tag.c_str());
    }

    // Host mode: copy the active slot's device KV into its host store.
    void host_dump_active_kv()
    {
        if (kv_active_slot_idx_ < 0 || kv_active_slot_idx_ >= (int)kv_slots_.size()) return;
        auto &s = kv_slots_[(size_t)kv_active_slot_idx_];
        s.host_k.assign(_attr.axmodel_num, {});
        s.host_v.assign(_attr.axmodel_num, {});
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
            s.host_k[(size_t)m].resize(k_elems);
            s.host_v[(size_t)m].resize(v_elems);
            if (k_elems) llm_d2h(s.host_k[(size_t)m].data(), LLM_RADDR(t_k), std::min(k_elems * sizeof(unsigned short), (size_t)t_k.nSize), devid);
            if (v_elems) llm_d2h(s.host_v[(size_t)m].data(), LLM_RADDR(t_v), std::min(v_elems * sizeof(unsigned short), (size_t)t_v.nSize), devid);
        }
    }

    // Host mode: load a slot's host KV into the (shared) device decode-group buffer.
    void host_load_kv(int idx)
    {
        auto &s = kv_slots_[(size_t)idx];
        if ((int)s.host_k.size() != _attr.axmodel_num) return; // fresh slot, nothing to load
        for (int m = 0; m < _attr.axmodel_num; ++m)
        {
            auto &lyr = llama_layers[(size_t)m];
            const int devid = LLM_DEVID(lyr);
            const int gid = decode_gid_for_layer(m, decode_grpid);
            auto &t_k = lyr.layer.get_input(gid, "K_cache");
            auto &t_v = lyr.layer.get_input(gid, "V_cache");
            if (!s.host_k[(size_t)m].empty())
                llm_h2d(LLM_WADDR(t_k), s.host_k[(size_t)m].data(), std::min(s.host_k[(size_t)m].size() * sizeof(unsigned short), (size_t)t_k.nSize), devid);
            if (!s.host_v[(size_t)m].empty())
                llm_h2d(LLM_WADDR(t_v), s.host_v[(size_t)m].data(), std::min(s.host_v[(size_t)m].size() * sizeof(unsigned short), (size_t)t_v.nSize), devid);
        }
    }

    void save_active_kv_slot()
    {
        if (!multi_slot_enabled_) return;
        if (kv_active_slot_idx_ < 0 || kv_active_slot_idx_ >= (int)kv_slots_.size()) return;
        auto &s = kv_slots_[(size_t)kv_active_slot_idx_];
        s.last_history_snapshot = last_history_snapshot;
        s.last_tokens_ids = last_tokens_ids;
        s.precompute_len = precompute_len;
        s.linear_state_snapshots = linear_state_snapshots_;
        s.cached_mrope_next_pos = cached_mrope_next_pos;
        s.full_cache_valid_slots = full_cache_valid_slots_;
        s.full_cache_has_sparse_slots = full_cache_has_sparse_slots_;
        s.used = (precompute_len > 0) || !last_tokens_ids.empty();
        if (kv_slot_location_ == KvSlotLocation::Host && s.used)
            host_dump_active_kv();
    }

    bool activate_kv_slot(int idx)
    {
        if (!multi_slot_enabled_) return false;
        if (idx < 0 || idx >= (int)kv_slots_.size()) return false;
        if (kv_slot_location_ == KvSlotLocation::Device)
        {
            for (int i = 0; i < _attr.axmodel_num; ++i)
            {
                if (llama_layers[i].layer.kv_cache_slots_activate(idx) != 0)
                {
                    ALOGE("kv_cache_slots_activate(layer=%d slot=%d) failed", i, idx);
                    return false;
                }
            }
        }
        else
        {
            host_load_kv(idx); // copy this slot's host KV onto the single device buffer
        }
        auto &s = kv_slots_[(size_t)idx];
        last_history_snapshot = s.last_history_snapshot;
        last_tokens_ids = s.last_tokens_ids;
        precompute_len = s.precompute_len;
        linear_state_snapshots_ = s.linear_state_snapshots;
        cached_mrope_next_pos = s.cached_mrope_next_pos;
        full_cache_valid_slots_ = s.full_cache_valid_slots;
        full_cache_has_sparse_slots_ = s.full_cache_has_sparse_slots;
        kv_active_slot_idx_ = idx;
        return true;
    }

    // Pick the slot to serve `sel_tokens`. Choose the used slot sharing the
    // longest leading-token prefix (>= kSlotReuseMinPrefix): the existing
    // token-level reuse logic then keeps that prefix's KV and recomputes only the
    // divergent tail. With no usable match, a free slot (or the LRU victim) is
    // reset for a full prefill. Output correctness holds for any choice because a
    // matched prefix is, by construction, an identical token sequence.
    void select_kv_slot(const std::vector<int> &sel_tokens)
    {
        multi_slot_active_request_ = false;
        if (!multi_slot_enabled_) return;
        multi_slot_active_request_ = true;

        int best = -1, best_off = 0;
        uint64_t best_lru = 0;
        int free_slot = -1;
        int lru_victim = 0;
        uint64_t lru_min = 0;
        bool lru_init = false;
        for (int i = 0; i < (int)kv_slots_.size(); ++i)
        {
            const auto &s = kv_slots_[(size_t)i];
            if (!s.used)
            {
                if (free_slot < 0) free_slot = i;
                continue;
            }
            if (!lru_init || s.lru < lru_min) { lru_min = s.lru; lru_victim = i; lru_init = true; }
            int off = 0;
            (void)diff_token_ids(s.last_tokens_ids, sel_tokens, off);
            if (off > best_off || (off == best_off && s.lru > best_lru))
            {
                best_off = off;
                best = i;
                best_lru = s.lru;
            }
        }

        int chosen;
        bool fresh;
        if (best >= 0 && best_off >= kSlotReuseMinPrefix)
        {
            chosen = best;            // reuse shared prefix (continuation or same system prompt)
            fresh = false;
        }
        else if (free_slot >= 0)
        {
            chosen = free_slot;       // new conversation gets its own slot
            fresh = true;
        }
        else if (best >= 0 && best_off > 0)
        {
            chosen = best;            // slots full: still salvage whatever prefix overlaps
            fresh = false;
        }
        else
        {
            chosen = lru_victim;      // slots full, nothing shared: evict LRU
            fresh = true;
        }

        // Perf: if the whole request fits the cheap (history-less) prefill group, a
        // fresh prefill is faster than reuse via the model's wider history group
        // (which attends over its full compiled width regardless of real history).
        // Keep slot affinity (`chosen`) but re-prefill instead of reusing.
        if (!fresh && best >= 0)
        {
            const int cheap_cap = cheap_prefill_capacity();
            if (cheap_cap > 0 && (int)sel_tokens.size() <= cheap_cap)
                fresh = true;
        }
        commit_kv_slot_choice(chosen, fresh, best_off, (int)sel_tokens.size());
    }

    // VLM slot selection: tokenizing VLM history needs the (expensive) vision
    // Prepare, so match on the chat history (Content) prefix instead. The chosen
    // slot's token-level reuse logic still refines actual KV reuse afterwards.
    void select_kv_slot_by_history(const std::vector<Content> &history)
    {
        multi_slot_active_request_ = false;
        if (!multi_slot_enabled_) return;
        multi_slot_active_request_ = true;

        int best = -1, best_common = 0;
        uint64_t best_lru = 0;
        int free_slot = -1, lru_victim = 0;
        uint64_t lru_min = 0;
        bool lru_init = false;
        for (int i = 0; i < (int)kv_slots_.size(); ++i)
        {
            const auto &s = kv_slots_[(size_t)i];
            if (!s.used)
            {
                if (free_slot < 0) free_slot = i;
                continue;
            }
            if (!lru_init || s.lru < lru_min) { lru_min = s.lru; lru_victim = i; lru_init = true; }
            const auto &h = s.last_history_snapshot;
            const int n = (int)std::min(h.size(), history.size());
            int common = 0;
            while (common < n && same_history_content(h[(size_t)common], history[(size_t)common])) ++common;
            if (common > best_common || (common == best_common && s.lru > best_lru))
            {
                best_common = common;
                best = i;
                best_lru = s.lru;
            }
        }

        int chosen;
        bool fresh;
        if (best >= 0 && best_common >= 1)   // share at least the system turn
        {
            chosen = best;
            fresh = false;
        }
        else if (free_slot >= 0)
        {
            chosen = free_slot;
            fresh = true;
        }
        else if (best >= 0 && best_common > 0)
        {
            chosen = best;
            fresh = false;
        }
        else
        {
            chosen = lru_victim;
            fresh = true;
        }
        commit_kv_slot_choice(chosen, fresh, best_common, (int)history.size());
    }

    void commit_kv_slot_choice(int chosen, bool fresh, int shared, int req_size)
    {
        if (chosen != kv_active_slot_idx_)
        {
            save_active_kv_slot();
            activate_kv_slot(chosen);
        }
        if (fresh)
        {
            ResetKVCache();
            kv_slots_[(size_t)chosen].used = false;
            kv_slots_[(size_t)chosen].host_k.clear();
            kv_slots_[(size_t)chosen].host_v.clear();
        }
        kv_slots_[(size_t)chosen].lru = ++kv_slot_lru_counter_;
        ALOGI("kv slot select: chosen=%d reuse=%d shared=%d req=%d (slots=%zu)",
              chosen, fresh ? 0 : 1, fresh ? 0 : shared, req_size, kv_slots_.size());
    }

    std::string Run(std::vector<unsigned short> &test_embed, int output_max_token = -1)
    {
        using DecodeClock = std::chrono::steady_clock;
        struct DecodeProfileStats
        {
            uint64_t per_layer_ns = 0;
            uint64_t shared_kv_ns = 0;
            uint64_t prep_ns = 0;
            uint64_t inference_ns = 0;
            uint64_t outcopy_ns = 0;
            uint64_t post_ns = 0;
        };

        b_stop.store(false, std::memory_order_relaxed); std::string final_out;
        utf8_filter = UTF8Filter();
        ChannelSectionFilter channel_filter;
        channel_filter.reset();
        const bool hide_channel_markup = tokenizer_uses_hidden_channel_markup(_attr.tokenizer_type);
        const bool debug_prefill =
            std::getenv("AXLLM_DEBUG_PREFILL") ||
            std::getenv("AXLLM_DEBUG_LAYER0_IO") ||
            std::getenv("AXLLM_DEBUG_DECODE_TRACE");
        std::string streamed_visible_text;
        std::string streamed_raw_visible_text;
        auto emit_stream_chunk = [&](const std::string &chunk, float tps)
        {
            if (!_attr.runing_callback || chunk.empty())
                return;

            streamed_raw_visible_text += chunk;
            const std::string visible_text = sanitize_utf8_text(streamed_raw_visible_text);
            if (visible_text.size() <= streamed_visible_text.size())
                return;
            if (visible_text.compare(0, streamed_visible_text.size(), streamed_visible_text) != 0)
            {
                ALOGW("streamed visible text prefix mismatch, suppressing incremental emit");
                streamed_visible_text = visible_text;
                return;
            }

            const std::string delta = visible_text.substr(streamed_visible_text.size());
            if (!delta.empty())
            {
                streamed_visible_text = visible_text;
                _attr.runing_callback(delta, tps, _attr.reserve);
            }
        };
        auto dump_hidden_trace = [&](const char *tag, int step, const unsigned short *hidden, int n)
        {
            if (!std::getenv("AXLLM_DEBUG_DECODE_TRACE") || !tag || !hidden || n <= 0)
                return;
            float min_v = 0.0f, max_v = 0.0f, mean_abs = 0.0f;
            summarize_bf16_buffer(hidden, n, min_v, max_v, mean_abs);
            ALOGI("DTRACE step=%d %s n=%d hash=0x%llx min=%.6f max=%.6f mean_abs=%.6f",
                  step,
                  tag,
                  n,
                  (unsigned long long)hash_u16_buffer(hidden, n),
                  min_v,
                  max_v,
                  mean_abs);
        };
        auto dump_decode_trace = [&](int step, const unsigned short *logits, int n, int chosen_token)
        {
            if (!std::getenv("AXLLM_DEBUG_DECODE_TRACE") || !logits || n <= 0)
                return;
            const auto topk = topk_bfloat16(const_cast<unsigned short *>(logits), n, std::min(5, n));
            std::string topk_str = "[";
            for (size_t i = 0; i < topk.size(); ++i)
            {
                const int token = topk[i].first;
                const float val = topk[i].second;
                if (i > 0) topk_str += ",";
                topk_str += "(" + std::to_string(token) + ":" + std::to_string(val) + ")";
            }
            topk_str += "]";
            ALOGI("DTRACE step=%d post_logits n=%d hash=0x%llx topk=%s",
                  step,
                  n,
                  (unsigned long long)hash_u16_buffer(logits, n),
                  topk_str.c_str());
            ALOGI("DTRACE step=%d token=%d piece='%s'",
                  step,
                  chosen_token,
                  safe_decode_token(tokenizer, chosen_token).c_str());
        };
        bfloat16 bf16 = -65536.f;
        bfloat16 bf16_one = 1.0f;

        int max_decode_cap = _attr.max_token_len;
        if (!decode_max_token_len_grp_.empty()) max_decode_cap = std::max(max_decode_cap, decode_max_token_len_grp_.back());
        if (max_decode_cap <= 0) max_decode_cap = _attr.kv_cache_num;
        std::vector<unsigned short> decode_mask((size_t)max_decode_cap + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
        std::vector<int> token_ids;
        int input_embed_num  = (int)(test_embed.size() / _attr.tokens_embed_size);
        int prefill_split_num = (int)ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
        timer t_cost, ttft_timer, decode_timer; ttft_timer.start();
        bool decode_timer_started = false;

        std::vector<int> prefill_grp_list;
        const int default_prefill_gid = prefill_grpids_.empty() ? 1 : prefill_grpids_.front();
        prefill_grp_list.resize(prefill_split_num, default_prefill_gid);
        for (int p = 0; p < prefill_split_num; ++p) {
            const int history_len = precompute_len + p * _attr.prefill_token_num;
            const int chunk_tokens = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;
            const bool prefer_symbolic_group = has_vision_state && history_len > 0;
            int g = select_prefill_group(history_len, chunk_tokens, prefer_symbolic_group);
            if (g < 0)
            {
                ALOGE("failed to select prefill group for history_len=%d chunk_tokens=%d", history_len, chunk_tokens);
                return final_out;
            }
            prefill_grp_list[p] = g;
        }

        const bool use_per_layer_input = gemma4_per_layer_helper.enabled();
        const int per_layer_hidden = gemma4_per_layer_helper.hidden_size_per_layer_input();
        const bool decode_profile_enabled = std::getenv("AXLLM_PROFILE_DECODE") != nullptr;
        DecodeProfileStats decode_profile{};
        std::vector<unsigned short> prefill_per_layer_inputs;
        std::vector<unsigned short> decode_per_layer_input;
        std::vector<unsigned short> per_layer_tmp;

        if (use_per_layer_input)
        {
            if ((int)run_input_token_ids.size() != input_embed_num)
            {
                ALOGE("Gemma4 requires run_input_token_ids for prefill: ids=%zu embeds=%d",
                      run_input_token_ids.size(),
                      input_embed_num);
                return final_out;
            }

            std::vector<int> per_layer_token_ids = run_input_token_ids;
            std::vector<unsigned short> per_layer_embed = test_embed;
            scale_all_embeds_inplace(per_layer_embed.data(), input_embed_num);
            if (has_vision_state && !vision_state.pos2vision.empty())
            {
                for (int i = 0; i < input_embed_num; ++i)
                {
                    const int abs_pos = (active_token_pos_start >= 0) ? (active_token_pos_start + i) : (precompute_len + i);
                    if ((size_t)abs_pos < vision_state.pos2vision.size() && vision_state.pos2vision[(size_t)abs_pos] >= 0)
                    {
                        per_layer_token_ids[(size_t)i] = gemma4_per_layer_helper.pad_token_id();
                    }
                }
            }

            if (!gemma4_per_layer_helper.Compute(per_layer_token_ids,
                                                 per_layer_embed.data(),
                                                 input_embed_num,
                                                 _attr.tokens_embed_size,
                                                 prefill_per_layer_inputs))
            {
                ALOGE("Gemma4 prefill per-layer input compute failed");
                return final_out;
            }
        }

        for (int p = 0; p < prefill_split_num; p++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;
            int input_num_token = (p == prefill_split_num - 1) ? input_embed_num - p * _attr.prefill_token_num : _attr.prefill_token_num;
            const int history_len = precompute_len + p * _attr.prefill_token_num;
            const int token_start_pos = (active_token_pos_start >= 0)
                                            ? active_token_pos_start + p * _attr.prefill_token_num
                                            : history_len;
            const int rope_start_pos = (active_prefill_pos_start >= 0)
                                           ? active_prefill_pos_start + p * _attr.prefill_token_num
                                           : history_len;

            const int prefill_grpid = prefill_grp_list[p];
            // ALOGI("prefill group for chunk %d: %d", p, prefill_grpid);
            int kv_cache_num_p = prefill_history_capacity_for_layer_group(cache_ref_full_layer_idx, prefill_grpid);
            if (kv_cache_num_p < 0) kv_cache_num_p = prefill_history_capacity_by_gid(prefill_grpid);
            ALOGI("prefill chunk p=%d history_len=%d grpid=%d kv_cache_num=%d input_tokens=%d",
                  p, history_len, prefill_grpid, kv_cache_num_p, input_num_token);

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            std::vector<unsigned short> mask_tmp;
            std::vector<unsigned short> linear_mask_tmp;
            if (use_per_layer_input) per_layer_tmp.assign((size_t)_attr.prefill_token_num * (size_t)per_layer_hidden, 0);

            size_t copy_tokens = (p == prefill_split_num - 1) ? (size_t)(input_embed_num - p * _attr.prefill_token_num) : (size_t)_attr.prefill_token_num;
            memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));
            scale_prefill_text_embeds_inplace(embed_tmp.data(), input_num_token, token_start_pos);
            if (std::getenv("AXLLM_DEBUG_DECODE_TRACE") && p == 0 && input_num_token > 0)
            {
                const unsigned short *last_token_embed =
                    embed_tmp.data() + (size_t)(input_num_token - 1) * (size_t)_attr.tokens_embed_size;
                dump_hidden_trace("prefill_input_last_embed", -1, last_token_embed, _attr.tokens_embed_size);
            }

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break;
                auto &lyr   = llama_layers[m]; int devid = LLM_DEVID(lyr);
                const int layer_prefill_grpid = prefill_gid_for_layer(m, prefill_grpid);
                int layer_kv_cache_num = prefill_history_capacity_for_layer_group(m, layer_prefill_grpid);
                if (layer_kv_cache_num < 0) layer_kv_cache_num = kv_cache_num_p;
                if (layer_kv_cache_num < history_len && !is_linear_layer(m))
                {
                    ALOGE("prefill layer %d group %d history_len=%d exceeds layer history cap=%d",
                          m,
                          layer_prefill_grpid,
                          history_len,
                          layer_kv_cache_num);
                    return final_out;
                }
                if (layer_prefill_grpid != prefill_grpid)
                {
                    ALOGD("prefill layer group remap: layer=%d global_gid=%d layer_gid=%d history_cap=%d",
                          m,
                          prefill_grpid,
                          layer_prefill_grpid,
                          layer_kv_cache_num);
                }
                auto &t_idx = lyr.layer.get_input(layer_prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr; memset(idx_ptr, 0, t_idx.nSize);
                {
                    const int seq_start_pos = rope_start_pos;
                    const int idx_elems = (int)(t_idx.nSize / (int)sizeof(unsigned int));
                    int idx_rows = idx_elems / _attr.prefill_token_num;
                    if (idx_rows <= 0) idx_rows = 1;
                    const bool use_pos_ids = has_vision_state &&
                                             idx_rows >= 3 &&
                                             vision_state.position_ids.size() >= 3 &&
                                             !vision_state.position_ids.empty();

                    for (int r = 0; r < idx_rows; ++r)
                    {
                        // ALOGI("r %d, idx_rows: %d", r, idx_rows);
                        for (int j = 0; j < input_num_token; ++j)
                        {
                            unsigned int v = (unsigned int)(seq_start_pos + j);
                            if (use_pos_ids)
                            {
                                if ((size_t)r < vision_state.position_ids.size())
                                {
                                    const auto &row = vision_state.position_ids[r];
                                    if ((size_t)(token_start_pos + j) < row.size())
                                        v = (unsigned int)row[token_start_pos + j];
                                }
                            }
                            idx_ptr[r * _attr.prefill_token_num + j] = v;
                        }
                    }
                }
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);
                auto &t_mask = lyr.layer.get_input(layer_prefill_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    fill_linear_prefill_mask(linear_mask_tmp, elems, input_num_token);
                    llm_h2d(LLM_WADDR(t_mask), linear_mask_tmp.data(), linear_mask_tmp.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    mask_tmp.assign((size_t)_attr.prefill_token_num * (size_t)(layer_kv_cache_num + _attr.prefill_token_num), bf16.data);
                    if (use_sparse_full_cache_mask())
                        build_sparse_layer_prefill_mask(mask_tmp, layer_kv_cache_num, _attr.prefill_token_num, history_len, input_num_token, m);
                    else
                        build_layer_prefill_mask(mask_tmp, layer_kv_cache_num, _attr.prefill_token_num, history_len, input_num_token, m);
                    llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
                }
                auto &t_in = lyr.layer.get_input(layer_prefill_grpid, "input");
                if (std::getenv("AXLLM_ZERO_LAYER0_PREFILL_INPUT") && m == 0 && p == 0)
                {
                    std::vector<unsigned short> zero_embed(embed_tmp.size(), 0);
                    llm_h2d(LLM_WADDR(t_in), zero_embed.data(), zero_embed.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), devid);
                }
                const int shared_src = shared_kv_source_for_layer(m);
                if (shared_src >= 0)
                {
                    auto &src_layer = llama_layers[(size_t)shared_src];
                    const int src_decode_grpid = decode_gid_for_layer(shared_src, decode_grpid);
                    auto &src_k = src_layer.layer.get_input(src_decode_grpid, "K_cache");
                    auto &src_v = src_layer.layer.get_input(src_decode_grpid, "V_cache");
                    auto &dst_k = lyr.layer.get_input(layer_prefill_grpid, "K_cache");
                    auto &dst_v = lyr.layer.get_input(layer_prefill_grpid, "V_cache");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    copy_shared_prefill_cache(dst_k,
                                              src_k,
                                              layer_kv,
                                              history_len,
                                              layer_kv_cache_num,
                                              history_len,
                                              input_num_token,
                                              devid);
                    copy_shared_prefill_cache(dst_v,
                                              src_v,
                                              layer_kv,
                                              history_len,
                                              layer_kv_cache_num,
                                              history_len,
                                              input_num_token,
                                              devid);
                }
                if (use_per_layer_input)
                {
                    const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, layer_prefill_grpid, "per_layer_input");
                    if (!t_per_layer)
                    {
                        ALOGE("Gemma4 decoder layer %d is missing per_layer_input for group %d", m, layer_prefill_grpid);
                        return final_out;
                    }
                    std::fill(per_layer_tmp.begin(), per_layer_tmp.end(), 0);
                    for (int j = 0; j < input_num_token; ++j)
                    {
                        const size_t src_off = (((size_t)p * (size_t)_attr.prefill_token_num + (size_t)j) * (size_t)_attr.axmodel_num + (size_t)m) * (size_t)per_layer_hidden;
                        const size_t dst_off = (size_t)j * (size_t)per_layer_hidden;
                        std::memcpy(per_layer_tmp.data() + dst_off,
                                    prefill_per_layer_inputs.data() + src_off,
                                    (size_t)per_layer_hidden * sizeof(unsigned short));
                    }
                    llm_h2d(LLM_WADDR(*t_per_layer),
                            per_layer_tmp.data(),
                            std::min((size_t)t_per_layer->nSize, per_layer_tmp.size() * sizeof(unsigned short)),
                            devid);
                }
                if (!ensure_layer_loaded(m)) return final_out;
                if (debug_prefill)
                {
                    ALOGI("prefill layer %d begin: gid=%d devid=%d input_tokens=%d history_len=%d layer_kv_cache_num=%d",
                          m,
                          layer_prefill_grpid,
                          devid,
                          input_num_token,
                          history_len,
                          layer_kv_cache_num);
                }
                if (m == 0 && p == 0)
                    dump_selected_prefill_tensors(lyr.layer, layer_prefill_grpid, -1);
                const bool linear_prefill = is_linear_layer(m);
                const int layer_decode_grpid = decode_gid_for_layer(m, decode_grpid);
                const int infer_ret = lyr.layer.inference(layer_prefill_grpid);
                if (infer_ret != 0)
                {
                    ALOGW("prefill layer %d gid=%d inference failed: 0x%x", m, layer_prefill_grpid, infer_ret);
                    return final_out;
                }
                if (debug_prefill) ALOGI("prefill layer %d inference done", m);
                if (std::getenv("AXLLM_DEBUG_LAYER0_IO") && m == 0 && p == 0)
                {
                    try
                    {
                        const auto &t_out0 = lyr.layer.get_output(layer_prefill_grpid, "output");
                        const int n = (int)(t_out0.nSize / sizeof(unsigned short));
                        if (n > 0 && t_out0.pVirAddr)
                        {
                            float min_v = 0.0f, max_v = 0.0f, mean_abs = 0.0f;
                            summarize_bf16_buffer((const unsigned short *)t_out0.pVirAddr, n, min_v, max_v, mean_abs);
                            ALOGI("DBGIO step=0 gid=%d output hash=0x%llx min=%.6f max=%.6f mean_abs=%.6f",
                                  layer_prefill_grpid,
                                  (unsigned long long)hash_u16_buffer((const unsigned short *)t_out0.pVirAddr, n),
                                  min_v,
                                  max_v,
                                  mean_abs);
                        }
                    }
                    catch (const std::exception &)
                    {
                    }
                }
                auto &dec_k = lyr.layer.get_input(layer_decode_grpid, "K_cache");
                auto &dec_v = lyr.layer.get_input(layer_decode_grpid, "V_cache");
                if (linear_prefill)
                {
                    (void)dec_k;
                    (void)dec_v;
                    sync_linear_input_state_from_group(m, lyr.layer, layer_prefill_grpid, devid, true);
                }
                else
                {
                    auto &out_k = lyr.layer.get_output(layer_prefill_grpid, "K_cache_out");
                    auto &out_v = lyr.layer.get_output(layer_prefill_grpid, "V_cache_out");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    int kv_off = history_len * layer_kv;
                    size_t kv_sz = (size_t)input_num_token * (size_t)layer_kv * sizeof(unsigned short);
                    llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                    llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                    std::vector<int> future_prefill_gids;
                    future_prefill_gids.reserve((size_t)std::max(0, prefill_split_num - p - 1));
                    for (int q = p + 1; q < prefill_split_num; ++q)
                    {
                        const int future_gid = prefill_gid_for_layer(m, prefill_grp_list[(size_t)q]);
                        if (std::find(future_prefill_gids.begin(), future_prefill_gids.end(), future_gid) == future_prefill_gids.end())
                            future_prefill_gids.push_back(future_gid);
                    }
                    for (const int gid : future_prefill_gids) {
                        auto &gk = lyr.layer.get_input(gid, "K_cache");
                        auto &gv = lyr.layer.get_input(gid, "V_cache");
                        const int cap_tokens_k = (int)(gk.nSize / (size_t)(layer_kv * (int)sizeof(unsigned short)));
                        const int cap_tokens_v = (int)(gv.nSize / (size_t)(layer_kv * (int)sizeof(unsigned short)));
                        if (kv_off / layer_kv + input_num_token <= cap_tokens_k) {
                            llm_d2d((unsigned short *)LLM_WADDR(gk) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                        }
                        if (kv_off / layer_kv + input_num_token <= cap_tokens_v) {
                            llm_d2d((unsigned short *)LLM_WADDR(gv) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                        }
                    }
                }

                auto &t_out = lyr.layer.get_output(layer_prefill_grpid, "output");
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), embed_tmp.size() * sizeof(unsigned short), devid);
                if (debug_prefill) ALOGI("prefill layer %d output d2h done", m);
                if (std::getenv("AXLLM_DEBUG_DECODE_TRACE") &&
                    p == prefill_split_num - 1 &&
                    input_num_token > 0 &&
                    (m == 0 || m == _attr.axmodel_num - 1))
                {
                    const unsigned short *last_token_hidden =
                        embed_tmp.data() + (size_t)(input_num_token - 1) * (size_t)_attr.tokens_embed_size;
                    dump_hidden_trace(m == 0 ? "prefill_layer0_last_hidden" : "prefill_last_layer_hidden",
                                      m,
                                      last_token_hidden,
                                      _attr.tokens_embed_size);
                }

                if (has_vision_state &&
                    !vision_state.deepstack_features.empty() &&
                    (size_t)m < vision_state.deepstack_features.size() &&
                    !vision_state.deepstack_features[m].empty())
                {
                    const int start_pos = token_start_pos;
                    const int emb_sz = _attr.tokens_embed_size;
                    const auto &feat = vision_state.deepstack_features[m];

                    for (int j = 0; j < input_num_token; ++j)
                    {
                        const int abs_pos = start_pos + j;
                        if ((size_t)abs_pos >= vision_state.pos2vision.size()) continue;
                        const int vidx = vision_state.pos2vision[abs_pos];
                        if (vidx < 0) continue;

                        const float *fv = feat.data() + (size_t)vidx * (size_t)emb_sz;
                        unsigned short *ev = embed_tmp.data() + (size_t)j * (size_t)emb_sz;

                        for (int di = 0; di < emb_sz; ++di)
                        {
                            unsigned int tmp_bf16 = ((unsigned int)ev[di]) << 16;
                            float fp32 = *reinterpret_cast<float *>(&tmp_bf16);
                            ev[di] = fp32_to_bfloat16_rne(fp32 + fv[di]);
                        }
                    }
                }
            }

            mark_full_cache_slots(history_len, input_num_token);
            capture_linear_state_snapshot(history_len + input_num_token);
            if (p == prefill_split_num - 1)
                memcpy(embed.data(), embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size, _attr.tokens_embed_size * sizeof(unsigned short));
        }

        int next_token = -1; t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
        bool b_hit_eos = false;
        if (use_per_layer_input)
            gemma4_per_layer_helper.reset_decode_stats(decode_profile_enabled);
        int last_shared_sync_decode_grpid = -1;
        {
            auto &t_in = llama_post.get_input("input");
            if (debug_prefill) ALOGI("post stage begin: input_bytes=%u device=%d", t_in.nSize, LLM_DEVID(llama_layers.back()));
            dump_hidden_trace("post_input_hidden", 0, embed.data(), _attr.tokens_embed_size);
            llm_h2d(LLM_WADDR(t_in), embed.data(), embed.size() * sizeof(unsigned short), LLM_DEVID(llama_layers.back()));
            if (debug_prefill) ALOGI("post stage h2d done");
            llama_post.inference();
            if (debug_prefill) ALOGI("post stage inference done");
            auto &t_out = llama_post.get_output("output");
            llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
            if (debug_prefill) ALOGI("post stage d2h done: output_bytes=%u", t_out.nSize);
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
            next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            dump_decode_trace(0, post_out, _attr.tokens_embed_num, next_token);
            const float ttft_text_ms = ttft_timer.cost();
            const float ttft_e2e_ms = CurrentRequestElapsedMs();
            if (ttft_e2e_ms >= 0.0f)
            {
                const float ttft_prepare_ms = std::max(0.0f, ttft_e2e_ms - ttft_text_ms);
                ALOGI("ttft: %.2f ms (prepare=%.2f ms, text=%.2f ms)", ttft_e2e_ms, ttft_prepare_ms, ttft_text_ms);
                ALOGI("ttft_prepare: %.2f ms", ttft_prepare_ms);
                ALOGI("ttft_text: %.2f ms", ttft_text_ms);
            }
            else
            {
                ALOGI("ttft: %.2f ms", ttft_text_ms);
            }
            if (debug_prefill) ALOGI("first decode token: id=%d", next_token);
            if (next_token < 0 || next_token >= _attr.tokens_embed_num)
            {
                ALOGE("first decode token out of range: token=%d vocab=%d", next_token, _attr.tokens_embed_num);
                return final_out;
            }
            b_hit_eos = tokenizer->is_stop(next_token);
            if (b_hit_eos)
            {
                ALOGW("first decode token hit stop: token=%d piece='%s' precompute_len=%d input_tokens=%d",
                      next_token,
                      safe_decode_token(tokenizer, next_token).c_str(),
                      precompute_len,
                      input_embed_num);
            }
            if (!b_hit_eos)
            {
                token_ids.push_back(next_token);
                decode_timer.start();
                decode_timer_started = true;
                if (_attr.runing_callback)
                {
                    auto str = safe_decode_token(tokenizer, next_token);
                    if (hide_channel_markup) str = channel_filter.filter(str);
                    emit_stream_chunk(str, -1);
                }
                if (output_max_token > 0 && (int)token_ids.size() >= output_max_token)
                {
                    b_hit_eos = true;
                }
            }
        }

        t_cost.start();
        const unsigned int dense_decode_start = (unsigned int)(precompute_len + input_embed_num);
        unsigned int decode_start = dense_decode_start;
        if (has_vision_state && vision_state.decode_start > 0) decode_start = (unsigned int)vision_state.decode_start;
        else if (active_prefill_pos_start >= 0) decode_start = (unsigned int)(active_prefill_pos_start + input_embed_num);
        if (decode_start != dense_decode_start)
        {
            ALOGI("VLM decode positions: rope_start=%u dense_kv_start=%u", decode_start, dense_decode_start);
        }
        for (unsigned int decode_pos = decode_start, kv_slot = dense_decode_start;
             !b_hit_eos && decode_pos < (unsigned int)_attr.max_token_len && kv_slot < (unsigned int)_attr.max_token_len;
             ++decode_pos, ++kv_slot)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;
            bool need_full_shared_sync = false;
            {
                const int want_gid = choose_decode_gid((int)kv_slot + 1);
                if (want_gid != decode_grpid)
                {
                    // ALOGI("switch decode_grpid: %d -> %d (kv_ctx=%u rope_pos=%u)", decode_grpid, want_gid, kv_slot + 1, decode_pos);
                    sync_device_kv_cache_from_decode(decode_grpid, want_gid, (int)kv_slot, false);
                    decode_grpid = want_gid;
                }
                need_full_shared_sync = (decode_grpid != last_shared_sync_decode_grpid);
            }
            embed_selector.getByIndex(next_token, embed);
            scale_all_embeds_inplace(embed.data(), 1);
            if (use_per_layer_input)
            {
                const auto t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                if (!gemma4_per_layer_helper.ComputeSingle(next_token,
                                                           embed.data(),
                                                           _attr.tokens_embed_size,
                                                           decode_per_layer_input))
                {
                    ALOGE("Gemma4 decode per-layer input compute failed");
                    return final_out;
                }
                if (decode_profile_enabled)
                    decode_profile.per_layer_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - t0).count();
            }

#ifdef USE_AXCL
            {
                const int layer0_decode_grpid = decode_gid_for_layer(0, decode_grpid);
                auto &l0_in = llama_layers[0].layer.get_input(layer0_decode_grpid, "input");
                llm_h2d(LLM_WADDR(l0_in), embed.data(), l0_in.nSize, llama_layers[0].layer.get_devid());
            }
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break; auto &lyr = llama_layers[m]; int devid = lyr.layer.get_devid();
                const int layer_decode_grpid = decode_gid_for_layer(m, decode_grpid);
                auto &in_k = lyr.layer.get_input(layer_decode_grpid, "K_cache");
                auto &in_v = lyr.layer.get_input(layer_decode_grpid, "V_cache");
                const int shared_src = shared_kv_source_for_layer(m);
                if (shared_src >= 0)
                {
                    auto &src_layer = llama_layers[(size_t)shared_src];
                    const int src_decode_grpid = decode_gid_for_layer(shared_src, decode_grpid);
                    auto &src_k = src_layer.layer.get_input(src_decode_grpid, "K_cache");
                    auto &src_v = src_layer.layer.get_input(src_decode_grpid, "V_cache");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                    const size_t visible_past = std::min((size_t)kv_slot, dst_tokens > 0 ? (dst_tokens - 1) : 0);
                    llm_memset(LLM_WADDR(in_k), 0, in_k.nSize, devid);
                    llm_memset(LLM_WADDR(in_v), 0, in_v.nSize, devid);
                    if (visible_past > 0)
                    {
                        const size_t past_bytes = visible_past * (size_t)layer_kv * sizeof(unsigned short);
                        llm_d2d(LLM_WADDR(in_k), LLM_RADDR(src_k), std::min(past_bytes, (size_t)src_k.nSize), devid);
                        llm_d2d(LLM_WADDR(in_v), LLM_RADDR(src_v), std::min(past_bytes, (size_t)src_v.nSize), devid);
                    }
                    if (dst_tokens > 0)
                    {
                        const size_t cur_off_bytes = (size_t)kv_slot * (size_t)layer_kv * sizeof(unsigned short);
                        const size_t dst_off_bytes = (dst_tokens - 1) * (size_t)layer_kv * sizeof(unsigned short);
                        llm_d2d((unsigned char *)LLM_WADDR(in_k) + dst_off_bytes,
                                (const unsigned char *)LLM_RADDR(src_k) + cur_off_bytes,
                                sizeof(unsigned short) * (size_t)layer_kv, devid);
                        llm_d2d((unsigned char *)LLM_WADDR(in_v) + dst_off_bytes,
                                (const unsigned char *)LLM_RADDR(src_v) + cur_off_bytes,
                                sizeof(unsigned short) * (size_t)layer_kv, devid);
                    }
                }
                auto &t_idx = lyr.layer.get_input(layer_decode_grpid, "indices"); llm_h2d(LLM_WADDR(t_idx), &decode_pos, sizeof(decode_pos), devid);
                auto &t_mask= lyr.layer.get_input(layer_decode_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    std::vector<unsigned short> linear_decode_mask(elems, bf16_one.data);
                    if (linear_decode_mask.empty()) linear_decode_mask.push_back(bf16_one.data);
                    llm_h2d(LLM_WADDR(t_mask), linear_decode_mask.data(), linear_decode_mask.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    const int mask_elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
                    if (use_sparse_full_cache_mask())
                        build_sparse_layer_decode_mask(decode_mask, mask_elems, (int)kv_slot, m);
                    else
                        build_layer_decode_mask(decode_mask, mask_elems, (int)kv_slot, m);
                    llm_h2d(LLM_WADDR(t_mask),
                            decode_mask.data(),
                            std::min((size_t)t_mask.nSize, (size_t)mask_elems * sizeof(unsigned short)),
                            devid);
                }
                if (use_per_layer_input)
                {
                    const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, layer_decode_grpid, "per_layer_input");
                    if (!t_per_layer)
                    {
                        ALOGE("Gemma4 decoder layer %d is missing decode per_layer_input", m);
                        return final_out;
                    }
                    const size_t src_off = (size_t)m * (size_t)per_layer_hidden;
                    llm_h2d(LLM_WADDR(*t_per_layer),
                            decode_per_layer_input.data() + src_off,
                            std::min((size_t)t_per_layer->nSize, (size_t)per_layer_hidden * sizeof(unsigned short)),
                            devid);
                }
                if (!ensure_layer_loaded(m)) return final_out;
                lyr.layer.inference(layer_decode_grpid);
                if (is_linear_layer(m))
                {
                    (void)in_k;
                    (void)in_v;
                    sync_linear_input_state_from_group(m, lyr.layer, layer_decode_grpid, devid, false);
                }
                else
                {
                    auto &out_k = lyr.layer.get_output(layer_decode_grpid, "K_cache_out"); auto &out_v = lyr.layer.get_output(layer_decode_grpid, "V_cache_out");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    if (shared_src >= 0)
                    {
                        const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                        if (dst_tokens > 0)
                        {
                            const size_t cur_off = (size_t)kv_slot * (size_t)layer_kv;
                            const size_t tail_off = (dst_tokens - 1) * (size_t)layer_kv;
                            const size_t copy_bytes = sizeof(unsigned short) * (size_t)layer_kv;
                            // Shared-KV layers consume the source layer's KV. Preserve that
                            // source KV in the past slot; their own K_cache_out is not the
                            // cache that should be visible to the next decode token.
                            llm_d2d((unsigned short *)LLM_WADDR(in_k) + cur_off,
                                    (unsigned short *)LLM_RADDR(in_k) + tail_off,
                                    copy_bytes,
                                    devid);
                            llm_d2d((unsigned short *)LLM_WADDR(in_v) + cur_off,
                                    (unsigned short *)LLM_RADDR(in_v) + tail_off,
                                    copy_bytes,
                                    devid);
                        }
                    }
                    else
                    {
                        llm_d2d((unsigned short *)LLM_WADDR(in_k) + kv_slot * layer_kv, LLM_RADDR(out_k), std::min((size_t)out_k.nSize, (size_t)layer_kv * sizeof(unsigned short)), devid);
                        llm_d2d((unsigned short *)LLM_WADDR(in_v) + kv_slot * layer_kv, LLM_RADDR(out_v), std::min((size_t)out_v.nSize, (size_t)layer_kv * sizeof(unsigned short)), devid);
                    }
                }
                auto &cur_out = lyr.layer.get_output(layer_decode_grpid, "output");
                if (m == _attr.axmodel_num - 1)
                {
                    auto &post_in = llama_post.get_input("input");
                    if (llama_post.get_devid() == devid) { llm_d2d(LLM_WADDR(post_in), LLM_RADDR(cur_out), post_in.nSize, devid); }
                    else { llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid); llm_h2d(LLM_WADDR(post_in), cur_out.pVirAddr, post_in.nSize, llama_post.get_devid()); }
                }
                else
                {
                    const int next_decode_grpid = decode_gid_for_layer(m + 1, decode_grpid);
                    auto &next_in = llama_layers[m + 1].layer.get_input(next_decode_grpid, "input"); int next_devid = llama_layers[m + 1].layer.get_devid();
                    if (next_devid == devid) { llm_d2d(LLM_WADDR(next_in), LLM_RADDR(cur_out), next_in.nSize, devid); }
                    else { llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid); llm_h2d(LLM_WADDR(next_in), cur_out.pVirAddr, next_in.nSize, next_devid); }
                }
            }
            if (use_sparse_full_cache_mask()) mark_full_cache_slot((int)kv_slot);
            llama_post.inference();
            {
                auto &t_out = llama_post.get_output("output"); llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
                unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            }
#else // AX650
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break; auto &lyr = llama_layers[m];
                const int layer_decode_grpid = decode_gid_for_layer(m, decode_grpid);
                auto &in_k = lyr.layer.get_input(layer_decode_grpid, "K_cache"); auto *in_k_ptr = (unsigned short *)in_k.pVirAddr;
                auto &in_v = lyr.layer.get_input(layer_decode_grpid, "V_cache"); auto *in_v_ptr = (unsigned short *)in_v.pVirAddr;
                const int shared_src = shared_kv_source_for_layer(m);
                const auto shared_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                if (shared_src >= 0)
                {
                    auto &src_layer = llama_layers[(size_t)shared_src];
                    const int src_decode_grpid = decode_gid_for_layer(shared_src, decode_grpid);
                    auto &src_k = src_layer.layer.get_input(src_decode_grpid, "K_cache");
                    auto &src_v = src_layer.layer.get_input(src_decode_grpid, "V_cache");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                    if (need_full_shared_sync)
                    {
                        const size_t visible_past = std::min((size_t)kv_slot, dst_tokens > 0 ? (dst_tokens - 1) : 0);
                        memset(in_k.pVirAddr, 0, in_k.nSize);
                        memset(in_v.pVirAddr, 0, in_v.nSize);
                        if (visible_past > 0)
                        {
                            const size_t past_bytes = visible_past * (size_t)layer_kv * sizeof(unsigned short);
                            memcpy(in_k.pVirAddr, src_k.pVirAddr, std::min(past_bytes, (size_t)src_k.nSize));
                            memcpy(in_v.pVirAddr, src_v.pVirAddr, std::min(past_bytes, (size_t)src_v.nSize));
                        }
                    }
                    if (dst_tokens > 0)
                    {
                        const size_t cur_off = (size_t)kv_slot * (size_t)layer_kv;
                        const size_t dst_off = (dst_tokens - 1) * (size_t)layer_kv;
                        memcpy(in_k_ptr + dst_off, (const unsigned short *)src_k.pVirAddr + cur_off, sizeof(unsigned short) * (size_t)layer_kv);
                        memcpy(in_v_ptr + dst_off, (const unsigned short *)src_v.pVirAddr + cur_off, sizeof(unsigned short) * (size_t)layer_kv);
                    }
                }
                if (decode_profile_enabled)
                    decode_profile.shared_kv_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - shared_t0).count();
                const auto prep_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                auto &t_idx = lyr.layer.get_input(layer_decode_grpid, "indices"); memcpy(t_idx.pVirAddr, &decode_pos, sizeof(decode_pos));
                auto &t_mask= lyr.layer.get_input(layer_decode_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    std::vector<unsigned short> linear_decode_mask(elems, bf16_one.data);
                    if (linear_decode_mask.empty()) linear_decode_mask.push_back(bf16_one.data);
                    memcpy(t_mask.pVirAddr, linear_decode_mask.data(), std::min((size_t)t_mask.nSize, linear_decode_mask.size() * sizeof(unsigned short)));
                }
                else
                {
                    const int mask_elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
                    if (use_sparse_full_cache_mask())
                        build_sparse_layer_decode_mask(decode_mask, mask_elems, (int)kv_slot, m);
                    else
                        build_layer_decode_mask(decode_mask, mask_elems, (int)kv_slot, m);
                    memcpy(t_mask.pVirAddr,
                           decode_mask.data(),
                           std::min((size_t)t_mask.nSize, (size_t)mask_elems * sizeof(unsigned short)));
                }
                if (use_per_layer_input)
                {
                    const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, layer_decode_grpid, "per_layer_input");
                    if (!t_per_layer)
                    {
                        ALOGE("Gemma4 decoder layer %d is missing decode per_layer_input", m);
                        return final_out;
                    }
                    const size_t src_off = (size_t)m * (size_t)per_layer_hidden;
                    memcpy(t_per_layer->pVirAddr,
                           decode_per_layer_input.data() + src_off,
                           std::min((size_t)t_per_layer->nSize, (size_t)per_layer_hidden * sizeof(unsigned short)));
                }
                auto &t_in  = lyr.layer.get_input(layer_decode_grpid, "input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                if (decode_profile_enabled)
                    decode_profile.prep_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - prep_t0).count();
                const auto infer_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                if (!ensure_layer_loaded(m)) return final_out;
                lyr.layer.inference(layer_decode_grpid);
                if (decode_profile_enabled)
                    decode_profile.inference_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - infer_t0).count();
                const auto out_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                if (is_linear_layer(m))
                {
                    (void)in_k;
                    (void)in_v;
                    sync_linear_input_state_from_group(m, lyr.layer, layer_decode_grpid, 0, false);
                }
                else
                {
                    auto &out_k = lyr.layer.get_output(layer_decode_grpid, "K_cache_out");
                    auto &out_v = lyr.layer.get_output(layer_decode_grpid, "V_cache_out");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    if (shared_src >= 0)
                    {
                        const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                        if (dst_tokens > 0)
                        {
                            const size_t cur_off = (size_t)kv_slot * (size_t)layer_kv;
                            const size_t tail_off = (dst_tokens - 1) * (size_t)layer_kv;
                            // See the AXCL branch above: shared layers must retain the
                            // source layer KV in their visible past cache.
                            memcpy(in_k_ptr + cur_off, in_k_ptr + tail_off, sizeof(unsigned short) * (size_t)layer_kv);
                            memcpy(in_v_ptr + cur_off, in_v_ptr + tail_off, sizeof(unsigned short) * (size_t)layer_kv);
                        }
                    }
                    else
                    {
                        memcpy(in_k_ptr + kv_slot * layer_kv, out_k.pVirAddr, sizeof(unsigned short) * layer_kv);
                        memcpy(in_v_ptr + kv_slot * layer_kv, out_v.pVirAddr, sizeof(unsigned short) * layer_kv);
                    }
                }
                auto &t_out= lyr.layer.get_output(layer_decode_grpid, "output"); memcpy(embed.data(), t_out.pVirAddr, embed.size() * sizeof(unsigned short));
                if (decode_profile_enabled)
                    decode_profile.outcopy_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - out_t0).count();
            }
            if (use_sparse_full_cache_mask()) mark_full_cache_slot((int)kv_slot);
            const auto post_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
            auto &t_in = llama_post.get_input("input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            dump_hidden_trace("post_input_hidden", (int)token_ids.size(), embed.data(), _attr.tokens_embed_size);
            llama_post.inference(); auto &t_out = llama_post.get_output("output");
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            dump_decode_trace((int)token_ids.size(), post_out, _attr.tokens_embed_num, next_token);
            if (decode_profile_enabled)
                decode_profile.post_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - post_t0).count();
            last_shared_sync_decode_grpid = decode_grpid;
#endif

            if (tokenizer->is_stop(next_token)) { b_hit_eos = true; break; }
            token_ids.push_back(next_token);
            if (_attr.runing_callback)
            {
                float tps = -1.0f;
                if (decode_timer_started)
                {
                    const int decode_tokens = std::max(0, (int)token_ids.size() - 1);
                    const float decode_ms = decode_timer.cost();
                    if (decode_tokens > 0 && decode_ms > 0.0f)
                        tps = decode_tokens / (decode_ms / 1000.0f);
                }
                auto  str   = safe_decode_token(tokenizer, next_token);
                if (hide_channel_markup) str = channel_filter.filter(str);
                emit_stream_chunk(str, tps);
            }
            if (output_max_token > 0 && (int)token_ids.size() >= output_max_token) { b_hit_eos = true; break; }
            if (_attr.runing_callback == nullptr) update_cqdm(&cqdm, kv_slot, "token", "");
        }

        const int generated_token_count = (int)token_ids.size();
        last_run_generated_token_ids = token_ids;
        final_out = tokenizer->decode(token_ids);
        if (hide_channel_markup)
        {
            final_out = strip_hidden_channel_sections(final_out);
        }
        final_out = sanitize_utf8_text(final_out);
        if (_attr.runing_callback)
        {
            std::string tail;
            if (hide_channel_markup)
                tail = channel_filter.flush();
            emit_stream_chunk(tail, -1);

            if (final_out.size() > streamed_visible_text.size() &&
                final_out.compare(0, streamed_visible_text.size(), streamed_visible_text) == 0)
            {
                _attr.runing_callback(final_out.substr(streamed_visible_text.size()), -1, _attr.reserve);
            }
        }
        printf("\n\n"); fflush(stdout);
        float avg_decode_tps = 0.0f;
        if (decode_timer_started)
        {
            const int decode_tokens = std::max(0, (int)token_ids.size() - 1);
            const float decode_ms = decode_timer.cost();
            if (decode_tokens > 0 && decode_ms > 0.0f)
                avg_decode_tps = decode_tokens / (decode_ms / 1000.0f);
        }
        if (decode_profile_enabled)
        {
            const int decode_tokens = std::max(0, (int)token_ids.size() - 1);
            if (decode_tokens > 0)
            {
                const auto avg_ms = [decode_tokens](uint64_t ns) {
                    return (double)ns / ((double)decode_tokens * 1e6);
                };
                ALOGN("decode profile avg/token: per_layer=%.3f ms, shared_kv=%.3f ms, prep=%.3f ms, inference=%.3f ms, outcopy=%.3f ms, post=%.3f ms\n",
                      avg_ms(decode_profile.per_layer_ns),
                      avg_ms(decode_profile.shared_kv_ns),
                      avg_ms(decode_profile.prep_ns),
                      avg_ms(decode_profile.inference_ns),
                      avg_ms(decode_profile.outcopy_ns),
                      avg_ms(decode_profile.post_ns));
                if (use_per_layer_input)
                {
                    ALOGN("decode profile cache: hits=%llu misses=%llu\n",
                          (unsigned long long)gemma4_per_layer_helper.decode_cache_hits(),
                          (unsigned long long)gemma4_per_layer_helper.decode_cache_misses());
                }
            }
        }
        ALOGN("hit eos,decode avg %.2f token/s\n", avg_decode_tps);
        if (generated_token_count >= 0)
        {
            precompute_len = dense_decode_start + generated_token_count;
            if ((has_vision_state && !vision_state.position_ids.empty()) || active_prefill_pos_start >= 0)
                cached_mrope_next_pos = (int)decode_start + generated_token_count;
            else
                cached_mrope_next_pos = -1;
        }
        active_prefill_pos_start = -1;
        active_token_pos_start = -1;
        return final_out;
    }

    std::vector<Content> Run(std::vector<Content> history, int output_max_token = -1)
    {
        return Run(std::move(history), {}, output_max_token);
    }

    std::vector<Content> Run(std::vector<Content> history, const std::vector<::MediaInputs> &media_inputs, int output_max_token = -1)
    {
        clear_last_error();
        has_vision_state = false;
        std::vector<::MediaInputs> effective_media_inputs = media_inputs;
        bool video_history_isolated = false;

        // Multi-slot prefix KV cache: pick the slot sharing the longest prefix with
        // this request; misses evict the LRU slot. Must run before the single-context
        // history-prefix reset below so the chosen slot's snapshot is in place when
        // the existing reuse logic runs. Text LLM matches on tokens; VLM matches on
        // chat history (tokenizing VLM history needs the expensive vision Prepare).
        if (multi_slot_enabled_)
        {
            if (_attr.vlm_type == VLMType::None)
                select_kv_slot(tokenizer->encode(history));
            else
                select_kv_slot_by_history(history);
        }

        if (request_has_video_media(history, effective_media_inputs))
        {
            const size_t old_history_size = history.size();
            const size_t old_media_size = effective_media_inputs.size();
            if (normalize_away_video_history(history, effective_media_inputs))
            {
                ALOGW("video history is isolated to current user turn: old_history=%zu new_history=%zu old_media_inputs=%zu new_media_inputs=%zu",
                      old_history_size,
                      history.size(),
                      old_media_size,
                      effective_media_inputs.size());
                ResetKVCache();
                video_history_isolated = true;
            }
        }

        std::vector<int> new_tokens;
        std::string response_prefix;
        bool used_cached_text_turn = false;
        bool used_cached_media_turn = false;
        size_t append_start = last_history_snapshot.size();
        if (!last_history_snapshot.empty() && !is_history_prefix(last_history_snapshot, history))
        {
            // Non-append TEXT histories (same system prompt, different question) can share a
            // long common token prefix -- let token-level prefix reuse keep it instead of
            // wiping the KV and recomputing the shared prefix every time. For VLM, keep the
            // single-context reset: image placeholder token IDs are identical across different
            // images, so a token-level prefix match would wrongly reuse a previous image's KV.
            if (!multi_slot_active_request_ && _attr.vlm_type != VLMType::None)
            {
                ALOGW("raw history not append. force ResetKVCache before request processing.");
                ResetKVCache();
            }
            append_start = 0;
        }

        const std::string tokenizer_key_for_cache = key_of(_attr.tokenizer_type);
        const bool is_minicpm_v46_cache =
            tokenizer_key_for_cache == "minicpmv46" ||
            tokenizer_key_for_cache == "minicpmv46vl";
        const bool allow_cached_text_turn =
            supports_cached_im_chat_turn_tokens() &&
            (!is_minicpm_v46_cache ||
             std::getenv("AXLLM_ENABLE_MINICPMV46_TEXT_KV_CACHE") != nullptr);
        const bool no_new_media_input = !has_new_media_input(effective_media_inputs, append_start);
        const bool cached_text_turn_requested = vision && vision->enabled() &&
                                                allow_cached_text_turn &&
                                                !last_history_snapshot.empty() &&
                                                precompute_len > 0 &&
                                                no_new_media_input &&
                                                history.size() > append_start &&
                                                appended_history_is_text_only_user_turn(history, append_start);
        const bool cached_media_turn_requested = vision && vision->enabled() &&
                                                 supports_cached_im_chat_turn_tokens() &&
                                                 !last_history_snapshot.empty() &&
                                                 precompute_len > 0 &&
                                                 !no_new_media_input &&
                                                 history.size() > append_start &&
                                                 appended_history_is_user_turn_with_media(history, append_start);
        const bool raw_history_appended = !last_history_snapshot.empty() && is_history_prefix(last_history_snapshot, history);
        const bool raw_history_rollback = !last_history_snapshot.empty() && is_history_prefix(history, last_history_snapshot);
        const bool raw_history_modified = !last_history_snapshot.empty() && !raw_history_appended && !raw_history_rollback;
        if (raw_history_modified)
        {
            if (cached_text_turn_requested || cached_media_turn_requested)
            {
                set_last_error("检测到历史被修改，无法复用已有 KV。请保持返回的 history 继续追加，或 /reset 后重新开始。");
                ALOGE("cached VLM turn cannot reuse KV because history is not append; refuse full recompute");
                return history;
            }
            // Non-append histories can still share a long common token prefix (e.g. same system prompt but
            // different user question). Leave KV intact here and let token-level prefix reuse decide later.
            ALOGI("raw history modified (not append / not rollback). try token-level KV prefix reuse.");
        }
        else if (raw_history_rollback)
        {
            ALOGI("raw history rollback detected: prev_contents=%zu new_contents=%zu, try KV reuse",
                  last_history_snapshot.size(),
                  history.size());
        }

        const bool history_appended = raw_history_appended && history.size() > append_start;

        if (cached_text_turn_requested)
        {
            if (!history_appended || !build_cached_im_chat_text_turn_tokens(history, append_start, new_tokens))
            {
                set_last_error("无法复用已有图像 KV，已拒绝重新编码历史。请 /reset 后重新开始。");
                ALOGE("failed to build cached text-only turn tokens; refuse vision Prepare/full recompute");
                return history;
            }
            used_cached_text_turn = true;
            ALOGI("reuse cached KV for text-only turn: cached_tokens=%zu append_contents=%zu input_tokens=%zu, skip vision Prepare",
                  last_tokens_ids.size(),
                  history.size() - append_start,
                  new_tokens.size() - last_tokens_ids.size());
        }
        else if (cached_media_turn_requested)
        {
            std::string verr;
            if (!history_appended || !build_cached_im_chat_media_turn_tokens(history, effective_media_inputs, append_start, new_tokens, vision_state, verr))
            {
                set_last_error("无法增量处理新增多模态输入，请 /reset 后重新开始。");
                ALOGE("failed to build cached media turn tokens: %s", verr.c_str());
                return history;
            }
            has_vision_state = true;
            used_cached_media_turn = true;
            ALOGI("reuse cached KV for media turn: cached_tokens=%zu append_contents=%zu input_tokens=%zu",
                  last_tokens_ids.size(),
                  history.size() - append_start,
                  new_tokens.size() - last_tokens_ids.size());
        }

        if (!used_cached_text_turn && !used_cached_media_turn && vision && vision->enabled())
        {
            // If caller provides media, we will fill num_media/num_media_tokens and build injection state.
            if (!effective_media_inputs.empty())
            {
                std::vector<vision::MediaInputs> vmins;
                vmins.reserve(effective_media_inputs.size());
                for (const auto &m : effective_media_inputs) vmins.push_back({m.content_index, m.uris});

                vision::PromptBudget budget;
                budget.last_tokens = last_tokens_ids;
                budget.precompute_len = precompute_len;
                budget.prefill_token_num = _attr.prefill_token_num;
                const int max_cap = !_attr.prefill_max_kv_cache_num_grp.empty()
                                        ? _attr.prefill_max_kv_cache_num_grp.back()
                                        : _attr.prefill_max_token_num;
                budget.max_total_tokens = max_cap;
                budget.max_history_tokens = (_attr.prefill_token_num > 0)
                                                ? std::max(0, max_cap - _attr.prefill_token_num)
                                                : std::max(0, max_cap);
                int remaining = std::max(0, max_cap - precompute_len);
                if (_attr.prefill_token_num > 0)
                {
                    remaining = ALIGN_DOWN(remaining, _attr.prefill_token_num);
                }
                budget.max_tail_tokens = remaining;

                std::vector<Content> prepared_history;
                std::vector<int> input_ids;
                vision::RunState st;
                vision::PrepareMetadata prepare_meta;
                std::string verr;
                if (!vision->Prepare(history, vmins, &budget, prepared_history, input_ids, st, verr, true, &prepare_meta))
                {
                    if (verr.find("exceeds current prefill budget") != std::string::npos ||
                        verr.find("exceeds current history budget") != std::string::npos ||
                        verr.find("history budget") != std::string::npos) {
                        set_context_limit_error();
                    } else {
                        set_last_error("多模态输入处理失败，请重新开始会话后再试一次。");
                    }
                    ALOGE("vision.Prepare failed: %s", verr.c_str());
                    return history;
                }
                if (prepare_meta.auto_reset_for_video)
                {
                    ALOGW("video request auto reset history for quality: current_frames=%d fresh_frames=%d",
                          prepare_meta.current_video_frames,
                          prepare_meta.fresh_video_frames);
                    ResetKVCache();
                    response_prefix = video_quality_reset_notice();
                    if (_attr.runing_callback)
                    {
                        _attr.runing_callback(response_prefix, -1, _attr.reserve);
                    }
                }
                history = std::move(prepared_history);
                new_tokens = std::move(input_ids);
                vision_state = std::move(st);
                has_vision_state = true;
            }
            else
            {
                // If history contains multimodal content, the caller must provide media inputs.
                bool need_media = false;
                for (const auto &c : history) if (c.type == IMAGE || c.type == VIDEO || c.type == AUDIO) { need_media = true; break; }
                if (need_media)
                {
                    ALOGE("vlm_type=%s/%d enabled but media_inputs is empty",
                          std::string(VLMTypeName(_attr.vlm_type)).c_str(),
                          (int)_attr.vlm_type);
                }
                new_tokens = tokenizer->encode(history);
            }
        }
        else if (!used_cached_text_turn && !used_cached_media_turn)
        {
            new_tokens = tokenizer->encode(history);
        }

        last_run_prompt_token_num_ = (int)new_tokens.size();
        int offset = 0;
        auto tokens_diff = diff_token_ids(last_tokens_ids, new_tokens, offset);
        const bool token_appended = (offset == (int)last_tokens_ids.size() && (int)new_tokens.size() >= (int)last_tokens_ids.size());
        const bool token_rollback = (offset == (int)new_tokens.size() && (int)new_tokens.size() <= (int)last_tokens_ids.size());
        const bool token_prefix_reuse = (!token_appended && !token_rollback &&
                                         offset > 0 &&
                                         precompute_len > 0 &&
                                         offset <= precompute_len);
        bool not_append = !(token_appended || token_rollback || token_prefix_reuse);
        if (not_append)
        {
            if (used_cached_text_turn)
            {
                set_last_error("KV 复用失败，已拒绝重新编码历史。请 /reset 后重新开始。");
                ALOGE("cached text-only turn token prefix mismatch: offset=%d cached_tokens=%zu new_tokens=%zu; refuse full recompute",
                      offset,
                      last_tokens_ids.size(),
                      new_tokens.size());
                return history;
            }
            if (used_cached_media_turn)
            {
                set_last_error("多模态 KV 复用失败，已拒绝重新编码历史。请 /reset 后重新开始。");
                ALOGE("cached media turn token prefix mismatch: offset=%d cached_tokens=%zu new_tokens=%zu; refuse full recompute",
                      offset,
                      last_tokens_ids.size(),
                      new_tokens.size());
                return history;
            }
            ALOGW("history diverged with no reusable prefix. force ResetKVCache and recompute.");
            ResetKVCache();
            tokens_diff = new_tokens;
            offset = 0;
        }
        else if (token_rollback)
        {
            const int new_len = (int)new_tokens.size();
            const int prev_tokens = (int)last_tokens_ids.size();
            const int prev_kv = precompute_len;
            int keep = new_len;

            if (has_linear_attention_layers())
            {
                const int target = tokens_diff.empty() ? std::max(0, new_len - 1) : new_len;
                keep = best_linear_state_snapshot_len(target);
                if (target > 0 && keep < 0)
                {
                    ALOGW("token rollback has no linear state snapshot <= %d. force ResetKVCache and recompute.", target);
                    ResetKVCache();
                    tokens_diff = new_tokens;
                    offset = 0;
                    not_append = false;
                }
                else
                {
                    keep = std::max(0, keep);
                    if (!restore_linear_state_snapshot_to_host_cache(keep))
                    {
                        ALOGW("failed to restore linear state snapshot at %d. force ResetKVCache and recompute.", keep);
                        ResetKVCache();
                        tokens_diff = new_tokens;
                        offset = 0;
                        not_append = false;
                    }
                    else
                    {
                        drop_linear_state_snapshots_after(keep);
                        if (prev_tokens != keep) last_tokens_ids.resize((size_t)keep);
                        if (precompute_len > keep) precompute_len = keep;
                        if (cached_mrope_next_pos >= keep) cached_mrope_next_pos = -1;
                        offset = keep;
                        tokens_diff.assign(new_tokens.begin() + keep, new_tokens.end());
                        ALOGI("token rollback: reuse KV prefix tokens=%d via linear snapshot (prev_tokens=%d prev_kv=%d) recompute_suffix=%zu",
                              keep,
                              prev_tokens,
                              prev_kv,
                              tokens_diff.size());
                    }
                }
            }
            else
            {
                if (prev_tokens != keep)
                {
                    last_tokens_ids.resize((size_t)keep);
                }
                if (precompute_len > keep)
                {
                    precompute_len = keep;
                }
                if (cached_mrope_next_pos >= keep)
                {
                    cached_mrope_next_pos = -1;
                }

                ALOGI("token rollback: reuse KV prefix tokens=%d (prev_tokens=%d prev_kv=%d)",
                      keep,
                      prev_tokens,
                      prev_kv);
            }
        }
        else if (token_prefix_reuse)
        {
            int keep = offset;
            const int prev_tokens = (int)last_tokens_ids.size();
            const int prev_kv = precompute_len;

            if (has_linear_attention_layers())
            {
                // Cap the reuse target to the max prefill history capacity before snapping to
                // a linear-state snapshot, so the resulting prefix fits a single prefill group.
                const int max_hist_lin = max_prefill_history_cap();
                const int want_lin = (max_hist_lin > 0 && offset > max_hist_lin) ? max_hist_lin : offset;
                keep = best_linear_state_snapshot_len(want_lin);
                if (keep < 0)
                {
                    ALOGW("token prefix reuse has no linear state snapshot <= %d. force ResetKVCache and recompute.", offset);
                    ResetKVCache();
                    tokens_diff = new_tokens;
                    offset = 0;
                    not_append = false;
                }
                else if (!restore_linear_state_snapshot_to_host_cache(keep))
                {
                    ALOGW("failed to restore linear state snapshot at %d. force ResetKVCache and recompute.", keep);
                    ResetKVCache();
                    tokens_diff = new_tokens;
                    offset = 0;
                    not_append = false;
                }
                else
                {
                    drop_linear_state_snapshots_after(keep);
                    if (prev_tokens != keep) last_tokens_ids.resize((size_t)keep);
                    if (precompute_len > keep) precompute_len = keep;
                    if (cached_mrope_next_pos >= keep) cached_mrope_next_pos = -1;
                    offset = keep;
                    tokens_diff.assign(new_tokens.begin() + keep, new_tokens.end());
                    ALOGI("token prefix reuse: reuse KV prefix tokens=%d via linear snapshot (prev_tokens=%d prev_kv=%d) recompute_suffix=%zu",
                          keep,
                          prev_tokens,
                          prev_kv,
                          tokens_diff.size());
                }
            }
            else
            {
                // Reuse whole prefill chunks only: round the reused prefix down to a
                // prefill-chunk boundary, and cap to what a single prefill can attend to as
                // history (max_prefill_history_cap, itself chunk-aligned). Recompute the
                // remaining tokens (the partial last chunk + the divergent suffix).
                const int step = std::max(1, _attr.prefill_token_num);
                const int cap = max_prefill_history_cap();
                int aligned = (keep / step) * step;
                if (cap > 0 && aligned > cap) aligned = (cap / step) * step;
                if (aligned != keep)
                {
                    ALOGW("token prefix reuse: chunk-align prefix %d -> %d (step=%d cap=%d)", keep, aligned, step, cap);
                    keep = aligned;
                    offset = keep;
                    tokens_diff.assign(new_tokens.begin() + keep, new_tokens.end());
                }
                if (prev_tokens != keep)
                {
                    last_tokens_ids.resize((size_t)keep);
                }
                if (precompute_len > keep)
                {
                    precompute_len = keep;
                }
                if (cached_mrope_next_pos >= keep)
                {
                    cached_mrope_next_pos = -1;
                }

                ALOGI("token prefix reuse: reuse KV prefix tokens=%d (prev_tokens=%d prev_kv=%d) recompute_suffix=%zu",
                      keep,
                      prev_tokens,
                      prev_kv,
                      tokens_diff.size());
            }
        }
        if (tokens_diff.empty())
        {
            if (used_cached_text_turn)
            {
                set_last_error("KV 复用失败：当前追问没有新增 token。已拒绝重新编码历史。");
                ALOGE("cached text-only turn has empty token diff; refuse full recompute");
                return history;
            }
            if (used_cached_media_turn)
            {
                set_last_error("多模态 KV 复用失败：当前追问没有新增 token。已拒绝重新编码历史。");
                ALOGE("cached media turn has empty token diff; refuse full recompute");
                return history;
            }
            if (!new_tokens.empty())
            {
                // Identical re-query: the reused prefix is the whole previous input minus one
                // token. Chunk-align + cap it to the prefill history capacity so it fits a
                // single prefill group and can be reused (otherwise SetKVCache would fail and
                // fall back to a full recompute). Linear models keep the exact value and rely
                // on the fallback (their state is snapshotted, not chunk-addressable).
                int keep = (int)new_tokens.size() - 1;
                if (!has_linear_attention_layers())
                {
                    const int step = std::max(1, _attr.prefill_token_num);
                    const int maxh = max_prefill_history_cap();
                    if (maxh > 0 && keep > maxh) keep = maxh;
                    keep = (keep / step) * step;
                    if (keep < 0) keep = 0;
                }
                precompute_len = keep;
                offset = keep;
                tokens_diff.assign(new_tokens.begin() + keep, new_tokens.end());
                if ((int)last_tokens_ids.size() > keep) last_tokens_ids.resize((size_t)keep);
                ALOGI("identical re-query: reuse KV prefix tokens=%d recompute_suffix=%zu", keep, tokens_diff.size());
            }
            else { ResetKVCache(); precompute_len = 0; }
        }
        if (!not_append && offset != precompute_len && precompute_len > 0)
        {
            if (used_cached_text_turn || used_cached_media_turn)
            {
                set_last_error("KV 复用失败：token 前缀长度与 KV 长度不一致。已拒绝重新编码历史。");
                ALOGE("cached turn token/KV mismatch: token_offset=%d precompute_len=%d; refuse full recompute",
                      offset,
                      precompute_len);
                return history;
            }
            ALOGW("token prefix/KV length mismatch: token_offset=%d precompute_len=%d, recompute full history",
                  offset,
                  precompute_len);
            ResetKVCache();
            tokens_diff = new_tokens;
            offset = 0;
        }
        int kv_ret = SetKVCache(k_caches, v_caches, precompute_len, (int)tokens_diff.size());
        if (kv_ret != 0 && precompute_len > 0 && !used_cached_text_turn && !used_cached_media_turn)
        {
            // The reused prefix could not be set up in a single prefill group. This is NOT a
            // real context overflow -- fall back to a full chunked recompute instead of
            // surfacing a misleading "context exceeded" error to the user.
            ALOGW("prefix-reuse SetKVCache failed (precompute_len=%d suffix=%zu); fall back to full recompute",
                  precompute_len, (int)tokens_diff.size());
            clear_last_error();
            ResetKVCache();
            tokens_diff = new_tokens;
            offset = 0;
            precompute_len = 0;
            kv_ret = SetKVCache(k_caches, v_caches, 0, (int)tokens_diff.size());
        }
        if (kv_ret != 0)
        {
            ALOGE("SetKVCache failed");
            return history;
        }
        active_token_pos_start = offset;
        active_prefill_pos_start = -1;
        if (has_vision_state && !vision_state.position_ids.empty() && offset != precompute_len)
        {
            ALOGI("VLM token/KV offset mismatch: token_offset=%d dense_kv_start=%d input_tokens=%zu",
                  offset,
                  precompute_len,
                  tokens_diff.size());
        }
        if (!(has_vision_state && !vision_state.position_ids.empty()) && cached_mrope_next_pos >= 0 && precompute_len > 0)
        {
            active_prefill_pos_start = cached_mrope_next_pos;
            ALOGI("VLM cached mRoPE positions: prefill_rope_start=%d dense_kv_start=%d input_tokens=%zu",
                  active_prefill_pos_start,
                  precompute_len,
                  tokens_diff.size());
        }
        std::vector<unsigned short> out_embed(tokens_diff.size() * _attr.tokens_embed_size);
        for (size_t i = 0; i < tokens_diff.size(); i++)
        {
            const int abs_pos = offset + (int)i;
            if (has_vision_state && (size_t)abs_pos < vision_state.pos2vision.size())
            {
                int vidx = vision_state.pos2vision[abs_pos];
                if (vidx >= 0)
                {
                    memcpy(out_embed.data() + i * _attr.tokens_embed_size,
                           vision_state.vision_embed.data() + (size_t)vidx * _attr.tokens_embed_size,
                           (size_t)_attr.tokens_embed_size * sizeof(unsigned short));
                    continue;
                }
            }
            embed_selector.getByIndex(tokens_diff[i], out_embed.data() + i * _attr.tokens_embed_size);
        }
        if (std::getenv("AXLLM_DEBUG_EMBED_TOKEN_IDS"))
        {
            std::string s;
            s.reserve(tokens_diff.size() * 8);
            for (size_t i = 0; i < tokens_diff.size(); ++i)
            {
                if (i) s.push_back(',');
                s += std::to_string(tokens_diff[i]);
            }
            ALOGI("Run history tokens_diff(len=%zu): %s", tokens_diff.size(), s.c_str());
        }
        std::vector<int> cached_prefix_tokens;
        if (precompute_len > 0)
        {
            const size_t prefix_len = std::min((size_t)precompute_len, last_tokens_ids.size());
            cached_prefix_tokens.assign(last_tokens_ids.begin(), last_tokens_ids.begin() + prefix_len);
            if ((int)cached_prefix_tokens.size() != precompute_len)
            {
                ALOGW("cached token prefix size mismatch before run: tokens=%zu precompute_len=%d",
                      cached_prefix_tokens.size(),
                      precompute_len);
            }
        }
        last_run_generated_token_ids.clear();
        run_input_token_ids = tokens_diff;
        auto reply = Run(out_embed, output_max_token);
        run_input_token_ids.clear();
        if (!response_prefix.empty())
        {
            reply = response_prefix + reply;
        }
        history.push_back({ASSISTANT, TEXT, reply});
        last_history_snapshot = history;
        last_tokens_ids = std::move(cached_prefix_tokens);
        last_tokens_ids.insert(last_tokens_ids.end(), tokens_diff.begin(), tokens_diff.end());
        last_tokens_ids.insert(last_tokens_ids.end(), last_run_generated_token_ids.begin(), last_run_generated_token_ids.end());
        if (video_history_isolated)
        {
            ALOGW("drop KV cache after isolated video-history request");
            ResetKVCache();
        }
        else
        {
            GetKVCache(k_caches, v_caches, precompute_len);
            if ((int)last_tokens_ids.size() != precompute_len)
            {
                ALOGW("exact cached token prefix length differs from KV: tokens=%zu precompute_len=%d generated=%zu input=%zu",
                      last_tokens_ids.size(),
                      precompute_len,
                      last_run_generated_token_ids.size(),
                      tokens_diff.size());
            }
        }

        has_vision_state = false;
        vision_state = {};

        // Persist the updated working state back into the active slot so the next
        // request can match/continue it. Device KV already lives in the slot's
        // own buffer (zero copy).
        save_active_kv_slot();

        return history;
    }
};

// Public LLM thin wrappers

LLM::LLM() : impl_(new Impl()) {}
LLM::~LLM() = default;

bool LLM::Init(LLMAttrType attr) { return impl_->Init(std::move(attr)); }
void LLM::Deinit() { impl_->Deinit(); }
void LLM::Stop() { impl_->Stop(); }

LLMAttrType *LLM::getAttr() { return &impl_->_attr; }
LLMPostprocess *LLM::getPostprocess() { return &impl_->postprocess; }
LLaMaEmbedSelector *LLM::getEmbedSelector() { return &impl_->embed_selector; }
void LLM::SetRequestSamplingOverride(bool has_temperature, float temperature, bool has_top_p, float top_p, bool has_frequency_penalty, float frequency_penalty, bool has_presence_penalty, float presence_penalty) { impl_->postprocess.set_request_sampling_override(has_temperature, temperature, has_top_p, top_p, has_frequency_penalty, frequency_penalty, has_presence_penalty, presence_penalty); }
void LLM::ClearRequestSamplingOverride() { impl_->postprocess.clear_request_sampling_override(); }
void LLM::SetRequestThinkingMode(ThinkingMode mode) { if (impl_->tokenizer) impl_->tokenizer->set_generation_thinking_mode(mode); }
void LLM::ClearRequestThinkingMode() { if (impl_->tokenizer) impl_->tokenizer->set_generation_thinking_mode(impl_->_attr.generation_thinking_mode); }
bool LLM::TokenizerSupportsThinkingToggle() const { return impl_->tokenizer ? impl_->tokenizer->supports_thinking_toggle() : false; }
void LLM::MarkRequestStart() { impl_->MarkRequestStart(); }
void LLM::ClearRequestStart() { impl_->ClearRequestStart(); }
std::string LLM::GetLastError() const { return impl_->get_last_error(); }
int LLM::GetLastPromptTokenNum() const { return impl_->get_last_prompt_token_num(); }
int LLM::GetLastCompletionTokenNum() const { return impl_->get_last_completion_token_num(); }

bool LLM::Embed(const std::string &text, std::vector<float> &out_embedding) { return impl_->EmbedText(text, out_embedding); }
bool LLM::Embed(const std::vector<Content> &history, const std::vector<MediaInputs> &media_inputs, std::vector<float> &out_embedding) { return impl_->EmbedHistory(history, media_inputs, out_embedding); }
bool LLM::EmbedBatch(const std::vector<std::string> &inputs, std::vector<std::vector<float>> &out_embeddings) { return impl_->EmbedBatch(inputs, out_embeddings); }

int LLM::GenerateKVCachePrefill(std::vector<int> &ids, std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int &pre_len) { return impl_->GenerateKVCachePrefill(ids, k, v, pre_len); }
int LLM::GetKVCache(std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int &pre_len) { return impl_->GetKVCache(k, v, pre_len); }
int LLM::SetKVCache(std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int pre_len, int in_tokens) { return impl_->SetKVCache(k, v, pre_len, in_tokens); }
void LLM::ResetKVCache() { impl_->ResetKVCache(); }

std::vector<Content> LLM::Run(std::vector<Content> history, int output_max_token) { return impl_->Run(std::move(history), output_max_token); }
std::vector<Content> LLM::Run(std::vector<Content> history, const std::vector<MediaInputs> &media_inputs, int output_max_token) { return impl_->Run(std::move(history), media_inputs, output_max_token); }
std::string LLM::Run(std::vector<unsigned short> &embed, int output_max_token) { return impl_->Run(embed, output_max_token); }
