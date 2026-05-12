#include "LLM.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>

#include "bfloat16.hpp"
#include "Gemma4PerLayerHelper.hpp"
#include "LLMEmbedSelector.hpp"
#include "LLMPostprocess.hpp"
#include "UTF8Filter.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "utils/memory_utils.hpp"
#include "sample_log.h"

#include "vision/vision_module.hpp"

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

class ChannelSectionFilter
{
public:
    void reset()
    {
        pending_.clear();
        skipping_ = false;
    }

    std::string filter(const std::string &chunk)
    {
        // Gemma4 hidden-thought section is "<|channel>thought\n...<channel|>" per tokenizer x-regex
        // and chat_template.jinja::strip_thinking. <|think|> is a prompt-only marker (emitted when
        // enable_thinking=true in the template); the model does not generate it. Matching it here
        // as a section-opener would make any model output containing the 9-byte literal "<|think|>"
        // be silently swallowed because no paired "<channel|>" would follow.
        static const std::string kChannelStart = "<|channel>";
        static const std::string kChannelEnd = "<channel|>";
        static const size_t kStartGuard = kChannelStart.size() - 1;
        static const size_t kEndGuard = kChannelEnd.size() - 1;

        pending_ += chunk;
        std::string out;

        while (true)
        {
            if (!skipping_)
            {
                size_t start = pending_.find(kChannelStart);

                if (start == std::string::npos)
                {
                    if (pending_.size() > kStartGuard)
                    {
                        const size_t emit_len = pending_.size() - kStartGuard;
                        out.append(pending_, 0, emit_len);
                        pending_.erase(0, emit_len);
                    }
                    break;
                }

                out.append(pending_, 0, start);
                pending_.erase(0, start);
                skipping_ = true;
            }

            const size_t end = pending_.find(kChannelEnd);
            if (end == std::string::npos)
            {
                if (pending_.size() > kEndGuard)
                    pending_.erase(0, pending_.size() - kEndGuard);
                break;
            }

            pending_.erase(0, end + kChannelEnd.size());
            skipping_ = false;
        }

        return out;
    }

    std::string flush()
    {
        if (skipping_)
        {
            pending_.clear();
            return {};
        }

        std::string out;
        out.swap(pending_);
        return out;
    }

private:
    std::string pending_;
    bool skipping_ = false;
};

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
    bool b_os_kvcache = false;
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    int precompute_len = 0;
    std::vector<int> prefill_history_kv_cache_num_grp;
    std::vector<int> prefill_symbolic_kv_cache_num_grp;

    LLMAttrType _attr;
    bool embedding_append_eos = false;
    int embedding_eos_token_id = -1;

    std::unique_ptr<vision::VisionModule> vision;
    vision::RunState vision_state;
    bool has_vision_state = false;

    struct LLMLayer {
        ax_runner_t layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    // Optional per-layer attention type (for models like Qwen3.5 that mix linear/full attention).
    std::vector<bool> layer_is_linear_attn;
    std::vector<int> layer_kv_cache_sizes;
    std::vector<int> shared_kv_source_layers;
    // Use a full-attention layer as reference for token-wise KV cache shapes.
    int cache_ref_full_layer_idx = 0;
    ax_runner_t llama_post;

    int decode_grpid = 0;
    std::vector<int> decode_grpids_;             // sorted by decode capacity (ascending)
    std::vector<int> decode_max_token_len_grp_;  // same length as decode_grpids_
    std::vector<int> prefill_grpids_;            // sorted by prefill capacity (ascending), aligns with _attr.prefill_max_kv_cache_num_grp
    std::atomic<bool> b_stop{false};
    LLMPostprocess postprocess;
    std::string last_error_message;

    static std::string context_limit_user_message()
    {
        return "当前会话上下文已超过该模型的长度上限，请使用 /clean 清空历史并重新开始对话。";
    }

    static std::string video_quality_reset_notice()
    {
        return "提示：为保证视频解析质量，本次视频请求已按新会话处理，未使用此前对话上下文。\n\n";
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
                auto &k = lyr.layer.get_input(gid, "K_cache");
                auto &v = lyr.layer.get_input(gid, "V_cache");
                llm_memset(LLM_WADDR(k), 0, k.nSize, devid);
                llm_memset(LLM_WADDR(v), 0, v.nSize, devid);
            }
        }
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
        if (prefill_gids.empty())
        {
            ALOGE("no prefill groups detected");
            return false;
        }

        // Detect prefill_token_num from the first prefill group.
        // Prefer `indices` last-dim (most consistent across models/backends). Fall back to `mask` shape.
        int detected_prefill_token_num = 0;
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
        for (const int gid : prefill_gids)
        {
            int total_cap = -1;
            int history_cap = -1;
            int symbolic_cap = -1;
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

    std::vector<int> diff_token_ids(const std::vector<int> &ids1, const std::vector<int> &ids2, int &offset)
    {
        int min_len = (int)std::min(ids1.size(), ids2.size());
        offset = 0;
        for (int i = 0; i < min_len; i++) { if (ids1[i] == ids2[i]) offset++; else break; }
        if (offset >= (int)ids2.size()) return {};
        return std::vector<int>(ids2.begin() + offset, ids2.end());
    }

    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        this->_attr = attr;
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
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");

#ifdef USE_AXCL
        llama_layers.resize(attr.axmodel_num);
        auto dev_assign = distributeModels((int)_attr.dev_ids.size(), attr.axmodel_num);
        std::vector<int> rets(attr.axmodel_num, 0);

        // Prepare filenames first (thread-safe, no I/O).
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            char path[1024];
            std::snprintf(path, sizeof(path), attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = path;
        }

        // Load models in parallel across devices (per-device sequential), while the main thread updates progress.
        struct LoadResult {
            int idx = -1;
            int ret = -1;
            int devid = -1;
            int remain_mb = -1;
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
        std::vector<std::thread> loaders;
        loaders.reserve(_attr.dev_ids.size());

        for (size_t dev_idx = 0; dev_idx < _attr.dev_ids.size(); ++dev_idx)
        {
            const int devid = _attr.dev_ids[dev_idx];
            loaders.emplace_back([&, dev_idx, devid]() {
                for (const int i : models_per_dev[dev_idx])
                {
                    const int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), devid);
                    const int remain = axcl_GetCMMRemain(devid);

                    char buf[256];
                    std::snprintf(buf, sizeof(buf), "init %d axmodel ok,devid(%d) remain_cmm(%d MB)", i, devid, remain);

                    LoadResult r;
                    r.idx = i;
                    r.ret = ret;
                    r.devid = devid;
                    r.remain_mb = remain;
                    r.msg = buf;

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
        while (finished < attr.axmodel_num)
        {
            LoadResult r;
            {
                std::unique_lock<std::mutex> lk(q_mu);
                q_cv.wait(lk, [&]() { return !q.empty(); });
                r = std::move(q.front());
                q.pop();
            }
            if (r.idx >= 0 && r.idx < attr.axmodel_num) rets[r.idx] = r.ret;
            update_cqdm(&cqdm, progress_step++, "count", r.msg.c_str());
            finished++;
        }

        for (auto &t : loaders)
        {
            if (t.joinable()) t.join();
        }

        for (int i = 0; i < attr.axmodel_num; i++) { if (rets[i] != 0) { ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str()); return false; } }
        {
            int post_devid = llama_layers.back().layer.get_devid();
            int ret = llama_post.init(attr.filename_post_axmodel.c_str(), post_devid);
            if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
            char path[1024];
            sprintf(path, "init post axmodel ok,remain_cmm(%d MB)", axcl_GetCMMRemain(post_devid));
            update_cqdm(&cqdm, attr.axmodel_num + 1, "count", path);
        }
#else
        llama_layers.resize(attr.axmodel_num);
        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;
            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), -1);
            if (ret != 0) { ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str()); return false; }
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 1, "count", axmodel_path);
        }
        {
            int ret = llama_post.init(attr.filename_post_axmodel.c_str(), -1);
            if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
            update_cqdm(&cqdm, attr.axmodel_num + 1, "count", axmodel_path);
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
            layer_kv_cache_sizes.assign(_attr.axmodel_num, _attr.kv_cache_size);
            for (int i = 0; i < _attr.axmodel_num; ++i)
            {
                auto &layer_k = llama_layers[(size_t)i].layer.get_input(decode_grpid, "K_cache");
                const int layer_kv_size = (int)(layer_k.nSize / sizeof(unsigned short) / std::max(1, _attr.kv_cache_num));
                if (layer_kv_size > 0)
                    layer_kv_cache_sizes[(size_t)i] = layer_kv_size;
            }
        }

        // embed file check → then init
        {
            auto &t_in0 = llama_layers[0].layer.get_input(decode_grpid, "input");
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
        ALOGI("LLM init ok");
        return true;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++) llama_layers[i].layer.deinit();
        llama_post.deinit();
        embed_selector.Deinit();
        gemma4_per_layer_helper.Deinit();
        if (vision) vision->Deinit();
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

                // indices
                const auto &t_idx = lyr.layer.get_input(prefill_grpid, "indices");
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
                const auto &t_mask = lyr.layer.get_input(prefill_grpid, "mask");
                llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);

                // input
                const auto &t_in = lyr.layer.get_input(prefill_grpid, "input");
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);

                // inference
                lyr.layer.inference(prefill_grpid);

                // KV cache update
                const size_t kv_off = (size_t)history_len * (size_t)_attr.kv_cache_size;
                const size_t kv_sz = (size_t)input_num_token * (size_t)_attr.kv_cache_size * sizeof(unsigned short);
                const auto &out_k = lyr.layer.get_output(prefill_grpid, "K_cache_out");
                const auto &out_v = lyr.layer.get_output(prefill_grpid, "V_cache_out");
                const auto &in_k  = lyr.layer.get_input(prefill_grpid, "K_cache");
                const auto &in_v  = lyr.layer.get_input(prefill_grpid, "V_cache");
                llm_d2d((unsigned short *)LLM_WADDR(in_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(in_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);

                // output -> embed_tmp for next layer
                const auto &t_out = lyr.layer.get_output(prefill_grpid, "output");
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
                if (!vision->Prepare(history_in, vmins, nullptr, prepared_history, token_ids, st, verr))
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
        const int prefill_grpid = select_stateless_prefill_group(input_embed_num);
        int kv_cache_num = prefill_history_capacity_by_gid(prefill_grpid);
        const int kv_from_mask = prefill_history_capacity_by_mask(prefill_grpid);
        if (kv_from_mask >= 0) kv_cache_num = kv_from_mask;
        ALOGI("input token num : %d, prefill_split_num : %d prefill_grpid : %d kv_cache_num:%d", input_embed_num, prefill_split_num, prefill_grpid, kv_cache_num);
        if (prefill_grpid < 0 || kv_cache_num < 0)
        {
            ALOGE("invalid prefill_grpid=%d", prefill_grpid);
            return -1;
        }

        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i];
            int devid = LLM_DEVID(lyr);
            llm_memset(LLM_WADDR(lyr.layer.get_input(prefill_grpid, "K_cache")), 0, lyr.layer.get_input(prefill_grpid, "K_cache").nSize, devid);
            llm_memset(LLM_WADDR(lyr.layer.get_input(prefill_grpid, "V_cache")), 0, lyr.layer.get_input(prefill_grpid, "V_cache").nSize, devid);
        }

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

        std::vector<unsigned short> test_embed(_token_ids.size() * _attr.tokens_embed_size);
        for (size_t i = 0; i < _token_ids.size(); i++) embed_selector.getByIndex(_token_ids[i], test_embed.data() + i * _attr.tokens_embed_size);

        for (int p = 0; p < prefill_split_num; p++)
        {
            int input_num_token = (p == prefill_split_num - 1) ? input_embed_num - p * _attr.prefill_token_num : _attr.prefill_token_num;
            std::vector<unsigned short> mask_tmp(_attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            for (int i = 0; i < _attr.prefill_token_num; i++) if (i < input_num_token)
            {
                auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);
                for (int j = 0; j < p * _attr.prefill_token_num; j++) mask_ptr[j] = 0;
                int cur_start = kv_cache_num; for (int j = cur_start; j < cur_start + i + 1; j++) mask_ptr[j] = 0;
            }
            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            size_t copy_tokens = (p == prefill_split_num - 1) ? (size_t)(input_embed_num - p * _attr.prefill_token_num) : (size_t)_attr.prefill_token_num;
            memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                auto &lyr = llama_layers[m]; int devid = LLM_DEVID(lyr);
                auto &t_idx = lyr.layer.get_input(prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr; memset(idx_ptr, 0, t_idx.nSize);
                int idx_i = 0; for (int i = 0; i < input_num_token; ++i) idx_ptr[idx_i++] = (unsigned int)(p * _attr.prefill_token_num + i);
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);
                auto &t_mask = lyr.layer.get_input(prefill_grpid, "mask"); llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
                auto &t_in = lyr.layer.get_input(prefill_grpid, "input"); llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);
                lyr.layer.inference(prefill_grpid);
                auto &out_k  = lyr.layer.get_output(prefill_grpid, "K_cache_out");
                auto &out_v  = lyr.layer.get_output(prefill_grpid, "V_cache_out");
                auto &pre_k  = lyr.layer.get_input(prefill_grpid, "K_cache");
                auto &pre_v  = lyr.layer.get_input(prefill_grpid, "V_cache");
                const int layer_kv = kv_cache_size_for_layer(m);
                int kv_off = p * _attr.prefill_token_num * layer_kv;
                size_t kv_sz = (size_t)_attr.prefill_token_num * (size_t)layer_kv * sizeof(unsigned short);
                llm_d2d((unsigned short *)LLM_WADDR(pre_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(pre_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                auto &t_out = lyr.layer.get_output(prefill_grpid, "output"); llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), std::min((size_t)t_out.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);
            }
        }

        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i]; int devid = LLM_DEVID(lyr);
            const int layer_kv = kv_cache_size_for_layer(i);
            k_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
            v_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
            auto &t_k = lyr.layer.get_input(prefill_grpid, "K_cache");
            auto &t_v = lyr.layer.get_input(prefill_grpid, "V_cache");
            const size_t kv_bytes = (size_t)prefill_precompute_len * (size_t)layer_kv * sizeof(unsigned short);
            llm_d2h(k_caches[i].data(), LLM_RADDR(t_k), kv_bytes, devid);
            llm_d2h(v_caches[i].data(), LLM_RADDR(t_v), kv_bytes, devid);
        }
        return 0;
    }

    int GetKVCache(std::vector<std::vector<unsigned short>> &kv_k, std::vector<std::vector<unsigned short>> &kv_v, int &kv_precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        int inferred_precompute_len = 0;
        auto &t_mask = llama_layers[(size_t)cache_ref_full_layer_idx].layer.get_input(decode_grpid, "mask");
        std::vector<unsigned short> mask(t_mask.nSize / sizeof(unsigned short), bf16.data);
        llm_d2h(mask.data(), LLM_RADDR(t_mask), t_mask.nSize, LLM_DEVID(llama_layers[(size_t)cache_ref_full_layer_idx]));
        for (size_t i = 0; i < mask.size(); i++) { if (mask[i] == bf16.data) { inferred_precompute_len = (int)i + 1; break; } }
        kv_precompute_len = precompute_len > 0 ? precompute_len : inferred_precompute_len;
        ALOGI("precompute_len:%d, remaining:%d%s",
              kv_precompute_len,
              _attr.prefill_max_kv_cache_num_grp.back() - kv_precompute_len,
              precompute_len > 0 ? " (tracked)" : " (mask inferred)");
        if (b_os_kvcache)
        {
            kv_k.resize(_attr.axmodel_num); kv_v.resize(_attr.axmodel_num);
            for (int i = 0; i < _attr.axmodel_num; i++)
            {
                auto &lyr = llama_layers[i]; int devid = LLM_DEVID(lyr);
                const int layer_kv = kv_cache_size_for_layer(i);
                kv_k[i].resize((size_t)kv_precompute_len * (size_t)layer_kv);
                kv_v[i].resize((size_t)kv_precompute_len * (size_t)layer_kv);
                auto &t_k = lyr.layer.get_input(decode_grpid, "K_cache"); auto &t_v = lyr.layer.get_input(decode_grpid, "V_cache");
                const size_t kv_bytes = (size_t)kv_precompute_len * (size_t)layer_kv * sizeof(unsigned short);
                llm_d2h(kv_k[i].data(), LLM_RADDR(t_k), kv_bytes, devid);
                llm_d2h(kv_v[i].data(), LLM_RADDR(t_v), kv_bytes, devid);
            }
        }
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
        if (_precompute_len == 0) { clear_all_group_kv_cache_tensors(); ALOGI("first run"); return 0; }
        if (!b_os_kvcache) return 0;
        if (kv_k.size() != kv_v.size() || (int)kv_k.size() != _attr.axmodel_num) {
            set_last_error("模型运行失败，请重新尝试。");
            ALOGE("kv cache size mismatch");
            return -1;
        }
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr  = llama_layers[i]; int devid = LLM_DEVID(lyr);
            auto &dk = lyr.layer.get_input(decode_grpid, "K_cache"); auto &dv = lyr.layer.get_input(decode_grpid, "V_cache");
            llm_memset(LLM_WADDR(dk), 0, dk.nSize, devid); llm_memset(LLM_WADDR(dv), 0, dv.nSize, devid);
            auto &pk = lyr.layer.get_input(_attr.prefill_grpid, "K_cache"); auto &pv = lyr.layer.get_input(_attr.prefill_grpid, "V_cache");
            llm_memset(LLM_WADDR(pk), 0, pk.nSize, devid); llm_memset(LLM_WADDR(pv), 0, pv.nSize, devid);
        }
        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            auto &lyr  = llama_layers[m]; int devid = LLM_DEVID(lyr);
            auto &kc = kv_k[m]; auto &vc = kv_v[m];
            const int layer_kv = kv_cache_size_for_layer(m);
            const size_t kv_elems = (size_t)_precompute_len * (size_t)layer_kv;
            const size_t kv_bytes = kv_elems * sizeof(unsigned short);
            if (kc.size() < kv_elems || vc.size() < kv_elems) {
                set_last_error("模型运行失败，请重新尝试。");
                ALOGE("kv_cache buffer too small for layer %d", m);
                return -1;
            }
            auto &dk = lyr.layer.get_input(decode_grpid, "K_cache"); auto &dv = lyr.layer.get_input(decode_grpid, "V_cache");
            llm_h2d(LLM_WADDR(dk), kc.data(), kv_bytes, devid); llm_h2d(LLM_WADDR(dv), vc.data(), kv_bytes, devid);
            auto &pk = lyr.layer.get_input(_attr.prefill_grpid, "K_cache"); auto &pv = lyr.layer.get_input(_attr.prefill_grpid, "V_cache");
            llm_h2d(LLM_WADDR(pk), kc.data(), kv_bytes, devid); llm_h2d(LLM_WADDR(pv), vc.data(), kv_bytes, devid);
        }
        return 0;
    }

    void ResetKVCache()
    {
        last_tokens_ids.clear(); last_history_snapshot.clear(); run_input_token_ids.clear(); k_caches.clear(); v_caches.clear(); precompute_len = 0;
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
        int max_prefill_gid = default_prefill_gid;
        for (int p = 0; p < prefill_split_num; ++p) {
            const int history_len = precompute_len + p * _attr.prefill_token_num;
            const int chunk_tokens = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;
            const bool prefer_symbolic_group = has_vision_state && history_len > 0;
            int g = select_prefill_group(history_len, chunk_tokens, prefer_symbolic_group);
            if (g <= 0)
            {
                ALOGE("failed to select prefill group for history_len=%d chunk_tokens=%d", history_len, chunk_tokens);
                return final_out;
            }
            prefill_grp_list[p] = g;
            if (g > max_prefill_gid) max_prefill_gid = g;
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
                    const int abs_pos = precompute_len + i;
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

            const int prefill_grpid = prefill_grp_list[p];
            int kv_cache_num_p = prefill_history_capacity_by_gid(prefill_grpid);
            {
                const auto &mask_t = llama_layers[(size_t)cache_ref_full_layer_idx].layer.get_input(prefill_grpid, "mask");
                const int mask_elems = (int)(mask_t.nSize / (int)sizeof(unsigned short));
                if (_attr.prefill_token_num > 0 && mask_elems > 0 && (mask_elems % _attr.prefill_token_num) == 0) {
                    const int cols = mask_elems / _attr.prefill_token_num;
                    const int kv_from_mask = cols - _attr.prefill_token_num;
                    if (kv_from_mask >= 0) kv_cache_num_p = kv_from_mask;
                }
            }
            ALOGI("prefill chunk p=%d history_len=%d grpid=%d kv_cache_num=%d input_tokens=%d",
                  p, history_len, prefill_grpid, kv_cache_num_p, input_num_token);

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            std::vector<unsigned short> mask_tmp(_attr.prefill_token_num * (kv_cache_num_p + _attr.prefill_token_num), bf16.data);
            std::vector<unsigned short> linear_mask_tmp;
            if (use_per_layer_input) per_layer_tmp.assign((size_t)_attr.prefill_token_num * (size_t)per_layer_hidden, 0);

            size_t copy_tokens = (p == prefill_split_num - 1) ? (size_t)(input_embed_num - p * _attr.prefill_token_num) : (size_t)_attr.prefill_token_num;
            memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));
            scale_prefill_text_embeds_inplace(embed_tmp.data(), input_num_token, history_len);

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break;
                auto &lyr   = llama_layers[m]; int devid = LLM_DEVID(lyr);
                auto &t_idx = lyr.layer.get_input(prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr; memset(idx_ptr, 0, t_idx.nSize);
                {
                    const int start_pos = history_len;
                    const int idx_elems = (int)(t_idx.nSize / (int)sizeof(unsigned int));
                    int idx_rows = idx_elems / _attr.prefill_token_num;
                    if (idx_rows <= 0) idx_rows = 1;
                    if (m == 0) {
                        ALOGI("prefill indices shape: p=%d idx_elems=%d idx_rows=%d pos_rows=%zu",
                              p, idx_elems, idx_rows, vision_state.position_ids.size());
                    }

                    const bool use_pos_ids = has_vision_state &&
                                             idx_rows >= 3 &&
                                             vision_state.position_ids.size() >= 3 &&
                                             !vision_state.position_ids.empty();

                    for (int r = 0; r < idx_rows; ++r)
                    {
                        for (int j = 0; j < input_num_token; ++j)
                        {
                            unsigned int v = (unsigned int)(start_pos + j);
                            if (use_pos_ids)
                            {
                                if ((size_t)r < vision_state.position_ids.size())
                                {
                                    const auto &row = vision_state.position_ids[r];
                                    if ((size_t)(start_pos + j) < row.size())
                                        v = (unsigned int)row[start_pos + j];
                                }
                            }
                            idx_ptr[r * _attr.prefill_token_num + j] = v;
                        }
                    }
                }
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);
                auto &t_mask = lyr.layer.get_input(prefill_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    linear_mask_tmp.assign(elems, 0);
                    const int n = std::min((int)elems, input_num_token);
                    for (int i = 0; i < n; ++i) linear_mask_tmp[(size_t)i] = bf16_one.data;
                    llm_h2d(LLM_WADDR(t_mask), linear_mask_tmp.data(), linear_mask_tmp.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    build_layer_prefill_mask(mask_tmp, kv_cache_num_p, _attr.prefill_token_num, history_len, input_num_token, m);
                    llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
                }
                auto &t_in = lyr.layer.get_input(prefill_grpid, "input"); llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), devid);
                const int shared_src = shared_kv_source_for_layer(m);
                if (shared_src >= 0)
                {
                    auto &src_layer = llama_layers[(size_t)shared_src];
                    auto &src_k = src_layer.layer.get_input(decode_grpid, "K_cache");
                    auto &src_v = src_layer.layer.get_input(decode_grpid, "V_cache");
                    auto &dst_k = lyr.layer.get_input(prefill_grpid, "K_cache");
                    auto &dst_v = lyr.layer.get_input(prefill_grpid, "V_cache");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    copy_shared_prefill_cache(dst_k,
                                              src_k,
                                              layer_kv,
                                              history_len,
                                              kv_cache_num_p,
                                              history_len,
                                              input_num_token,
                                              devid);
                    copy_shared_prefill_cache(dst_v,
                                              src_v,
                                              layer_kv,
                                              history_len,
                                              kv_cache_num_p,
                                              history_len,
                                              input_num_token,
                                              devid);
                }
                if (use_per_layer_input)
                {
                    const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, prefill_grpid, "per_layer_input");
                    if (!t_per_layer)
                    {
                        ALOGE("Gemma4 decoder layer %d is missing per_layer_input for group %d", m, prefill_grpid);
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
                lyr.layer.inference(prefill_grpid);
                auto &out_k = lyr.layer.get_output(prefill_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(prefill_grpid, "V_cache_out");
                auto &dec_k = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &dec_v = lyr.layer.get_input(decode_grpid, "V_cache");
                if (is_linear_layer(m))
                {
                    const size_t kbytes = std::min((size_t)dec_k.nSize, (size_t)out_k.nSize);
                    const size_t vbytes = std::min((size_t)dec_v.nSize, (size_t)out_v.nSize);
                    llm_d2d(LLM_WADDR(dec_k), LLM_RADDR(out_k), kbytes, devid);
                    llm_d2d(LLM_WADDR(dec_v), LLM_RADDR(out_v), vbytes, devid);

                    const int ng = (int)lyr.layer.get_num_input_groups();
                    const int max_gid = std::min(max_prefill_gid, ng - 1);
                    for (int gid = prefill_grpid + 1; gid <= max_gid; ++gid)
                    {
                        auto &gk = lyr.layer.get_input(gid, "K_cache");
                        auto &gv = lyr.layer.get_input(gid, "V_cache");
                        llm_d2d(LLM_WADDR(gk), LLM_RADDR(out_k), std::min((size_t)gk.nSize, (size_t)out_k.nSize), devid);
                        llm_d2d(LLM_WADDR(gv), LLM_RADDR(out_v), std::min((size_t)gv.nSize, (size_t)out_v.nSize), devid);
                    }
                }
                else
                {
                    const int layer_kv = kv_cache_size_for_layer(m);
                    int kv_off = history_len * layer_kv;
                    size_t kv_sz = (size_t)input_num_token * (size_t)layer_kv * sizeof(unsigned short);
                    llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                    llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                    const int ng = (int)lyr.layer.get_num_input_groups();
                    const int max_gid = std::min(max_prefill_gid, ng - 1);
                    for (int gid = prefill_grpid + 1; gid <= max_gid; ++gid) {
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

                auto &t_out = lyr.layer.get_output(prefill_grpid, "output");
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), embed_tmp.size() * sizeof(unsigned short), devid);

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
            llm_h2d(LLM_WADDR(t_in), embed.data(), embed.size() * sizeof(unsigned short), LLM_DEVID(llama_layers.back()));
            llama_post.inference();
            auto &t_out = llama_post.get_output("output");
            llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
            next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
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
            }
        }

        t_cost.start();
        unsigned int decode_start = (unsigned int)(precompute_len + input_embed_num);
        if (has_vision_state && vision_state.decode_start > 0) decode_start = (unsigned int)vision_state.decode_start;
        for (unsigned int indices = decode_start; !b_hit_eos && indices < (unsigned int)_attr.max_token_len; indices++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;
            bool need_full_shared_sync = false;
            {
                const int want_gid = choose_decode_gid((int)indices + 1);
                if (want_gid != decode_grpid)
                {
                    ALOGI("switch decode_grpid: %d -> %d (ctx=%u)", decode_grpid, want_gid, indices + 1);
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
                auto &l0_in = llama_layers[0].layer.get_input(decode_grpid, "input");
                llm_h2d(LLM_WADDR(l0_in), embed.data(), l0_in.nSize, llama_layers[0].layer.get_devid());
            }
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break; auto &lyr = llama_layers[m]; int devid = lyr.layer.get_devid();
                auto &in_k = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &in_v = lyr.layer.get_input(decode_grpid, "V_cache");
                const int shared_src = shared_kv_source_for_layer(m);
                if (shared_src >= 0)
                {
                    auto &src_layer = llama_layers[(size_t)shared_src];
                    auto &src_k = src_layer.layer.get_input(decode_grpid, "K_cache");
                    auto &src_v = src_layer.layer.get_input(decode_grpid, "V_cache");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                    const size_t visible_past = std::min((size_t)indices, dst_tokens > 0 ? (dst_tokens - 1) : 0);
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
                        const size_t cur_off_bytes = (size_t)indices * (size_t)layer_kv * sizeof(unsigned short);
                        const size_t dst_off_bytes = (dst_tokens - 1) * (size_t)layer_kv * sizeof(unsigned short);
                        llm_d2d((unsigned char *)LLM_WADDR(in_k) + dst_off_bytes,
                                (const unsigned char *)LLM_RADDR(src_k) + cur_off_bytes,
                                sizeof(unsigned short) * (size_t)layer_kv, devid);
                        llm_d2d((unsigned char *)LLM_WADDR(in_v) + dst_off_bytes,
                                (const unsigned char *)LLM_RADDR(src_v) + cur_off_bytes,
                                sizeof(unsigned short) * (size_t)layer_kv, devid);
                    }
                }
                auto &t_idx = lyr.layer.get_input(decode_grpid, "indices"); llm_h2d(LLM_WADDR(t_idx), &indices, sizeof(indices), devid);
                auto &t_mask= lyr.layer.get_input(decode_grpid, "mask");
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
                    build_layer_decode_mask(decode_mask, mask_elems, (int)indices, m);
                    llm_h2d(LLM_WADDR(t_mask),
                            decode_mask.data(),
                            std::min((size_t)t_mask.nSize, (size_t)mask_elems * sizeof(unsigned short)),
                            devid);
                }
                if (use_per_layer_input)
                {
                    const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, decode_grpid, "per_layer_input");
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
                lyr.layer.inference(decode_grpid);
                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out"); auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                if (is_linear_layer(m))
                {
                    llm_d2d(LLM_WADDR(in_k), LLM_RADDR(out_k), std::min((size_t)in_k.nSize, (size_t)out_k.nSize), devid);
                    llm_d2d(LLM_WADDR(in_v), LLM_RADDR(out_v), std::min((size_t)in_v.nSize, (size_t)out_v.nSize), devid);
                }
                else
                {
                    const int layer_kv = kv_cache_size_for_layer(m);
                    if (shared_src >= 0)
                    {
                        const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                        if (dst_tokens > 0)
                        {
                            const size_t cur_off = (size_t)indices * (size_t)layer_kv;
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
                        llm_d2d((unsigned short *)LLM_WADDR(in_k) + indices * layer_kv, LLM_RADDR(out_k), std::min((size_t)out_k.nSize, (size_t)layer_kv * sizeof(unsigned short)), devid);
                        llm_d2d((unsigned short *)LLM_WADDR(in_v) + indices * layer_kv, LLM_RADDR(out_v), std::min((size_t)out_v.nSize, (size_t)layer_kv * sizeof(unsigned short)), devid);
                    }
                }
                auto &cur_out = lyr.layer.get_output(decode_grpid, "output");
                if (m == _attr.axmodel_num - 1)
                {
                    auto &post_in = llama_post.get_input("input");
                    if (llama_post.get_devid() == devid) { llm_d2d(LLM_WADDR(post_in), LLM_RADDR(cur_out), post_in.nSize, devid); }
                    else { llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid); llm_h2d(LLM_WADDR(post_in), cur_out.pVirAddr, post_in.nSize, llama_post.get_devid()); }
                }
                else
                {
                    auto &next_in = llama_layers[m + 1].layer.get_input(decode_grpid, "input"); int next_devid = llama_layers[m + 1].layer.get_devid();
                    if (next_devid == devid) { llm_d2d(LLM_WADDR(next_in), LLM_RADDR(cur_out), next_in.nSize, devid); }
                    else { llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid); llm_h2d(LLM_WADDR(next_in), cur_out.pVirAddr, next_in.nSize, next_devid); }
                }
            }
            llama_post.inference();
            {
                auto &t_out = llama_post.get_output("output"); llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
                unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            }
#else // AX650
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break; auto &lyr = llama_layers[m];
                auto &in_k = lyr.layer.get_input(decode_grpid, "K_cache"); auto *in_k_ptr = (unsigned short *)in_k.pVirAddr;
                auto &in_v = lyr.layer.get_input(decode_grpid, "V_cache"); auto *in_v_ptr = (unsigned short *)in_v.pVirAddr;
                const int shared_src = shared_kv_source_for_layer(m);
                const auto shared_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                if (shared_src >= 0)
                {
                    auto &src_layer = llama_layers[(size_t)shared_src];
                    auto &src_k = src_layer.layer.get_input(decode_grpid, "K_cache");
                    auto &src_v = src_layer.layer.get_input(decode_grpid, "V_cache");
                    const int layer_kv = kv_cache_size_for_layer(m);
                    const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                    if (need_full_shared_sync)
                    {
                        const size_t visible_past = std::min((size_t)indices, dst_tokens > 0 ? (dst_tokens - 1) : 0);
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
                        const size_t cur_off = (size_t)indices * (size_t)layer_kv;
                        const size_t dst_off = (dst_tokens - 1) * (size_t)layer_kv;
                        memcpy(in_k_ptr + dst_off, (const unsigned short *)src_k.pVirAddr + cur_off, sizeof(unsigned short) * (size_t)layer_kv);
                        memcpy(in_v_ptr + dst_off, (const unsigned short *)src_v.pVirAddr + cur_off, sizeof(unsigned short) * (size_t)layer_kv);
                    }
                }
                if (decode_profile_enabled)
                    decode_profile.shared_kv_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - shared_t0).count();
                const auto prep_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                auto &t_idx = lyr.layer.get_input(decode_grpid, "indices"); memcpy(t_idx.pVirAddr, &indices, sizeof(indices));
                auto &t_mask= lyr.layer.get_input(decode_grpid, "mask");
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
                    build_layer_decode_mask(decode_mask, mask_elems, (int)indices, m);
                    memcpy(t_mask.pVirAddr,
                           decode_mask.data(),
                           std::min((size_t)t_mask.nSize, (size_t)mask_elems * sizeof(unsigned short)));
                }
                if (use_per_layer_input)
                {
                    const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, decode_grpid, "per_layer_input");
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
                auto &t_in  = lyr.layer.get_input(decode_grpid, "input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                if (decode_profile_enabled)
                    decode_profile.prep_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - prep_t0).count();
                const auto infer_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                lyr.layer.inference(decode_grpid);
                if (decode_profile_enabled)
                    decode_profile.inference_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - infer_t0).count();
                const auto out_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                if (is_linear_layer(m))
                {
                    memcpy(in_k.pVirAddr, out_k.pVirAddr, std::min((size_t)in_k.nSize, (size_t)out_k.nSize));
                    memcpy(in_v.pVirAddr, out_v.pVirAddr, std::min((size_t)in_v.nSize, (size_t)out_v.nSize));
                }
                else
                {
                    const int layer_kv = kv_cache_size_for_layer(m);
                    if (shared_src >= 0)
                    {
                        const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                        if (dst_tokens > 0)
                        {
                            const size_t cur_off = (size_t)indices * (size_t)layer_kv;
                            const size_t tail_off = (dst_tokens - 1) * (size_t)layer_kv;
                            // See the AXCL branch above: shared layers must retain the
                            // source layer KV in their visible past cache.
                            memcpy(in_k_ptr + cur_off, in_k_ptr + tail_off, sizeof(unsigned short) * (size_t)layer_kv);
                            memcpy(in_v_ptr + cur_off, in_v_ptr + tail_off, sizeof(unsigned short) * (size_t)layer_kv);
                        }
                    }
                    else
                    {
                        memcpy(in_k_ptr + indices * layer_kv, out_k.pVirAddr, sizeof(unsigned short) * layer_kv);
                        memcpy(in_v_ptr + indices * layer_kv, out_v.pVirAddr, sizeof(unsigned short) * layer_kv);
                    }
                }
                auto &t_out= lyr.layer.get_output(decode_grpid, "output"); memcpy(embed.data(), t_out.pVirAddr, embed.size() * sizeof(unsigned short));
                if (decode_profile_enabled)
                    decode_profile.outcopy_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - out_t0).count();
            }
            const auto post_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
            auto &t_in = llama_post.get_input("input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference(); auto &t_out = llama_post.get_output("output");
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
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
            if (_attr.runing_callback == nullptr) update_cqdm(&cqdm, indices, "token", "");
        }

        const int generated_token_count = (int)token_ids.size();
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
            precompute_len = decode_start + generated_token_count;
        }
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

        const bool raw_history_not_append = !last_history_snapshot.empty() && !is_history_prefix(last_history_snapshot, history);
        if (raw_history_not_append)
        {
            ALOGW("raw history not append. force ResetKVCache before request processing.");
            ResetKVCache();
        }

        std::vector<int> new_tokens;
        std::string response_prefix;

        if (vision && vision->enabled())
        {
            // If caller provides media, we will fill num_media/num_media_tokens and build injection state.
            if (!media_inputs.empty())
            {
                std::vector<vision::MediaInputs> vmins;
                vmins.reserve(media_inputs.size());
                for (const auto &m : media_inputs) vmins.push_back({m.content_index, m.uris});

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
                if (!vision->Prepare(history, vmins, &budget, prepared_history, input_ids, st, verr, &prepare_meta))
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
        else
        {
            new_tokens = tokenizer->encode(history);
        }

        int offset = 0;
        auto tokens_diff = diff_token_ids(last_tokens_ids, new_tokens, offset);
        bool not_append = !(offset == (int)last_tokens_ids.size() && (int)new_tokens.size() >= (int)last_tokens_ids.size());
        if (not_append) { ALOGW("history not append (rollback/modify). force ResetKVCache and recompute."); ResetKVCache(); tokens_diff = new_tokens; offset = 0; }
        if (tokens_diff.empty())
        {
            if (!new_tokens.empty()) { precompute_len = (int)new_tokens.size() - 1; tokens_diff = {new_tokens.back()}; }
            else { ResetKVCache(); precompute_len = 0; }
        }
        const int kv_ret = SetKVCache(k_caches, v_caches, precompute_len, (int)tokens_diff.size());
        if (kv_ret != 0)
        {
            ALOGE("SetKVCache failed");
            return history;
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
        run_input_token_ids = tokens_diff;
        auto reply = Run(out_embed, output_max_token);
        run_input_token_ids.clear();
        if (!response_prefix.empty())
        {
            reply = response_prefix + reply;
        }
        history.push_back({ASSISTANT, TEXT, reply});
        last_history_snapshot = history;
        last_tokens_ids = tokenizer->encode(history);
        if (last_tokens_ids.size() >= 2) last_tokens_ids.erase(last_tokens_ids.end() - 2, last_tokens_ids.end());
        GetKVCache(k_caches, v_caches, precompute_len);

        has_vision_state = false;
        vision_state = {};

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
void LLM::SetRequestSamplingOverride(bool has_temperature, float temperature, bool has_top_p, float top_p) { impl_->postprocess.set_request_sampling_override(has_temperature, temperature, has_top_p, top_p); }
void LLM::ClearRequestSamplingOverride() { impl_->postprocess.clear_request_sampling_override(); }
std::string LLM::GetLastError() const { return impl_->get_last_error(); }

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
