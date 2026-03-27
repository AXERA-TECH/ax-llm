#include "LLM.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>

#include "bfloat16.hpp"
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

struct LLM::Impl {
    UTF8Filter utf8_filter;
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;

    std::vector<int> last_tokens_ids;
    bool b_os_kvcache = false;
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    int precompute_len = 0;

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
    // Use a full-attention layer as reference for token-wise KV cache shapes.
    int cache_ref_full_layer_idx = 0;
    ax_runner_t llama_post;

    int decode_grpid = 0;
    std::atomic<bool> b_stop{false};
    LLMPostprocess postprocess;

    // ---- small helpers ----
    static int post_process(LLMPostprocess &postprocess, unsigned short *p, int n, std::vector<int> &history, float *val = 0)
    {
        std::vector<float> logits(n);
        for (int i = 0; i < n; i++)
        {
            unsigned int proc = p[i] << 16;
            logits[i] = *reinterpret_cast<float *>(&proc);
        }
        return postprocess.apply(logits, history);
    }

    static inline void fill_indices(unsigned int *dst, int start, int count)
    { for (int i = 0; i < count; ++i) dst[i] = (unsigned int)(start + i); }

    static inline void build_prefill_mask(std::vector<unsigned short> &mask_tmp,
                                          int kv_cache_num,
                                          int token_rows,
                                          int history_len,
                                          int valid_rows)
    {
        bfloat16 bf16 = -65536.f;
        std::fill(mask_tmp.begin(), mask_tmp.end(), bf16.data);
        const int rows = std::max(0, std::min(token_rows, valid_rows));
        for (int r = 0; r < rows; ++r) {
            auto row = mask_tmp.data() + r * (kv_cache_num + token_rows);
            for (int j = 0; j < history_len; ++j) row[j] = 0;
            int cur = kv_cache_num; for (int j = cur; j < cur + r + 1; ++j) row[j] = 0;
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

    void init_layer_types()
    {
        layer_is_linear_attn.assign(_attr.axmodel_num, false);
        const int interval = _attr.full_attention_interval;
        if (interval <= 0) return;
        for (int i = 0; i < _attr.axmodel_num; ++i)
        {
            const bool is_full = (((i + 1) % interval) == 0);
            layer_is_linear_attn[i] = !is_full;
        }
    }

    bool is_linear_layer(int layer_idx) const
    {
        return layer_idx >= 0 &&
               layer_idx < (int)layer_is_linear_attn.size() &&
               layer_is_linear_attn[(size_t)layer_idx];
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
        cache_ref_full_layer_idx = first_full_layer_idx();
        if (cache_ref_full_layer_idx < 0) cache_ref_full_layer_idx = 0;
        if (_attr.full_attention_interval > 0)
        {
            ALOGI("mixed attention enabled: full_attention_interval=%d ref_full_layer_idx=%d",
                  _attr.full_attention_interval,
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
        tokenizer->set_think_in_prompt(true);
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
            _attr.prefill_token_num = ref_layer.get_input(1, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
            _attr.prefill_max_kv_cache_num_grp.clear();
            for (size_t i = 0; i < ref_layer.get_num_input_groups() - 1; i++)
            {
                int n = ref_layer.get_input((int)i + 1, "K_cache").vShape[1];
                ALOGI("grp: %zu, prefill_max_kv_cache_num : %d", i + 1, n);
                _attr.prefill_max_kv_cache_num_grp.push_back(n);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();
            _attr.prefill_grpid = (int)_attr.prefill_max_kv_cache_num_grp.size();
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
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
        ALOGI("LLM init ok");
        return true;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++) llama_layers[i].layer.deinit();
        llama_post.deinit();
        embed_selector.Deinit();
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

        const int input_embed_num = (int)token_ids.size();
        if (_attr.prefill_token_num <= 0 || _attr.tokens_embed_size <= 0 || _attr.kv_cache_size <= 0)
        {
            ALOGE("LLM embedding not initialized correctly (prefill_token_num/embed_size/kv_cache_size)");
            return false;
        }

        const int prefill_split_num = (int)std::ceil((double)input_embed_num / (double)_attr.prefill_token_num);

        int prefill_grpid = (int)_attr.prefill_max_kv_cache_num_grp.size();
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (input_embed_num <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                prefill_grpid = (int)i + 1;
                break;
            }
        }
        if (prefill_grpid <= 0)
        {
            ALOGE("invalid prefill_grpid=%d", prefill_grpid);
            return false;
        }

        // Clear KV caches for this run (embedding is stateless).
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i];
            const int devid = LLM_DEVID(lyr);
            const auto &k = lyr.layer.get_input(prefill_grpid, "K_cache");
            const auto &v = lyr.layer.get_input(prefill_grpid, "V_cache");
            llm_memset(LLM_WADDR(k), 0, (size_t)k.nSize, devid);
            llm_memset(LLM_WADDR(v), 0, (size_t)v.nSize, devid);
        }

        const int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];

        std::vector<unsigned short> all_embed((size_t)input_embed_num * (size_t)_attr.tokens_embed_size);
        for (int i = 0; i < input_embed_num; i++)
        {
            embed_selector.getByIndex((unsigned int)token_ids[(size_t)i], all_embed.data() + (size_t)i * (size_t)_attr.tokens_embed_size);
        }

        std::vector<unsigned short> last_hidden((size_t)_attr.tokens_embed_size, 0);
        std::vector<unsigned short> embed_tmp((size_t)_attr.prefill_token_num * (size_t)_attr.tokens_embed_size, 0);
        std::vector<unsigned short> mask_tmp((size_t)_attr.prefill_token_num * (size_t)(kv_cache_num + _attr.prefill_token_num), bfloat16(-65536.f).data);

        for (int p = 0; p < prefill_split_num; p++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;

            const int history_len = p * _attr.prefill_token_num;
            const int input_num_token = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;

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
                fill_indices(idx_ptr, history_len, _attr.prefill_token_num);
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, (size_t)t_idx.nSize, devid);

                // mask
                const auto &t_mask = lyr.layer.get_input(prefill_grpid, "mask");
                llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short), devid);

                // input
                const auto &t_in = lyr.layer.get_input(prefill_grpid, "input");
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), devid);

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
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), embed_tmp.size() * sizeof(unsigned short), devid);
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

        // For now, use last hidden state directly as embeddings (common for Qwen3-Embedding).
        out_embedding.resize((size_t)_attr.tokens_embed_size);
        for (int i = 0; i < _attr.tokens_embed_size; i++)
        {
            out_embedding[(size_t)i] = bfloat16(last_hidden[(size_t)i]).fp32();
        }
        out_embedding = l2norm(std::move(out_embedding));
        return true;
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
        int prefill_grpid = (int)_attr.prefill_max_kv_cache_num_grp.size();
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++) { if (input_embed_num <= _attr.prefill_max_kv_cache_num_grp[i]) { prefill_grpid = (int)i + 1; break; } }
        ALOGI("input token num : %d, prefill_split_num : %d prefill_grpid : %d", input_embed_num, prefill_split_num, prefill_grpid);

        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i];
            int devid = LLM_DEVID(lyr);
            llm_memset(LLM_WADDR(lyr.layer.get_input(prefill_grpid, "K_cache")), 0, lyr.layer.get_input(prefill_grpid, "K_cache").nSize, devid);
            llm_memset(LLM_WADDR(lyr.layer.get_input(prefill_grpid, "V_cache")), 0, lyr.layer.get_input(prefill_grpid, "V_cache").nSize, devid);
        }

        if (input_embed_num == 0)
        {
            for (int i = 0; i < _attr.axmodel_num; i++) { k_caches[i].resize(prefill_precompute_len * _attr.kv_cache_size); v_caches[i].resize(prefill_precompute_len * _attr.kv_cache_size); }
            ALOGI("input token num is 0, skip");
            return 0;
        }

        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];
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
                auto &t_mask = lyr.layer.get_input(prefill_grpid, "mask"); llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short), devid);
                auto &t_in = lyr.layer.get_input(prefill_grpid, "input"); llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), devid);
                lyr.layer.inference(prefill_grpid);
                auto &out_k  = lyr.layer.get_output(prefill_grpid, "K_cache_out");
                auto &out_v  = lyr.layer.get_output(prefill_grpid, "V_cache_out");
                auto &pre_k  = lyr.layer.get_input(prefill_grpid, "K_cache");
                auto &pre_v  = lyr.layer.get_input(prefill_grpid, "V_cache");
                int kv_off = p * _attr.prefill_token_num * _attr.kv_cache_size;
                size_t kv_sz = (size_t)_attr.prefill_token_num * _attr.kv_cache_size * sizeof(unsigned short);
                llm_d2d((unsigned short *)LLM_WADDR(pre_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(pre_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                auto &t_out = lyr.layer.get_output(prefill_grpid, "output"); llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), embed_tmp.size() * sizeof(unsigned short), devid);
            }
        }

        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i]; int devid = LLM_DEVID(lyr);
            k_caches[i].resize(prefill_precompute_len * _attr.kv_cache_size);
            v_caches[i].resize(prefill_precompute_len * _attr.kv_cache_size);
            auto &t_k = lyr.layer.get_input(prefill_grpid, "K_cache");
            auto &t_v = lyr.layer.get_input(prefill_grpid, "V_cache");
            llm_d2h(k_caches[i].data(), LLM_RADDR(t_k), prefill_precompute_len * _attr.kv_cache_size * sizeof(unsigned short), devid);
            llm_d2h(v_caches[i].data(), LLM_RADDR(t_v), prefill_precompute_len * _attr.kv_cache_size * sizeof(unsigned short), devid);
        }
        return 0;
    }

    int GetKVCache(std::vector<std::vector<unsigned short>> &kv_k, std::vector<std::vector<unsigned short>> &kv_v, int &kv_precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        auto &t_mask = llama_layers[(size_t)cache_ref_full_layer_idx].layer.get_input(decode_grpid, "mask");
        std::vector<unsigned short> mask(t_mask.nSize / sizeof(unsigned short), bf16.data);
        llm_d2h(mask.data(), LLM_RADDR(t_mask), t_mask.nSize, LLM_DEVID(llama_layers[(size_t)cache_ref_full_layer_idx]));
        kv_precompute_len = 0; for (size_t i = 0; i < mask.size(); i++) { if (mask[i] == bf16.data) { kv_precompute_len = (int)i + 1; break; } }
        ALOGI("precompute_len:%d, remaining:%d", kv_precompute_len, _attr.prefill_max_kv_cache_num_grp.back() - kv_precompute_len);
        if (b_os_kvcache)
        {
            kv_k.resize(_attr.axmodel_num); kv_v.resize(_attr.axmodel_num);
            for (int i = 0; i < _attr.axmodel_num; i++)
            {
                auto &lyr = llama_layers[i]; int devid = LLM_DEVID(lyr);
                kv_k[i].resize(kv_precompute_len * _attr.kv_cache_size); kv_v[i].resize(kv_precompute_len * _attr.kv_cache_size);
                auto &t_k = lyr.layer.get_input(decode_grpid, "K_cache"); auto &t_v = lyr.layer.get_input(decode_grpid, "V_cache");
                llm_d2h(kv_k[i].data(), LLM_RADDR(t_k), kv_precompute_len * _attr.kv_cache_size * sizeof(unsigned short), devid);
                llm_d2h(kv_v[i].data(), LLM_RADDR(t_v), kv_precompute_len * _attr.kv_cache_size * sizeof(unsigned short), devid);
            }
        }
        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();
        return 0;
    }

    int SetKVCache(std::vector<std::vector<unsigned short>> &kv_k,
                   std::vector<std::vector<unsigned short>> &kv_v,
                   int _precompute_len, int input_num_token)
    {
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (_precompute_len + input_num_token <= _attr.prefill_max_kv_cache_num_grp[i]) { _attr.prefill_grpid = (int)i + 1; break; }
        }
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];
        ALOGI("prefill_grpid:%d kv_cache_num:%d precompute_len:%d input_num_token:%d", _attr.prefill_grpid, kv_cache_num, _precompute_len, input_num_token);
        _attr.prefill_max_token_num = ALIGN_DOWN(_attr.prefill_max_token_num - _precompute_len, _attr.prefill_token_num);
        ALOGI("current prefill_max_token_num:%d", _attr.prefill_max_token_num);
        if (_precompute_len + input_num_token > kv_cache_num) { ALOGE("precompute_len(%d) + input_num_token(%d) > kv_cache_num(%d)", _precompute_len, input_num_token, kv_cache_num); return -1; }
        if (input_num_token > _attr.prefill_max_token_num) { ALOGE("input_num_token(%d) > prefill_max_token_num(%d)", input_num_token, _attr.prefill_max_token_num); return -1; }
        if (_precompute_len == 0) { ALOGI("first run"); return 0; }
        if (!b_os_kvcache) return 0;
        if (kv_k.size() != kv_v.size() || (int)kv_k.size() != _attr.axmodel_num) { ALOGE("kv cache size mismatch"); return -1; }
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr  = llama_layers[i]; int devid = LLM_DEVID(lyr);
            auto &dk = lyr.layer.get_input(decode_grpid, "K_cache"); auto &dv = lyr.layer.get_input(decode_grpid, "V_cache");
            llm_memset(LLM_WADDR(dk), 0, dk.nSize, devid); llm_memset(LLM_WADDR(dv), 0, dv.nSize, devid);
            auto &pk = lyr.layer.get_input(_attr.prefill_grpid, "K_cache"); auto &pv = lyr.layer.get_input(_attr.prefill_grpid, "V_cache");
            llm_memset(LLM_WADDR(pk), 0, pk.nSize, devid); llm_memset(LLM_WADDR(pv), 0, pv.nSize, devid);
        }
        size_t kv_bytes = (size_t)_precompute_len * _attr.kv_cache_size * sizeof(unsigned short);
        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            auto &lyr  = llama_layers[m]; int devid = LLM_DEVID(lyr);
            auto &kc = kv_k[m]; auto &vc = kv_v[m];
            if ((int)kc.size() < _precompute_len * _attr.kv_cache_size || (int)vc.size() < _precompute_len * _attr.kv_cache_size) { ALOGE("kv_cache buffer too small for layer %d", m); return -1; }
            auto &dk = lyr.layer.get_input(decode_grpid, "K_cache"); auto &dv = lyr.layer.get_input(decode_grpid, "V_cache");
            llm_h2d(LLM_WADDR(dk), kc.data(), kv_bytes, devid); llm_h2d(LLM_WADDR(dv), vc.data(), kv_bytes, devid);
            auto &pk = lyr.layer.get_input(_attr.prefill_grpid, "K_cache"); auto &pv = lyr.layer.get_input(_attr.prefill_grpid, "V_cache");
            llm_h2d(LLM_WADDR(pk), kc.data(), kv_bytes, devid); llm_h2d(LLM_WADDR(pv), vc.data(), kv_bytes, devid);
        }
        return 0;
    }

    void ResetKVCache()
    {
        last_tokens_ids.clear(); k_caches.clear(); v_caches.clear(); precompute_len = 0;
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
        b_stop.store(false, std::memory_order_relaxed); std::string final_out;
        bfloat16 bf16 = -65536.f;
        bfloat16 bf16_one = 1.0f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];
        std::vector<int> token_ids;
        int input_embed_num  = (int)(test_embed.size() / _attr.tokens_embed_size);
        int prefill_split_num = (int)ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
        mask[_attr.kv_cache_num] = 0; for (int i = 0; i < precompute_len + input_embed_num; i++) mask[i] = 0;
        timer t_cost, ttft_timer; ttft_timer.start();

        // Prefill
        // Pre-compute which prefill group id we will use per chunk, so we can propagate KV caches
        // to the future groups (axcl-qwen3-vl behavior).
        std::vector<int> prefill_grp_list;
        prefill_grp_list.resize(prefill_split_num, 1);
        int max_prefill_gid = 1;
        for (int p = 0; p < prefill_split_num; ++p) {
            const int history_len = precompute_len + p * _attr.prefill_token_num;
            int g = 1;
            for (size_t gi = 0; gi < _attr.prefill_max_kv_cache_num_grp.size(); ++gi) {
                if (history_len <= _attr.prefill_max_kv_cache_num_grp[gi]) { g = (int)gi + 1; break; }
            }
            prefill_grp_list[p] = g;
            if (g > max_prefill_gid) max_prefill_gid = g;
        }

        for (int p = 0; p < prefill_split_num; p++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;
            int input_num_token = (p == prefill_split_num - 1) ? input_embed_num - p * _attr.prefill_token_num : _attr.prefill_token_num;
            const int history_len = precompute_len + p * _attr.prefill_token_num;

            // Pick the smallest prefill group that can cover `history_len` cached tokens.
            // Qwen-VL models are sensitive to group/kv_cache_num mismatch.
            const int prefill_grpid = prefill_grp_list[p];
            // Derive kv_cache_num from mask tensor shape when possible (more reliable than K_cache vShape on some models).
            int kv_cache_num_p = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];
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

            build_prefill_mask(mask_tmp, kv_cache_num_p, _attr.prefill_token_num, history_len, input_num_token);
            size_t copy_tokens = (p == prefill_split_num - 1) ? (size_t)(input_embed_num - p * _attr.prefill_token_num) : (size_t)_attr.prefill_token_num;
            memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));

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
                    llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short), devid);
                }
                auto &t_in = lyr.layer.get_input(prefill_grpid, "input"); llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), devid);
                lyr.layer.inference(prefill_grpid);
                auto &out_k = lyr.layer.get_output(prefill_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(prefill_grpid, "V_cache_out");
                auto &dec_k = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &dec_v = lyr.layer.get_input(decode_grpid, "V_cache");
                if (is_linear_layer(m))
                {
                    // Linear-attention layers keep a persistent state, not token-wise KV cache.
                    // Copy the whole cache state tensor.
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
                    int kv_off = history_len * _attr.kv_cache_size;
                    size_t kv_sz = (size_t)input_num_token * _attr.kv_cache_size * sizeof(unsigned short);
                    // Sync current prefill chunk K/V into decode group on both AXCL and AX650.
                    // Missing this causes decode stage to ignore prefill history (AX650 output degrades badly).
                    llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                    llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                    // axcl-qwen3-vl behavior: do not write back to the current prefill group
                    // (group-1 K/V cache capacity can be much smaller than one prefill chunk).
                    // Only propagate to future prefill groups so the next chunk can reuse history.
                    const int ng = (int)lyr.layer.get_num_input_groups();
                    const int max_gid = std::min(max_prefill_gid, ng - 1);
                    for (int gid = prefill_grpid + 1; gid <= max_gid; ++gid) {
                        auto &gk = lyr.layer.get_input(gid, "K_cache");
                        auto &gv = lyr.layer.get_input(gid, "V_cache");
                        const int cap_tokens_k = (int)(gk.nSize / (size_t)(_attr.kv_cache_size * (int)sizeof(unsigned short)));
                        const int cap_tokens_v = (int)(gv.nSize / (size_t)(_attr.kv_cache_size * (int)sizeof(unsigned short)));
                        if (kv_off + input_num_token <= cap_tokens_k) {
                            llm_d2d((unsigned short *)LLM_WADDR(gk) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                        }
                        if (kv_off + input_num_token <= cap_tokens_v) {
                            llm_d2d((unsigned short *)LLM_WADDR(gv) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                        }
                    }
                }

                auto &t_out = lyr.layer.get_output(prefill_grpid, "output");
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), embed_tmp.size() * sizeof(unsigned short), devid);

                // Optional Qwen3VL "deepstack" feature addition (legacy branches behavior).
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
                            // bf16 -> fp32
                            unsigned int tmp_bf16 = ((unsigned int)ev[di]) << 16;
                            float fp32 = *reinterpret_cast<float *>(&tmp_bf16);
                            // fp32 + deepstack -> bf16
                            ev[di] = bfloat16(fp32 + fv[di]).data;
                        }
                    }
                }
            }

            if (p == prefill_split_num - 1)
                memcpy(embed.data(), embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size, _attr.tokens_embed_size * sizeof(unsigned short));
        }

        int next_token = -1; t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
        {
            auto &t_in = llama_post.get_input("input");
            llm_h2d(LLM_WADDR(t_in), embed.data(), embed.size() * sizeof(unsigned short), LLM_DEVID(llama_layers.back()));
            llama_post.inference();
            auto &t_out = llama_post.get_output("output");
            llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
            next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            token_ids.push_back(next_token);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
            if (_attr.runing_callback)
            {
                auto str = utf8_filter.filter(tokenizer->decode(next_token));
                if (!str.empty()) _attr.runing_callback(str, -1, _attr.reserve);
            }
        }

        t_cost.start(); bool b_hit_eos = false;
        unsigned int decode_start = (unsigned int)(precompute_len + input_embed_num);
        if (has_vision_state && vision_state.decode_start > 0) decode_start = (unsigned int)vision_state.decode_start;
        for (unsigned int indices = decode_start; indices < (unsigned int)_attr.max_token_len; indices++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;
            embed_selector.getByIndex(next_token, embed);

#ifdef USE_AXCL
            {
                auto &l0_in = llama_layers[0].layer.get_input(decode_grpid, "input");
                llm_h2d(LLM_WADDR(l0_in), embed.data(), l0_in.nSize, llama_layers[0].layer.get_devid());
            }
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break; auto &lyr = llama_layers[m]; int devid = lyr.layer.get_devid();
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
                    llm_h2d(LLM_WADDR(t_mask), mask.data(), mask.size() * sizeof(unsigned short), devid);
                }
                lyr.layer.inference(decode_grpid);
                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out"); auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                auto &in_k  = lyr.layer.get_input(decode_grpid, "K_cache"); auto &in_v  = lyr.layer.get_input(decode_grpid, "V_cache");
                if (is_linear_layer(m))
                {
                    llm_d2d(LLM_WADDR(in_k), LLM_RADDR(out_k), std::min((size_t)in_k.nSize, (size_t)out_k.nSize), devid);
                    llm_d2d(LLM_WADDR(in_v), LLM_RADDR(out_v), std::min((size_t)in_v.nSize, (size_t)out_v.nSize), devid);
                }
                else
                {
                    llm_d2d((unsigned short *)LLM_WADDR(in_k) + indices * _attr.kv_cache_size, LLM_RADDR(out_k), out_k.nSize, devid);
                    llm_d2d((unsigned short *)LLM_WADDR(in_v) + indices * _attr.kv_cache_size, LLM_RADDR(out_v), out_v.nSize, devid);
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
                    memcpy(t_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));
                }
                auto &t_in  = lyr.layer.get_input(decode_grpid, "input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                lyr.layer.inference(decode_grpid);
                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                if (is_linear_layer(m))
                {
                    memcpy(in_k.pVirAddr, out_k.pVirAddr, std::min((size_t)in_k.nSize, (size_t)out_k.nSize));
                    memcpy(in_v.pVirAddr, out_v.pVirAddr, std::min((size_t)in_v.nSize, (size_t)out_v.nSize));
                }
                else
                {
                    memcpy(in_k_ptr + indices * _attr.kv_cache_size, out_k.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                    memcpy(in_v_ptr + indices * _attr.kv_cache_size, out_v.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                }
                auto &t_out= lyr.layer.get_output(decode_grpid, "output"); memcpy(embed.data(), t_out.pVirAddr, embed.size() * sizeof(unsigned short));
            }
            auto &t_in = llama_post.get_input("input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference(); auto &t_out = llama_post.get_output("output");
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
#endif

            mask[indices] = 0;
            if (tokenizer->is_stop(next_token)) { b_hit_eos = true; break; }
            token_ids.push_back(next_token);
            if (_attr.runing_callback)
            {
                float t_ms  = t_cost.cost(); float tps   = token_ids.size() / (t_ms / 1000.0f);
                auto  str   = utf8_filter.filter(tokenizer->decode(next_token));
                if (!str.empty()) _attr.runing_callback(str, tps, _attr.reserve);
            }
            if (output_max_token > 0 && (int)token_ids.size() >= output_max_token) { b_hit_eos = true; break; }
            if (_attr.runing_callback == nullptr) update_cqdm(&cqdm, indices, "token", "");
        }

        printf("\n\n"); fflush(stdout); float t_ms = t_cost.cost(); ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_ms / 1000.0f));
        final_out = tokenizer->decode(token_ids); return final_out;
    }

    std::vector<Content> Run(std::vector<Content> history, int output_max_token = -1)
    {
        return Run(std::move(history), {}, output_max_token);
    }

    std::vector<Content> Run(std::vector<Content> history, const std::vector<::MediaInputs> &media_inputs, int output_max_token = -1)
    {
        has_vision_state = false;

        std::vector<int> new_tokens;

        if (vision && vision->enabled())
        {
            // If caller provides media, we will fill num_media/num_media_tokens and build injection state.
            if (!media_inputs.empty())
            {
                std::vector<vision::MediaInputs> vmins;
                vmins.reserve(media_inputs.size());
                for (const auto &m : media_inputs) vmins.push_back({m.content_index, m.uris});

                std::vector<Content> prepared_history;
                std::vector<int> input_ids;
                vision::RunState st;
                std::string verr;
                if (!vision->Prepare(history, vmins, prepared_history, input_ids, st, verr))
                {
                    ALOGE("vision.Prepare failed: %s", verr.c_str());
                    return history;
                }
                history = std::move(prepared_history);
                new_tokens = std::move(input_ids);
                vision_state = std::move(st);
                has_vision_state = true;
            }
            else
            {
                // If history contains IMAGE/VIDEO, the caller must provide media inputs.
                bool need_media = false;
                for (const auto &c : history) if (c.type == IMAGE || c.type == VIDEO) { need_media = true; break; }
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

        int offset = 0; auto tokens_diff = diff_token_ids(last_tokens_ids, new_tokens, offset);
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
        auto reply = Run(out_embed, output_max_token);
        GetKVCache(k_caches, v_caches, precompute_len);
        history.push_back({ASSISTANT, TEXT, reply});
        last_tokens_ids = tokenizer->encode(history);
        if (last_tokens_ids.size() >= 2) last_tokens_ids.erase(last_tokens_ids.end() - 2, last_tokens_ids.end());

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

bool LLM::Embed(const std::string &text, std::vector<float> &out_embedding) { return impl_->EmbedText(text, out_embedding); }
bool LLM::EmbedBatch(const std::vector<std::string> &inputs, std::vector<std::vector<float>> &out_embeddings) { return impl_->EmbedBatch(inputs, out_embeddings); }

int LLM::GenerateKVCachePrefill(std::vector<int> &ids, std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int &pre_len) { return impl_->GenerateKVCachePrefill(ids, k, v, pre_len); }
int LLM::GetKVCache(std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int &pre_len) { return impl_->GetKVCache(k, v, pre_len); }
int LLM::SetKVCache(std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int pre_len, int in_tokens) { return impl_->SetKVCache(k, v, pre_len, in_tokens); }
void LLM::ResetKVCache() { impl_->ResetKVCache(); }

std::vector<Content> LLM::Run(std::vector<Content> history, int output_max_token) { return impl_->Run(std::move(history), output_max_token); }
std::vector<Content> LLM::Run(std::vector<Content> history, const std::vector<MediaInputs> &media_inputs, int output_max_token) { return impl_->Run(std::move(history), media_inputs, output_max_token); }
std::string LLM::Run(std::vector<unsigned short> &embed, int output_max_token) { return impl_->Run(embed, output_max_token); }
