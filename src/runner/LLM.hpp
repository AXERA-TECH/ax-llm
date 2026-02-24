#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "BaseTokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "UTF8Filter.hpp"
#include "LLMPostprocess.hpp"

#ifdef USE_AXCL
#include <atomic>
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

// ---------------------------------------------------------------------------
// Backend-agnostic memory helpers
//   llm_memset  : zero-fill device/CMM buffer
//   llm_h2d     : host → device (or host → CMM via pVirAddr on ax650)
//   llm_d2h     : device → host
//   llm_d2d     : device → device (same device; falls back to memcpy on ax650)
//   LLM_WADDR   : write address (phyAddr on AXCL, pVirAddr on ax650)
//   LLM_RADDR   : read address  (same)
//   LLM_DEVID   : device id of a LLMLayer (always 0 on ax650)
// ---------------------------------------------------------------------------
#ifdef USE_AXCL
static inline void llm_memset(void *phy, int val, size_t n, int devid)
{
    axcl_Memset(phy, (uint8_t)val, n, devid);
}
static inline void llm_h2d(void *phy_dst, const void *src, size_t n, int devid)
{
    axcl_Memcpy(phy_dst, src, n, AXCL_MEMCPY_HOST_TO_DEVICE, devid);
}
static inline void llm_d2h(void *dst, const void *phy_src, size_t n, int devid)
{
    axcl_Memcpy(dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_HOST, devid);
}
static inline void llm_d2d(void *phy_dst, const void *phy_src, size_t n, int devid)
{
    axcl_Memcpy(phy_dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_DEVICE, devid);
}
#define LLM_WADDR(t)      ((void *)(t).phyAddr)
#define LLM_RADDR(t)      ((const void *)(t).phyAddr)
#define LLM_DEVID(layer_obj) ((layer_obj).layer.get_devid())
#else
static inline void llm_memset(void *vir, int val, size_t n, int /*devid*/)
{
    memset(vir, val, n);
}
static inline void llm_h2d(void *vir_dst, const void *src, size_t n, int /*devid*/)
{
    memcpy(vir_dst, src, n);
}
static inline void llm_d2h(void *dst, const void *vir_src, size_t n, int /*devid*/)
{
    memcpy(dst, vir_src, n);
}
static inline void llm_d2d(void *vir_dst, const void *vir_src, size_t n, int /*devid*/)
{
    memcpy(vir_dst, vir_src, n);
}
#define LLM_WADDR(t)      ((t).pVirAddr)
#define LLM_RADDR(t)      ((const void *)(t).pVirAddr)
#define LLM_DEVID(layer_obj) (0)
#endif

// ---------------------------------------------------------------------------

using LLMRuningCallback = std::function<void(std::string str, float token_per_sec, void *reserve)>;

struct LLMAttrType
{
    std::string system_prompt;
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    std::string tokenizer_type;
    std::string url_tokenizer_model = "http://127.0.0.1:12345";
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    std::vector<int> prefill_max_kv_cache_num_grp;

    int prefill_grpid = -1;

    std::string post_config_path = "post_config.json";

    bool b_use_mmap_load_embed = false;

#ifndef USE_AXCL
    bool b_use_mmap_load_layer = true;
#endif

#ifdef USE_AXCL
    // AXCL 多卡设备 ID 列表，默认单卡 0
    std::vector<int> dev_ids = {0};
#endif

    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;
};

class LLM
{
private:
    UTF8Filter utf8_filter;
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;

    std::vector<int> last_tokens_ids;
    bool b_os_kvcache = false;
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    int precompute_len = 0;

    LLMAttrType _attr;

    struct LLMLayer
    {
        ax_runner_t layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_t llama_post;

    int decode_grpid = 0;

    bool b_stop = false;

    LLMPostprocess postprocess;

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

#ifdef USE_AXCL
    // 将 modelCount 个模型均匀分配到 cardCount 张卡上
    std::vector<int> distributeModels(int cardCount, int modelCount)
    {
        std::vector<int> assign(modelCount, 0);
        if (cardCount <= 0 || modelCount <= 0)
            return assign;
        int base = modelCount / cardCount;
        int rem  = modelCount % cardCount;
        int idx  = 0;
        for (int c = 0; c < cardCount; c++)
        {
            int cnt = base + (c < rem ? 1 : 0);
            for (int i = 0; i < cnt; i++)
                assign[idx++] = c;
        }
        return assign;
    }
#endif

    std::vector<int> diff_token_ids(std::vector<int> ids1, std::vector<int> ids2)
    {
        if (ids1.size() >= ids2.size()) return {};
        for (int i = 0; i < (int)ids1.size(); i++)
            if (ids1[i] != ids2[i]) return {};
        return std::vector<int>(ids2.begin() + ids1.size(), ids2.end());
    }

    std::vector<int> diff_token_ids(const std::vector<int> &ids1, const std::vector<int> &ids2, int &offset)
    {
        int min_len = std::min(ids1.size(), ids2.size());
        offset = 0;
        for (int i = 0; i < min_len; i++)
        {
            if (ids1[i] == ids2[i]) offset++;
            else break;
        }
        if (offset >= (int)ids2.size()) return {};
        return std::vector<int>(ids2.begin() + offset, ids2.end());
    }

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;

        tokenizer = create_tokenizer(this->_attr.tokenizer_type);
        if (!tokenizer)
        {
            ALOGE("create_tokenizer(%s) failed", this->_attr.tokenizer_type.c_str());
            return false;
        }
        if (!tokenizer->load(attr.url_tokenizer_model))
        {
            ALOGE("tokenizer.init(%s) failed", attr.url_tokenizer_model.c_str());
            return false;
        }
        // match older behavior: keep think content inside prompt
        tokenizer->set_think_in_prompt(true);
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");
#ifdef USE_AXCL
        // 初始化所有 AXCL 设备
        for (auto &devid : _attr.dev_ids)
        {
            if (axcl_Init(devid) != 0)
            {
                ALOGE("axcl_Init(%d) failed", devid);
                return false;
            }
        }

        llama_layers.resize(attr.axmodel_num);
        auto dev_assign = distributeModels((int)_attr.dev_ids.size(), attr.axmodel_num);

        std::vector<int> rets(attr.axmodel_num, 0);
        std::atomic<int> process_idx(2);
#pragma omp parallel for if (_attr.dev_ids.size() > 1)
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            char path[1024];
            sprintf(path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = path;

            int devid = _attr.dev_ids[dev_assign[i]];
            rets[i] = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), devid);

            int remain = axcl_GetCMMRemain(devid);
            sprintf(path, "init %d axmodel ok,devid(%d) remain_cmm(%d MB)", i, devid, remain);
            update_cqdm(&cqdm, process_idx++, "count", path);
        }
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            if (rets[i] != 0)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
        }

        {
            int post_devid = llama_layers.back().layer.get_devid();
            int ret = llama_post.init(attr.filename_post_axmodel.c_str(), post_devid);
            if (ret != 0)
            {
                ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
                return false;
            }
            char path[1024];
            sprintf(path, "init post axmodel ok,remain_cmm(%d MB)", axcl_GetCMMRemain(post_devid));
            update_cqdm(&cqdm, attr.axmodel_num + 2, "count", path);
        }

        // 打印各卡剩余 CMM
        printf(MACRO_PURPLE "________________________\n");
        printf("|%6s|%15s|\n", "ID", "remain cmm(MB)");
        printf("========================\n");
        for (auto devid : _attr.dev_ids)
            printf("|%6d|%15d|\n", devid, axcl_GetCMMRemain(devid));
        printf("________________________\n" MACRO_END);

#else // AX650 on-chip backend
        llama_layers.resize(attr.axmodel_num);
        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), -1);
            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 2, "count", axmodel_path);
        }

        {
            int ret = llama_post.init(attr.filename_post_axmodel.c_str(), -1);
            if (ret != 0)
            {
                ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
                return false;
            }
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
            update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);
        }
#endif

        // 从模型自动推断参数（两个后端相同）
        {
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            ALOGI("max_token_len : %d", _attr.max_token_len);

            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num  = llama_layers[0].layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);

            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = llama_layers[0].layer.get_input(1, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);

            for (size_t i = 0; i < llama_layers[0].layer.get_num_input_groups() - 1; i++)
            {
                int n = llama_layers[0].layer.get_input((int)i + 1, "K_cache").vShape[1];
                ALOGI("grp: %zu, prefill_max_kv_cache_num : %d", i + 1, n);
                _attr.prefill_max_kv_cache_num_grp.push_back(n);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();
            // default prefill group is the largest one unless later overridden
            _attr.prefill_grpid = (int)_attr.prefill_max_kv_cache_num_grp.size();
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }

        {
            auto &t_in0 = llama_layers[0].layer.get_input(decode_grpid, "input");
            int model_embed_sz = t_in0.nSize / (int)sizeof(unsigned short);
            if (model_embed_sz != _attr.tokens_embed_size)
            {
                ALOGE("tokens_embed_size mismatch: config(%d) != model(%d). Please fix config or embed file.",
                      _attr.tokens_embed_size, model_embed_sz);
                return false;
            }
            if (!embed_selector.Init(_attr.filename_tokens_embed, _attr.tokens_embed_num, _attr.tokens_embed_size, _attr.b_use_mmap_load_embed))
            {
                ALOGE("embed_selector.Init(%s, %d, %d) failed", _attr.filename_tokens_embed.c_str(), _attr.tokens_embed_num, _attr.tokens_embed_size);
                return false;
            }
            update_cqdm(&cqdm, 1, "count", "embed_selector init ok");
            printf("\n");
        }

        if (!postprocess.load_config(attr.post_config_path))
        {
            ALOGW("load postprocess config(%s) failed", attr.post_config_path.c_str());
        }

        ALOGI("LLM init ok");
        return true;
    }

    LLMAttrType *getAttr() { return &_attr; }
    LLMPostprocess *getPostprocess() { return &postprocess; }
    LLaMaEmbedSelector *getEmbedSelector() { return &embed_selector; }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
            llama_layers[i].layer.deinit();
        llama_post.deinit();
        embed_selector.Deinit();
#ifdef USE_AXCL
        for (auto &devid : _attr.dev_ids)
            axcl_Exit(devid);
#endif
    }

    void Stop() { b_stop = true; }

    // -----------------------------------------------------------------------
    // GenerateKVCachePrefill
    //   对给定 token 序列做 prefill，将结果 KV 缓存回传给调用方
    // -----------------------------------------------------------------------
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
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (input_embed_num <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                prefill_grpid = (int)i + 1;
                break;
            }
        }
        ALOGI("input token num : %d, prefill_split_num : %d prefill_grpid : %d",
              input_embed_num, prefill_split_num, prefill_grpid);

        // 清空 KV 缓存
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i];
            int devid = LLM_DEVID(lyr);
            llm_memset(LLM_WADDR(lyr.layer.get_input(prefill_grpid, "K_cache")),
                       0, lyr.layer.get_input(prefill_grpid, "K_cache").nSize, devid);
            llm_memset(LLM_WADDR(lyr.layer.get_input(prefill_grpid, "V_cache")),
                       0, lyr.layer.get_input(prefill_grpid, "V_cache").nSize, devid);
        }

        if (input_embed_num == 0)
        {
            for (int i = 0; i < _attr.axmodel_num; i++)
            {
                k_caches[i].resize(prefill_precompute_len * _attr.kv_cache_size);
                v_caches[i].resize(prefill_precompute_len * _attr.kv_cache_size);
            }
            ALOGI("input token num is 0, skip");
            return 0;
        }

        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];

        std::vector<unsigned short> test_embed(_token_ids.size() * _attr.tokens_embed_size);
        for (size_t i = 0; i < _token_ids.size(); i++)
            embed_selector.getByIndex(_token_ids[i], test_embed.data() + i * _attr.tokens_embed_size);

        for (int p = 0; p < prefill_split_num; p++)
        {
            int input_num_token = (p == prefill_split_num - 1)
                                  ? input_embed_num - p * _attr.prefill_token_num
                                  : _attr.prefill_token_num;

            // 构造 mask_tmp
            std::vector<unsigned short> mask_tmp(
                _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            for (int i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);
                    for (int j = 0; j < p * _attr.prefill_token_num; j++)
                        mask_ptr[j] = 0;
                    int cur_start = kv_cache_num;
                    for (int j = cur_start; j < cur_start + i + 1; j++)
                        mask_ptr[j] = 0;
                }
            }

            // 构造 embed_tmp
            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            size_t copy_tokens = (p == prefill_split_num - 1)
                                 ? (size_t)(input_embed_num - p * _attr.prefill_token_num)
                                 : (size_t)_attr.prefill_token_num;
            memcpy(embed_tmp.data(),
                   test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                   copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                auto &lyr = llama_layers[m];
                int devid = LLM_DEVID(lyr);

                // indices
                auto &t_idx = lyr.layer.get_input(prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr;
                memset(idx_ptr, 0, t_idx.nSize);
                int idx_i = 0;
                for (int i = 0; i < input_num_token; ++i)
                    idx_ptr[idx_i++] = (unsigned int)(p * _attr.prefill_token_num + i);
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);

                // mask
                auto &t_mask = lyr.layer.get_input(prefill_grpid, "mask");
                llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short), devid);

                // input embed
                auto &t_in = lyr.layer.get_input(prefill_grpid, "input");
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), devid);

                lyr.layer.inference(prefill_grpid);

                auto &out_k  = lyr.layer.get_output(prefill_grpid, "K_cache_out");
                auto &out_v  = lyr.layer.get_output(prefill_grpid, "V_cache_out");
                auto &dec_k  = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &dec_v  = lyr.layer.get_input(decode_grpid, "V_cache");
                auto &pre_k  = lyr.layer.get_input(prefill_grpid, "K_cache");
                auto &pre_v  = lyr.layer.get_input(prefill_grpid, "V_cache");

                int kv_off = p * _attr.prefill_token_num * _attr.kv_cache_size;
                size_t kv_sz = (size_t)_attr.prefill_token_num * _attr.kv_cache_size * sizeof(unsigned short);

                // 将本轮 prefill 输出的 KV 写入 decoder KV cache 和 prefill KV cache
#ifdef USE_AXCL
                llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
#endif
                llm_d2d((unsigned short *)LLM_WADDR(pre_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(pre_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);

                // 读取本层输出作为下一层输入（host 中转）
                auto &t_out = lyr.layer.get_output(prefill_grpid, "output");
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), embed_tmp.size() * sizeof(unsigned short), devid);
            }
        }

        // 将 prefill 后的 KV cache 拷回调用方
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i];
            int devid = LLM_DEVID(lyr);
            k_caches[i].resize(prefill_precompute_len * _attr.kv_cache_size);
            v_caches[i].resize(prefill_precompute_len * _attr.kv_cache_size);
            auto &t_k = lyr.layer.get_input(prefill_grpid, "K_cache");
            auto &t_v = lyr.layer.get_input(prefill_grpid, "V_cache");
            llm_d2h(k_caches[i].data(), LLM_RADDR(t_k),
                    prefill_precompute_len * _attr.kv_cache_size * sizeof(unsigned short), devid);
            llm_d2h(v_caches[i].data(), LLM_RADDR(t_v),
                    prefill_precompute_len * _attr.kv_cache_size * sizeof(unsigned short), devid);
        }

        return 0;
    }

#ifdef USE_AXCL
    // -----------------------------------------------------------------------
    // GenerateKVCache (AXCL 专用)
    //   通过 decode 模式循环对 token 序列做 KV cache 预热
    // -----------------------------------------------------------------------
    int GenerateKVCache(std::vector<int> &_token_ids)
    {
        // 清空 decode KV cache
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr = llama_layers[i];
            int devid = lyr.layer.get_devid();
            llm_memset(LLM_WADDR(lyr.layer.get_input(decode_grpid, "K_cache")),
                       0, lyr.layer.get_input(decode_grpid, "K_cache").nSize, devid);
            llm_memset(LLM_WADDR(lyr.layer.get_input(decode_grpid, "V_cache")),
                       0, lyr.layer.get_input(decode_grpid, "V_cache").nSize, devid);
        }

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        mask[_attr.kv_cache_num] = 0;
        std::vector<unsigned short> embed;

        int next_token = _token_ids[0];
        t_cqdm cqdm = create_cqdm((int)_token_ids.size(), 32);

        for (unsigned int indices = 0; indices < _token_ids.size(); indices++)
        {
            embed_selector.getByIndex(next_token, embed);

            // 将 embed 写入第 0 层的 input
            auto &l0_in = llama_layers[0].layer.get_input(decode_grpid, "input");
            llm_h2d(LLM_WADDR(l0_in), embed.data(), l0_in.nSize, llama_layers[0].layer.get_devid());

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop) break;
                auto &lyr   = llama_layers[m];
                int   devid = lyr.layer.get_devid();

                auto &t_idx = lyr.layer.get_input(decode_grpid, "indices");
                llm_h2d(LLM_WADDR(t_idx), &indices, sizeof(indices), devid);

                auto &t_mask = lyr.layer.get_input(decode_grpid, "mask");
                llm_h2d(LLM_WADDR(t_mask), mask.data(), mask.size() * sizeof(unsigned short), devid);

                lyr.layer.inference(decode_grpid);

                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                auto &in_k  = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &in_v  = lyr.layer.get_input(decode_grpid, "V_cache");
                llm_d2d((unsigned short *)LLM_WADDR(in_k) + indices * _attr.kv_cache_size,
                        LLM_RADDR(out_k), out_k.nSize, devid);
                llm_d2d((unsigned short *)LLM_WADDR(in_v) + indices * _attr.kv_cache_size,
                        LLM_RADDR(out_v), out_v.nSize, devid);

                // 跨层数据路由
                if (m < _attr.axmodel_num - 1)
                {
                    auto &next_in = llama_layers[m + 1].layer.get_input(decode_grpid, "input");
                    auto &cur_out = lyr.layer.get_output(decode_grpid, "output");
                    int next_devid = llama_layers[m + 1].layer.get_devid();
                    if (next_devid == devid)
                    {
                        llm_d2d(LLM_WADDR(next_in), LLM_RADDR(cur_out), next_in.nSize, devid);
                    }
                    else
                    {
                        llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid);
                        llm_h2d(LLM_WADDR(next_in), cur_out.pVirAddr, next_in.nSize, next_devid);
                    }
                }
            }

            mask[indices] = 0;
            if (indices + 1 < _token_ids.size())
                next_token = _token_ids[indices + 1];
            update_cqdm(&cqdm, indices, "token", "");
        }
        return 0;
    }
#endif

    // -----------------------------------------------------------------------
    // GetKVCache  —  将当前 decode KV cache 内容读回主机
    // -----------------------------------------------------------------------
    int GetKVCache(std::vector<std::vector<unsigned short>> &kv_k,
                   std::vector<std::vector<unsigned short>> &kv_v,
                   int &kv_precompute_len)
    {
        bfloat16 bf16 = -65536.f;

        // 读取 mask 以推断已有的 token 数量
        auto &t_mask = llama_layers[0].layer.get_input(decode_grpid, "mask");
        std::vector<unsigned short> mask(t_mask.nSize / sizeof(unsigned short), bf16.data);
        llm_d2h(mask.data(), LLM_RADDR(t_mask), t_mask.nSize, LLM_DEVID(llama_layers[0]));

        kv_precompute_len = 0;
        for (size_t i = 0; i < mask.size(); i++)
        {
            if (mask[i] == bf16.data)
            {
                kv_precompute_len = (int)i + 1;
                break;
            }
        }
        ALOGI("precompute_len:%d, remaining:%d", kv_precompute_len,
              _attr.prefill_max_kv_cache_num_grp.back() - kv_precompute_len);

        if (b_os_kvcache)
        {
            kv_k.resize(_attr.axmodel_num);
            kv_v.resize(_attr.axmodel_num);
            for (int i = 0; i < _attr.axmodel_num; i++)
            {
                auto &lyr = llama_layers[i];
                int devid = LLM_DEVID(lyr);
                kv_k[i].resize(kv_precompute_len * _attr.kv_cache_size);
                kv_v[i].resize(kv_precompute_len * _attr.kv_cache_size);
                auto &t_k = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &t_v = lyr.layer.get_input(decode_grpid, "V_cache");
                llm_d2h(kv_k[i].data(), LLM_RADDR(t_k),
                        kv_precompute_len * _attr.kv_cache_size * sizeof(unsigned short), devid);
                llm_d2h(kv_v[i].data(), LLM_RADDR(t_v),
                        kv_precompute_len * _attr.kv_cache_size * sizeof(unsigned short), devid);
            }
        }

        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();
        return 0;
    }

    // -----------------------------------------------------------------------
    // SetKVCache  —  将主机端 KV cache 写入模型缓冲区
    // -----------------------------------------------------------------------
    int SetKVCache(std::vector<std::vector<unsigned short>> &kv_k,
                   std::vector<std::vector<unsigned short>> &kv_v,
                   int _precompute_len, int input_num_token)
    {
        // 选择合适的 prefill group
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (_precompute_len + input_num_token <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                _attr.prefill_grpid = (int)i + 1;
                break;
            }
        }
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];
        ALOGI("prefill_grpid:%d kv_cache_num:%d precompute_len:%d input_num_token:%d",
              _attr.prefill_grpid, kv_cache_num, _precompute_len, input_num_token);

        _attr.prefill_max_token_num = ALIGN_DOWN(_attr.prefill_max_token_num - _precompute_len,
                                                  _attr.prefill_token_num);
        ALOGI("current prefill_max_token_num:%d", _attr.prefill_max_token_num);

        if (_precompute_len == 0)
        {
            ALOGI("first run");
            return 0;
        }

        if (_precompute_len + input_num_token > kv_cache_num)
        {
            ALOGE("precompute_len(%d) + input_num_token(%d) > kv_cache_num(%d)",
                  _precompute_len, input_num_token, kv_cache_num);
            return -1;
        }
        if (input_num_token > _attr.prefill_max_token_num)
        {
            ALOGE("input_num_token(%d) > prefill_max_token_num(%d)", input_num_token, _attr.prefill_max_token_num);
            return -1;
        }

        if (!b_os_kvcache) return 0;

        if (kv_k.size() != kv_v.size() || (int)kv_k.size() != _attr.axmodel_num)
        {
            ALOGE("kv cache size mismatch");
            return -1;
        }

        // 清空并写入 decode + prefill KV cache
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr  = llama_layers[i];
            int   devid = LLM_DEVID(lyr);

            auto &dk = lyr.layer.get_input(decode_grpid, "K_cache");
            auto &dv = lyr.layer.get_input(decode_grpid, "V_cache");
            llm_memset(LLM_WADDR(dk), 0, dk.nSize, devid);
            llm_memset(LLM_WADDR(dv), 0, dv.nSize, devid);

#ifdef USE_AXCL
            auto &pk = lyr.layer.get_input(_attr.prefill_grpid, "K_cache");
            auto &pv = lyr.layer.get_input(_attr.prefill_grpid, "V_cache");
            llm_memset(LLM_WADDR(pk), 0, pk.nSize, devid);
            llm_memset(LLM_WADDR(pv), 0, pv.nSize, devid);
#endif
        }

        size_t kv_bytes = (size_t)_precompute_len * _attr.kv_cache_size * sizeof(unsigned short);
        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            auto &lyr  = llama_layers[m];
            int   devid = LLM_DEVID(lyr);
            auto &kc = kv_k[m];
            auto &vc = kv_v[m];

            if ((int)kc.size() < _precompute_len * _attr.kv_cache_size ||
                (int)vc.size() < _precompute_len * _attr.kv_cache_size)
            {
                ALOGE("kv_cache buffer too small for layer %d", m);
                return -1;
            }

            auto &dk = lyr.layer.get_input(decode_grpid, "K_cache");
            auto &dv = lyr.layer.get_input(decode_grpid, "V_cache");
            llm_h2d(LLM_WADDR(dk), kc.data(), kv_bytes, devid);
            llm_h2d(LLM_WADDR(dv), vc.data(), kv_bytes, devid);

#ifdef USE_AXCL
            auto &pk = lyr.layer.get_input(_attr.prefill_grpid, "K_cache");
            auto &pv = lyr.layer.get_input(_attr.prefill_grpid, "V_cache");
            llm_h2d(LLM_WADDR(pk), kc.data(), kv_bytes, devid);
            llm_h2d(LLM_WADDR(pv), vc.data(), kv_bytes, devid);
#endif
        }
        return 0;
    }

    void ResetKVCache()
    {
        last_tokens_ids.clear();
        k_caches.clear();
        v_caches.clear();
        precompute_len = 0;

        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            auto &lyr  = llama_layers[i];
            int   devid = LLM_DEVID(lyr);
            auto &dk = lyr.layer.get_input(decode_grpid, "K_cache");
            auto &dv = lyr.layer.get_input(decode_grpid, "V_cache");
            llm_memset(LLM_WADDR(dk), 0, dk.nSize, devid);
            llm_memset(LLM_WADDR(dv), 0, dv.nSize, devid);
        }
    }

    // -----------------------------------------------------------------------
    // Run (低层接口) — 给定 embed 序列，推理并流式回调输出
    // -----------------------------------------------------------------------
    std::string Run(std::vector<unsigned short> &test_embed, int output_max_token = -1)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];

        std::vector<int> token_ids;

        int input_embed_num  = (int)(test_embed.size() / _attr.tokens_embed_size);
        int prefill_split_num = (int)ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);

        mask[_attr.kv_cache_num] = 0;
        for (int i = 0; i < precompute_len + input_embed_num; i++)
            mask[i] = 0;

        timer t_cost, ttft_timer;
        ttft_timer.start();

        // ---- Prefill 阶段 ----
        for (int p = 0; p < prefill_split_num; p++)
        {
            if (b_stop) break;

            int input_num_token = (p == prefill_split_num - 1)
                                  ? input_embed_num - p * _attr.prefill_token_num
                                  : _attr.prefill_token_num;

            std::vector<unsigned short> mask_tmp(
                _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            for (int i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);
                    for (int j = 0; j < precompute_len + p * _attr.prefill_token_num; j++)
                        mask_ptr[j] = 0;
                    int cur_start = kv_cache_num;
                    for (int j = cur_start; j < cur_start + i + 1; j++)
                        mask_ptr[j] = 0;
                }
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            size_t copy_tokens = (p == prefill_split_num - 1)
                                 ? (size_t)(input_embed_num - p * _attr.prefill_token_num)
                                 : (size_t)_attr.prefill_token_num;
            memcpy(embed_tmp.data(),
                   test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                   copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop) break;
                auto &lyr   = llama_layers[m];
                int   devid = LLM_DEVID(lyr);

                // indices
                auto &t_idx = lyr.layer.get_input(_attr.prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr;
                memset(idx_ptr, 0, t_idx.nSize);
                int idx_i = 0;
                for (int i = 0; i < input_num_token; ++i)
                    idx_ptr[idx_i++] = (unsigned int)(precompute_len + p * _attr.prefill_token_num + i);
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);

                // mask
                auto &t_mask = lyr.layer.get_input(_attr.prefill_grpid, "mask");
                llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(),
                        mask_tmp.size() * sizeof(unsigned short), devid);

                // input embed
                auto &t_in = lyr.layer.get_input(_attr.prefill_grpid, "input");
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(),
                        embed_tmp.size() * sizeof(unsigned short), devid);

                lyr.layer.inference(_attr.prefill_grpid);

                auto &out_k = lyr.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(_attr.prefill_grpid, "V_cache_out");
                auto &dec_k = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &dec_v = lyr.layer.get_input(decode_grpid, "V_cache");
                auto &pre_k = lyr.layer.get_input(_attr.prefill_grpid, "K_cache");
                auto &pre_v = lyr.layer.get_input(_attr.prefill_grpid, "V_cache");

                int kv_off = (precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;
#ifdef USE_AXCL
                size_t kv_sz = (size_t)_attr.prefill_token_num * _attr.kv_cache_size * sizeof(unsigned short);
#else
                size_t kv_sz = (size_t)input_num_token * _attr.kv_cache_size * sizeof(unsigned short);
#endif

#ifdef USE_AXCL
                llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
#endif
                llm_d2d((unsigned short *)LLM_WADDR(pre_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(pre_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);

                auto &t_out = lyr.layer.get_output(_attr.prefill_grpid, "output");
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out),
                        embed_tmp.size() * sizeof(unsigned short), devid);
            }

            if (p == prefill_split_num - 1)
            {
                memcpy(embed.data(),
                       embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            }
        }

        // ---- 首 token post-process ----
        int next_token = -1;
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
        {
            auto &t_in = llama_post.get_input("input");
            llm_h2d(LLM_WADDR(t_in), embed.data(), embed.size() * sizeof(unsigned short),
                    LLM_DEVID(llama_layers.back()) /* post 与最后一层同卡 */);
            llama_post.inference();

            auto &t_out = llama_post.get_output("output");
            llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize,
                    LLM_DEVID(llama_layers.back()));

            unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
            next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

            token_ids.push_back(next_token);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());

            if (_attr.runing_callback)
            {
                auto str = utf8_filter.filter(tokenizer->decode(next_token));
                if (!str.empty())
                    _attr.runing_callback(str, -1, _attr.reserve);
            }
        }
        t_cost.start();

        // ---- Decode 阶段 ----
        bool b_hit_eos = false;
        for (unsigned int indices = (unsigned int)(precompute_len + input_embed_num);
             indices < (unsigned int)_attr.max_token_len; indices++)
        {
            if (b_stop) break;

            embed_selector.getByIndex(next_token, embed);

#ifdef USE_AXCL
            // AXCL: embed 写入第 0 层 input，层间 D2D（或跨卡 D2H+H2D）路由
            {
                auto &l0_in = llama_layers[0].layer.get_input(decode_grpid, "input");
                llm_h2d(LLM_WADDR(l0_in), embed.data(), l0_in.nSize,
                        llama_layers[0].layer.get_devid());
            }

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop) break;
                auto &lyr   = llama_layers[m];
                int   devid = lyr.layer.get_devid();

                auto &t_idx = lyr.layer.get_input(decode_grpid, "indices");
                llm_h2d(LLM_WADDR(t_idx), &indices, sizeof(indices), devid);

                auto &t_mask = lyr.layer.get_input(decode_grpid, "mask");
                llm_h2d(LLM_WADDR(t_mask), mask.data(), mask.size() * sizeof(unsigned short), devid);

                lyr.layer.inference(decode_grpid);

                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                auto &in_k  = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &in_v  = lyr.layer.get_input(decode_grpid, "V_cache");
                llm_d2d((unsigned short *)LLM_WADDR(in_k) + indices * _attr.kv_cache_size,
                        LLM_RADDR(out_k), out_k.nSize, devid);
                llm_d2d((unsigned short *)LLM_WADDR(in_v) + indices * _attr.kv_cache_size,
                        LLM_RADDR(out_v), out_v.nSize, devid);

                // 路由到下一层或 post
                auto &cur_out = lyr.layer.get_output(decode_grpid, "output");
                if (m == _attr.axmodel_num - 1)
                {
                    auto &post_in = llama_post.get_input("input");
                    int post_devid = llama_layers.back().layer.get_devid();
                    if (llama_post.get_devid() == devid)
                    {
                        llm_d2d(LLM_WADDR(post_in), LLM_RADDR(cur_out), post_in.nSize, devid);
                    }
                    else
                    {
                        llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid);
                        llm_h2d(LLM_WADDR(post_in), cur_out.pVirAddr, post_in.nSize, llama_post.get_devid());
                    }
                }
                else
                {
                    auto &next_in = llama_layers[m + 1].layer.get_input(decode_grpid, "input");
                    int next_devid = llama_layers[m + 1].layer.get_devid();
                    if (next_devid == devid)
                    {
                        llm_d2d(LLM_WADDR(next_in), LLM_RADDR(cur_out), next_in.nSize, devid);
                    }
                    else
                    {
                        llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid);
                        llm_h2d(LLM_WADDR(next_in), cur_out.pVirAddr, next_in.nSize, next_devid);
                    }
                }
            }

            // post (input 已在上面路由写入)
            llama_post.inference();
            {
                auto &t_out = llama_post.get_output("output");
                llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
                unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
                next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            }

#else // AX650: 通过 host embed buffer 依次传递
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop) break;
                auto &lyr = llama_layers[m];

                auto &in_k = lyr.layer.get_input(decode_grpid, "K_cache");
                auto *in_k_ptr = (unsigned short *)in_k.pVirAddr;
                auto &in_v = lyr.layer.get_input(decode_grpid, "V_cache");
                auto *in_v_ptr = (unsigned short *)in_v.pVirAddr;

                auto &t_idx = lyr.layer.get_input(decode_grpid, "indices");
                memcpy(t_idx.pVirAddr, &indices, sizeof(indices));

                auto &t_mask = lyr.layer.get_input(decode_grpid, "mask");
                memcpy(t_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                auto &t_in = lyr.layer.get_input(decode_grpid, "input");
                memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                lyr.layer.inference(decode_grpid);

                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out");
                memcpy(in_k_ptr + indices * _attr.kv_cache_size, out_k.pVirAddr,
                       sizeof(unsigned short) * _attr.kv_cache_size);

                auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                memcpy(in_v_ptr + indices * _attr.kv_cache_size, out_v.pVirAddr,
                       sizeof(unsigned short) * _attr.kv_cache_size);

                auto &t_out = lyr.layer.get_output(decode_grpid, "output");
                memcpy(embed.data(), t_out.pVirAddr, embed.size() * sizeof(unsigned short));
            }

            // post
            {
                auto &t_in = llama_post.get_input("input");
                memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                llama_post.inference();
                auto &t_out = llama_post.get_output("output");
                unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
                next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            }
#endif

            mask[indices] = 0;

            // EOS / callback 处理
            if (tokenizer->is_stop(next_token))
            {
                b_hit_eos = true;
                break;
            }
            token_ids.push_back(next_token);

            if (_attr.runing_callback)
            {
                float t_ms  = t_cost.cost();
                float tps   = token_ids.size() / (t_ms / 1000.0f);
                auto  str   = utf8_filter.filter(tokenizer->decode(next_token));
                if (!str.empty())
                    _attr.runing_callback(str, tps, _attr.reserve);
            }

            if (output_max_token > 0 && (int)token_ids.size() >= output_max_token)
            {
                b_hit_eos = true;
                break;
            }

            if (_attr.runing_callback == nullptr)
                update_cqdm(&cqdm, indices, "token", "");
        }

        printf("\n\n");
        fflush(stdout);
        float t_ms = t_cost.cost();
        ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_ms / 1000.0f));

        final_out = tokenizer->decode(token_ids);
        return final_out;
    }

    // -----------------------------------------------------------------------
    // Run (高层接口) — 接收 Content 历史，自动管理 KV cache 增量
    // -----------------------------------------------------------------------
    std::vector<Content> Run(std::vector<Content> history, int output_max_token = -1)
    {
        auto new_tokens = tokenizer->encode(history);

        int offset = 0;
        auto tokens_diff = diff_token_ids(last_tokens_ids, new_tokens, offset);

        bool not_append = !(offset == (int)last_tokens_ids.size() &&
                            (int)new_tokens.size() >= (int)last_tokens_ids.size());
        if (not_append)
        {
            ALOGW("history not append (rollback/modify). force ResetKVCache and recompute.");
            ResetKVCache();
            tokens_diff   = new_tokens;
            offset        = 0;
        }

        if (tokens_diff.empty())
        {
            if (!new_tokens.empty())
            {
                precompute_len = (int)new_tokens.size() - 1;
                tokens_diff    = {new_tokens.back()};
            }
            else
            {
                ResetKVCache();
                precompute_len = 0;
            }
        }

        SetKVCache(k_caches, v_caches, precompute_len, (int)tokens_diff.size());

        std::vector<unsigned short> out_embed(tokens_diff.size() * _attr.tokens_embed_size);
        for (size_t i = 0; i < tokens_diff.size(); i++)
            embed_selector.getByIndex(tokens_diff[i], out_embed.data() + i * _attr.tokens_embed_size);

        auto reply = Run(out_embed, output_max_token);

        GetKVCache(k_caches, v_caches, precompute_len);

        history.push_back({ASSISTANT, TEXT, reply});
        last_tokens_ids = tokenizer->encode(history);
        // 删除最后两个 token（"<|im_start|>assistant\n"），避免重复计算
        if (last_tokens_ids.size() >= 2)
            last_tokens_ids.erase(last_tokens_ids.end() - 2, last_tokens_ids.end());

        return history;
    }
};
