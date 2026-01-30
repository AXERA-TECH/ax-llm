#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "BaseTokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "UTF8Filter.hpp"

#include <ax_sys_api.h>
#include "LLMPostprocess.hpp"

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))

typedef void (*LLMRuningCallback)(const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string system_prompt;
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    // std::string template_prefill_filename_axmodel = "minicpmv/prefill_axmodel/minicpm_p96_l%d.axmodel";
    // int prefill_axmodel_num = 40;
    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    // std::string filename_vpm_resampler_axmodedl = "minicpmv/vpm_resampler_version0_fp16.axmodel";
    // int vpm_width = 280;
    // int vpm_height = 280;

    std::string url_tokenizer_model = "http://127.0.0.1:12345";
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    // int precompute_len = 1202;
    std::vector<int> prefill_max_kv_cache_num_grp;

    int prefill_grpid = -1;

    std::string post_config_path = "post_config.json";

    bool b_use_mmap_load_embed = false;

    bool b_use_mmap_load_layer = true;

    // bool b_live_print = true;
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
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;

    //
    int decode_grpid = 0;

    // ax_runner_ax650 vpm_resampler;

    // std::vector<std::vector<unsigned short>> k_caches, v_caches;

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

    // ids1: {1,2,3}
    // ids2: {1,2,3,4,5}
    // diff_ids: {4,5}
    // 这是用来比较两个 token id 序列的差异的函数，当新一轮对话用户输入prompt的时候，编码出来的token id 序列与上一轮的差异，
    // 得到新增的token id 序列，这样历史对话的token id 序列就不需要重复计算kvcached了
    std::vector<int> diff_token_ids(std::vector<int> ids1, std::vector<int> ids2)
    {
        if (ids1.size() >= ids2.size())
        {
            return {};
        }
        for (int i = 0; i < ids1.size(); i++)
        {
            if (ids1[i] != ids2[i])
            {
                return {};
            }
        }
        std::vector<int> diff_ids(ids2.begin() + ids1.size(), ids2.end());
        return diff_ids;
    }

    // ids1: {1,2,3,4,5}
    // ids2: {1,2,3,6,7}
    // diff_ids: {6,7}
    std::vector<int> diff_token_ids(const std::vector<int> &ids1, const std::vector<int> &ids2, int &offset)
    {
        int min_len = std::min(ids1.size(), ids2.size());
        offset = 0;

        // 1. 找到公共前缀的长度
        for (int i = 0; i < min_len; i++)
        {
            if (ids1[i] == ids2[i])
            {
                offset++;
            }
            else
            {
                // 2. 一旦发现不匹配，立即停止！
                break;
            }
        }

        // 3. 安全检查：如果 offset 已经等于 ids2 的长度，说明 ids2 完全被包含在 ids1 里（没有新内容）
        if (offset >= ids2.size())
        {
            return {}; // 返回空 vector
        }

        // 4. 截取 ids2 中 offset 之后的部分
        // 此时 offset 肯定 <= ids2.size()，所以是安全的
        std::vector<int> diff_ids(ids2.begin() + offset, ids2.end());
        return diff_ids;
    }

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer = create_tokenizer(Qwen3);
        if (!tokenizer->load(attr.url_tokenizer_model))
        {
            ALOGE("tokenizer.load(%s) failed", attr.url_tokenizer_model.c_str());
            return false;
        }
        tokenizer->set_think_in_prompt(true);

        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");
        // test code
        // {
        //     std::vector<int> output;
        //     tokenizer.Encode("Today is National", output);
        //     // print output
        //     for (size_t i = 0; i < output.size(); i++)
        //     {
        //         printf("%d ", output[i]);
        //     }
        //     printf("\n");
        // }

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size, attr.b_use_mmap_load_embed))
        {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num, attr.tokens_embed_size);
            return false;
        }
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");
        // test code
        // {
        //     std::vector<unsigned short> embed = embed_selector.getByIndex(123);
        //     printf("embed size: %d\n", embed.size());
        //     for (int i = 0; i < embed.size(); i++)
        //     {
        //         bfloat16 bf16 = bfloat16(embed[i]);
        //         float val = bf16;
        //         printf("%d %0.22f\n", embed[i], val);
        //     }
        // }

        llama_layers.resize(attr.axmodel_num);
        // prefill_layers.resize(attr.prefill_axmodel_num);

        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), false);
            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 2, "count", axmodel_path);
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        // int remain_cmm = get_remaining_cmm_size();
        // sprintf(axmodel_path, "init vpm axmodel ok,remain_cmm(%d MB)", remain_cmm);
        // update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        {
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            ALOGI("max_token_len : %d", _attr.max_token_len);
            // auto &input_k_cache = llama_layers[0].layer.get_input("K_cache");
            // auto &output_k_cache_out = llama_layers[0].layer.get_output("K_cache_out");
            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = llama_layers[0].layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
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
                int prefill_max_kv_cache_num = llama_layers[0].layer.get_input(i + 1, "K_cache").vShape[1];
                ALOGI("grp: %d, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
                _attr.prefill_max_kv_cache_num_grp.push_back(prefill_max_kv_cache_num);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }

        if (!postprocess.load_config(attr.post_config_path))
        {
            ALOGW("load postprocess config(%s) failed", attr.post_config_path.c_str());
        }

        // Reset();
        ALOGI("LLM init ok");
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    LLMPostprocess *getPostprocess()
    {
        return &postprocess;
    }

    LLaMaEmbedSelector *getEmbedSelector()
    {
        return &embed_selector;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            llama_layers[i].layer.deinit();
        }
        llama_post.deinit();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    // for test only
    int GenerateKVCachePrefill(std::vector<int> &_token_ids, std::vector<std::vector<unsigned short>> &k_caches, std::vector<std::vector<unsigned short>> &v_caches, int &precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        int input_embed_num = _token_ids.size();
        precompute_len = _token_ids.size();

        k_caches.resize(_attr.axmodel_num);
        v_caches.resize(_attr.axmodel_num);
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);

        int prefill_grpid = _attr.prefill_max_kv_cache_num_grp.size();

        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (input_embed_num <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                prefill_grpid = i + 1;
                break;
            }
        }
        ALOGI("input token num : %d, prefill_split_num : %d prefill_grpid : %d", input_embed_num, prefill_split_num, prefill_grpid);

        // clear kv cache
        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            memset((void *)llama_layers[i].layer.get_input(prefill_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(prefill_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(prefill_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(prefill_grpid, "V_cache").nSize);
        }

        if (input_embed_num == 0)
        {
            for (size_t i = 0; i < _attr.axmodel_num; i++)
            {
                k_caches[i].resize(precompute_len * _attr.kv_cache_size);
                v_caches[i].resize(precompute_len * _attr.kv_cache_size);
            }
            ALOGI("input token num is 0, skip");
            return 0;
        }

        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];

        std::vector<unsigned short> test_embed;
        test_embed.resize(_token_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < _token_ids.size(); i++)
        {
            embed_selector.getByIndex(_token_ids[i], test_embed.data() + i * _attr.tokens_embed_size);
        }

        for (size_t p = 0; p < prefill_split_num; p++)
        {
            std::vector<unsigned short> mask_tmp;
            mask_tmp.resize(1 * _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            int input_num_token = _attr.prefill_token_num;
            if (p == prefill_split_num - 1)
            {
                input_num_token = input_embed_num - p * _attr.prefill_token_num;
            }

            ALOGI("input_num_token:%d", input_num_token);
            for (size_t i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    int mask_current_start = kv_cache_num;
                    auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);

                    for (int j = 0; j < p * _attr.prefill_token_num; j++)
                    {
                        mask_ptr[j] = 0;
                    }

                    for (int j = mask_current_start; j < mask_current_start + i + 1; j++)
                    {
                        mask_ptr[j] = 0;
                    }
                }
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            }
            else
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
            {
                auto &layer = llama_layers[m];
                // set indices
                auto &input_indices = layer.layer.get_input(prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                int idx = 0;
                for (unsigned int i = p * _attr.prefill_token_num; i < (p + 1) * _attr.prefill_token_num; i++)
                {
                    input_indices_ptr[idx] = i;
                    idx++;
                }
                // memcpy((void *)input_indices.phyAddr, input_indices_ptr, input_indices.nSize);

                // set mask
                auto &input_mask = layer.layer.get_input(prefill_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(prefill_grpid, "input");
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(prefill_grpid);

                // auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                // auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer.get_input(prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(prefill_grpid, "V_cache_out");

                int kv_offset = (p * _attr.prefill_token_num) * _attr.kv_cache_size;

                // memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                //        (void *)output_k_cache.pVirAddr,
                //        sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                // memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                //        (void *)output_v_cache.pVirAddr,
                //        sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(prefill_grpid, "output");
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
        }

        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            auto &layer = llama_layers[i];
            k_caches[i].resize(precompute_len * _attr.kv_cache_size);
            v_caches[i].resize(precompute_len * _attr.kv_cache_size);
            auto &input_k_cache = layer.layer.get_input(prefill_grpid, "K_cache");
            auto &input_v_cache = layer.layer.get_input(prefill_grpid, "V_cache");
            memcpy((void *)k_caches[i].data(), (void *)input_k_cache.pVirAddr, precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
            memcpy((void *)v_caches[i].data(), (void *)input_v_cache.pVirAddr, precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
        }

        return 0;
    }

    int GetKVCache(std::vector<std::vector<unsigned short>> &k_caches, std::vector<std::vector<unsigned short>> &v_caches, int &precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        // std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        auto &input_mask = llama_layers[0].layer.get_input(decode_grpid, "mask");
        unsigned short *mask = (unsigned short *)input_mask.pVirAddr;
        // memcpy(mask.data(), (void *)input_mask.pVirAddr, input_mask.nSize);
        for (size_t i = 0; i < input_mask.nSize / sizeof(unsigned short); i++)
        {
            if (mask[i] == bf16.data)
            {
                precompute_len = i + 1;
                break;
            }
        }
        ALOGI("precompute_len:%d, remaining:%d", precompute_len, _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1] - precompute_len);
        if (b_os_kvcache)
        {
            k_caches.resize(_attr.axmodel_num);
            v_caches.resize(_attr.axmodel_num);
            for (size_t i = 0; i < _attr.axmodel_num; i++)
            {
                auto &layer = llama_layers[i];
                k_caches[i].resize(precompute_len * _attr.kv_cache_size);
                v_caches[i].resize(precompute_len * _attr.kv_cache_size);
                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                memcpy((void *)k_caches[i].data(), (void *)input_k_cache.pVirAddr, precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                memcpy((void *)v_caches[i].data(), (void *)input_v_cache.pVirAddr, precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
            }
        }

        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];

        return 0;
    }

    int SetKVCache(std::vector<std::vector<unsigned short>> &k_caches, std::vector<std::vector<unsigned short>> &v_caches, int _precompute_len, int input_num_token)
    {
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (_precompute_len + input_num_token <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                _attr.prefill_grpid = i + 1;
                break;
            }
        }
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];
        ALOGI("prefill_grpid:%d kv_cache_num:%d precompute_len:%d input_num_token:%d", _attr.prefill_grpid, kv_cache_num, _precompute_len, input_num_token);

        _attr.prefill_max_token_num = ALIGN_DOWN(_attr.prefill_max_token_num - _precompute_len, _attr.prefill_token_num);
        ALOGI("current prefill_max_token_num:%d", _attr.prefill_max_token_num);

        if (_precompute_len == 0)
        {
            ALOGI("first run");
            return 0;
        }

        if (_precompute_len + input_num_token > kv_cache_num)
        {
            ALOGE("precompute_len(%d) + input_num_token(%d) > _attr.prefill_max_kv_cache_num_grp[%d]", _precompute_len, input_num_token, _attr.prefill_grpid - 1);
            return -1;
        }

        if (input_num_token > _attr.prefill_max_token_num)
        {
            ALOGE("input_num_token(%d) > _attr.prefill_max_token_num(%d)", input_num_token, _attr.prefill_max_token_num);
            return -1;
        }
        if (b_os_kvcache)
        {
            if (k_caches.size() != v_caches.size())
            {
                ALOGE("k_caches.size(%d) != v_caches.size(%d)", k_caches.size(), v_caches.size());
                return -1;
            }

            if (k_caches.size() != _attr.axmodel_num)
            {
                ALOGE("k_caches.size(%d) != _attr.axmodel_num(%d)", k_caches.size(), _attr.axmodel_num);
                return -1;
            }

            // clear kv cache

            for (size_t i = 0; i < _attr.axmodel_num; i++)
            {
                memset((void *)llama_layers[i].layer.get_input(decode_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize);
                memset((void *)llama_layers[i].layer.get_input(decode_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize);
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
            {
                auto &layer = llama_layers[m];

                auto &k_cache = k_caches[m];
                auto &v_cache = v_caches[m];

                if (k_cache.size() < _precompute_len * _attr.kv_cache_size)
                {
                    ALOGE("k_cache.size(%d) < precompute_len(%d) * _attr.kv_cache_size(%d)", k_cache.size(), _precompute_len, _attr.kv_cache_size);
                    return -1;
                }
                if (v_cache.size() < _precompute_len * _attr.kv_cache_size)
                {
                    ALOGE("v_cache.size(%d) < precompute_len(%d) * _attr.kv_cache_size(%d)", v_cache.size(), _precompute_len, _attr.kv_cache_size);
                    return -1;
                }

                {
                    auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                    unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                    auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                    unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;

                    memcpy(input_k_cache_ptr, k_cache.data(), _precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                    memcpy(input_v_cache_ptr, v_cache.data(), _precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                }
            }
        }
        return 0;
    }

    void ResetKVCache()
    {
        last_tokens_ids.clear();
        k_caches.clear();
        v_caches.clear();
        precompute_len = 0;

        // reset kv cache
        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize);
        }
    }
    std::vector<Content> Run(std::vector<Content> history, int output_max_token = -1)
    {
        auto new_tokens = tokenizer->encode(history);

        int offset = 0;
        auto tokens_diff = diff_token_ids(last_tokens_ids, new_tokens, offset);

        bool not_append = !(offset == (int)last_tokens_ids.size() && new_tokens.size() >= last_tokens_ids.size());

        if (not_append)
        {
            ALOGW("history not append (rollback/modify). force ResetKVCache and recompute.");
            ResetKVCache();
            last_tokens_ids.clear();
            k_caches.clear();
            v_caches.clear();
            precompute_len = 0;

            // 重算 diff（此时 old 为空 => diff = 全量 tokens）
            tokens_diff = new_tokens;
            offset = 0;
        }

        if (tokens_diff.empty())
        {
            if (!new_tokens.empty())
            {
                // 重新跑最后一个 token，以获得正确的 embed
                precompute_len = (int)new_tokens.size() - 1;
                tokens_diff = {new_tokens.back()};
            }
            else
            {
                // 只有 system 之类的极端情况
                ResetKVCache();
                tokens_diff.clear();
                precompute_len = 0;
            }
        }

        SetKVCache(k_caches, v_caches, precompute_len, tokens_diff.size());
        std::vector<unsigned short> out_embed(tokens_diff.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < tokens_diff.size(); i++)
        {
            embed_selector.getByIndex(tokens_diff[i], out_embed.data() + i * _attr.tokens_embed_size);
        }
        auto reply = Run(out_embed, output_max_token);

        GetKVCache(k_caches, v_caches, precompute_len);

        history.push_back({ASSISTANT, TEXT, reply});
        last_tokens_ids = tokenizer->encode(history);
        // 删除最后两个token_ids， “<|im_start|>assistant\n”，否则会跳过计算
        last_tokens_ids.erase(last_tokens_ids.end() - 2, last_tokens_ids.end());
        return history;
    }

    std::string Run(std::vector<unsigned short> &test_embed, int output_max_token = -1)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];

        // std::vector<int> cached_token;
        std::vector<int> token_ids;

        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < precompute_len + input_embed_num; i++)
        {
            mask[i] = 0;
        }
        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        for (size_t p = 0; p < prefill_split_num; p++)
        {
            if (b_stop)
            {
                break;
            }

            std::vector<unsigned short> mask_tmp;
            mask_tmp.resize(1 * _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            int input_num_token = _attr.prefill_token_num;
            if (p == prefill_split_num - 1)
            {
                input_num_token = input_embed_num - p * _attr.prefill_token_num;
            }

            ALOGI("input_num_token:%d", input_num_token);
            for (size_t i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    int mask_current_start = kv_cache_num;
                    auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);

                    for (int j = 0; j < precompute_len + p * _attr.prefill_token_num; j++)
                    {
                        mask_ptr[j] = 0;
                    }

                    for (int j = mask_current_start; j < mask_current_start + i + 1; j++)
                    {
                        mask_ptr[j] = 0;
                    }
                }
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            }
            else
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                // set indices
                auto &input_indices = layer.layer.get_input(_attr.prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                int idx = 0;
                for (unsigned int i = precompute_len + p * _attr.prefill_token_num; i < precompute_len + (p + 1) * _attr.prefill_token_num; i++)
                {
                    input_indices_ptr[idx] = i;
                    idx++;
                }
                // memcpy((void *)input_indices.phyAddr, input_indices_ptr, input_indices.nSize, AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                // set mask
                auto &input_mask = layer.layer.get_input(_attr.prefill_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));

                // set input
                auto &input_input = layer.layer.get_input(_attr.prefill_grpid, "input");
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(_attr.prefill_grpid);

                // auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                // auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer.get_input(_attr.prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(_attr.prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                int kv_offset = (precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;

                // memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                //        (void *)output_k_cache.pVirAddr,
                //        sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                // memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                //        (void *)output_v_cache.pVirAddr,
                //        sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(_attr.prefill_grpid, "output");
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed.data(),
                       embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            }
        }

        int next_token = -1;
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);

        {

            // post process
            auto &input = llama_post.get_input("input");
            memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference();
            int max_index;

            auto &output_post = llama_post.get_output("output");
            // AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
            unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
            float max_val = -MAXFLOAT;
            // max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
            max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

            next_token = max_index;

            token_ids.push_back(max_index);
            // cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
            if (_attr.runing_callback)
            {
                auto str = utf8_filter.filter(tokenizer->decode(max_index));
                if (!str.empty())
                {
                    _attr.runing_callback(str.c_str(), -1, _attr.reserve);
                }
            }
        }
        t_cost.start();

        bool b_hit_eos = false;
        for (unsigned int indices = precompute_len + input_embed_num; indices < _attr.max_token_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

            // ALOGI("out %d %d", indices, next_token);
            embed_selector.getByIndex(next_token, embed);
            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                // memcpy(input_k_cache.pVirAddr, k_caches[m].data(), sizeof(unsigned short) * k_caches[m].size());
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;
                // memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy(input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(decode_grpid, "input");
                memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                // AX_SYS_MinvalidateCache(output_k_cache.phyAddr, output_k_cache.pVirAddr, output_k_cache.nSize);
                memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                // AX_SYS_MinvalidateCache(output_v_cache.phyAddr, output_v_cache.pVirAddr, output_v_cache.nSize);
                memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(decode_grpid, "output");
                // AX_SYS_MinvalidateCache(output.phyAddr, output.pVirAddr, output.nSize);
                memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            // ALOGI("");
            mask[indices] = 0;
            {
                // post process
                auto &input = llama_post.get_input("input");
                memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                llama_post.inference();
                int max_index;

                auto &output_post = llama_post.get_output("output");
                // AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val = -MAXFLOAT;
                // max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
                max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

                next_token = max_index;

                if (tokenizer->is_stop(max_index))
                {
                    // if (cached_token.size() && _attr.runing_callback)
                    // {
                    //     float t_cost_ms = t_cost.cost();
                    //     float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                    //     auto tmp_out = tokenizer->decode(cached_token);
                    //     _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec, _attr.reserve);
                    //     cached_token.clear();
                    // }
                    b_hit_eos = true;
                    break;
                }
                token_ids.push_back(max_index);

                if (_attr.runing_callback)
                {
                    // cached_token.push_back(max_index);
                    // if (cached_token.size() >= 3)
                    // {
                    float t_cost_ms = t_cost.cost();
                    float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                    auto tmp_out = tokenizer->decode(max_index);
                    if (!tmp_out.empty())
                    {
                        _attr.runing_callback(tmp_out.c_str(), token_per_sec, _attr.reserve);
                    }
                }
                if (output_max_token > 0 && token_ids.size() >= output_max_token)
                {
                    b_hit_eos = true;
                    break;
                }
            }

            if (_attr.runing_callback == nullptr)
                update_cqdm(&cqdm, indices, "token", "");
            if (b_hit_eos)
            {
                break;
            }
        }
        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_cost_ms / 1000));

        // 去掉 len_of_input 那部分
        // token_ids.erase(token_ids.begin(), token_ids.begin() + len_of_input);

        final_out = tokenizer->decode(token_ids);

        return final_out;
    }
};
