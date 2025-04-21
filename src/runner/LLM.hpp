#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"

#include <ax_sys_api.h>
#include "LLMPostprocess.hpp"

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

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

    TokenizerType tokenizer_type = TKT_LLaMa;
    std::string url_tokenizer_model = "http://127.0.0.1:12345";
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    int precompute_len = 1202;
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
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;

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

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer = CreateTokenizer(attr.tokenizer_type);
        if (!tokenizer->Init(attr.url_tokenizer_model))
        {
            ALOGE("tokenizer.Init(%s) failed", attr.url_tokenizer_model.c_str());
            return false;
        }
        std::vector<int> _token_ids;
        tokenizer->Reset(attr.system_prompt, _token_ids);
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

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            llama_layers[i].layer.release();
        }
        llama_post.release();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int SetSystemPrompt(std::string system_prompt, std::vector<int> &_token_ids)
    {
        tokenizer->Reset(system_prompt, _token_ids);
        _attr.system_prompt = system_prompt;
        return 0;
    }

    int GenerateKVCachePrefill(std::vector<int> &_token_ids, std::vector<std::vector<unsigned short>> &k_caches, std::vector<std::vector<unsigned short>> &v_caches, int &precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        int input_embed_num = _token_ids.size();
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

        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];

        std::vector<unsigned short> test_embed;
        test_embed.resize(_token_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < _token_ids.size(); i++)
        {
            embed_selector.getByIndex(_token_ids[i], test_embed.data() + i * _attr.tokens_embed_size);
        }

        // axcl_Memcpy((void *)llama_layers[0].layer.get_input(_attr.prefill_grpid, "input").phyAddr, test_embed.data(), test_embed.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, llama_layers[0].layer.get_devid());
        // test_embed.resize(_attr.prefill_token_num * _attr.tokens_embed_size);

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
                if (b_stop)
                {
                    break;
                }

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

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer.get_input(prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(prefill_grpid, "V_cache_out");

                int kv_offset = (p * _attr.prefill_token_num) * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

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

        precompute_len = _token_ids.size();

        k_caches.resize(_attr.axmodel_num);
        v_caches.resize(_attr.axmodel_num);
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

    int GenerateKVCache(std::vector<int> &_token_ids)
    {
        // clear kv cache
        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize);
        }

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        mask[_attr.kv_cache_num] = 0;
        std::vector<unsigned short> embed;

        int next_token = _token_ids[0];

        t_cqdm cqdm = create_cqdm(_token_ids.size(), 32);

        for (unsigned int indices = 0; indices < _token_ids.size(); indices++)
        {
            // ALOGI("out %d %d", indices, next_token);
            embed_selector.getByIndex(next_token, embed);

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy(input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(decode_grpid, "input");
                memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(decode_grpid, "output");
                memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            mask[indices] = 0;
            next_token = _token_ids[indices + 1];
            update_cqdm(&cqdm, indices, "token", "");
            // ALOGI("");
        }
        return 0;
    }

    int GetKVCache(std::vector<std::vector<unsigned short>> &k_caches, std::vector<std::vector<unsigned short>> &v_caches, int &precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        auto &input_mask = llama_layers[0].layer.get_input(decode_grpid, "mask");
        memcpy(mask.data(), (void *)input_mask.pVirAddr, input_mask.nSize);
        for (size_t i = 0; i < mask.size(); i++)
        {
            if (mask[i] == bf16.data)
            {
                precompute_len = i + 1;
                break;
            }
        }
        ALOGI("precompute_len:%d, remaining:%d", precompute_len, _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1] - precompute_len);
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

        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];

        return 0;
    }

    int SetKVCache(std::vector<std::vector<unsigned short>> &k_caches, std::vector<std::vector<unsigned short>> &v_caches, int precompute_len, int input_num_token)
    {
        _attr.precompute_len = precompute_len;
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (_attr.precompute_len + input_num_token <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                _attr.prefill_grpid = i + 1;
                break;
            }
        }
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];
        ALOGI("prefill_grpid:%d kv_cache_num:%d precompute_len:%d input_num_token:%d", _attr.prefill_grpid, kv_cache_num, precompute_len, input_num_token);

        _attr.prefill_max_token_num = ALIGN_DOWN(_attr.prefill_max_token_num - _attr.precompute_len, _attr.prefill_token_num);
        ALOGI("current prefill_max_token_num:%d", _attr.prefill_max_token_num);

        if (precompute_len == 0)
        {
            ALOGI("first run");
            return 0;
        }

        if (precompute_len + input_num_token > kv_cache_num)
        {
            ALOGE("precompute_len(%d) + input_num_token(%d) > _attr.prefill_max_kv_cache_num_grp[%d]", precompute_len, input_num_token, _attr.prefill_grpid - 1);
            return -1;
        }

        if (input_num_token > _attr.prefill_max_token_num)
        {
            ALOGE("input_num_token(%d) > _attr.prefill_max_token_num(%d)", input_num_token, _attr.prefill_max_token_num);
            return -1;
        }

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
            memset((void *)llama_layers[i].layer.get_input(_attr.prefill_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(_attr.prefill_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(_attr.prefill_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(_attr.prefill_grpid, "V_cache").nSize);

            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize);
        }

        // int prefill_grpid = llama_layers[0].layer.get_num_input_groups() - 1;

        for (unsigned int m = 0; m < _attr.axmodel_num; m++)
        {
            auto &layer = llama_layers[m];

            auto &k_cache = k_caches[m];
            auto &v_cache = v_caches[m];

            if (k_cache.size() != _attr.precompute_len * _attr.kv_cache_size)
            {
                ALOGE("k_cache.size(%d) != precompute_len(%d) * _attr.kv_cache_size(%d)", k_cache.size(), _attr.precompute_len, _attr.kv_cache_size);
                return -1;
            }
            if (v_cache.size() < _attr.precompute_len * _attr.kv_cache_size)
            {
                ALOGE("v_cache.size(%d) < precompute_len(%d) * _attr.kv_cache_size(%d)", v_cache.size(), _attr.precompute_len, _attr.kv_cache_size);
                return -1;
            }

            // set kv cache inputs
            {
                auto &input_k_cache = layer.layer.get_input(_attr.prefill_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                auto &input_v_cache = layer.layer.get_input(_attr.prefill_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;

                memcpy(input_k_cache_ptr, k_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                memcpy(input_v_cache_ptr, v_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                // axcl_Memcpy((void *)input_k_cache_ptr, (void *)k_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
                // axcl_Memcpy((void *)input_v_cache_ptr, (void *)v_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
            }

            {
                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;

                memcpy(input_k_cache_ptr, k_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                memcpy(input_v_cache_ptr, v_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                // axcl_Memcpy((void *)input_k_cache_ptr, (void *)k_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
                // axcl_Memcpy((void *)input_v_cache_ptr, (void *)v_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
            }
        }

        return 0;
    }

    int Encode(std::vector<unsigned short> &out_embed, std::string prompt, std::string last_reply, std::vector<int> &tokens_ids, std::vector<int> &tokens_diff)
    {
        if (!tokenizer->Encode(prompt, last_reply, tokens_ids, tokens_diff))
        {
            ALOGE("encode failed");
            return -1;
        }

        out_embed.resize(tokens_diff.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < tokens_diff.size(); i++)
        {
            embed_selector.getByIndex(tokens_diff[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        // memcpy(out_embed.data() + 5 * _attr.tokens_embed_size, vpm_resampler.get_output("output").pVirAddr, vpm_resampler.get_output("output").nSize);

        return 0;
    }

    std::string Run(std::vector<unsigned short> test_embed)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];

        std::vector<int> cached_token;
        std::vector<int> token_ids;

        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < _attr.precompute_len + input_embed_num; i++)
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

                    for (int j = 0; j < _attr.precompute_len + p * _attr.prefill_token_num; j++)
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
                for (unsigned int i = _attr.precompute_len + p * _attr.prefill_token_num; i < _attr.precompute_len + (p + 1) * _attr.prefill_token_num; i++)
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

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer.get_input(_attr.prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(_attr.prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                int kv_offset = (_attr.precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

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
            cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;
        for (unsigned int indices = _attr.precompute_len + input_embed_num; indices < _attr.max_token_len; indices++)
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

                if (tokenizer->isEnd(max_index))
                {
                    if (cached_token.size() && _attr.runing_callback)
                    {
                        float t_cost_ms = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec, _attr.reserve);
                        cached_token.clear();
                    }
                    b_hit_eos = true;
                    break;
                }
                token_ids.push_back(max_index);

                if (_attr.runing_callback)
                {
                    cached_token.push_back(max_index);
                    if (cached_token.size() >= 3)
                    {
                        float t_cost_ms = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec, _attr.reserve);
                        cached_token.clear();
                    }
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

        final_out = tokenizer->Decode(token_ids);

        return final_out;
    }
};
