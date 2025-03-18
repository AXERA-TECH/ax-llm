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
#include "LLMPostprocess.hpp"

#include <axcl.h>
#include <axcl_rt_memory.h>

#define USE_SET_INPUT 1

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    TokenizerType tokenizer_type = TKT_LLaMa;
    std::string filename_tokenizer_model = "tokenizer.model";
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    bool b_use_mmap_load_embed = false;

    bool b_use_mmap_load_layer = true;

    std::string post_config_path = "post_config.json";

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

    // std::vector<std::vector<unsigned short>> k_caches, v_caches;

    bool b_stop = false;

#if USE_SET_INPUT
    unsigned int *p_indices_list;
    unsigned short *p_mask_list;
#endif

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
        if (!tokenizer->Init(attr.filename_tokenizer_model, attr.b_bos, attr.b_eos))
        {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.filename_tokenizer_model.c_str(), attr.b_bos, attr.b_eos);
            return false;
        }
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
        axclrtDeviceList dev_list = {0};
        if (auto ret = axclrtGetDeviceList(&dev_list); ret != 0 || dev_list.num == 0)
        {
            ALOGE("axclrtGetDeviceList failed");
            return false;
        }

        llama_layers.resize(attr.axmodel_num);

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
            int remain_cmm = get_pcie_remaining_cmm_size(dev_list.devices[0]);
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 2, "count", axmodel_path);
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);

        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", "init post axmodel ok\n");

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
        }

#if USE_SET_INPUT
        // 类似人体蜈蚣，将输入输出串联起来，减少内存拷贝
        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            auto &layer = llama_layers[m];

            if (m == _attr.axmodel_num - 1)
            {
                llama_post.set_input(llama_post.get_input("input").nIdx, layer.layer.get_output("output").phyAddr, layer.layer.get_output("output").nSize);
            }
            else if (m < _attr.axmodel_num - 1)
            {
                llama_layers[m + 1].layer.set_input(llama_layers[m + 1].layer.get_input("input").nIdx, layer.layer.get_output("output").phyAddr, layer.layer.get_output("output").nSize);
            }
        }
        {
            bfloat16 bf16 = -65536.f;
            axclrtMalloc((void **)&p_indices_list, _attr.max_token_len * sizeof(unsigned int), axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
            axclrtMalloc((void **)&p_mask_list, _attr.max_token_len * (_attr.kv_cache_num + 1) * sizeof(unsigned short), axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);

            std::vector<unsigned int> indices_list(_attr.max_token_len, 0);

            std::vector<unsigned short> tmp_mask_list(_attr.max_token_len * (_attr.kv_cache_num + 1), bf16.data);

            std::vector<unsigned short> tmp_mask(_attr.kv_cache_num + 1, bf16.data);
            tmp_mask[_attr.kv_cache_num] = 0;
            for (unsigned int indices = 0; indices < _attr.max_token_len; indices++)
            {
                indices_list[indices] = indices;
                tmp_mask[indices] = 0;
                // printf("%d %d %d\n", indices, tmp_mask.size(), tmp_mask_list.size());
                memcpy(tmp_mask_list.data() + indices * tmp_mask.size(), tmp_mask.data(), tmp_mask.size() * sizeof(unsigned short));
                // axclrtMemcpy(p_mask_list + indices * tmp_mask.size(), tmp_mask.size() * sizeof(unsigned short), tmp_mask.data(), tmp_mask.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE);
            }
            axclrtMemcpy(p_indices_list, indices_list.data(), _attr.max_token_len * sizeof(unsigned int), AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(p_mask_list, tmp_mask_list.data(), tmp_mask_list.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE);
        }
#endif

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
#if USE_SET_INPUT
        axclrtFree(p_indices_list);
        axclrtFree(p_mask_list);
#endif
        axclFinalize();
    }

    // void Reset()
    // {
    //     k_caches.resize(_attr.axmodel_num, std::vector<unsigned short>(_attr.kv_cache_num * _attr.kv_cache_size, 0));
    //     v_caches.resize(_attr.axmodel_num, std::vector<unsigned short>(_attr.kv_cache_num * _attr.kv_cache_size, 0));
    // }

    void Stop()
    {
        b_stop = true;
    }

    std::string Run(std::string input_str)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        mask[_attr.kv_cache_num] = 0;
        std::vector<int> cached_token;
        std::vector<int> token_ids = tokenizer->Encode(input_str);
        int len_of_input = token_ids.size();
        timer t_cost;
        // print token_ids
        // printf("%s\n", input_str.c_str());
        // for (size_t i = 0; i < token_ids.size(); i++)
        // {
        //     printf("%d ", token_ids[i]);
        // }
        // printf("\n");

        int next_token = token_ids[0];
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
        std::vector<unsigned short> embed;

        bool b_hit_eos = false;
        for (unsigned int indices = 0; indices < _attr.max_token_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

            embed_selector.getByIndex(next_token, embed);

            // embed_selector.getByIndex(next_token, embed);

            axclrtMemcpy((void *)llama_layers[0].layer.get_input("input").phyAddr, embed.data(), llama_layers[0].layer.get_input("input").nSize, AXCL_MEMCPY_HOST_TO_DEVICE);

            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            axclrtEngineSequence seq = nullptr;
            if (auto ret = axclrtEngineCreateSequence(&seq); ret != 0)
            {
                ALOGE("axclrtEngineCreateSequence failed");
                return "";
            }
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

#if USE_SET_INPUT
                // axclrtMemcpy((void *)layer.layer.get_input("indices").phyAddr, sizeof(indices), &indices, sizeof(indices), AXCL_MEMCPY_HOST_TO_DEVICE);
                layer.layer.set_input(layer.layer.get_input("indices").nIdx, (unsigned long long)(p_indices_list + indices), sizeof(unsigned int) * _attr.max_token_len);
                // axclrtMemcpy((void *)layer.layer.get_input("mask").phyAddr, mask.size() * sizeof(unsigned short), mask.data(), mask.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE);
                layer.layer.set_input(layer.layer.get_input("mask").nIdx, (unsigned long long)(p_mask_list + indices * (_attr.kv_cache_num + 1)), mask.size() * sizeof(unsigned short));
#else
                axclrtMemcpy((void *)layer.layer.get_input("indices").phyAddr, &indices, sizeof(indices), AXCL_MEMCPY_HOST_TO_DEVICE);
                axclrtMemcpy((void *)layer.layer.get_input("mask").phyAddr, mask.data(), mask.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE);
#endif

                // if (m == 0)
                //     axclrtMemcpy((void *)layer.layer.get_input("input").phyAddr, layer.layer.get_input("input").nSize, embed, layer.layer.get_input("input").nSize, AXCL_MEMCPY_HOST_TO_DEVICE);

#if USE_SET_INPUT
                {
                    unsigned short *input_k_cache_ptr = (unsigned short *)layer.layer.get_input("K_cache").phyAddr;
                    unsigned short *input_v_cache_ptr = (unsigned short *)layer.layer.get_input("V_cache").phyAddr;
                    layer.layer.set_output(layer.layer.get_output("K_cache_out").nIdx, (unsigned long long)(input_k_cache_ptr + indices * _attr.kv_cache_size), sizeof(unsigned short) * _attr.kv_cache_size);
                    layer.layer.set_output(layer.layer.get_output("V_cache_out").nIdx, (unsigned long long)(input_v_cache_ptr + indices * _attr.kv_cache_size), sizeof(unsigned short) * _attr.kv_cache_size);
                }
#endif
                // layer.layer.inference();
                if (auto ret = axclrtEnginePushModelTask(seq, layer.layer.getModelID(), layer.layer.getContextID(), 0, layer.layer.getIO()); ret != 0)
                {
                    ALOGE("axclrtEnginePushModelTask failed");
                    return "";
                }
#if !USE_SET_INPUT
                {
                    unsigned short *input_k_cache_ptr = (unsigned short *)layer.layer.get_input("K_cache").phyAddr;
                    unsigned short *input_v_cache_ptr = (unsigned short *)layer.layer.get_input("V_cache").phyAddr;

                    axclrtMemcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, (void *)layer.layer.get_output("K_cache_out").phyAddr, sizeof(unsigned short) * _attr.kv_cache_size, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                    axclrtMemcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, (void *)layer.layer.get_output("V_cache_out").phyAddr, sizeof(unsigned short) * _attr.kv_cache_size, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                    if (m == _attr.axmodel_num - 1)
                        axclrtMemcpy((void *)llama_post.get_input("input").phyAddr,
                                     (void *)layer.layer.get_output("output").phyAddr, llama_post.get_input("input").nSize, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                    else if (m < _attr.axmodel_num - 1)
                    {
                        axclrtMemcpy((void *)llama_layers[m + 1].layer.get_input("input").phyAddr,
                                     (void *)layer.layer.get_output("output").phyAddr, layer.layer.get_input("input").nSize, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                    }
                }
#endif
            }

            mask[indices] = 0;
            if (indices + 1 < token_ids.size())
            {
                if (auto ret = axclrtEngineSubmitSequence(seq); ret != 0)
                {
                    ALOGE("axclrtEngineSubmitSequence failed");
                    return "";
                }
                if (auto ret = axclrtEngineDestroySequence(seq); ret != 0)
                {
                    ALOGE("axclrtEngineDestroySequence failed");
                    return "";
                }

                next_token = token_ids[indices + 1];
            }
            else
            {
                // post process
                // llama_post.inference();
                if (auto ret = axclrtEnginePushModelTask(seq, llama_post.getModelID(), llama_post.getContextID(), 0, llama_post.getIO()); ret != 0)
                {
                    ALOGE("axclrtEnginePushModelTask failed");
                    return "";
                }
                if (auto ret = axclrtEngineSubmitSequence(seq); ret != 0)
                {
                    ALOGE("axclrtEngineSubmitSequence failed");
                    return "";
                }
                if (auto ret = axclrtEngineDestroySequence(seq); ret != 0)
                {
                    ALOGE("axclrtEngineDestroySequence failed");
                    return "";
                }

                auto &output_post = llama_post.get_output("output");
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                axclrtMemcpy(post_out, (void *)output_post.phyAddr, output_post.nSize, AXCL_MEMCPY_DEVICE_TO_HOST);

                auto max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
                next_token = max_index;

                if (tokenizer->isEnd(max_index))
                {
                    if (cached_token.size())
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
        token_ids.erase(token_ids.begin(), token_ids.begin() + len_of_input);

        final_out = tokenizer->Decode(token_ids);

        return final_out;
    }
};
