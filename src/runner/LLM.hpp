#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_model_runner/ax_model_runner_ax650_host.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"

#include "axcl/rt/axcl_rt_memory.h"

#define HOST_DEBUG 0

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
    bool b_dynamic_load_axmodel_layer = false;

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
#if HOST_DEBUG
        ax_runner_ax650_host layer_host;
#endif
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;
#if HOST_DEBUG
    ax_runner_ax650_host llama_post_host;
#endif

    // std::vector<std::vector<unsigned short>> k_caches, v_caches;

    bool b_stop = false;

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

        llama_layers.resize(attr.axmodel_num);

        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            if (!attr.b_dynamic_load_axmodel_layer)
            {
                int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), false);
#if HOST_DEBUG
                ret = llama_layers[i].layer_host.init(llama_layers[i].filename.c_str(), false);
#endif
                if (ret != 0)
                {
                    ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                    return false;
                }
                int remain_cmm = get_remaining_cmm_size();
                sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
                update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            }
            else
            {
                if (!attr.b_use_mmap_load_layer)
                {
                    if (!read_file(llama_layers[i].filename, llama_layers[i].layer_buffer_vec))
                    {
                        ALOGE("read_file(%s) failed", llama_layers[i].filename.c_str());
                        return false;
                    }
                }
                else
                {
                    llama_layers[i].layer_buffer.open_file(llama_layers[i].filename.c_str());
                }

                sprintf(axmodel_path, "read_file %s ok", llama_layers[i].filename.c_str());
                update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            }
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);
#if HOST_DEBUG
        ret = llama_post_host.init(attr.filename_post_axmodel.c_str(), false);
#endif
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", "init post axmodel ok\n");

        if (attr.b_dynamic_load_axmodel_layer)
        {
            // 加载第一层获取shape信息
            auto &layer = llama_layers[0];
            int ret;
            if (_attr.b_use_mmap_load_layer)
            {
                ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size());
            }
            else
            {
                ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size());
            }
            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", layer.filename.c_str());
            }
        }

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
        if (attr.b_dynamic_load_axmodel_layer)
        {
            auto &layer = llama_layers[0];
            layer.layer.deinit();
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
#if HOST_DEBUG
            llama_layers[i].layer_host.release();
#endif
        }
        llama_post.release();
#if HOST_DEBUG
        llama_post_host.release();
#endif
        embed_selector.Deinit();
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
#if HOST_DEBUG
        std::vector<unsigned short> embed_host;
#endif
        bool b_hit_eos = false;
        for (unsigned int indices = 0; indices < _attr.max_token_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

            embed_selector.getByIndex(next_token, embed);
#if HOST_DEBUG
            embed_host = embed;
#endif

            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    int ret;
                    if (_attr.b_use_mmap_load_layer)
                    {
                        ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size());
                    }
                    else
                    {
                        ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size());
                    }
                    if (ret != 0)
                    {
                        ALOGE("init axmodel(%s) failed", layer.filename.c_str());
                    }
                }

                // auto &input_k_cache = layer.layer.get_input("K_cache");
                // unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                // // memcpy(input_k_cache.pVirAddr, k_caches[m].data(), sizeof(unsigned short) * k_caches[m].size());
                // auto &input_v_cache = layer.layer.get_input("V_cache");
                // unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;
                // // memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());

                // memcpy(layer.layer.get_input("indices").pVirAddr, &indices, sizeof(indices));
                axclrtMemcpy((void *)layer.layer.get_input("indices").phyAddr, sizeof(indices), &indices, sizeof(indices), AXCL_MEMCPY_HOST_TO_DEVICE);
                // memcpy(layer.layer.get_input("mask").pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));
                axclrtMemcpy((void *)layer.layer.get_input("mask").phyAddr, mask.size() * sizeof(unsigned short), mask.data(), mask.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE);
                // memcpy(layer.layer.get_input("input").pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                axclrtMemcpy((void *)layer.layer.get_input("input").phyAddr, embed.size() * sizeof(unsigned short), embed.data(), embed.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE);

                // memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());
#if HOST_DEBUG
                memcpy(layer.layer_host.get_input("indices").pVirAddr, &indices, sizeof(indices));
                memcpy(layer.layer_host.get_input("mask").pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));
                memcpy(layer.layer_host.get_input("input").pVirAddr, embed_host.data(), embed_host.size() * sizeof(unsigned short));
                layer.layer_host.inference();
#endif

                layer.layer.inference();

                {
                    unsigned short *input_k_cache_ptr = (unsigned short *)layer.layer.get_input("K_cache").phyAddr;
                    unsigned short *input_v_cache_ptr = (unsigned short *)layer.layer.get_input("V_cache").phyAddr;
                    // memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, layer.layer.get_output("K_cache_out").pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                    axclrtMemcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, sizeof(unsigned short) * _attr.kv_cache_size, (void *)layer.layer.get_output("K_cache_out").phyAddr, sizeof(unsigned short) * _attr.kv_cache_size, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                    // memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, layer.layer.get_output("V_cache_out").pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                    axclrtMemcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, sizeof(unsigned short) * _attr.kv_cache_size, (void *)layer.layer.get_output("V_cache_out").phyAddr, sizeof(unsigned short) * _attr.kv_cache_size, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                    // memcpy(embed.data(), layer.layer.get_output("output").pVirAddr, embed.size() * sizeof(unsigned short));
                    axclrtMemcpy(embed.data(), embed.size() * sizeof(unsigned short), (void *)layer.layer.get_output("output").phyAddr, embed.size() * sizeof(unsigned short), AXCL_MEMCPY_DEVICE_TO_HOST);
                }
#if HOST_DEBUG
                {
                    // ALOGI("slave:%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
                    unsigned short *input_k_cache_ptr_host = (unsigned short *)layer.layer_host.get_input("K_cache").pVirAddr;
                    unsigned short *input_v_cache_ptr_host = (unsigned short *)layer.layer_host.get_input("V_cache").pVirAddr;
                    memcpy(input_k_cache_ptr_host + indices * _attr.kv_cache_size, layer.layer_host.get_output("K_cache_out").pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                    memcpy(input_v_cache_ptr_host + indices * _attr.kv_cache_size, layer.layer_host.get_output("V_cache_out").pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                    memcpy(embed_host.data(), layer.layer_host.get_output("output").pVirAddr, embed_host.size() * sizeof(unsigned short));
                    // ALOGI(" host:%f %f %f %f %f", bfloat16(embed_host[0]).fp32(), bfloat16(embed_host[1]).fp32(), bfloat16(embed_host[2]).fp32(), bfloat16(embed_host[3]).fp32(), bfloat16(embed_host[4]).fp32());
                }
#endif
                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    layer.layer.deinit();
                }
            }
#if HOST_DEBUG
            ALOGI("");
#endif
            mask[indices] = 0;
            if (indices + 1 < token_ids.size())
            {
                next_token = token_ids[indices + 1];
            }
            else
            {
                // post process
                auto &input = llama_post.get_input("input");
                // memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                axclrtMemcpy((void *)input.phyAddr, embed.size() * sizeof(unsigned short), embed.data(), embed.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE);
                llama_post.inference();
                auto &output_post = llama_post.get_output("output");
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                axclrtMemcpy(post_out, output_post.nSize, (void *)output_post.phyAddr, output_post.nSize, AXCL_MEMCPY_DEVICE_TO_HOST);
#if HOST_DEBUG
                {
                    printf("slave: %f %f %f %f %f\n", bfloat16(post_out[0]).fp32(), bfloat16(post_out[1]).fp32(), bfloat16(post_out[2]).fp32(), bfloat16(post_out[3]).fp32(), bfloat16(post_out[4]).fp32());
                    auto &input = llama_post_host.get_input("input");
                    memcpy(input.pVirAddr, embed_host.data(), embed_host.size() * sizeof(unsigned short));
                    llama_post_host.inference();
                    auto &output_post = llama_post_host.get_output("output");
                    unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                    printf("host: %f %f %f %f %f\n", bfloat16(post_out[0]).fp32(), bfloat16(post_out[1]).fp32(), bfloat16(post_out[2]).fp32(), bfloat16(post_out[3]).fp32(), bfloat16(post_out[4]).fp32());
                }
#endif
                float max_val = -MAXFLOAT;
                int max_index = 0;
                for (int i = 0; i < _attr.tokens_embed_num; i++)
                {
                    float tmp = bfloat16(post_out[i]).fp32();
                    if (tmp > max_val)
                    {
                        max_val = tmp;
                        max_index = i;
                    }
                }
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
