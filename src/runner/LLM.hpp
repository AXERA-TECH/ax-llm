#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"

#include "cqdm.h"

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

    int max_token_len = 127;

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256;

    bool b_use_mmap_load_embed = false;
    bool b_dynamic_load_axmodel_layer = false;

    bool b_use_mmap_load_layer = true;

    bool b_live_print = true;
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

    std::vector<std::vector<unsigned short>> k_caches, v_caches;

    bool b_stop = false;

public:
    bool Init(LLMAttrType attr)
    {
        this->_attr = attr;
        tokenizer = CreateTokenizer(attr.tokenizer_type);
        if (!tokenizer->Init(attr.filename_tokenizer_model, attr.b_bos, attr.b_eos))
        {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.filename_tokenizer_model.c_str(), attr.b_bos, attr.b_eos);
            return false;
        }
        ALOGI("tokenizer init ok");
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
        ALOGI("embed_selector init ok");
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
                int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str());
                if (ret != 0)
                {
                    ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                    return false;
                }
                ALOGI("init axmodel(%s) ok", llama_layers[i].filename.c_str());
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

                ALOGI("read_file(%s) ok", llama_layers[i].filename.c_str());
            }
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str());
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        ALOGI("init post axmodel(%s) ok", attr.filename_post_axmodel.c_str());

        if (!attr.b_dynamic_load_axmodel_layer)
        {
            auto &input_k_cache = llama_layers[0].layer.get_input("K_cache");
            _attr.kv_cache_num = input_k_cache.vShape[0] / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_num: %d", _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }
        }

        Reset();
        return true;
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

    void Reset()
    {
        k_caches.resize(_attr.axmodel_num, std::vector<unsigned short>(_attr.kv_cache_num * _attr.kv_cache_size, 0));
        v_caches.resize(_attr.axmodel_num, std::vector<unsigned short>(_attr.kv_cache_num * _attr.kv_cache_size, 0));
    }

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

        std::vector<int> token_ids = tokenizer->Encode(input_str);
        // print token_ids
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
                        ret = layer.layer.init(layer.layer_buffer);
                    }
                    else
                    {
                        ret = layer.layer.init(layer.layer_buffer_vec);
                    }
                    if (ret != 0)
                    {
                        ALOGE("init axmodel(%s) failed", layer.filename.c_str());
                    }
                    // ALOGI("init axmodel(%s) ok", layer.filename.c_str());

                    static bool b_first = true;
                    if (m == 0 && indices == 0 && b_first)
                    {
                        b_first = false;
                        auto &input_k_cache = layer.layer.get_input("K_cache");
                        _attr.kv_cache_num = input_k_cache.vShape[0] / _attr.kv_cache_size / sizeof(unsigned short);
                        ALOGI("kv_cache_num: %d", _attr.kv_cache_num);
                        if (_attr.max_token_len > _attr.kv_cache_num)
                        {
                            ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                            return "";
                        }
                        /* code */
                    }
                }

                auto &input_k_cache = layer.layer.get_input("K_cache");
                memcpy(input_k_cache.pVirAddr, k_caches[m].data(), sizeof(unsigned short) * k_caches[m].size());
                auto &input_v_cache = layer.layer.get_input("V_cache");
                memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());

                auto &input_indices = layer.layer.get_input("indices");
                memcpy(input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input("mask");
                memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input("input");
                memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                layer.layer.inference();

                auto &output_k_cache = layer.layer.get_output("K_cache_out");
                memcpy(k_caches[m].data() + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output("V_cache_out");
                memcpy(v_caches[m].data() + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output = layer.layer.get_output("output");
                memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));
                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    layer.layer.deinit();
                }
                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            // ALOGI("");
            mask[indices] = 0;
            if (indices + 1 < token_ids.size())
            {
                next_token = token_ids[indices + 1];
            }
            else
            {
                // post process
                auto &input = llama_post.get_input("input");
                memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                llama_post.inference();
                auto &output_post = llama_post.get_output("output");
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;

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
                token_ids.push_back(max_index);
                if (_attr.b_live_print)
                {
                    static std::vector<int> cached_token;
                    cached_token.push_back(max_index);
                    if (cached_token.size() >= 3)
                    {
                        auto tmp_out = tokenizer->Decode({cached_token});
                        fprintf(stdout, "%s", tmp_out.c_str());
                        fflush(stdout);
                        cached_token.clear();
                    }
                }

                if (max_index == tokenizer->GetEosID())
                {
                    printf("\n");
                    ALOGN("hit eos\n");
                    b_hit_eos = true;
                    break;
                }
            }
            if (!_attr.b_live_print)
                update_cqdm(&cqdm, indices);
            if (b_hit_eos)
            {
                break;
            }
        }
        final_out = tokenizer->Decode(token_ids);
        printf("\n");
        return final_out;
    }
};
