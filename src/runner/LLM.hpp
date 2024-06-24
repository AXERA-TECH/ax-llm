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
#include "opencv2/opencv.hpp"

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    std::string template_prefill_filename_axmodel = "minicpmv/prefill_axmodel/minicpm_p96_l%d.axmodel";
    int prefill_axmodel_num = 40;
    int prefill_feat_pad_num = 96;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    std::string filename_vpm_resampler_axmodedl = "minicpmv/vpm_resampler_version0_fp16.axmodel";
    int vpm_width = 280;
    int vpm_height = 280;

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
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers, prefill_layers;
    ax_runner_ax650 llama_post;

    ax_runner_ax650 vpm_resampler;

    // std::vector<std::vector<unsigned short>> k_caches, v_caches;

    bool b_stop = false;

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + attr.prefill_axmodel_num + 4, 32);
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
        prefill_layers.resize(attr.prefill_axmodel_num);

        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            if (!attr.b_dynamic_load_axmodel_layer)
            {
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

        for (int i = 0; i < attr.prefill_axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_prefill_filename_axmodel.c_str(), i);
            prefill_layers[i].filename = axmodel_path;

            if (!attr.b_dynamic_load_axmodel_layer)
            {
                int ret = prefill_layers[i].layer.init(prefill_layers[i].filename.c_str(), false);
                if (ret != 0)
                {
                    ALOGE("init prefill axmodel(%s) failed", prefill_layers[i].filename.c_str());
                    return false;
                }
                int remain_cmm = get_remaining_cmm_size();
                sprintf(axmodel_path, "init prefill %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
                update_cqdm(&cqdm, i + attr.axmodel_num + 2, "count", axmodel_path);
            }
            else
            {
                if (!attr.b_use_mmap_load_layer)
                {
                    if (!read_file(prefill_layers[i].filename, prefill_layers[i].layer_buffer_vec))
                    {
                        ALOGE("read_file(%s) failed", prefill_layers[i].filename.c_str());
                        return false;
                    }
                }
                else
                {
                    prefill_layers[i].layer_buffer.open_file(prefill_layers[i].filename.c_str());
                }

                sprintf(axmodel_path, "read_file %s ok", prefill_layers[i].filename.c_str());
                update_cqdm(&cqdm, i + attr.axmodel_num + 2, "count", axmodel_path);
            }
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + attr.prefill_axmodel_num + 2, "count", axmodel_path);

        ret = vpm_resampler.init(attr.filename_vpm_resampler_axmodedl.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init vpm axmodel(%s) failed", attr.filename_vpm_resampler_axmodedl.c_str());
            return false;
        }
        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init vpm axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + attr.prefill_axmodel_num + 3, "count", axmodel_path);

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
        }
        llama_post.release();
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

    int RunVpm(cv::Mat src, std::vector<unsigned short> &out_embed)
    {
        if (src.empty())
        {
            return -1;
        }

        cv::Mat dst;
        cv::resize(src, dst, cv::Size(_attr.vpm_width, _attr.vpm_height), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

        void *data = vpm_resampler.get_input("input").pVirAddr;
        memcpy(data, dst.data, dst.rows * dst.cols * 3);

        static bool isfirst = true;
        if (isfirst)
        {
            std::vector<int> pos_embed_ids(400, 0);
            int pos = 0;
            for (int i = 0; i < 20; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    pos_embed_ids[pos++] = i * 70 + j;
                }
            }
            memcpy(vpm_resampler.get_input("pos_embed_ids").pVirAddr, pos_embed_ids.data(), 400 * sizeof(int));

            isfirst = false;
        }

        vpm_resampler.inference();

        out_embed.resize(vpm_resampler.get_output("output").nSize / sizeof(unsigned short));
        memcpy(out_embed.data(), vpm_resampler.get_output("output").pVirAddr, vpm_resampler.get_output("output").nSize);

        return 0;
    }

    std::string Run(std::vector<unsigned short> &test_embed)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> mask_p(_attr.prefill_feat_pad_num * _attr.prefill_feat_pad_num, bf16.data);

        for (size_t i = 0; i < _attr.prefill_feat_pad_num; i++)
        {
            for (size_t j = 0; j < i + 1; j++)
            {
                mask_p[i * _attr.prefill_feat_pad_num + j] = 0;
            }
        }

        std::vector<int> cached_token;
        std::vector<int> token_ids;
        // std::vector<int> token_ids = tokenizer->Encode(input_str);
        // int len_of_input = token_ids.size();
        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++)
        {
            mask[i] = 0;
        }

        for (unsigned int m = 0; m < _attr.prefill_axmodel_num; m++)
        {
            if (b_stop)
            {
                break;
            }

            auto &layer = prefill_layers[m];
            auto &layer_llama = llama_layers[m];

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

            auto &input_indices = layer.layer.get_input("indices");
            unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
            for (unsigned int i = 0; i < input_embed_num; i++)
            {
                input_indices_ptr[i] = i;
            }

            auto &input_mask = layer.layer.get_input("mask");
            memcpy(input_mask.pVirAddr, mask_p.data(), mask_p.size() * sizeof(unsigned short));

            auto &input_input = layer.layer.get_input("input");
            memcpy(input_input.pVirAddr, test_embed.data(), test_embed.size() * sizeof(unsigned short));
            if (m == 0)
            {
                test_embed.resize(_attr.prefill_feat_pad_num * _attr.tokens_embed_size);
            }

            layer.layer.inference();

            auto &output_k_cache = layer.layer.get_output("K_cache_out");
            auto &input_k_cache = layer_llama.layer.get_input("K_cache");
            memcpy(input_k_cache.pVirAddr, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.prefill_feat_pad_num * _attr.tokens_embed_size);

            auto &output_v_cache = layer.layer.get_output("V_cache_out");
            auto &input_v_cache = layer_llama.layer.get_input("V_cache");
            memcpy(input_v_cache.pVirAddr, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.prefill_feat_pad_num * _attr.tokens_embed_size);

            auto &output = layer.layer.get_output("output");
            memcpy(test_embed.data(), output.pVirAddr, test_embed.size() * sizeof(unsigned short));
            if (_attr.b_dynamic_load_axmodel_layer)
            {
                layer.layer.deinit();
            }
            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
        }

        timer t_cost;

        // print token_ids
        // printf("%s\n", input_str.c_str());
        // for (size_t i = 0; i < token_ids.size(); i++)
        // {
        //     printf("%d ", token_ids[i]);
        // }
        // printf("\n");

        int next_token = -1;
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);

        memcpy(embed.data(),
               test_embed.data() + (input_embed_num - 1) * _attr.tokens_embed_size,
               _attr.tokens_embed_size * sizeof(unsigned short));

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
            cached_token.push_back(max_index);
        }

        bool b_hit_eos = false;
        for (unsigned int indices = 80; indices < _attr.max_token_len; indices++)
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

                auto &input_k_cache = layer.layer.get_input("K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                // memcpy(input_k_cache.pVirAddr, k_caches[m].data(), sizeof(unsigned short) * k_caches[m].size());
                auto &input_v_cache = layer.layer.get_input("V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;
                // memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());

                auto &input_indices = layer.layer.get_input("indices");
                memcpy(input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input("mask");
                memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                // if (indices == 0 && m == 0)
                // {
                //     memcpy(layer.layer.get_input("input").pVirAddr,
                //            test_embed.data() + (input_embed_num - 1) * _attr.tokens_embed_size,
                //            _attr.tokens_embed_size * sizeof(unsigned short));
                // }

                auto &input_input = layer.layer.get_input("input");
                memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                layer.layer.inference();

                auto &output_k_cache = layer.layer.get_output("K_cache_out");
                memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output("V_cache_out");
                memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

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
        // token_ids.erase(token_ids.begin(), token_ids.begin() + len_of_input);

        final_out = tokenizer->Decode(token_ids);

        return final_out;
    }
};
