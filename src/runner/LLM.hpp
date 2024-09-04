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
#include "ax_sys_api.h"

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    // std::string template_prefill_filename_axmodel = "minicpmv/prefill_axmodel/minicpm_p96_l%d.axmodel";
    // int prefill_axmodel_num = 40;
    int prefill_token_num = 96;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    std::string filename_vpm_encoder_axmodedl = "minicpmv/vpm_resampler_version0_fp16.axmodel";
    std::string filename_vpm_resampler_axmodedl = "minicpmv/vpm_resampler_version0_fp16.axmodel";
    int vpm_width = 280;
    int vpm_height = 280;
    bool b_vpm_two_stage = false;

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

    bool b_use_topk = false;

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

    ax_runner_ax650 vpm_encoder, vpm_resampler;

    int prefill_grpid = 1;
    int decode_grpid = 0;

    // std::vector<std::vector<unsigned short>> k_caches, v_caches;

    bool b_stop = false;

    static int FindMax(unsigned short *p, int n, float *val = 0)
    {
        float max_val = -MAXFLOAT;
        int max_index = 0;
        for (int i = 0; i < n; i++)
        {
            unsigned int proc = p[i] << 16;
            float tmp = *reinterpret_cast<float *>(&proc);
            if (tmp > max_val)
            {
                max_val = tmp;
                max_index = i;
            }
        }

        // for (int i = 0; i < n; i += 4)
        // {
        //     uint16x4_t bf16_data = vld1_u16(&p[i]);
        //     uint32x4_t float_data = vmovl_u16(bf16_data);
        //     float32x4_t tmp_floats = vreinterpretq_f32_u32(vshlq_n_u32(float_data, 16));

        //     for (int j = 0; j < 4; j++)
        //     {
        //         float tmp = vgetq_lane_f32(tmp_floats, j);
        //         if (tmp > max_val)
        //         {
        //             max_val = tmp;
        //             max_index = i + j;
        //         }
        //     }
        // }

        if (val)
            *val = max_val;
        return max_index;
    }

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 4, 32);
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
        // prefill_layers.resize(attr.prefill_axmodel_num);

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

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        if (_attr.b_vpm_two_stage)
        {
            ret = vpm_encoder.init(attr.filename_vpm_encoder_axmodedl.c_str(), false);
            if (ret != 0)
            {
                ALOGE("init vpm axmodel(%s) failed", attr.filename_vpm_encoder_axmodedl.c_str());
                return false;
            }

            ret = vpm_resampler.init(attr.filename_vpm_resampler_axmodedl.c_str(), false);
            if (ret != 0)
            {
                ALOGE("init vpm axmodel(%s) failed", attr.filename_vpm_resampler_axmodedl.c_str());
                return false;
            }
        }
        else
        {
            ret = vpm_resampler.init(attr.filename_vpm_resampler_axmodedl.c_str(), false);
            if (ret != 0)
            {
                ALOGE("init vpm axmodel(%s) failed", attr.filename_vpm_resampler_axmodedl.c_str());
                return false;
            }
        }
        //fill pos_embed_ids
        {
            std::vector<int> pos_embed_ids(vpm_resampler.get_input("pos_embed_ids").nSize / sizeof(int), 0);
            int pos = 0;
            for (int i = 0; i < _attr.vpm_height / 14; i++)
            {
                for (int j = 0; j < _attr.vpm_width / 14; j++)
                {
                    pos_embed_ids[pos++] = i * 70 + j;
                }
            }
            memcpy(vpm_resampler.get_input("pos_embed_ids").pVirAddr, pos_embed_ids.data(), pos_embed_ids.size() * sizeof(int));
        }

        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init vpm axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 3, "count", axmodel_path);

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

            _attr.prefill_token_num = llama_layers[0].layer.get_input(prefill_grpid, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
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
        vpm_encoder.release();
        vpm_resampler.release();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int Encode(cv::Mat src, std::vector<unsigned short> &out_embed)
    {
        timer t;
        t.start();
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(_attr.vpm_width, _attr.vpm_height), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

        if (_attr.b_vpm_two_stage)
        {
            void *data = vpm_encoder.get_input(0).pVirAddr;
            memcpy(data, dst.data, dst.rows * dst.cols * 3);
            vpm_encoder.inference();
            AX_SYS_MinvalidateCache(vpm_encoder.get_output(0).phyAddr, vpm_encoder.get_output(0).pVirAddr, vpm_encoder.get_output(0).nSize);
            memcpy(vpm_resampler.get_input("input").pVirAddr, vpm_encoder.get_output(0).pVirAddr, vpm_encoder.get_output(0).nSize);
        }
        else
        {
            void *data = vpm_resampler.get_input("input").pVirAddr;
            memcpy(data, dst.data, dst.rows * dst.cols * 3);
        }

        vpm_resampler.inference();
        out_embed.resize(vpm_resampler.get_output("output").nSize / sizeof(unsigned short));
        AX_SYS_MinvalidateCache(vpm_resampler.get_output("output").phyAddr, vpm_resampler.get_output("output").pVirAddr, vpm_resampler.get_output("output").nSize);
        memcpy(out_embed.data(), vpm_resampler.get_output("output").pVirAddr, vpm_resampler.get_output("output").nSize);
        ALOGI("image encode time : %f ms", t.cost());
        return 0;
    }

    int Encode(std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    {
        std::vector<int> input_ids = tokenizer->Encode(prompt, true);
        if (input_ids.size() > _attr.prefill_token_num)
        {
            ALOGE("input_ids(%d) > prefill_token_num(%d)", input_ids.size(), _attr.prefill_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++)
        {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        // memcpy(out_embed.data() + 5 * _attr.tokens_embed_size, vpm_resampler.get_output("output").pVirAddr, vpm_resampler.get_output("output").nSize);

        return 0;
    }

    int Encode(std::vector<unsigned short> &img_embed, std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    {
        std::vector<int> input_ids = tokenizer->Encode(prompt, true);
        std::vector<int> tmp_img_ids((img_embed.size() / _attr.tokens_embed_size), 0);
        input_ids.insert(input_ids.begin() + 5, tmp_img_ids.begin(), tmp_img_ids.end());
        /*
        <用户><image><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk></image>
        What is in the image?<AI>
        1, 95396, 4194, 95388, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 5, 5856, 1410, 1377, 1358, 3766, 74, 95396, 10850, 95388
        */
        // printf("%s\n", prompt.c_str());
        // printf("input_ids: ");
        // for (size_t i = 0; i < input_ids.size(); i++)
        // {
        //     printf("%d, ", input_ids[i]);
        // }
        // printf("\n");

        if (input_ids.size() > _attr.prefill_token_num)
        {
            ALOGE("input_ids(%d) > prefill_token_num(%d)", input_ids.size(), _attr.prefill_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++)
        {
            // std::vector<unsigned short> embed;
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);

            if (i <= 5 || i > 69)
            {
                unsigned short *p = out_embed.data() + i * _attr.tokens_embed_size;
                for (size_t j = 0; j < _attr.tokens_embed_size; j++)
                {
                    bfloat16 bf16(bfloat16(p[j]).fp32() * 12);
                    p[j] = bf16.data;
                }
            }
            // memcpy(, embed.data(), _attr.tokens_embed_size * sizeof(unsigned short));
        }

        // std::vector<unsigned short> image_embed;
        // image_embed.resize(vpm_resampler.get_output("output").nSize / sizeof(unsigned short));
        memcpy(out_embed.data() + 5 * _attr.tokens_embed_size, img_embed.data(), img_embed.size() * sizeof(unsigned short));

        return 0;
    }

    std::string Run(std::string input_str)
    {
        std::vector<unsigned short> test_embed;
        Encode(test_embed, input_str);
        return Run(test_embed);
    }

    std::string Run(std::vector<unsigned short> test_embed)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> mask_p(_attr.prefill_token_num * _attr.prefill_token_num, bf16.data);

        for (size_t i = 0; i < _attr.prefill_token_num; i++)
        {
            for (size_t j = 0; j < i + 1; j++)
            {
                mask_p[i * _attr.prefill_token_num + j] = 0;
            }
        }

        std::vector<int> cached_token;
        std::vector<int> token_ids;
        // std::vector<int> token_ids = tokenizer->Encode(input_str);
        // int len_of_input = token_ids.size();
        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        // ALOGI("input_embed_num(%d)", input_embed_num);

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++)
        {
            mask[i] = 0;
        }
        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        for (unsigned int m = 0; m < _attr.axmodel_num; m++)
        {
            if (b_stop)
            {
                break;
            }

            auto &layer = llama_layers[m];
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

            auto &input_indices = layer.layer.get_input(prefill_grpid, "indices");
            unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
            for (unsigned int i = 0; i < input_embed_num; i++)
            {
                input_indices_ptr[i] = i;
            }

            auto &input_mask = layer.layer.get_input(prefill_grpid, "mask");
            memcpy(input_mask.pVirAddr, mask_p.data(), mask_p.size() * sizeof(unsigned short));

            auto &input_input = layer.layer.get_input(prefill_grpid, "input");
            memcpy(input_input.pVirAddr, test_embed.data(), test_embed.size() * sizeof(unsigned short));
            if (m == 0)
            {
                test_embed.resize(_attr.prefill_token_num * _attr.tokens_embed_size);
            }

            layer.layer.inference(prefill_grpid);

            auto &output_k_cache = layer.layer.get_output(prefill_grpid, "K_cache_out");
            AX_SYS_MinvalidateCache(output_k_cache.phyAddr, output_k_cache.pVirAddr, output_k_cache.nSize);
            auto &input_k_cache = layer_llama.layer.get_input(decode_grpid, "K_cache");
            memcpy(input_k_cache.pVirAddr, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

            auto &output_v_cache = layer.layer.get_output(prefill_grpid, "V_cache_out");
            AX_SYS_MinvalidateCache(output_v_cache.phyAddr, output_v_cache.pVirAddr, output_v_cache.nSize);
            auto &input_v_cache = layer_llama.layer.get_input(decode_grpid, "V_cache");
            memcpy(input_v_cache.pVirAddr, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

            auto &output = layer.layer.get_output(prefill_grpid, "output");
            AX_SYS_MinvalidateCache(output.phyAddr, output.pVirAddr, output.nSize);
            memcpy(test_embed.data(), output.pVirAddr, test_embed.size() * sizeof(unsigned short));
            if (_attr.b_dynamic_load_axmodel_layer)
            {
                layer.layer.deinit();
            }
            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
        }

        // ALOGI("prefill time cost: %.2f s", t_cost.cost() / 1000);

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
            int max_index;
            if (_attr.b_use_topk)
            {
                AX_SYS_MinvalidateCache(llama_post.get_output("indices").phyAddr, llama_post.get_output("indices").pVirAddr, llama_post.get_output("indices").nSize);
                max_index = *(int *)llama_post.get_output("indices").pVirAddr;
            }
            else
            {
                auto &output_post = llama_post.get_output("output");
                AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val = -MAXFLOAT;
                max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
            }
            next_token = max_index;

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;
        for (unsigned int indices = input_embed_num; indices < _attr.max_token_len; indices++)
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
                AX_SYS_MinvalidateCache(output_k_cache.phyAddr, output_k_cache.pVirAddr, output_k_cache.nSize);
                memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                AX_SYS_MinvalidateCache(output_v_cache.phyAddr, output_v_cache.pVirAddr, output_v_cache.nSize);
                memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(decode_grpid, "output");
                AX_SYS_MinvalidateCache(output.phyAddr, output.pVirAddr, output.nSize);
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
                int max_index;
                if (_attr.b_use_topk)
                {
                    AX_SYS_MinvalidateCache(llama_post.get_output("indices").phyAddr, llama_post.get_output("indices").pVirAddr, llama_post.get_output("indices").nSize);
                    max_index = *(int *)llama_post.get_output("indices").pVirAddr;
                }
                else
                {
                    auto &output_post = llama_post.get_output("output");
                    AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
                    unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                    float max_val = -MAXFLOAT;
                    max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
                }
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
