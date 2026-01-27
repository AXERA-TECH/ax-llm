#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>

#include <opencv2/opencv.hpp>

#include "bfloat16.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "ax_cmm_utils.hpp"

#include "BaseTokenizer.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"

#include "LLMEmbedSelector.hpp"
#include "LLMPostprocess.hpp"

/**
 * @brief 说明图像编码器输入数据的格式和预处理要求
 *
 * 当定义了 IMAGE_ENCODER_INPUT_NCHW 且其值为 1 时，图像编码器的输入数据格式为 [1*3*h*w] 的浮点型数据，
 * 该数据需要进行归一化处理，具体操作是 (像素值/255 - 均值) / 标准差。
 *
 * 当 IMAGE_ENCODER_INPUT_NCHW 的值为 0 时，图像编码器的输入数据格式为 [1*h*w*3] 的无符号 8 位整型数据。
 */
int IMAGE_ENCODER_INPUT_NCHW = -1;

/**
 * @brief 说明图像编码器输出数据的格式
 *
 * 当 IMAGE_ENCODER_OUTPUT_BF16 为 1 时，图像编码器的输出数据格式为 bfloat16 半精度浮点型数据，
 * 否则为float32数据。
 */
int IMAGE_ENCODER_OUTPUT_BF16 = -1;

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    std::string filename_image_encoder_axmodedl = "minicpmv/vpm_resampler_version0_fp16.axmodel";
    int image_encoder_width = 448;
    int image_encoder_height = 448;

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;
    std::vector<int> prefill_max_kv_cache_num_grp;
    int precompute_len = 0;
    int prefill_grpid = -1;

    std::string filename_tokenizer_txt = "fastvlm_tokenizer.txt";
    bool b_bos = false, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    bool b_use_mmap_load_embed = false;

    std::string post_config_path = "post_config.json";

    // std::vector<int> dev_ids = {0, 1, 2, 3};

    // bool b_live_print = true;
    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;

    /**
     * 151667 for InternVL 2.5/3
     * 92546 for InternVL 2.5-8B-MPO
     * 
     */
    int IMAGE_CONTEXT_TOKEN = 151646;

    /**
     * 151665 for InternVL 2.5/3
     * 92544 for InternVL 2.5-8B-MPO
     */
    int IMAGE_START_TOKEN = 151644;
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
    ax_runner_ax650 image_encoder;

    // int prefill_grpid = 1;
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

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer = create_tokenizer(FastVLM);
        if (!tokenizer->load(attr.filename_tokenizer_txt))
        {
            ALOGE("tokenizer.load(%s) failed", attr.filename_tokenizer_txt.c_str());
            return false;
        }

        if (bool ret = tokenizer->add_stop_token("<|im_end|>"); !ret)
        {
            ALOGE("tokenizer.add_stop_token(<|im_end|>) failed");
            return false;
        }
        auto stop_tokens = tokenizer->get_stop_tokens();
        // printf("stop_tokens size: %d\n", stop_tokens.size());
        // for (auto &token : stop_tokens)
        // {
        //     printf("%d\n", token);
        // }
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
        printf("\n");
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

        std::vector<int> rets(attr.axmodel_num);
        std::atomic<int> process_idx = 2;
        // #pragma omp parallel for
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            char axmodel_path[1024];
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str());
            // llama_layers[i].layer.set_auto_sync_after_inference(true);
            // llama_layers[i].layer.set_auto_sync_before_inference(true);
            rets[i] = ret;
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, process_idx++, "count", axmodel_path);
        }

        for (int i = 0; i < attr.axmodel_num; i++)
        {
            if (rets[i] != 0)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str());

        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = get_remaining_cmm_size();
        char axmodel_path[1024];
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        ret = image_encoder.init(attr.filename_image_encoder_axmodedl.c_str());
        if (ret != 0)
        {
            ALOGE("init vpm axmodel(%s) failed", attr.filename_image_encoder_axmodedl.c_str());
            return false;
        }

        auto ids = tokenizer->encode("<image>");
        if (ids.size() != 1)
        {
            ALOGE("encode <image> failed");
            return false;
        }
        _attr.IMAGE_CONTEXT_TOKEN = ids[0];
        // ids = tokenizer->encode("<img>");
        // if (ids.size() != 1)
        // {
        //     ALOGE("encode <img> failed");
        //     return false;
        // }
        // _attr.IMAGE_START_TOKEN = ids[0];

        // ALOGI("IMAGE_CONTEXT_TOKEN: %d, IMAGE_START_TOKEN: %d", _attr.IMAGE_CONTEXT_TOKEN, _attr.IMAGE_START_TOKEN);
        // ALOGI("IMAGE_CONTEXT_TOKEN: %d", _attr.IMAGE_CONTEXT_TOKEN);


        IMAGE_ENCODER_INPUT_NCHW = -1;
        for (size_t i = 1; i < image_encoder.get_input(0).vShape.size(); i++)
        {
            if (image_encoder.get_input(0).vShape[i] == 3)
            {
                if (i == 1)
                {
                    IMAGE_ENCODER_INPUT_NCHW = 1;
                }
                else if (i == 3)
                {
                    IMAGE_ENCODER_INPUT_NCHW = 0;
                }
            }
        }
        if (IMAGE_ENCODER_INPUT_NCHW == -1)
        {
            ALOGE("image encoder input nchw or nhwc not found");
            return false;
        }

        if (IMAGE_ENCODER_INPUT_NCHW)
        {
            ALOGI("image encoder input nchw@float32");
            _attr.image_encoder_height = image_encoder.get_input(0).vShape[2];
            _attr.image_encoder_width = image_encoder.get_input(0).vShape[3];
        }
        else
        {
            ALOGI("image encoder input nhwc@uint8");
            _attr.image_encoder_height = image_encoder.get_input(0).vShape[1];
            _attr.image_encoder_width = image_encoder.get_input(0).vShape[2];
        }

        if (_attr.image_encoder_height != _attr.image_encoder_width)
        {
            ALOGE("image encoder height != width");
            return false;
        }
        int output_elem_size = 1;
        for (int i = 0; i < image_encoder.get_output(0).vShape.size(); i++)
        {
            output_elem_size *= image_encoder.get_output(0).vShape[i];
        }

        if (output_elem_size * 2 == image_encoder.get_output(0).nSize)
        {
            IMAGE_ENCODER_OUTPUT_BF16 = 1;
            ALOGI("image encoder output bf16");
        }
        else if (output_elem_size * 4 == image_encoder.get_output(0).nSize)
        {
            IMAGE_ENCODER_OUTPUT_BF16 = 0;
            ALOGI("image encoder output float32");
        }
        else
        {
            ALOGE("image encoder output not support");
            return false;
        }

        printf("\n");
        {
            ALOGI("image_encoder_height : %d, image_encoder_width: %d", _attr.image_encoder_height, _attr.image_encoder_width);
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            ALOGI("max_token_len : %d", _attr.max_token_len);
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
                ALOGI("grp: %ld, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
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

    LLMPostprocess *getPostprocess()
    {
        return &postprocess;
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

    int Encode(cv::Mat src, std::vector<unsigned short> &out_embed)
    {
        timer t;
        t.start();
        if (IMAGE_ENCODER_INPUT_NCHW)
        {
            std::vector<float> mean = {0.485, 0.456, 0.406};
            std::vector<float> scale = {0.229, 0.224, 0.225};

            cv::Mat dst;
            cv::resize(src, dst, cv::Size(_attr.image_encoder_width, _attr.image_encoder_height));
            cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

            // std::vector<float> input_data(dst.rows * dst.cols * 3);

            float *input_data = (float *)image_encoder.get_input(0).pVirAddr;

            unsigned char *img_data = dst.data;
            int letterbox_rows = dst.rows;
            int letterbox_cols = dst.cols;

            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        int in_index = h * letterbox_cols * 3 + w * 3 + c;
                        int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                        input_data[out_index] = (float(img_data[in_index]) / 255.0 - mean[c]) / scale[c];
                    }
                }
            }
            image_encoder.inference();
        }
        else
        {
            cv::Mat dst;
            cv::resize(src, dst, cv::Size(_attr.image_encoder_width, _attr.image_encoder_height));
            cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
            void *data = image_encoder.get_input(0).pVirAddr;
            memcpy(data, dst.data, dst.rows * dst.cols * 3);
            image_encoder.inference();
        }

        int size = 1;
        for (size_t i = 0; i < image_encoder.get_output(0).vShape.size(); i++)
        {
            size *= image_encoder.get_output(0).vShape[i];
        }

        out_embed.resize(size);

        if (IMAGE_ENCODER_OUTPUT_BF16)
            memcpy(out_embed.data(), image_encoder.get_output(0).pVirAddr, image_encoder.get_output(0).nSize);
        else
        {
            float *out_data = (float *)image_encoder.get_output(0).pVirAddr;
            for (size_t i = 0; i < size; i++)
            {
                out_embed[i] = bfloat16(out_data[i]).data;
            }
        }

        ALOGI("image encode time : %0.2f ms, size : %ld", t.cost(), out_embed.size());
        return 0;
    }

    int Encode(std::vector<cv::Mat> srcs, std::vector<std::vector<unsigned short>> &out_embeds)
    {
        out_embeds.resize(srcs.size());
        for (size_t i = 0; i < srcs.size(); i++)
        {
            auto ret = Encode(srcs[i], out_embeds[i]);
            if (ret != 0)
            {
                ALOGE("Encode image failed");
                return -1;
            }
        }

        return 0;
    }

    int Encode(std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    {   
        std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are a helpful assistant."},
            {USER, TEXT, prompt},
        };
        std::vector<int> input_ids = tokenizer->encode(contents);

        ALOGI("input_ids size: %ld", input_ids.size());
        // for (size_t i = 0; i < input_ids.size(); i++)
        // {
        //     printf("%d ", input_ids[i]);
        // }
        // printf("\n");
        
        if (input_ids.size() > _attr.prefill_token_num)
        {
            ALOGE("input_ids(%ld) > prefill_token_num(%d)", input_ids.size(), _attr.prefill_token_num);
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

    int Encode(std::vector<std::vector<unsigned short>> &imgs_embed, std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    {
        ALOGI("imgs_embed.size() : %ld, media token size : %ld", imgs_embed.size(), int(imgs_embed[0].size() / _attr.tokens_embed_size));
        std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are a helpful assistant."},
            {USER, IMAGE, prompt, (int)imgs_embed.size(), int(imgs_embed[0].size() / _attr.tokens_embed_size)},
        };

        std::vector<int> input_ids = tokenizer->encode(contents);

        int img_start_index = 0;
        for (size_t i = 0; i < input_ids.size(); i++)
        {
            if (input_ids[i] == _attr.IMAGE_CONTEXT_TOKEN)
            {
                img_start_index = i;
                break;
            }
        }
        // printf("input_ids : ");
        //     for (size_t i = 0; i < input_ids.size(); i++)
        //     {
        //         printf("%d ", input_ids[i]);
        //     }
        //     printf("\n");

        // if (img_start_index.size() != imgs_embed.size())
        // {
        //     ALOGE("img_start_index.size() != imgs_embed.size(), img_start_index.size() : %ld, imgs_embed.size() : %ld", img_start_index.size(), imgs_embed.size());

        //     printf("input_ids : ");
        //     for (size_t i = 0; i < input_ids.size(); i++)
        //     {
        //         printf("%d ", input_ids[i]);
        //     }
        //     printf("\n");

        //     return -1;
        // }

        if (input_ids.size() > _attr.prefill_max_token_num)
        {
            ALOGE("input_ids(%ld) > prefill_max_token_num(%d)", input_ids.size(), _attr.prefill_max_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++)
        {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        // ALOGI("imgs_embed.size() : %ld", imgs_embed.size());
        // ALOGI("input_ids.size() : %ld", input_ids.size());
        // ALOGI("img_start_index : %d", img_start_index);
        for (size_t i = 0; i < imgs_embed.size(); i++)
        {
            // int offset = img_start_index[i] + 1;
            int offset = img_start_index;
            auto &img_embed = imgs_embed[i];
            // ALOGI("img %ld embed size : %ld", i, img_embed.size());

            int img_context_count = 0;
            for (size_t j = offset; j < input_ids.size(); j++)
            {
                if (input_ids[j] == _attr.IMAGE_CONTEXT_TOKEN)
                {
                    img_context_count++;
                }
                // else
                // {
                //     break;
                // }
            }
            // ALOGI("img %ld context token count : %d", i, img_context_count);

            if (img_context_count != img_embed.size() / _attr.tokens_embed_size)
            {
                ALOGE("img_context_count(%d) != img_embed.size() / tokens_embed_size(%ld)", img_context_count, img_embed.size() / _attr.tokens_embed_size);
                return -1;
            }

            memcpy(out_embed.data() + offset * _attr.tokens_embed_size, img_embed.data(), img_embed.size() * sizeof(unsigned short));
            // ALOGI("idx:%ld offset : %d out_embed.size() : %ld", i, offset, out_embed.size());
        }

        return 0;
    }

    int Encode(std::vector<unsigned short> &img_embed, std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    {
        std::vector<std::vector<unsigned short>> imgs_embed = {img_embed};
        return Encode(imgs_embed, out_embed, prompt);
    }

    std::string Run(std::vector<unsigned short> &test_embed)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);

        std::vector<int> cached_token;
        std::vector<int> token_ids;
        // std::vector<int> token_ids = tokenizer->Encode(input_str);
        // int len_of_input = token_ids.size();
        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
        if (input_embed_num > _attr.prefill_max_token_num)
        {
            ALOGE("input token num(%d) > prefill_max_token_num(%d)", input_embed_num, _attr.prefill_max_token_num);
            return "";
        }

        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (input_embed_num <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                _attr.prefill_grpid = i + 1;
                break;
            }
        }
        ALOGI("prefill grpid %d", _attr.prefill_grpid);
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++)
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
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

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
            auto &input = llama_post.get_input(0);
            // memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            memcpy((void *)input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference();

            int max_index;

            auto &output_post = llama_post.get_output(0);
            memcpy(output_post.pVirAddr, (void *)output_post.pVirAddr, output_post.nSize);
            unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
            float max_val = -MAXFLOAT;
            // max_index = post_process(post_out, _attr.tokens_embed_num, &max_val);
            max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

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

            memcpy((void *)llama_layers[0].layer.get_input(decode_grpid, "input").pVirAddr, embed.data(), llama_layers[0].layer.get_input(decode_grpid, "input").nSize);
            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy((void *)input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                memcpy((unsigned short *)input_k_cache.pVirAddr + indices * _attr.kv_cache_size, (void *)output_k_cache.pVirAddr, output_k_cache.nSize);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                memcpy((unsigned short *)input_v_cache.pVirAddr + indices * _attr.kv_cache_size, (void *)output_v_cache.pVirAddr, output_v_cache.nSize);

                if (m == _attr.axmodel_num - 1)
                {
                    memcpy((void *)llama_post.get_input(0).pVirAddr,
                           (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, llama_post.get_input(0).nSize);
                }
                else if (m < _attr.axmodel_num - 1)
                {
                    memcpy((void *)llama_layers[m + 1].layer.get_input(decode_grpid, "input").pVirAddr,
                           (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, layer.layer.get_input(decode_grpid, "input").nSize);
                }
                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            // ALOGI("");
            mask[indices] = 0;
            {
                llama_post.inference();

                auto &output_post = llama_post.get_output(0);
                memcpy(output_post.pVirAddr, (void *)output_post.pVirAddr, output_post.nSize);
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val = -MAXFLOAT;
                // max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
                auto max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

                next_token = max_index;

                if (tokenizer->is_stop(max_index))
                {
                    if (cached_token.size() && _attr.runing_callback)
                    {
                        float t_cost_ms = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out = tokenizer->decode(cached_token);
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
                        auto tmp_out = tokenizer->decode(cached_token);
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

        final_out = tokenizer->decode(token_ids);

        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            for (size_t j = 0; j < llama_layers[i].layer.get_num_input_groups(); j++)
            {
                memset((void *)llama_layers[i].layer.get_input(j, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(j, "K_cache").nSize);
                memset((void *)llama_layers[i].layer.get_input(j, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(j, "V_cache").nSize);
            }
        }

        return final_out;
    }
};
