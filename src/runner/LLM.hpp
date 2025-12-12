#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
// #include "Tokenizer/Tokenizer.hpp"
#include "BaseTokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "opencv2/opencv.hpp"
#include "ax_sys_api.h"
#include "LLMPostprocess.hpp"
#include "image_processor.hpp"
#include "mrope.hpp"

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

    std::string filename_image_encoder_axmodedl = "image_encoder.axmodel";
    int image_encoder_width = 448;
    int image_encoder_height = 448;

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;
    std::vector<int> prefill_max_kv_cache_num_grp;
    int precompute_len = 0;
    int prefill_grpid = -1;

    // TokenizerType tokenizer_type = TKT_HTTP;
    std::string filename_tokenizer_model = "http://127.0.0.1:12345";
    bool b_bos = false, b_eos = false;
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
    std::string post_config_path = "post_config.json";

    // bool b_live_print = true;
    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;

    /**
     * 151667 for InternVL 2.5/3
     * 92546 for InternVL 2.5-8B-MPO
     */
    int IMAGE_CONTEXT_TOKEN = 151667;

    /**
     * 151665 for InternVL 2.5/3
     * 92544 for InternVL 2.5-8B-MPO
     */
    int IMAGE_START_TOKEN = 151665;
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
        int remain_cmm = get_remaining_cmm_size();
        ALOGI("Total CMM:%d MB", remain_cmm);

        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer = create_tokenizer(Qwen3VL);
        if (!tokenizer->load(attr.filename_tokenizer_model))
        {
            ALOGE("tokenizer.load(%s) failed", attr.filename_tokenizer_model.c_str());
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
        ALOGI("attr.axmodel_num:%d", attr.axmodel_num);
        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            if (!attr.b_dynamic_load_axmodel_layer)
            {
                int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), true);
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
        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        ret = image_encoder.init(attr.filename_image_encoder_axmodedl.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init image_encoder axmodel(%s) failed", attr.filename_image_encoder_axmodedl.c_str());
            return false;
        }

        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init vpm axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 3, "count", axmodel_path);

        // _attr.IMAGE_CONTEXT_TOKEN = tokenizer->GetImgContextID();
        // _attr.IMAGE_START_TOKEN = tokenizer->GetImgStartID();

        // ALOGI("IMAGE_CONTEXT_TOKEN: %d, IMAGE_START_TOKEN: %d", _attr.IMAGE_CONTEXT_TOKEN, _attr.IMAGE_START_TOKEN);

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

        if (IMAGE_ENCODER_INPUT_NCHW == 1)
        {
            ALOGE("Qwen2.5_VL Image Encoder just support NHWC");
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
            printf("\n");
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
                ALOGI("grp: %ld, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
                _attr.prefill_max_kv_cache_num_grp.push_back(prefill_max_kv_cache_num);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }
        if (attr.b_dynamic_load_axmodel_layer)
        {
            for (int i = 0; i < attr.axmodel_num; i++)
            {
                auto &layer = llama_layers[i];
                layer.layer.deinit();
            }
        }

        if (!postprocess.load_config(attr.post_config_path))
        {
            ALOGW("load postprocess config(%s) failed", attr.post_config_path.c_str());
        }

        // Reset();
        ALOGI("LLM init ok");
        remain_cmm = get_remaining_cmm_size();
        ALOGI("Left CMM:%d MB", remain_cmm);
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

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            llama_layers[i].layer.deinit();
        }
        llama_post.deinit();
        image_encoder.deinit();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int EncodeImage(std::vector<cv::Mat> &src, bool b_video, Config &cfg,
                    std::vector<std::vector<unsigned short>> &out_embed)
    {
        int temporal_patch_size = cfg.vision_config.temporal_patch_size;
        int merge_size = cfg.vision_config.spatial_merge_size;
        int patch_size = cfg.vision_config.patch_size;
        int ret;
        timer t;
        t.start();

        int grid_h = cfg.vision_config.height / cfg.vision_config.patch_size;
        int grid_w = cfg.vision_config.width / cfg.vision_config.patch_size;

        std::vector<std::vector<unsigned char>> pixel_values;

        int w = cfg.vision_config.width, h = cfg.vision_config.height;

        int channel = src[0].channels();
        int hwc = grid_h * grid_w * temporal_patch_size * patch_size * patch_size * channel;

        if (!b_video)
        {
            for (int i = 0; i < src.size(); i++)
            {
                std::vector<std::vector<unsigned char>> img_values;
                std::vector<cv::Mat> si{src[i]};
                Qwen2VideoProcessor(si, img_values,
                                    h, w,
                                    temporal_patch_size, merge_size, patch_size);
                pixel_values.push_back(img_values[0]);
            }
            for (int i = 0; i < pixel_values.size(); i++)
            {
                cfg.image_grid_thw.push_back({1, grid_h, grid_w});
            }
        }
        else
        {
            // 只支持一个视频
            Qwen2VideoProcessor(src, pixel_values,
                                h, w,
                                temporal_patch_size, merge_size, patch_size);
            cfg.video_grid_thw = {{(int)pixel_values.size(), grid_h, grid_w}};
        }

        ALOGI("pixel_values size %d", pixel_values.size());
        ALOGI("grid_h %d grid_w %d", grid_h, grid_w);
        int cnt = 0;
        if (out_embed.empty())
        {
            out_embed.resize(pixel_values.size());
        }

        for (int i = 0; i < pixel_values.size(); i++)
        {

            void *data = image_encoder.get_input(0).pVirAddr;
            memcpy(data, pixel_values[i].data(), hwc);
            image_encoder.inference();

            size_t size = image_encoder.get_output(0).nSize / sizeof(float);
            if (out_embed[i].empty())
            {
                out_embed[i].resize(size);
            }

            float *output_data = (float *)image_encoder.get_output(0).pVirAddr;
            for (size_t j = 0; j < size; j++)
            {
                out_embed[i][j] = bfloat16(output_data[j]).data;
            }
        }

        ALOGI("image encode time : %f ms, size : %d", t.cost(), out_embed.size());
        return 0;
    }

    int GetPositionIds(std::vector<int> &input_ids, std::vector<std::vector<int>> &position_ids, Config &cfg)
    {
        std::vector<double> second_per_grid_ts = {cfg.vision_config.temporal_patch_size/cfg.vision_config.fps}; // temporal_patch_size / fps
        position_ids = get_rope_index(cfg, input_ids, cfg.image_grid_thw, cfg.video_grid_thw, second_per_grid_ts);
        return 0;
    }

    int Encode(std::vector<unsigned short> &out_embed, std::vector<std::vector<int>> &position_ids, Config &cfg, std::string prompt = "What is in the image?")
    {
        std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are a helpful assistant."},
            {USER, TEXT, prompt},
        };

        // ImageInfo img_info;
        // img_info.img_prompt = false;
        std::vector<int> input_ids = tokenizer->encode(contents);
        if (input_ids.size() > _attr.prefill_max_token_num)
        {
            ALOGE("input_ids(%d) > prefill_max_token_num(%d)", input_ids.size(), _attr.prefill_max_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++)
        {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        cfg.image_grid_thw.clear();
        cfg.video_grid_thw.clear();
        GetPositionIds(input_ids, position_ids, cfg);
        return 0;
    }

    int Encode(std::vector<std::vector<unsigned short>> &img_embed, bool b_video, std::vector<unsigned short> &out_embed,
               std::vector<std::vector<int>> &position_ids, Config &cfg, std::string prompt = "What is in the image?")
    {
        std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are a helpful assistant."},
            {USER, b_video ? VIDEO : IMAGE, prompt, (int)img_embed.size(), int(img_embed[0].size() / _attr.tokens_embed_size)},
        };

        // ImageInfo img_info;
        // img_info.img_prompt = true;
        // img_info.video_prompt = b_video;
        // img_info.img_token_num = img_embed[0].size() / _attr.tokens_embed_size;
        // img_info.num_img = img_embed.size();
        std::vector<int> input_ids = tokenizer->encode(contents);
        ALOGI("input_ids size:%d", input_ids.size());
        std::vector<int> offsets;
        int vision_start_token_id = cfg.vision_start_token_id;
        for (size_t i = 0; i < input_ids.size() - 1; i++)
        {
            if (input_ids[i] == vision_start_token_id)
            {
                int offset = i + 1;
                ALOGI("offset %d", offset);
                offsets.push_back(offset);
            }
        }

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
        ALOGI("img_embed.size:%d, %d", img_embed.size(), img_embed[0].size());

        if (offsets.size() == 1 && img_embed.size() > 1)
        {
            for (int i = 1; i < img_embed.size(); i++)
            {
                offsets.push_back(offsets[i - 1] + img_embed[i - 1].size() / _attr.tokens_embed_size);
                ALOGI("offset:%d", offsets[i - 1] + img_embed[i - 1].size() / _attr.tokens_embed_size);
            }
        }

        for (int i = 0; i < img_embed.size(); i++)
        {
            memcpy(out_embed.data() + offsets[i] * _attr.tokens_embed_size, img_embed[i].data(), img_embed[i].size() * sizeof(unsigned short));
        }

        ALOGI("out_embed size:%d", out_embed.size());
        ALOGI("input_ids size %d", input_ids.size());
        GetPositionIds(input_ids, position_ids, cfg);
        ALOGI("position_ids size:%d", position_ids[0].size());
        return 0;
    }

    std::string Run(std::vector<unsigned short> &test_embed,
                    std::vector<std::vector<int>> &position_ids)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);

        std::vector<int> cached_token;
        std::vector<int> token_ids;
        // std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
        if (input_embed_num > _attr.prefill_max_token_num)
        {
            ALOGE("input token num(%d) > prefill_max_token_num(%d)", input_embed_num, _attr.prefill_max_token_num);
            return "";
        }

        int kv_cache_num;
        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++)
        {
            mask[i] = 0;
        }
        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        int max_pos_id = 0;
        for (size_t p = 0; p < prefill_split_num; p++)
        {
            if (b_stop)
            {
                break;
            }
            _attr.prefill_grpid = p + 1;
            kv_cache_num = p * _attr.prefill_token_num;
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
            int start, offset;

            start = p * _attr.prefill_token_num;
            if (p == (prefill_split_num - 1))
            {
                offset = (input_embed_num - p * _attr.prefill_token_num);
                memcpy(embed_tmp.data(), test_embed.data() + start * _attr.tokens_embed_size, offset * _attr.tokens_embed_size * sizeof(unsigned short));
            }
            else
            {
                offset = _attr.prefill_token_num;
                memcpy(embed_tmp.data(), test_embed.data() + start * _attr.tokens_embed_size, offset * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
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

                // set indices
                auto &input_indices = layer.layer.get_input(_attr.prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                // ALOGI("position_ids");
                for (unsigned int i = 0; i < position_ids.size(); i++)
                {
                    for (unsigned int j = _attr.precompute_len + p * _attr.prefill_token_num, jj = 0; j < _attr.precompute_len + (p + 1) * _attr.prefill_token_num; j++, jj++)
                    {
                        if (j < position_ids[i].size())
                        {
                            input_indices_ptr[i * _attr.prefill_token_num + jj] = position_ids[i][j];
                            if (position_ids[i][j] > max_pos_id)
                            {
                                max_pos_id = position_ids[i][j];
                            }
                        }
                    }
                }
                // axcl_Memcpy((void *)input_indices.phyAddr, input_indices_ptr, input_indices.nSize, AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                // set mask
                auto &input_mask = layer.layer.get_input(_attr.prefill_grpid, "mask");
                // axcl_Memcpy((void *)input_mask.phyAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));
                // set input
                auto &input_input = layer.layer.get_input(_attr.prefill_grpid, "input");
                // axcl_Memcpy((void *)input_input.phyAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(_attr.prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                int kv_offset = (_attr.precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;

                // axcl_Memcpy((unsigned short *)input_decoder_k_cache.phyAddr + kv_offset,
                //             (void *)output_k_cache.phyAddr,
                //             sizeof(unsigned short) * input_num_token * _attr.kv_cache_size,
                //             AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());
                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                // axcl_Memcpy((unsigned short *)input_decoder_v_cache.phyAddr + kv_offset,
                //             (void *)output_v_cache.phyAddr,
                //             sizeof(unsigned short) * input_num_token * _attr.kv_cache_size,
                //             AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());
                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                for (int gid = _attr.prefill_grpid + 1; gid < prefill_split_num + 1; gid++)
                {
                    auto &input_prefill_k_cache = layer.layer.get_input(gid, "K_cache");
                    // axcl_Memcpy((unsigned short *)input_prefill_k_cache.phyAddr + kv_offset,
                    //             (void *)output_k_cache.phyAddr,
                    //             sizeof(unsigned short) * input_num_token * _attr.kv_cache_size,
                    //             AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());
                    memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                           (void *)output_k_cache.pVirAddr,
                           sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);
                }

                for (int gid = _attr.prefill_grpid + 1; gid < prefill_split_num + 1; gid++)
                {
                    auto &input_prefill_v_cache = layer.layer.get_input(gid, "V_cache");
                    // axcl_Memcpy((unsigned short *)input_prefill_v_cache.phyAddr + kv_offset,
                    //             (void *)output_v_cache.phyAddr,
                    //             sizeof(unsigned short) * input_num_token * _attr.kv_cache_size,
                    //             AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());
                    memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                           (void *)output_v_cache.pVirAddr,
                           sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);
                }

                auto &output = layer.layer.get_output(_attr.prefill_grpid, "output");
                // axcl_Memcpy(embed_tmp.data(), (void *)output.phyAddr, embed_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    layer.layer.deinit();
                }
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
            if (_attr.b_use_topk)
            {
                AX_SYS_MinvalidateCache(llama_post.get_output("indices").phyAddr, llama_post.get_output("indices").pVirAddr, llama_post.get_output("indices").nSize);
                max_index = *(int *)llama_post.get_output("indices").pVirAddr;
            }
            else
            {
                auto &output_post = llama_post.get_output(0);
                // AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
                memcpy(output_post.pVirAddr, (void *)output_post.pVirAddr, output_post.nSize);
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val = -MAXFLOAT;
                max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, &max_val);
                // max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
            }
            next_token = max_index;

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;

        for (unsigned int indices = max_pos_id + 1; indices < _attr.max_token_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

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
