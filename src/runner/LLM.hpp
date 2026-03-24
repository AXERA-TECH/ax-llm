#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
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
#include "utils/files.hpp"
#include "utils/json.hpp"

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
    std::vector<bool> layer_is_linear_attn;
    bool has_layer_types_from_config = false;
    ax_runner_ax650 llama_post;
    ax_runner_ax650 image_encoder;
    // int prefill_grpid = 1;
    int decode_grpid = 0;

    bool b_stop = false;

    LLMPostprocess postprocess;

    static bool _endswith(const std::string &s, const std::string &suffix)
    {
        if (suffix.size() > s.size())
        {
            return false;
        }
        return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
    }

    static std::string _parent_dir(const std::string &path)
    {
        auto pos = path.find_last_of('/');
        if (pos == std::string::npos)
        {
            return ".";
        }
        if (pos == 0)
        {
            return "/";
        }
        return path.substr(0, pos);
    }

    static std::string _resolve_config_path(const std::string &tokenizer_model_path)
    {
        if (tokenizer_model_path.empty())
        {
            return "";
        }
        if (is_directory(tokenizer_model_path))
        {
            std::string cfg = tokenizer_model_path + "/config.json";
            if (is_file(cfg))
            {
                return cfg;
            }
            return "";
        }
        if (is_file(tokenizer_model_path) && _endswith(tokenizer_model_path, ".json"))
        {
            return tokenizer_model_path;
        }
        std::string cfg = _parent_dir(tokenizer_model_path) + "/config.json";
        if (is_file(cfg))
        {
            return cfg;
        }
        return "";
    }

    void _load_layer_types_from_model_config()
    {
        has_layer_types_from_config = false;
        layer_is_linear_attn.assign(_attr.axmodel_num, false);
        std::string config_path = _resolve_config_path(_attr.filename_tokenizer_model);
        if (config_path.empty())
        {
            ALOGW("cannot resolve model config from tokenizer path: %s", _attr.filename_tokenizer_model.c_str());
            return;
        }

        try
        {
            std::ifstream f(config_path);
            if (!f.is_open())
            {
                ALOGW("open config failed: %s", config_path.c_str());
                return;
            }
            nlohmann::json j = nlohmann::json::parse(f);
            nlohmann::json text_cfg = j;
            if (j.contains("text_config"))
            {
                text_cfg = j["text_config"];
            }
            else if (j.contains("llm_config"))
            {
                text_cfg = j["llm_config"];
            }
            else if (j.contains("language_config"))
            {
                text_cfg = j["language_config"];
            }

            std::vector<std::string> layer_types;
            if (text_cfg.contains("layer_types") && text_cfg["layer_types"].is_array())
            {
                for (auto &it : text_cfg["layer_types"])
                {
                    if (it.is_string())
                    {
                        layer_types.push_back(it.get<std::string>());
                    }
                }
            }

            if (layer_types.empty() && text_cfg.contains("full_attention_interval") && text_cfg.contains("num_hidden_layers"))
            {
                int interval = text_cfg["full_attention_interval"].get<int>();
                int num_layers = text_cfg["num_hidden_layers"].get<int>();
                if (interval > 0 && num_layers > 0)
                {
                    layer_types.resize(num_layers, "linear_attention");
                    for (int i = 0; i < num_layers; ++i)
                    {
                        layer_types[i] = (((i + 1) % interval) == 0) ? "full_attention" : "linear_attention";
                    }
                }
            }

            int n = std::min((int)layer_types.size(), _attr.axmodel_num);
            for (int i = 0; i < n; ++i)
            {
                layer_is_linear_attn[i] = (layer_types[i] == "linear_attention");
            }
            has_layer_types_from_config = (n > 0);
            ALOGI("load layer_types from %s, parsed=%d, applied=%d", config_path.c_str(), (int)layer_types.size(), n);
        }
        catch (const std::exception &e)
        {
            ALOGW("parse model config failed (%s): %s", config_path.c_str(), e.what());
        }
    }

    void _fallback_infer_layer_types_from_io()
    {
        if ((int)layer_is_linear_attn.size() != _attr.axmodel_num)
        {
            layer_is_linear_attn.assign(_attr.axmodel_num, false);
        }
        for (int i = 0; i < _attr.axmodel_num; ++i)
        {
            try
            {
                auto &layer = llama_layers[i].layer;
                auto &k_in = layer.get_input(decode_grpid, "K_cache");
                auto &k_out = layer.get_output(decode_grpid, "K_cache_out");
                bool by_size = (k_in.nSize == k_out.nSize);
                bool by_shape = (k_in.vShape.size() >= 2 && k_out.vShape.size() >= 2 &&
                                 k_in.vShape[1] == 1 && k_out.vShape[1] == 1);
                layer_is_linear_attn[i] = by_size || by_shape;
            }
            catch (const std::exception &e)
            {
                ALOGW("infer layer type by io failed at layer %d: %s", i, e.what());
                layer_is_linear_attn[i] = false;
            }
        }
        ALOGW("layer_types fallback by io done");
    }

    bool _is_linear_layer(int layer_idx) const
    {
        return layer_idx >= 0 && layer_idx < (int)layer_is_linear_attn.size() && layer_is_linear_attn[layer_idx];
    }

    int _first_full_layer_idx() const
    {
        for (int i = 0; i < (int)layer_is_linear_attn.size(); ++i)
        {
            if (!layer_is_linear_attn[i])
            {
                return i;
            }
        }
        return -1;
    }

    bool _try_set_layer_input_mask(const ax_runner_tensor_t &input_mask, const std::vector<unsigned short> &mask_data)
    {
        size_t bytes = std::min((size_t)input_mask.nSize, mask_data.size() * sizeof(unsigned short));
        if (bytes == 0)
        {
            return false;
        }
        memcpy((void *)input_mask.pVirAddr, (void *)mask_data.data(), bytes);
        return true;
    }

    void _fill_linear_prefill_mask(std::vector<unsigned short> &mask_data, int valid_tokens, unsigned short one_bf16)
    {
        std::fill(mask_data.begin(), mask_data.end(), 0);
        int n = std::min((int)mask_data.size(), valid_tokens);
        for (int i = 0; i < n; ++i)
        {
            mask_data[i] = one_bf16;
        }
    }

    void _copy_linear_cache_state(const ax_runner_tensor_t &dst_k_cache, const ax_runner_tensor_t &dst_v_cache,
                                  const ax_runner_tensor_t &src_k_out, const ax_runner_tensor_t &src_v_out)
    {
        memcpy((void *)dst_k_cache.pVirAddr, (void *)src_k_out.pVirAddr, std::min(dst_k_cache.nSize, src_k_out.nSize));
        memcpy((void *)dst_v_cache.pVirAddr, (void *)src_v_out.pVirAddr, std::min(dst_v_cache.nSize, src_v_out.nSize));
    }

    void _copy_full_cache_tokens(const ax_runner_tensor_t &dst_k_cache, const ax_runner_tensor_t &dst_v_cache,
                                 const ax_runner_tensor_t &src_k_out, const ax_runner_tensor_t &src_v_out,
                                 int kv_offset_tokens, int valid_token_num)
    {
        if (_attr.kv_cache_size <= 0 || valid_token_num <= 0)
        {
            return;
        }
        int max_dst_tokens = dst_k_cache.nSize / (int)sizeof(unsigned short) / _attr.kv_cache_size;
        if (kv_offset_tokens >= max_dst_tokens)
        {
            return;
        }
        int copy_tokens = std::min(valid_token_num, max_dst_tokens - kv_offset_tokens);
        int src_tokens = src_k_out.nSize / (int)sizeof(unsigned short) / _attr.kv_cache_size;
        copy_tokens = std::min(copy_tokens, src_tokens);
        if (copy_tokens <= 0)
        {
            return;
        }
        size_t copy_bytes = (size_t)copy_tokens * _attr.kv_cache_size * sizeof(unsigned short);
        memcpy((unsigned short *)dst_k_cache.pVirAddr + (size_t)kv_offset_tokens * _attr.kv_cache_size,
               (void *)src_k_out.pVirAddr, copy_bytes);
        memcpy((unsigned short *)dst_v_cache.pVirAddr + (size_t)kv_offset_tokens * _attr.kv_cache_size,
               (void *)src_v_out.pVirAddr, copy_bytes);
    }

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

        // tokenizer = CreateTokenizer(attr.tokenizer_type);
        // if (!tokenizer->Init(attr.filename_tokenizer_model, attr.b_bos, attr.b_eos))
        // {
        //     ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.filename_tokenizer_model.c_str(), attr.b_bos, attr.b_eos);
        //     return false;
        // }

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

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), true);
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        ret = image_encoder.init(attr.filename_image_encoder_axmodedl.c_str(), true);
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

        _load_layer_types_from_model_config();
        if (!has_layer_types_from_config && !attr.b_dynamic_load_axmodel_layer)
        {
            _fallback_infer_layer_types_from_io();
        }

        int ref_full_layer_idx = _first_full_layer_idx();
        if (ref_full_layer_idx < 0)
        {
            ref_full_layer_idx = 0;
            ALOGW("no full-attention layer found, fallback to layer-0 for kv settings");
        }
        ALOGI("cache reference layer idx=%d (%s)", ref_full_layer_idx, _is_linear_layer(ref_full_layer_idx) ? "linear" : "full");

        if (attr.b_dynamic_load_axmodel_layer)
        {
            // 仅加载参考层获取 shape 信息
            auto &layer = llama_layers[ref_full_layer_idx];
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
                return false;
            }
            if (!has_layer_types_from_config)
            {
                try
                {
                    auto &k_in = layer.layer.get_input(decode_grpid, "K_cache");
                    auto &k_out = layer.layer.get_output(decode_grpid, "K_cache_out");
                    layer_is_linear_attn[ref_full_layer_idx] = (k_in.nSize == k_out.nSize);
                }
                catch (const std::exception &e)
                {
                    ALOGW("infer reference layer type failed: %s", e.what());
                }
            }
        }

        {
            auto &ref_layer = llama_layers[ref_full_layer_idx].layer;
            _attr.max_token_len = ref_layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            printf("\n");
            ALOGI("max_token_len : %d", _attr.max_token_len);
            // 基于 full-attention 参考层计算 token-wise KV cache 参数
            _attr.kv_cache_size = ref_layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = ref_layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = ref_layer.get_input(1, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
            _attr.prefill_max_kv_cache_num_grp.clear();
            for (size_t i = 0; i < ref_layer.get_num_input_groups() - 1; i++)
            {
                int prefill_max_kv_cache_num = ref_layer.get_input(i + 1, "K_cache").vShape[1];
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
        position_ids = get_rope_index(cfg, input_ids, cfg.image_grid_thw, cfg.video_grid_thw);
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
        // std::vector<int> input_ids = tokenizer->Encode(prompt, img_info);
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
               std::vector<std::vector<int>> &position_ids,
               Config &cfg, std::string prompt = "What is in the image?")
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
        // img_info.imgsz = 384;
        std::vector<int> input_ids = tokenizer->encode(contents);
        // std::vector<int> input_ids = tokenizer->Encode(prompt, img_info);
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

        bfloat16 bf16_neg = -65536.f;
        bfloat16 bf16_one = 1.f;
        std::vector<unsigned short> full_decode_mask(_attr.kv_cache_num + 1, bf16_neg.data);
        std::vector<unsigned short> linear_decode_mask_scalar(1, bf16_one.data);
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
        full_decode_mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++)
        {
            full_decode_mask[i] = 0;
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
            std::vector<unsigned short> full_prefill_mask_tmp;
            full_prefill_mask_tmp.resize(1 * _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16_neg.data);
            std::vector<unsigned short> linear_prefill_mask_tmp(_attr.prefill_token_num, 0);
            int input_num_token = _attr.prefill_token_num;
            if (p == prefill_split_num - 1)
            {
                input_num_token = input_embed_num - p * _attr.prefill_token_num;
            }

            ALOGI("input_num_token:%d", input_num_token);
            _fill_linear_prefill_mask(linear_prefill_mask_tmp, input_num_token, bf16_one.data);
            for (size_t i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    int mask_current_start = kv_cache_num;
                    auto mask_ptr = full_prefill_mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);

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
                int idx_rows = input_indices.vShape.size() >= 1 ? (int)input_indices.vShape[0] : 1;
                int idx_cols = input_indices.vShape.size() >= 2 ? (int)input_indices.vShape[1] : _attr.prefill_token_num;
                if (idx_rows <= 0)
                {
                    idx_rows = 1;
                }
                if (idx_cols <= 0)
                {
                    idx_cols = _attr.prefill_token_num;
                }
                for (int i = 0; i < idx_rows; i++)
                {
                    const std::vector<int> *pos_row = nullptr;
                    if (i < (int)position_ids.size())
                    {
                        pos_row = &position_ids[i];
                    }
                    else if (!position_ids.empty())
                    {
                        pos_row = &position_ids[0];
                    }
                    if (pos_row == nullptr)
                    {
                        break;
                    }
                    for (int j = _attr.precompute_len + p * _attr.prefill_token_num, jj = 0;
                         j < _attr.precompute_len + (p + 1) * _attr.prefill_token_num && jj < idx_cols;
                         j++, jj++)
                    {
                        if (j < (int)pos_row->size())
                        {
                            input_indices_ptr[i * idx_cols + jj] = (*pos_row)[j];
                            if ((*pos_row)[j] > max_pos_id)
                            {
                                max_pos_id = (*pos_row)[j];
                            }
                        }
                    }
                }

                // set mask
                auto &input_mask = layer.layer.get_input(_attr.prefill_grpid, "mask");
                if (_is_linear_layer(m))
                {
                    _try_set_layer_input_mask(input_mask, linear_prefill_mask_tmp);
                }
                else
                {
                    _try_set_layer_input_mask(input_mask, full_prefill_mask_tmp);
                }
                // set input
                auto &input_input = layer.layer.get_input(_attr.prefill_grpid, "input");
                // axcl_Memcpy((void *)input_input.phyAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(_attr.prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                if (_is_linear_layer(m))
                {
                    _copy_linear_cache_state(input_decoder_k_cache, input_decoder_v_cache, output_k_cache, output_v_cache);
                    for (int gid = _attr.prefill_grpid + 1; gid < prefill_split_num + 1; gid++)
                    {
                        auto &input_prefill_k_cache = layer.layer.get_input(gid, "K_cache");
                        auto &input_prefill_v_cache = layer.layer.get_input(gid, "V_cache");
                        _copy_linear_cache_state(input_prefill_k_cache, input_prefill_v_cache, output_k_cache, output_v_cache);
                    }
                }
                else
                {
                    int kv_offset_tokens = _attr.precompute_len + p * _attr.prefill_token_num;
                    _copy_full_cache_tokens(input_decoder_k_cache, input_decoder_v_cache, output_k_cache, output_v_cache,
                                            kv_offset_tokens, input_num_token);

                    for (int gid = _attr.prefill_grpid + 1; gid < prefill_split_num + 1; gid++)
                    {
                        auto &input_prefill_k_cache = layer.layer.get_input(gid, "K_cache");
                        auto &input_prefill_v_cache = layer.layer.get_input(gid, "V_cache");
                        _copy_full_cache_tokens(input_prefill_k_cache, input_prefill_v_cache, output_k_cache, output_v_cache,
                                                kv_offset_tokens, input_num_token);
                    }
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

                if (m == 0)
                {
                    auto &input_decode_embed = layer.layer.get_input(decode_grpid, "input");
                    memcpy((void *)input_decode_embed.pVirAddr, embed.data(), input_decode_embed.nSize);
                }

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                unsigned int *decode_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                int decode_indices_num = input_indices.nSize / (int)sizeof(unsigned int);
                for (int i = 0; i < decode_indices_num; ++i)
                {
                    decode_indices_ptr[i] = indices;
                }

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                if (_is_linear_layer(m))
                {
                    std::vector<unsigned short> linear_decode_mask(input_mask.nSize / sizeof(unsigned short), bf16_one.data);
                    if (linear_decode_mask.empty())
                    {
                        linear_decode_mask = linear_decode_mask_scalar;
                    }
                    _try_set_layer_input_mask(input_mask, linear_decode_mask);
                }
                else
                {
                    _try_set_layer_input_mask(input_mask, full_decode_mask);
                }

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                if (_is_linear_layer(m))
                {
                    _copy_linear_cache_state(input_k_cache, input_v_cache, output_k_cache, output_v_cache);
                }
                else
                {
                    int max_full_tokens = input_k_cache.nSize / (int)sizeof(unsigned short) / std::max(1, _attr.kv_cache_size);
                    if ((int)indices < max_full_tokens)
                    {
                        memcpy((unsigned short *)input_k_cache.pVirAddr + (size_t)indices * _attr.kv_cache_size,
                               (void *)output_k_cache.pVirAddr,
                               std::min((size_t)output_k_cache.nSize, (size_t)_attr.kv_cache_size * sizeof(unsigned short)));
                        memcpy((unsigned short *)input_v_cache.pVirAddr + (size_t)indices * _attr.kv_cache_size,
                               (void *)output_v_cache.pVirAddr,
                               std::min((size_t)output_v_cache.nSize, (size_t)_attr.kv_cache_size * sizeof(unsigned short)));
                    }
                }

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
            full_decode_mask[indices] = 0;
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
                // if (tokenizer->isEnd(max_index))
                {
                    if (cached_token.size() && _attr.runing_callback)
                    {
                        float t_cost_ms = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out = tokenizer->decode(cached_token);
                        // auto tmp_out = tokenizer->Decode(cached_token);
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
                        // auto tmp_out = tokenizer->Decode(cached_token);
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
        // final_out = tokenizer->Decode(token_ids);

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
