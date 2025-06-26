#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>

#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"

#include "cqdm.h"
#include "timer.hpp"
#include "LLMPostprocess.hpp"

#include "utils/axcl_manager.h"

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;

    TokenizerType tokenizer_type = TKT_LLaMa;
    std::string url_tokenizer_model = "http://127.0.0.1:12345";
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    // int prefill_max_kv_cache_num = 2048;
    std::vector<int> prefill_max_kv_cache_num_grp;
    int precompute_len = 1202;

    int prefill_grpid = -1;

    bool b_use_mmap_load_embed = false;

    std::string post_config_path = "post_config.json";

    std::vector<int> dev_ids = {0, 1, 2, 3};

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
    // ax_runner_ax650 image_encoder;

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

    std::vector<int> distributeModels(int cardCount, int modelCount)
    {
        std::vector<int> cardAssignments(modelCount);
        if (cardCount <= 0 || modelCount <= 0)
            return cardAssignments; // 返回空的或未初始化的 vector

        // 计算每张卡至少分配的模型数量
        int baseCount = modelCount / cardCount;
        // 计算余数，多出的模型会依次分配给前面的卡
        int remainder = modelCount % cardCount;

        int startIndex = 0;
        for (int card = 0; card < cardCount; ++card)
        {
            // 如果当前卡号在前 remainder 张卡中，则多分配一个模型
            int modelsOnThisCard = baseCount + (card < remainder ? 1 : 0);
            for (int i = 0; i < modelsOnThisCard; ++i)
            {
                cardAssignments[startIndex + i] = card;
            }
            startIndex += modelsOnThisCard;
        }

        return cardAssignments;
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
        for (auto &devid : _attr.dev_ids)
        {
            if (axcl_Init(devid) != 0)
            {
                ALOGE("axcl_Init(%d) failed", devid);
                return false;
            }
        }

        llama_layers.resize(attr.axmodel_num);

        auto dev_assignments = distributeModels(_attr.dev_ids.size(), attr.axmodel_num);

        std::vector<int> rets(attr.axmodel_num);
        std::atomic<int> process_idx = 2;
#pragma omp parallel for
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            char axmodel_path[1024];
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), _attr.dev_ids[dev_assignments[i]]);
            rets[i] = ret;
            // if (ret != 0)
            // {
            //     ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
            //     return false;
            // }
            int remain_cmm = axcl_GetCMMRemain(_attr.dev_ids[dev_assignments[i]]);
            sprintf(axmodel_path, "init %d axmodel ok,devid(%d) remain_cmm(%d MB)", i, _attr.dev_ids[dev_assignments[i]], remain_cmm);
            update_cqdm(&cqdm, process_idx++, "count", axmodel_path);
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), llama_layers[llama_layers.size() - 1].layer.get_devid());

        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = axcl_GetCMMRemain(llama_post.get_devid());
        char axmodel_path[1024];
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        printf("\n");
        {
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

        std::vector<int> v_remain_cmm;
        for (int i = 0; i < _attr.dev_ids.size(); i++)
        {
            v_remain_cmm.push_back(axcl_GetCMMRemain(_attr.dev_ids[i]));
        }
        printf(MACRO_PURPLE "________________________\n");
        printf("|%6s|%15s|\n", "ID", "remain cmm(MB)");
        printf("========================\n");
        for (int i = 0; i < _attr.dev_ids.size(); i++)
        {
            printf("|%6d|%15d|\n", _attr.dev_ids[i], v_remain_cmm[i]);
        }
        printf("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n" MACRO_END);

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

        for (auto &devid : _attr.dev_ids)
            axcl_Exit(devid);
    }

    LLMPostprocess *getPostprocess()
    {
        return &postprocess;
    }

    void Stop()
    {
        b_stop = true;
    }

    int SetSystemPrompt(std::string system_prompt, std::vector<int> &_token_ids)
    {
        tokenizer->Reset(system_prompt, _token_ids);
        _attr.system_prompt = system_prompt;
        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
        return 0;
    }

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
            axcl_Memset((void *)llama_layers[i].layer.get_input(prefill_grpid, "K_cache").phyAddr, 0, llama_layers[i].layer.get_input(prefill_grpid, "K_cache").nSize, llama_layers[i].layer.get_devid());
            axcl_Memset((void *)llama_layers[i].layer.get_input(prefill_grpid, "V_cache").phyAddr, 0, llama_layers[i].layer.get_input(prefill_grpid, "V_cache").nSize, llama_layers[i].layer.get_devid());
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

        // axcl_Memcpy((void *)llama_layers[0].layer.get_input(_attr.prefill_grpid, "input").phyAddr, test_embed.data(), test_embed.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, llama_layers[0].layer.get_devid());
        // test_embed.resize(_attr.prefill_token_num * _attr.tokens_embed_size);

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
                axcl_Memcpy((void *)input_indices.phyAddr, input_indices_ptr, input_indices.nSize, AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                // set mask
                auto &input_mask = layer.layer.get_input(prefill_grpid, "mask");
                // memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));
                axcl_Memcpy((void *)input_mask.phyAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                auto &input_input = layer.layer.get_input(prefill_grpid, "input");
                axcl_Memcpy((void *)input_input.phyAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                layer.layer.inference(prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer.get_input(prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(prefill_grpid, "V_cache_out");

                int kv_offset = (p * _attr.prefill_token_num) * _attr.kv_cache_size;

                axcl_Memcpy((unsigned short *)input_decoder_k_cache.phyAddr + kv_offset,
                            (void *)output_k_cache.phyAddr,
                            sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size,
                            AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                axcl_Memcpy((unsigned short *)input_decoder_v_cache.phyAddr + kv_offset,
                            (void *)output_v_cache.phyAddr,
                            sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size,
                            AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                axcl_Memcpy((unsigned short *)input_prefill_k_cache.phyAddr + kv_offset,
                            (void *)output_k_cache.phyAddr,
                            sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size,
                            AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                axcl_Memcpy((unsigned short *)input_prefill_v_cache.phyAddr + kv_offset,
                            (void *)output_v_cache.phyAddr,
                            sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size,
                            AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                auto &output = layer.layer.get_output(prefill_grpid, "output");
                axcl_Memcpy(embed_tmp.data(), (void *)output.phyAddr, embed_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

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
            axcl_Memcpy((void *)k_caches[i].data(), (void *)input_k_cache.phyAddr, precompute_len * _attr.kv_cache_size * sizeof(unsigned short), AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());
            axcl_Memcpy((void *)v_caches[i].data(), (void *)input_v_cache.phyAddr, precompute_len * _attr.kv_cache_size * sizeof(unsigned short), AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());
        }

        return 0;
    }

    int GenerateKVCache(std::vector<int> &_token_ids)
    {
        // clear kv cache
        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            axcl_Memset((void *)llama_layers[i].layer.get_input(decode_grpid, "K_cache").phyAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize, llama_layers[i].layer.get_devid());
            axcl_Memset((void *)llama_layers[i].layer.get_input(decode_grpid, "V_cache").phyAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize, llama_layers[i].layer.get_devid());
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

            axcl_Memcpy((void *)llama_layers[0].layer.get_input(decode_grpid, "input").phyAddr, embed.data(), llama_layers[0].layer.get_input(decode_grpid, "input").nSize, AXCL_MEMCPY_HOST_TO_DEVICE, llama_layers[0].layer.get_devid());
            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                // unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                // memcpy(input_k_cache.pVirAddr, k_caches[m].data(), sizeof(unsigned short) * k_caches[m].size());
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                // unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;
                // memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                // memcpy(input_indices.pVirAddr, &indices, sizeof(indices));
                axcl_Memcpy((void *)input_indices.phyAddr, &indices, sizeof(indices), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                // memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));
                axcl_Memcpy((void *)input_mask.phyAddr, mask.data(), mask.size() * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                // auto &input_input = layer.layer.get_input(decode_grpid, "input");
                // memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                // axcl_Memcpy((void *)input_input.phyAddr, embed.data(), embed.size() * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                // memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                axcl_Memcpy((unsigned short *)input_k_cache.phyAddr + indices * _attr.kv_cache_size, (void *)output_k_cache.phyAddr, output_k_cache.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                // memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                axcl_Memcpy((unsigned short *)input_v_cache.phyAddr + indices * _attr.kv_cache_size, (void *)output_v_cache.phyAddr, output_v_cache.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                // auto &output = layer.layer.get_output(decode_grpid, "output");
                // memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));
                // axcl_Memcpy(embed.data(), (void *)output.phyAddr, output.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

                if (m < _attr.axmodel_num - 1)
                {
                    if (llama_layers[m + 1].layer.get_devid() == layer.layer.get_devid())
                    {
                        axcl_Memcpy((void *)llama_layers[m + 1].layer.get_input(decode_grpid, "input").phyAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").phyAddr, layer.layer.get_input(decode_grpid, "input").nSize, AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());
                    }
                    else
                    {
                        axcl_Memcpy((void *)layer.layer.get_output(decode_grpid, "output").pVirAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").phyAddr, layer.layer.get_output(decode_grpid, "output").nSize, AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

                        axcl_Memcpy((void *)llama_layers[m + 1].layer.get_input(decode_grpid, "input").phyAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, layer.layer.get_input(decode_grpid, "input").nSize, AXCL_MEMCPY_HOST_TO_DEVICE, llama_layers[m + 1].layer.get_devid());
                    }
                }

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
        axcl_Memcpy(mask.data(), (void *)input_mask.phyAddr, input_mask.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, llama_layers[0].layer.get_devid());
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
            axcl_Memcpy((void *)k_caches[i].data(), (void *)input_k_cache.phyAddr, precompute_len * _attr.kv_cache_size * sizeof(unsigned short), AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());
            axcl_Memcpy((void *)v_caches[i].data(), (void *)input_v_cache.phyAddr, precompute_len * _attr.kv_cache_size * sizeof(unsigned short), AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());
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
            ALOGE("k_caches.size(%ld) != v_caches.size(%ld)", k_caches.size(), v_caches.size());
            return -1;
        }

        if (k_caches.size() != _attr.axmodel_num)
        {
            ALOGE("k_caches.size(%ld) != _attr.axmodel_num(%d)", k_caches.size(), _attr.axmodel_num);
            return -1;
        }

        // clear kv cache
        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            // int prefill_split_num = _attr.prefill_max_token_num / _attr.prefill_token_num;
            // for (size_t p = 0; p < prefill_split_num; p++)
            // {
            // int prefill_grpid = p + 1;
            axcl_Memset((void *)llama_layers[i].layer.get_input(_attr.prefill_grpid, "K_cache").phyAddr, 0, llama_layers[i].layer.get_input(_attr.prefill_grpid, "K_cache").nSize, llama_layers[i].layer.get_devid());
            axcl_Memset((void *)llama_layers[i].layer.get_input(_attr.prefill_grpid, "V_cache").phyAddr, 0, llama_layers[i].layer.get_input(_attr.prefill_grpid, "V_cache").nSize, llama_layers[i].layer.get_devid());
            // }

            axcl_Memset((void *)llama_layers[i].layer.get_input(decode_grpid, "K_cache").phyAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize, llama_layers[i].layer.get_devid());
            axcl_Memset((void *)llama_layers[i].layer.get_input(decode_grpid, "V_cache").phyAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize, llama_layers[i].layer.get_devid());
        }

        // int prefill_grpid = llama_layers[0].layer.get_num_input_groups() - 1;

        for (unsigned int m = 0; m < _attr.axmodel_num; m++)
        {
            auto &layer = llama_layers[m];

            auto &k_cache = k_caches[m];
            auto &v_cache = v_caches[m];

            if (k_cache.size() != _attr.precompute_len * _attr.kv_cache_size)
            {
                ALOGE("k_cache.size(%ld) != precompute_len(%d) * _attr.kv_cache_size(%d)", k_cache.size(), _attr.precompute_len, _attr.kv_cache_size);
                return -1;
            }
            if (v_cache.size() < _attr.precompute_len * _attr.kv_cache_size)
            {
                ALOGE("v_cache.size(%ld) < precompute_len(%d) * _attr.kv_cache_size(%d)", v_cache.size(), _attr.precompute_len, _attr.kv_cache_size);
                return -1;
            }

            // set kv cache inputs
            {
                auto &input_k_cache = layer.layer.get_input(_attr.prefill_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.phyAddr;
                auto &input_v_cache = layer.layer.get_input(_attr.prefill_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.phyAddr;

                // memcpy(input_k_cache_ptr, k_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                // memcpy(input_v_cache_ptr, v_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                axcl_Memcpy((void *)input_k_cache_ptr, (void *)k_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
                axcl_Memcpy((void *)input_v_cache_ptr, (void *)v_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
            }

            {
                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.phyAddr;
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.phyAddr;

                // memcpy(input_k_cache_ptr, k_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                // memcpy(input_v_cache_ptr, v_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                axcl_Memcpy((void *)input_k_cache_ptr, (void *)k_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
                axcl_Memcpy((void *)input_v_cache_ptr, (void *)v_cache.data(), _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
            }
        }

        return 0;
    }

    // int Encode(cv::Mat src, std::vector<unsigned short> &out_embed)
    // {
    //     std::vector<float> mean = {0.485, 0.456, 0.406};
    //     std::vector<float> scale = {0.229, 0.224, 0.225};
    //     timer t;
    //     t.start();
    //     cv::Mat dst;
    //     cv::resize(src, dst, cv::Size(_attr.image_encoder_width, _attr.image_encoder_height));
    //     cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    //     // std::vector<float> input_data(dst.rows * dst.cols * 3);

    //     float *input_data = (float *)image_encoder.get_input(0).pVirAddr;

    //     unsigned char *img_data = dst.data;
    //     int letterbox_rows = dst.rows;
    //     int letterbox_cols = dst.cols;

    //     for (int h = 0; h < letterbox_rows; h++)
    //     {
    //         for (int w = 0; w < letterbox_cols; w++)
    //         {
    //             for (int c = 0; c < 3; c++)
    //             {
    //                 int in_index = h * letterbox_cols * 3 + w * 3 + c;
    //                 int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
    //                 input_data[out_index] = (float(img_data[in_index]) / 255.0 - mean[c]) / scale[c];
    //             }
    //         }
    //     }

    //     // void *data = image_encoder.get_input("input").pVirAddr;
    //     // memcpy(data, dst.data, dst.rows * dst.cols * 3);

    //     // std::vector<char> vit_in;
    //     // if (!read_file("/home/axera/internvl2_5-8b-mpo_ax-infer/img.bin", vit_in))
    //     // {
    //     //     ALOGE("read img.bin failed");
    //     //     return -1;
    //     // }
    //     // memcpy(input_data, vit_in.data(), image_encoder.get_input(0).nSize);

    //     image_encoder.inference();
    //     int size = 1;
    //     for (size_t i = 0; i < image_encoder.get_output(0).vShape.size(); i++)
    //     {
    //         size *= image_encoder.get_output(0).vShape[i];
    //     }

    //     out_embed.resize(size);

    //     float *out_data = (float *)image_encoder.get_output(0).pVirAddr;

    //     for (size_t i = 0; i < size; i++)
    //     {
    //         out_embed[i] = bfloat16(out_data[i]).data;
    //     }

    //     // memcpy(out_embed.data(), image_encoder.get_output(0).pVirAddr, image_encoder.get_output(0).nSize);
    //     ALOGI("image encode time : %0.2f ms, size : %ld", t.cost(), out_embed.size());
    //     return 0;
    // }

    // int Encode(std::vector<unsigned short> &img_embed, std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    // {
    //     std::vector<int> input_ids = tokenizer->Encode(prompt, true);

    //     // constexpr int IMG_CONTEXT = 151648;	// InternVL2
    //     // constexpr int IMG_CONTEXT = 151667; // InternVL2.5
    //     constexpr int IMG_CONTEXT = 92546; // InternVL2.5-8B-MPO
    //     int offset = 0;

    //     for (size_t i = 0; i < input_ids.size(); i++)
    //     {
    //         if (input_ids[i] == IMG_CONTEXT)
    //         {
    //             offset = i;
    //             break;
    //         }
    //     }

    //     // for (size_t i = 0; i < input_ids.size(); i++)
    //     // {
    //     //     printf("%d ", input_ids[i]);
    //     // }
    //     // printf("\n");

    //     if (input_ids.size() > _attr.prefill_token_num)
    //     {
    //         ALOGE("input_ids(%ld) > prefill_token_num(%d)", input_ids.size(), _attr.prefill_token_num);
    //         return -1;
    //     }
    //     out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

    //     for (size_t i = 0; i < input_ids.size(); i++)
    //     {
    //         embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
    //     }
    //     memcpy(out_embed.data() + offset * _attr.tokens_embed_size, img_embed.data(), img_embed.size() * sizeof(unsigned short));
    //     ALOGI("offset : %d out_embed.size() : %ld", offset, out_embed.size());
    //     return 0;
    // }

    int Encode(std::vector<unsigned short> &out_embed, std::string prompt, std::string last_reply, std::vector<int> &tokens_ids, std::vector<int> &tokens_diff)
    {
        if (!tokenizer->Encode(prompt, last_reply, tokens_ids, tokens_diff))
        {
            ALOGE("encode failed");
            return -1;
        }

        // if (tokens_diff.size() > _attr.prefill_token_num)
        // {
        //     ALOGE("input_ids(%ld) > prefill_token_num(%d)", tokens_diff.size(), _attr.prefill_token_num);
        //     return -1;
        // }
        out_embed.resize(tokens_diff.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < tokens_diff.size(); i++)
        {
            embed_selector.getByIndex(tokens_diff[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        // memcpy(out_embed.data() + 5 * _attr.tokens_embed_size, vpm_resampler.get_output("output").pVirAddr, vpm_resampler.get_output("output").nSize);

        return 0;
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

        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < _attr.precompute_len + input_embed_num; i++)
        {
            mask[i] = 0;
        }

        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];

        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

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
                axcl_Memcpy((void *)input_indices.phyAddr, input_indices_ptr, input_indices.nSize, AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                // set mask
                auto &input_mask = layer.layer.get_input(_attr.prefill_grpid, "mask");
                axcl_Memcpy((void *)input_mask.phyAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                // set input
                auto &input_input = layer.layer.get_input(_attr.prefill_grpid, "input");
                axcl_Memcpy((void *)input_input.phyAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                layer.layer.inference(_attr.prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer.get_input(_attr.prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(_attr.prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                int kv_offset = (_attr.precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;

                axcl_Memcpy((unsigned short *)input_decoder_k_cache.phyAddr + kv_offset,
                            (void *)output_k_cache.phyAddr,
                            sizeof(unsigned short) * input_num_token * _attr.kv_cache_size,
                            AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                axcl_Memcpy((unsigned short *)input_decoder_v_cache.phyAddr + kv_offset,
                            (void *)output_v_cache.phyAddr,
                            sizeof(unsigned short) * input_num_token * _attr.kv_cache_size,
                            AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                axcl_Memcpy((unsigned short *)input_prefill_k_cache.phyAddr + kv_offset,
                            (void *)output_k_cache.phyAddr,
                            sizeof(unsigned short) * input_num_token * _attr.kv_cache_size,
                            AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                axcl_Memcpy((unsigned short *)input_prefill_v_cache.phyAddr + kv_offset,
                            (void *)output_v_cache.phyAddr,
                            sizeof(unsigned short) * input_num_token * _attr.kv_cache_size,
                            AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                auto &output = layer.layer.get_output(_attr.prefill_grpid, "output");
                axcl_Memcpy(embed_tmp.data(), (void *)output.phyAddr, embed_tmp.size() * sizeof(unsigned short), AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

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
            // memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            axcl_Memcpy((void *)input.phyAddr, embed.data(), embed.size() * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, llama_post.get_devid());
            llama_post.inference();

            int max_index;

            auto &output_post = llama_post.get_output("output");
            axcl_Memcpy(output_post.pVirAddr, (void *)output_post.phyAddr, output_post.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, llama_post.get_devid());
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
        for (unsigned int indices = _attr.precompute_len + input_embed_num; indices < _attr.max_token_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

            // ALOGI("out %d %d", indices, next_token);
            embed_selector.getByIndex(next_token, embed);

            axcl_Memcpy((void *)llama_layers[0].layer.get_input(decode_grpid, "input").phyAddr, embed.data(), llama_layers[0].layer.get_input(decode_grpid, "input").nSize, AXCL_MEMCPY_HOST_TO_DEVICE, llama_layers[0].layer.get_devid());
            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                // unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                // memcpy(input_k_cache.pVirAddr, k_caches[m].data(), sizeof(unsigned short) * k_caches[m].size());
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                // unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;
                // memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                // memcpy(input_indices.pVirAddr, &indices, sizeof(indices));
                axcl_Memcpy((void *)input_indices.phyAddr, &indices, sizeof(indices), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                // memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));
                axcl_Memcpy((void *)input_mask.phyAddr, mask.data(), mask.size() * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                // auto &input_input = layer.layer.get_input(decode_grpid, "input");
                // memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                // axcl_Memcpy((void *)input_input.phyAddr, embed.data(), embed.size() * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                // memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                axcl_Memcpy((unsigned short *)input_k_cache.phyAddr + indices * _attr.kv_cache_size, (void *)output_k_cache.phyAddr, output_k_cache.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                // memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                axcl_Memcpy((unsigned short *)input_v_cache.phyAddr + indices * _attr.kv_cache_size, (void *)output_v_cache.phyAddr, output_v_cache.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());

                // auto &output = layer.layer.get_output(decode_grpid, "output");
                // memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));
                // axcl_Memcpy(embed.data(), (void *)output.phyAddr, output.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

                if (m == _attr.axmodel_num - 1)
                {
                    if (llama_post.get_devid() == layer.layer.get_devid())
                    {
                        axcl_Memcpy((void *)llama_post.get_input("input").phyAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").phyAddr, llama_post.get_input("input").nSize, AXCL_MEMCPY_DEVICE_TO_DEVICE, llama_post.get_devid());
                    }
                    else
                    {
                        axcl_Memcpy((void *)layer.layer.get_output(decode_grpid, "output").pVirAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").phyAddr, layer.layer.get_output(decode_grpid, "output").nSize, AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

                        axcl_Memcpy((void *)llama_post.get_input("input").phyAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, llama_post.get_input("input").nSize, AXCL_MEMCPY_HOST_TO_DEVICE, llama_post.get_devid());
                    }
                }
                else if (m < _attr.axmodel_num - 1)
                {
                    if (llama_layers[m + 1].layer.get_devid() == layer.layer.get_devid())
                    {
                        axcl_Memcpy((void *)llama_layers[m + 1].layer.get_input(decode_grpid, "input").phyAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").phyAddr, layer.layer.get_input(decode_grpid, "input").nSize, AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid());
                    }
                    else
                    {
                        axcl_Memcpy((void *)layer.layer.get_output(decode_grpid, "output").pVirAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").phyAddr, layer.layer.get_output(decode_grpid, "output").nSize, AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

                        axcl_Memcpy((void *)llama_layers[m + 1].layer.get_input(decode_grpid, "input").phyAddr,
                                    (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, layer.layer.get_input(decode_grpid, "input").nSize, AXCL_MEMCPY_HOST_TO_DEVICE, llama_layers[m + 1].layer.get_devid());
                    }
                }

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            // ALOGI("");
            mask[indices] = 0;
            {
                // post process
                // auto &input = llama_post.get_input("input");
                // memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                // axcl_Memcpy((void *)input.phyAddr, embed.data(), embed.size() * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, llama_post.get_devid());
                llama_post.inference();

                auto &output_post = llama_post.get_output("output");
                axcl_Memcpy(output_post.pVirAddr, (void *)output_post.phyAddr, output_post.nSize, axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, llama_post.get_devid());
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val = -MAXFLOAT;
                // max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
                auto max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

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
