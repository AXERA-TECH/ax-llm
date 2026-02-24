#pragma once
#include <functional>
#include <string>
#include <vector>
#include <memory>

#include "BaseTokenizer.hpp"  // for Content/RoleType definitions

class LLMPostprocess;
class LLaMaEmbedSelector;

using LLMRuningCallback = std::function<void(std::string str, float token_per_sec, void *reserve)>;

struct LLMAttrType {
    std::string system_prompt;
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    std::string tokenizer_type;
    std::string url_tokenizer_model = "http://127.0.0.1:12345";
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc
    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    std::vector<int> prefill_max_kv_cache_num_grp;
    int prefill_grpid = -1;
    std::string post_config_path = "post_config.json";
    bool b_use_mmap_load_embed = false;

#ifndef USE_AXCL
    bool b_use_mmap_load_layer = true;
#endif

#ifdef USE_AXCL
    std::vector<int> dev_ids = {0};
#endif

    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;
};

class LLM {
public:
    LLM();
    ~LLM();

    bool Init(LLMAttrType attr);
    void Deinit();
    void Stop();

    LLMAttrType *getAttr();
    LLMPostprocess *getPostprocess();
    LLaMaEmbedSelector *getEmbedSelector();

    int GenerateKVCachePrefill(std::vector<int> &ids,
                               std::vector<std::vector<unsigned short>> &k,
                               std::vector<std::vector<unsigned short>> &v,
                               int &pre_len);

    int GetKVCache(std::vector<std::vector<unsigned short>> &k,
                   std::vector<std::vector<unsigned short>> &v,
                   int &pre_len);

    int SetKVCache(std::vector<std::vector<unsigned short>> &k,
                   std::vector<std::vector<unsigned short>> &v,
                   int precompute_len, int input_num_token);

    void ResetKVCache();

    std::vector<Content> Run(std::vector<Content> history, int output_max_token = -1);
    std::string Run(std::vector<unsigned short> &embed, int output_max_token = -1);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
