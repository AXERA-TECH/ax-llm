#pragma once
#include <functional>
#include <string>
#include <vector>
#include <memory>

#include "BaseTokenizer.hpp"  // for Content/RoleType definitions
#include "VLMType.hpp"

class LLMPostprocess;
class LLaMaEmbedSelector;

using LLMRuningCallback = std::function<void(std::string str, float token_per_sec, void *reserve)>;

// Multimodal (VLM) media inputs are passed out-of-band, aligned to `history` by index.
// `history[content_index].type` must be IMAGE or VIDEO.
// For IMAGE: each uri can be a file or a directory of images (sorted).
// For VIDEO: uri is typically a directory of frames (sorted).
struct MediaInputs {
    size_t content_index = 0;
    std::vector<std::string> uris;
};

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

    // ---- vision / VLM (optional, runtime switch by `vlm_type`) ----
    // If `vlm_type != VLMType::None`, vision encoder will be initialized and used.
    // See `VLMType` in `src/runner/VLMType.hpp`.
    // Names/ids (via magic_enum): `None(0)`, `Qwen2_5VL(1)`, `Qwen3VL(2)`, `InternVL3(3)`, `FastVLM(4)`, `SmolVLM2(5)`.
    VLMType vlm_type = VLMType::None;

    // Vision encoder axmodel (image/video encoder). Required if `vlm_type != VLMType::None`.
    std::string filename_image_encoder_axmodel = "image_encoder.axmodel";

    // Optional: vision embedding cache directory. If empty: memory-only cache for the process lifetime.
    // If set: read/write encoded embeddings for repeated images across runs.
    std::string vision_cache_dir;

    // Qwen-VL patchifier params (also used to compute mRoPE indices).
    int vision_width = 448;
    int vision_height = 448;
    int vision_temporal_patch_size = 2;
    int vision_spatial_merge_size = 2;
    int vision_patch_size = 14;
    int vision_fps = 1;              // for qwen2.5-vl time scaling
    int vision_tokens_per_second = 1;

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
    std::vector<Content> Run(std::vector<Content> history, const std::vector<MediaInputs> &media_inputs, int output_max_token = -1);
    std::string Run(std::vector<unsigned short> &embed, int output_max_token = -1);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
