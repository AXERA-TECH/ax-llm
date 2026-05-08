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

// Multimodal media inputs are passed out-of-band, aligned to `history` by index.
// `history[content_index].type` can be IMAGE / VIDEO / AUDIO.
// For IMAGE: each uri can be a file or a directory of images (sorted).
// For VIDEO: uri is typically a directory of frames (sorted).
// For AUDIO: current Gemma4 support expects one audio file per message.
struct MediaInputs {
    size_t content_index = 0;
    std::vector<std::string> uris;
};

struct LLMAttrType {
    std::string system_prompt;
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    // Optional: for models with mixed attention layer types (e.g., Qwen3.5).
    // If > 0, every Nth layer (1-indexed) is treated as full-attention and the others as linear-attention.
    // When 0, all layers are treated as full-attention (default).
    int full_attention_interval = 0;
    int num_kv_shared_layers = 0;
    int sliding_window = 0;
    std::vector<std::string> layer_types;

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    std::string tokenizer_type;
    std::string url_tokenizer_model = "http://127.0.0.1:12345";
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;
    int pad_token_id = 0;
    int hidden_size_per_layer_input = 0;
    float rms_norm_eps = 1e-6f;
    std::string filename_tokens_embed_per_layer;
    std::string filename_per_layer_model_projection;
    std::string filename_per_layer_projection_norm;

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
    // Names/ids (via magic_enum): `None(0)`, `Qwen2_5VL(1)`, `Qwen3VL(2)`, `InternVL3(3)`, `FastVLM(4)`, `SmolVLM2(5)`, `PaddleOCRVL(6)`, `Gemma4VL(7)`.
    VLMType vlm_type = VLMType::None;

    // Vision encoder axmodel (image/video encoder). Required if `vlm_type != VLMType::None`.
    std::string filename_image_encoder_axmodel = "image_encoder.axmodel";
    std::string filename_audio_encoder_axmodel_5s = "gemma4_audio_5s.axmodel";
    std::string filename_audio_encoder_axmodel_30s = "gemma4_audio_30s.axmodel";

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
    int vision_num_frames = 0;       // for frame-sampled video input
    bool vision_do_sample_frames = true;

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
    void SetRequestSamplingOverride(bool has_temperature, float temperature, bool has_top_p, float top_p);
    void ClearRequestSamplingOverride();
    std::string GetLastError() const;

    bool Embed(const std::string &text, std::vector<float> &out_embedding);
    bool Embed(const std::vector<Content> &history, const std::vector<MediaInputs> &media_inputs, std::vector<float> &out_embedding);
    bool EmbedBatch(const std::vector<std::string> &inputs, std::vector<std::vector<float>> &out_embeddings);

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
