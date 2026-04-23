#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <optional>
#include "runner/LLM.hpp"
#include "utils/json.hpp"
#ifdef USE_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

static std::string resolve_path(const std::string &base, const std::string &p) {
    if (p.empty()) return p;
    if (p.rfind("http://",0)==0 || p.rfind("https://",0)==0) return p;
    namespace fs = std::filesystem;
    if (fs::path(p).is_absolute()) return p;
    return base + "/" + p;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: llm_smoke <model_dir> [max_tokens] [--image <path>] [--video <frames_dir>] [--audio <path>] [--prompt <text>] [--repeat <N>] [--quiet]\n";
        return 1;
    }
    std::string model_dir = argv[1];
    int max_tokens = (argc>=3)? std::atoi(argv[2]) : 16;
    std::string image_path;
    std::string video_dir;
    std::string audio_path;
    std::string prompt = "Describe the image.";
    int repeat = 1;
    bool quiet = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (a == "--video" && i + 1 < argc) video_dir = argv[++i];
        else if (a == "--audio" && i + 1 < argc) audio_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--repeat" && i + 1 < argc) repeat = std::max(1, std::atoi(argv[++i]));
        else if (a == "--quiet") quiet = true;
    }
    if (repeat > 1)
    {
        std::string rep;
        rep.reserve(prompt.size() * (size_t)repeat + (size_t)repeat);
        for (int i = 0; i < repeat; ++i)
        {
            if (i) rep.push_back(' ');
            rep += prompt;
        }
        prompt = std::move(rep);
    }
    std::string cfg = model_dir + "/config.json";
    if (!std::filesystem::exists(cfg)) { std::cerr << "config.json not found in " << model_dir << "\n"; return 2; }

    std::ifstream f(cfg); nlohmann::json j; f >> j;

    LLMAttrType attr;
    attr.template_filename_axmodel = resolve_path(model_dir, j["template_filename_axmodel"].get<std::string>());
    attr.filename_post_axmodel     = resolve_path(model_dir, j["filename_post_axmodel"].get<std::string>());
    attr.url_tokenizer_model       = resolve_path(model_dir, j["url_tokenizer_model"].get<std::string>());
    attr.tokenizer_type            = j.contains("tokenizer_type")? j["tokenizer_type"].get<std::string>() : std::string("Qwen2_5");
    attr.filename_tokens_embed     = resolve_path(model_dir, j["filename_tokens_embed"].get<std::string>());
    attr.post_config_path          = resolve_path(model_dir, j["post_config_path"].get<std::string>());
    attr.axmodel_num               = j["axmodel_num"].get<int>();
    attr.full_attention_interval   = 0;
    if (j.contains("full_attention_interval")) {
        attr.full_attention_interval = j["full_attention_interval"].get<int>();
    } else if (j.contains("text_config") && j["text_config"].contains("full_attention_interval")) {
        attr.full_attention_interval = j["text_config"]["full_attention_interval"].get<int>();
    }
    attr.tokens_embed_num          = j["tokens_embed_num"].get<int>();
    attr.tokens_embed_size         = j["tokens_embed_size"].get<int>();
    if (j.contains("pad_token_id")) attr.pad_token_id = j["pad_token_id"].get<int>();
    if (j.contains("hidden_size_per_layer_input")) attr.hidden_size_per_layer_input = j["hidden_size_per_layer_input"].get<int>();
    if (j.contains("rms_norm_eps")) attr.rms_norm_eps = j["rms_norm_eps"].get<float>();
    if (j.contains("filename_tokens_embed_per_layer")) {
        attr.filename_tokens_embed_per_layer = resolve_path(model_dir, j["filename_tokens_embed_per_layer"].get<std::string>());
    }
    if (j.contains("filename_per_layer_model_projection")) {
        attr.filename_per_layer_model_projection = resolve_path(model_dir, j["filename_per_layer_model_projection"].get<std::string>());
    }
    if (j.contains("filename_per_layer_projection_norm")) {
        attr.filename_per_layer_projection_norm = resolve_path(model_dir, j["filename_per_layer_projection_norm"].get<std::string>());
    }
    if (j.contains("b_use_mmap_load_embed")) attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();

    // Optional VLM config (match src/main.cpp behavior).
    if (j.contains("vlm_type") || j.contains("VLM_TYPE")) {
        const auto &v = j.contains("vlm_type") ? j["vlm_type"] : j["VLM_TYPE"];
        std::optional<VLMType> parsed;
        if (v.is_number_integer()) parsed = VLMTypeFromInt(v.get<int>());
        else if (v.is_string()) parsed = VLMTypeFromString(v.get<std::string>());
        if (parsed.has_value()) attr.vlm_type = *parsed;
    }
    if (j.contains("filename_image_encoder_axmodel")) {
        attr.filename_image_encoder_axmodel = resolve_path(model_dir, j["filename_image_encoder_axmodel"].get<std::string>());
    } else if (j.contains("filename_image_encoder_axmodedl")) {
        attr.filename_image_encoder_axmodel = resolve_path(model_dir, j["filename_image_encoder_axmodedl"].get<std::string>());
    }
    if (j.contains("vision_cache_dir")) attr.vision_cache_dir = resolve_path(model_dir, j["vision_cache_dir"].get<std::string>());
    if (j.contains("vision_width")) attr.vision_width = j["vision_width"].get<int>();
    if (j.contains("vision_height")) attr.vision_height = j["vision_height"].get<int>();
    if (j.contains("vision_temporal_patch_size")) attr.vision_temporal_patch_size = j["vision_temporal_patch_size"].get<int>();
    if (j.contains("vision_spatial_merge_size")) attr.vision_spatial_merge_size = j["vision_spatial_merge_size"].get<int>();
    if (j.contains("vision_patch_size")) attr.vision_patch_size = j["vision_patch_size"].get<int>();
    if (j.contains("vision_fps")) attr.vision_fps = j["vision_fps"].get<int>();
    if (j.contains("vision_tokens_per_second")) attr.vision_tokens_per_second = j["vision_tokens_per_second"].get<int>();

    // Auto-pick a smoke image if this is a VLM model and caller didn't specify one.
    if (attr.vlm_type != VLMType::None && image_path.empty() && video_dir.empty() && audio_path.empty())
    {
        std::string p = model_dir + "/smoke_image.png";
        if (std::filesystem::exists(p)) {
            image_path = p;
        } else {
            // Repo-local fallback asset (run from repo root).
            std::string p2 = "tools/smoke_assets/qwen3vl_smoke_image.png";
            if (std::filesystem::exists(p2)) image_path = p2;
        }
    }

    // ---- system init (参考 main) ----
#ifdef USE_AXCL
    {
        auto ret = axclInit(nullptr);
        if (0 != ret) { std::cerr << "axclInit failed: " << ret << "\n"; return ret; }
    }
#else
    AX_ENGINE_NPU_ATTR_T npu_attr; memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    {
        auto ret = AX_ENGINE_Init(&npu_attr);
        if (0 != ret) { std::cerr << "AX_ENGINE_Init failed: " << ret << "\n"; return ret; }
    }
#endif

    LLM llm;
    if (!llm.Init(attr)) {
        std::cerr << "LLM.Init failed\n";
#ifdef USE_AXCL
        axclFinalize();
#else
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
#endif
        return 3;
    }

    std::vector<Content> history;
    history.push_back({SYSTEM, TEXT, std::string("You are a helpful assistant.")});
    std::vector<MediaInputs> media_inputs;
    if (!video_dir.empty()) {
        history.push_back({USER, VIDEO, prompt});
        media_inputs.push_back({1, {video_dir}});
    } else if (!audio_path.empty()) {
        history.push_back({USER, AUDIO, prompt});
        media_inputs.push_back({1, {audio_path}});
    } else if (!image_path.empty()) {
        history.push_back({USER, IMAGE, prompt});
        media_inputs.push_back({1, {image_path}});
    } else {
        history.push_back({USER, TEXT, prompt});
    }

    if (quiet)
    {
        auto cb = [](std::string, float, void*){};
        llm.getAttr()->runing_callback = cb;
    }
    else
    {
        auto cb = [](std::string s, float, void*){ std::cout << s << std::flush; };
        llm.getAttr()->runing_callback = cb;
    }
    if (!media_inputs.empty()) llm.Run(history, media_inputs, max_tokens);
    else llm.Run(history, max_tokens);
    std::cout << "\n[SMOKE OK]\n";
    llm.Deinit();

    // ---- system deinit ----
#ifdef USE_AXCL
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif
    return 0;
}
