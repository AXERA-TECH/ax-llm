#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "runner/LLM.hpp"
#include "utils/json.hpp"

static std::string resolve_path(const std::string &base, const std::string &p) {
    if (p.empty()) return p;
    if (p.rfind("http://",0)==0 || p.rfind("https://",0)==0) return p;
    namespace fs = std::filesystem;
    if (fs::path(p).is_absolute()) return p;
    return base + "/" + p;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: llm_smoke <model_dir> [max_tokens]\n";
        return 1;
    }
    std::string model_dir = argv[1];
    int max_tokens = (argc>=3)? std::atoi(argv[2]) : 16;
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
    attr.tokens_embed_num          = j["tokens_embed_num"].get<int>();
    attr.tokens_embed_size         = j["tokens_embed_size"].get<int>();
    if (j.contains("b_use_mmap_load_embed")) attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();

    LLM llm;
    if (!llm.Init(attr)) { std::cerr << "LLM.Init failed\n"; return 3; }

    std::vector<Content> history;
    history.push_back({SYSTEM, TEXT, std::string("You are a helpful assistant.")});
    history.push_back({USER, TEXT, std::string("hello")});

    auto cb = [](std::string s, float tps, void*){ std::cout << s << std::flush; };
    llm.getAttr()->runing_callback = cb;
    llm.Run(history, max_tokens);
    std::cout << "\n[SMOKE OK]\n";
    llm.Deinit();
    return 0;
}
