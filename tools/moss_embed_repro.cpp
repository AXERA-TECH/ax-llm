// Reproducer for issue #72: MOSS-Transcribe-Diarize drives the decoder through
// the raw LLM::Run(embed) injection path, feeding one audio segment per call.
// This tool reconstructs that calling pattern WITHOUT the audio encoder: each
// "segment" is a run of legitimate token embeddings pulled from the embedding
// table (the decoder-side KV/prefill behavior does not care whether the
// vectors came from Whisper or from the embed table).
//
// Modes exercised in one process:
//   A) issue pattern  : N segments, consecutive Run(embed), NO ResetKVCache
//   B) per-seg reset  : same segments, ResetKVCache() before each call
//   C) determinism    : same segment M times, reset each time, outputs compared
//
// Usage: moss_embed_repro <model_dir> [--segments N] [--seg-tokens T] [--repeat M]
#include "runner/LLM.hpp"
#include "runner/LLMEmbedSelector.hpp"
#include "utils/json.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef USE_AXCL
#include <axcl.h>
#include "runner/utils/axcl_manager.h"
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

using nlohmann::json;

static std::string resolve_path(const std::string &base, const std::string &p)
{
    if (p.empty()) return p;
    if (p.rfind("http://", 0) == 0 || p.rfind("https://", 0) == 0) return p;
    if (std::filesystem::path(p).is_absolute()) return p;
    return base + "/" + p;
}

static std::string jstr(const json &j, const char *k, const std::string &def = "")
{
    auto it = j.find(k);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : def;
}

static bool load_config(const std::string &model_dir, LLMAttrType &attr)
{
    const std::string cfg = model_dir + "/config.json";
    std::ifstream f(cfg);
    if (!f) { std::cerr << "config.json not found in " << model_dir << "\n"; return false; }
    json j; f >> j;
    attr.template_filename_axmodel = resolve_path(model_dir, jstr(j, "template_filename_axmodel"));
    attr.filename_post_axmodel     = resolve_path(model_dir, jstr(j, "filename_post_axmodel"));
    attr.url_tokenizer_model       = resolve_path(model_dir, jstr(j, "url_tokenizer_model"));
    attr.tokenizer_type            = jstr(j, "tokenizer_type", "Qwen3");
    attr.filename_tokens_embed     = resolve_path(model_dir, jstr(j, "filename_tokens_embed"));
    attr.post_config_path          = resolve_path(model_dir, jstr(j, "post_config_path", "post_config.json"));
    attr.axmodel_num               = j["axmodel_num"].get<int>();
    attr.tokens_embed_num          = j["tokens_embed_num"].get<int>();
    attr.tokens_embed_size         = j["tokens_embed_size"].get<int>();
    if (j.contains("b_use_mmap_load_embed")) attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: moss_embed_repro <model_dir> [--segments N] [--seg-tokens T] [--repeat M] [--max-tokens K]\n";
        return 2;
    }
    std::string model_dir = argv[1];
    int segments = 13, seg_tokens = 600, repeat = 6, max_tokens = 32;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--segments" && i + 1 < argc) segments = std::atoi(argv[++i]);
        else if (a == "--seg-tokens" && i + 1 < argc) seg_tokens = std::atoi(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
    }

    LLMAttrType attr;
    if (!load_config(model_dir, attr)) return 2;

#ifdef USE_AXCL
    if (axclInit(nullptr) != 0) { std::cerr << "axclInit failed\n"; return 3; }
    attr.dev_ids = {0};
#else
    AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (int r = AX_ENGINE_Init(&npu_attr)) { std::cerr << "AX_ENGINE_Init failed\n"; return 3; }
#endif

    LLM llm;
    if (!llm.Init(attr)) { std::cerr << "LLM.Init failed\n"; return 3; }

    std::string streamed;
    llm.getAttr()->runing_callback = [](std::string s, float, void *ud) {
        (void)s; (void)ud; // silent streaming; we read the Run() return value
    };
    llm.getAttr()->runing_callback = nullptr;

    auto *sel = llm.getEmbedSelector();
    const int esz = attr.tokens_embed_size;

    // Build one deterministic "segment" of embeddings from real vocab rows.
    // Token ids stride over a benign ASCII-ish region of the vocab.
    auto build_segment = [&](int seed, int tokens) {
        std::vector<unsigned short> seg((size_t)tokens * (size_t)esz);
        std::vector<unsigned short> row((size_t)esz);
        for (int t = 0; t < tokens; ++t) {
            const unsigned int tok = (unsigned int)(1000 + ((seed * 131 + t * 7) % 20000));
            sel->getByIndex(tok, row);
            memcpy(seg.data() + (size_t)t * esz, row.data(), (size_t)esz * sizeof(unsigned short));
        }
        return seg;
    };

    llm.SetRequestSamplingOverride(true, 0.0f, false, 0.0f, false, 0.0f, false, 0.0f); // greedy

    auto pre_len = [&]() {
        std::vector<std::vector<unsigned short>> k, v;
        int pre = 0;
        llm.GetKVCache(k, v, pre);
        return pre;
    };

    printf("=== MODE A: %d consecutive Run(embed) WITHOUT reset (issue #72 pattern) ===\n", segments);
    llm.ResetKVCache();
    for (int s = 0; s < segments; ++s) {
        auto seg = build_segment(s, seg_tokens);
        std::string out = llm.Run(seg, max_tokens);
        printf("[A seg %02d] pre_after=%d out_len=%zu out='%.48s'%s\n",
               s, pre_len(), out.size(), out.c_str(), out.size() > 48 ? "..." : "");
        fflush(stdout);
    }

    printf("=== MODE B: same segments WITH ResetKVCache() per segment ===\n");
    for (int s = 0; s < segments; ++s) {
        llm.ResetKVCache();
        auto seg = build_segment(s, seg_tokens);
        std::string out = llm.Run(seg, max_tokens);
        printf("[B seg %02d] pre_after=%d out_len=%zu out='%.48s'%s\n",
               s, pre_len(), out.size(), out.c_str(), out.size() > 48 ? "..." : "");
        fflush(stdout);
    }

    printf("=== MODE C: identical segment x%d, reset each time (determinism) ===\n", repeat);
    std::string first;
    int mismatches = 0, empties = 0;
    for (int r = 0; r < repeat; ++r) {
        llm.ResetKVCache();
        auto seg = build_segment(7, seg_tokens);
        std::string out = llm.Run(seg, max_tokens);
        if (out.empty()) ++empties;
        if (r == 0) first = out;
        else if (out != first) ++mismatches;
        printf("[C run %02d] out_len=%zu %s\n", r, out.size(), out == first ? "SAME" : "<<<DIFF");
        fflush(stdout);
    }
    printf("MODE C: mismatches=%d empties=%d of %d\n", mismatches, empties, repeat);

    llm.ClearRequestSamplingOverride();
    llm.Deinit();
#ifdef USE_AXCL
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif
    printf("REPRO_DONE\n");
    return 0;
}
