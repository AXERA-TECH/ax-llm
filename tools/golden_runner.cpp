// golden_runner: deterministic multi-turn regression harness for behavior-preserving
// refactors (esp. the KV-cache / prefix-reuse subsystem).
//
// It drives a scripted multi-turn conversation on ONE persistent LLM instance,
// greedy-decoded (temperature 0) so output is reproducible, feeding the model's own
// reply back as history each turn -- which exercises the REAL prefix-KV reuse path
// that single-turn smoke tests never touch. Per turn it records the output text,
// prompt/completion token counts, and a fingerprint (hash) of the live KV cache.
//
//   Record a golden:   golden_runner <model_dir> --convo c.json --save gold.json
//   Check against it:  golden_runner <model_dir> --convo c.json --compare gold.json
//
// A behavior-preserving refactor MUST reproduce the golden byte-for-byte. The KV
// fingerprint catches state corruption that hasn't yet changed the output token
// (e.g. a slot write that bites several turns later). Exit 0 = match, 1 = mismatch.
//
// convo JSON:  { "system": "...", "max_tokens": 64, "turns": ["u1", "u2", ...] }
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "runner/LLM.hpp"
#include "utils/json.hpp"
#ifdef USE_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

using nlohmann::json;

static std::string resolve_path(const std::string &base, const std::string &p) {
    if (p.empty()) return p;
    if (p.rfind("http://", 0) == 0 || p.rfind("https://", 0) == 0) return p;
    namespace fs = std::filesystem;
    if (fs::path(p).is_absolute()) return p;
    return base + "/" + p;
}

// Load a model dir's config.json into LLMAttrType (mirrors llm_smoke / main.cpp).
static bool load_config(const std::string &model_dir, LLMAttrType &attr,
                        std::optional<std::vector<int>> devices_override) {
    std::string cfg = model_dir + "/config.json";
    if (!std::filesystem::exists(cfg)) { std::cerr << "config.json not found in " << model_dir << "\n"; return false; }
    std::ifstream f(cfg); json j; f >> j;

    attr.template_filename_axmodel = resolve_path(model_dir, j["template_filename_axmodel"].get<std::string>());
    attr.filename_post_axmodel     = resolve_path(model_dir, j["filename_post_axmodel"].get<std::string>());
    attr.url_tokenizer_model       = resolve_path(model_dir, j["url_tokenizer_model"].get<std::string>());
    attr.tokenizer_type            = j.contains("tokenizer_type") ? j["tokenizer_type"].get<std::string>() : std::string("Qwen2_5");
    attr.filename_tokens_embed     = resolve_path(model_dir, j["filename_tokens_embed"].get<std::string>());
    attr.post_config_path          = resolve_path(model_dir, j["post_config_path"].get<std::string>());
    attr.axmodel_num               = j["axmodel_num"].get<int>();
    if (j.contains("dynamic_load_enable")) attr.dynamic_load_enable = j["dynamic_load_enable"].get<bool>();
    if (j.contains("dynamic_load_pool_size")) attr.dynamic_load_pool_size = j["dynamic_load_pool_size"].get<int>();
    if (attr.dynamic_load_enable && attr.dynamic_load_pool_size <= 0) attr.dynamic_load_pool_size = 2;
    attr.full_attention_interval = 0;
    if (j.contains("full_attention_interval")) attr.full_attention_interval = j["full_attention_interval"].get<int>();
    else if (j.contains("text_config") && j["text_config"].contains("full_attention_interval"))
        attr.full_attention_interval = j["text_config"]["full_attention_interval"].get<int>();
    attr.tokens_embed_num  = j["tokens_embed_num"].get<int>();
    attr.tokens_embed_size = j["tokens_embed_size"].get<int>();
    if (j.contains("pad_token_id")) attr.pad_token_id = j["pad_token_id"].get<int>();
    if (j.contains("hidden_size_per_layer_input")) attr.hidden_size_per_layer_input = j["hidden_size_per_layer_input"].get<int>();
    if (j.contains("rms_norm_eps")) attr.rms_norm_eps = j["rms_norm_eps"].get<float>();
    if (j.contains("filename_tokens_embed_per_layer")) attr.filename_tokens_embed_per_layer = resolve_path(model_dir, j["filename_tokens_embed_per_layer"].get<std::string>());
    if (j.contains("filename_per_layer_model_projection")) attr.filename_per_layer_model_projection = resolve_path(model_dir, j["filename_per_layer_model_projection"].get<std::string>());
    if (j.contains("filename_per_layer_projection_norm")) attr.filename_per_layer_projection_norm = resolve_path(model_dir, j["filename_per_layer_projection_norm"].get<std::string>());
    if (j.contains("b_use_mmap_load_embed")) attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();
    if (j.contains("kv_cache_slots")) attr.kv_cache_slots = j["kv_cache_slots"].get<int>();
#ifdef USE_AXCL
    if (j.contains("devices")) attr.dev_ids = j["devices"].get<std::vector<int>>();
    if (devices_override.has_value()) attr.dev_ids = *devices_override;
#else
    (void)devices_override;
#endif
    return true;
}

// 64-bit FNV-1a over raw bytes (order-sensitive -> good state fingerprint).
static uint64_t fnv1a(const void *data, size_t n, uint64_t h = 1469598103934665603ULL) {
    const unsigned char *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

// Fingerprint the live KV cache (per-layer K/V tensors + precompute length).
static uint64_t kv_fingerprint(LLM &llm, int &pre_len_out) {
    std::vector<std::vector<unsigned short>> k, v;
    int pre_len = -1;
    llm.GetKVCache(k, v, pre_len);
    pre_len_out = pre_len;
    uint64_t h = fnv1a(&pre_len, sizeof(pre_len));
    for (const auto &kk : k) if (!kk.empty()) h = fnv1a(kk.data(), kk.size() * sizeof(unsigned short), h);
    for (const auto &vv : v) if (!vv.empty()) h = fnv1a(vv.data(), vv.size() * sizeof(unsigned short), h);
    return h;
}

static std::string hex64(uint64_t x) { char b[19]; std::snprintf(b, sizeof(b), "0x%016llx", (unsigned long long)x); return b; }

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: golden_runner <model_dir> --convo <c.json> (--save <out.json> | --compare <ref.json>)\n"
                     "                     [--max-tokens N] [--no-kv-hash] [--devices <csv>]\n";
        return 1;
    }
    std::string model_dir = argv[1];
    std::string convo_path, save_path, compare_path;
    int cli_max_tokens = -1;
    bool kv_hash = true;
    std::optional<std::vector<int>> devices_override;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--convo" && i + 1 < argc) convo_path = argv[++i];
        else if (a == "--save" && i + 1 < argc) save_path = argv[++i];
        else if (a == "--compare" && i + 1 < argc) compare_path = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) cli_max_tokens = std::atoi(argv[++i]);
        else if (a == "--no-kv-hash") kv_hash = false;
        else if (a == "--devices" && i + 1 < argc) {
            std::string s = argv[++i]; std::vector<int> devs; std::string cur;
            for (char c : s) { if (c == ',' || c == ';' || c == ' ') { if (!cur.empty()) { devs.push_back(std::atoi(cur.c_str())); cur.clear(); } } else cur.push_back(c); }
            if (!cur.empty()) devs.push_back(std::atoi(cur.c_str()));
            if (!devs.empty()) devices_override = std::move(devs);
        }
    }
    if (convo_path.empty() || (save_path.empty() == compare_path.empty())) {
        std::cerr << "error: need --convo and exactly one of --save / --compare\n";
        return 1;
    }
    const bool compare_mode = !compare_path.empty();

    // ---- load conversation script ----
    json convo;
    { std::ifstream cf(convo_path); if (!cf) { std::cerr << "cannot open convo: " << convo_path << "\n"; return 2; } cf >> convo; }
    std::string system_prompt = convo.value("system", std::string("You are a helpful assistant."));
    int max_tokens = cli_max_tokens > 0 ? cli_max_tokens : convo.value("max_tokens", 64);
    std::vector<std::string> turns;
    for (const auto &t : convo.at("turns")) turns.push_back(t.get<std::string>());
    if (turns.empty()) { std::cerr << "convo has no turns\n"; return 2; }

    // ---- config + system init ----
    LLMAttrType attr;
    if (!load_config(model_dir, attr, devices_override)) return 2;
#ifdef USE_AXCL
    if (int r = axclInit(nullptr)) { std::cerr << "axclInit failed: " << r << "\n"; return r; }
#else
    AX_ENGINE_NPU_ATTR_T npu_attr; memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    if (int r = AX_ENGINE_Init(&npu_attr)) { std::cerr << "AX_ENGINE_Init failed: " << r << "\n"; return r; }
#endif

    LLM llm;
    if (!llm.Init(attr)) { std::cerr << "LLM.Init failed\n"; return 3; }

    std::string turn_out;
    llm.getAttr()->runing_callback = [&turn_out](std::string s, float, void *) { turn_out += s; };

    // ---- drive the conversation (persistent KV, greedy, feed reply back) ----
    std::vector<Content> history;
    history.push_back({SYSTEM, TEXT, system_prompt});
    json out_turns = json::array();
    for (size_t i = 0; i < turns.size(); ++i) {
        history.push_back({USER, TEXT, turns[i]});
        turn_out.clear();
        llm.MarkRequestStart();
        llm.SetRequestSamplingOverride(true, 0.0f, false, 0.0f, false, 0.0f, false, 0.0f); // greedy
        history = llm.Run(history, max_tokens);
        llm.ClearRequestSamplingOverride();
        llm.ClearRequestStart();

        int pre_len = -1;
        uint64_t kvh = kv_hash ? kv_fingerprint(llm, pre_len) : 0;
        json rec = {
            {"i", (int)i},
            {"user", turns[i]},
            {"output", turn_out},
            {"prompt_tokens", llm.GetLastPromptTokenNum()},
            {"completion_tokens", llm.GetLastCompletionTokenNum()},
            {"kv_pre_len", pre_len},
            {"kv_hash", kv_hash ? hex64(kvh) : std::string("off")},
        };
        out_turns.push_back(rec);
        std::cout << "[turn " << i << "] comp_tokens=" << llm.GetLastCompletionTokenNum()
                  << " kv_pre_len=" << pre_len << " kv=" << (kv_hash ? hex64(kvh) : "off")
                  << "\n           out: " << turn_out.substr(0, 80) << (turn_out.size() > 80 ? "..." : "") << "\n";
    }

    llm.Deinit();
#ifdef USE_AXCL
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif

    // ---- save or compare ----
    if (!compare_mode) {
        json doc = {{"model_dir", model_dir}, {"convo", convo_path}, {"max_tokens", max_tokens}, {"turns", out_turns}};
        std::ofstream of(save_path); of << doc.dump(2) << "\n";
        std::cout << "\n[GOLDEN SAVED] " << save_path << " (" << out_turns.size() << " turns)\n";
        return 0;
    }

    json ref; { std::ifstream rf(compare_path); if (!rf) { std::cerr << "cannot open golden: " << compare_path << "\n"; return 2; } rf >> ref; }
    const auto &ref_turns = ref.at("turns");
    int fails = 0;
    if (ref_turns.size() != out_turns.size()) { std::cout << "FAIL: turn count " << out_turns.size() << " != golden " << ref_turns.size() << "\n"; ++fails; }
    size_t n = std::min(ref_turns.size(), out_turns.size());
    for (size_t i = 0; i < n; ++i) {
        const auto &r = ref_turns[i]; const auto &o = out_turns[i];
        if (r.value("output", std::string()) != o.value("output", std::string())) {
            std::cout << "FAIL turn " << i << ": OUTPUT differs\n  golden: " << r.value("output", std::string()).substr(0, 120)
                      << "\n  got   : " << o.value("output", std::string()).substr(0, 120) << "\n";
            ++fails;
        }
        if (kv_hash && r.contains("kv_hash") && r.value("kv_hash", std::string()) != std::string("off") &&
            r.value("kv_hash", std::string()) != o.value("kv_hash", std::string())) {
            std::cout << "FAIL turn " << i << ": KV fingerprint differs (golden " << r.value("kv_hash", std::string())
                      << " got " << o.value("kv_hash", std::string()) << ")\n";
            ++fails;
        }
        if (r.value("completion_tokens", -1) != o.value("completion_tokens", -2))
            std::cout << "WARN turn " << i << ": completion_tokens " << o.value("completion_tokens", -2)
                      << " != golden " << r.value("completion_tokens", -1) << "\n";
    }
    if (fails == 0) { std::cout << "\nPASS: " << n << " turns match golden (output" << (kv_hash ? " + KV fingerprint" : "") << ")\n"; return 0; }
    std::cout << "\nRESULT: FAIL (" << fails << " mismatch(es))\n";
    return 1;
}
