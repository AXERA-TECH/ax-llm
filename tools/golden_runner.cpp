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
#include <map>
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

// Read an optional string key; missing / null / non-string -> default. Avoids a
// json type_error on configs that omit e.g. post_config_path (issue #63 class).
static std::string jstr(const nlohmann::json &j, const char *k, const std::string &def = "") {
    auto it = j.find(k);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : def;
}

// Load a model dir's config.json into LLMAttrType (mirrors llm_smoke / main.cpp).
static bool load_config(const std::string &model_dir, LLMAttrType &attr,
                        std::optional<std::vector<int>> devices_override) {
    std::string cfg = model_dir + "/config.json";
    if (!std::filesystem::exists(cfg)) { std::cerr << "config.json not found in " << model_dir << "\n"; return false; }
    std::ifstream f(cfg); json j; f >> j;

    attr.template_filename_axmodel = resolve_path(model_dir, jstr(j, "template_filename_axmodel"));
    attr.filename_post_axmodel     = resolve_path(model_dir, jstr(j, "filename_post_axmodel"));
    attr.url_tokenizer_model       = resolve_path(model_dir, jstr(j, "url_tokenizer_model"));
    attr.tokenizer_type            = jstr(j, "tokenizer_type", "Qwen2_5");
    attr.filename_tokens_embed     = resolve_path(model_dir, jstr(j, "filename_tokens_embed"));
    attr.post_config_path          = resolve_path(model_dir, jstr(j, "post_config_path", "post_config.json"));
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
    // VLM config (mirror llm_smoke / main.cpp) -- REQUIRED or vlm_type stays None
    // and images are silently ignored (vision encoder never initialized).
    if (j.contains("vlm_type") || j.contains("VLM_TYPE")) {
        const auto &v = j.contains("vlm_type") ? j["vlm_type"] : j["VLM_TYPE"];
        std::optional<VLMType> parsed;
        if (v.is_number_integer()) parsed = VLMTypeFromInt(v.get<int>());
        else if (v.is_string()) parsed = VLMTypeFromString(v.get<std::string>());
        if (parsed.has_value()) attr.vlm_type = *parsed;
    }
    if (j.contains("filename_image_encoder_axmodel")) attr.filename_image_encoder_axmodel = resolve_path(model_dir, jstr(j, "filename_image_encoder_axmodel"));
    else if (j.contains("filename_image_encoder_axmodedl")) attr.filename_image_encoder_axmodel = resolve_path(model_dir, jstr(j, "filename_image_encoder_axmodedl"));
    if (j.contains("filename_audio_encoder_axmodel_5s")) attr.filename_audio_encoder_axmodel_5s = resolve_path(model_dir, jstr(j, "filename_audio_encoder_axmodel_5s"));
    if (j.contains("filename_audio_encoder_axmodel_30s")) attr.filename_audio_encoder_axmodel_30s = resolve_path(model_dir, jstr(j, "filename_audio_encoder_axmodel_30s"));
    if (j.contains("vision_cache_dir")) attr.vision_cache_dir = resolve_path(model_dir, jstr(j, "vision_cache_dir"));
    if (j.contains("vision_width")) attr.vision_width = j["vision_width"].get<int>();
    if (j.contains("vision_height")) attr.vision_height = j["vision_height"].get<int>();
    if (j.contains("vision_temporal_patch_size")) attr.vision_temporal_patch_size = j["vision_temporal_patch_size"].get<int>();
    if (j.contains("vision_spatial_merge_size")) attr.vision_spatial_merge_size = j["vision_spatial_merge_size"].get<int>();
    if (j.contains("vision_patch_size")) attr.vision_patch_size = j["vision_patch_size"].get<int>();
    if (j.contains("vision_fps")) attr.vision_fps = j["vision_fps"].get<int>();
    if (j.contains("vision_tokens_per_second")) attr.vision_tokens_per_second = j["vision_tokens_per_second"].get<int>();
    if (j.contains("vision_num_frames")) attr.vision_num_frames = j["vision_num_frames"].get<int>();
    if (j.contains("vision_do_sample_frames")) attr.vision_do_sample_frames = j["vision_do_sample_frames"].get<bool>();
#ifdef USE_AXCL
    if (j.contains("devices")) attr.dev_ids = j["devices"].get<std::vector<int>>();
    if (devices_override.has_value()) attr.dev_ids = *devices_override;
#else
    (void)devices_override;
#endif
    return true;
}

// Fingerprint the ACTIVE decode-group KV *content* via the engine hook (real K/V
// tensor bytes). GetKVCache only sets precompute_len, so we call it for pre_len
// then hash actual KV via LLM::HashActiveKV().
static uint64_t kv_fingerprint(LLM &llm, int &pre_len_out) {
    std::vector<std::vector<unsigned short>> k, v;
    int pre_len = -1;
    llm.GetKVCache(k, v, pre_len);
    pre_len_out = pre_len;
    return llm.HashActiveKV();
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
    int cli_kv_slots = -1;
    std::string image_path;
    ContentType media_type = IMAGE;
    bool kv_hash = true;
    std::optional<std::vector<int>> devices_override;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--convo" && i + 1 < argc) convo_path = argv[++i];
        else if (a == "--save" && i + 1 < argc) save_path = argv[++i];
        else if (a == "--compare" && i + 1 < argc) compare_path = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) cli_max_tokens = std::atoi(argv[++i]);
        else if (a == "--kv-slots" && i + 1 < argc) cli_kv_slots = std::atoi(argv[++i]);
        else if (a == "--image" && i + 1 < argc) { image_path = argv[++i]; media_type = IMAGE; }
        else if (a == "--video" && i + 1 < argc) { image_path = argv[++i]; media_type = VIDEO; }
        else if (a == "--audio" && i + 1 < argc) { image_path = argv[++i]; media_type = AUDIO; }
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
    if (!convo.contains("conversations")) {
        for (const auto &t : convo.at("turns")) turns.push_back(t.get<std::string>());
        if (turns.empty()) { std::cerr << "convo has no turns\n"; return 2; }
    }

    // ---- config + system init ----
    LLMAttrType attr;
    if (!load_config(model_dir, attr, devices_override)) return 2;
    if (cli_kv_slots > 0) attr.kv_cache_slots = cli_kv_slots;
    else if (convo.value("kv_cache_slots", 0) > 0) attr.kv_cache_slots = convo.value("kv_cache_slots", 0);
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

    // ---- drive: run one turn (persistent KV, greedy, feed reply back), record it ----
    json out_turns = json::array();
    int step = 0;
    auto run_step = [&](std::vector<Content> &hist, const std::string &user, json extra, const std::string &image) {
        if (image.empty()) hist.push_back({USER, TEXT, user});
        else hist.push_back({USER, media_type, user});
        turn_out.clear();
        llm.MarkRequestStart();
        llm.SetRequestSamplingOverride(true, 0.0f, false, 0.0f, false, 0.0f, false, 0.0f); // greedy
        if (image.empty()) {
            hist = llm.Run(hist, max_tokens);
        } else {
            std::vector<MediaInputs> media; media.push_back({hist.size() - 1, {image}});
            hist = llm.Run(hist, media, max_tokens);
        }
        llm.ClearRequestSamplingOverride();
        llm.ClearRequestStart();
        int pre_len = -1;
        uint64_t kvh = kv_hash ? kv_fingerprint(llm, pre_len) : 0;
        extra["step"] = step;
        extra["user"] = user;
        extra["output"] = turn_out;
        extra["prompt_tokens"] = llm.GetLastPromptTokenNum();
        extra["completion_tokens"] = llm.GetLastCompletionTokenNum();
        extra["kv_pre_len"] = pre_len;
        extra["kv_hash"] = kv_hash ? hex64(kvh) : std::string("off");
        extra["vis_hash"] = image.empty() ? std::string("off") : hex64(llm.HashLastVisionEmbed());
        out_turns.push_back(extra);
        std::cout << "[step " << step << "] "
                  << (out_turns.back().contains("conv") ? out_turns.back()["conv"].get<std::string>() : std::string("-"))
                  << " comp=" << llm.GetLastCompletionTokenNum() << " kv_pre=" << pre_len
                  << " kv=" << (kv_hash ? hex64(kvh) : "off")
                  << " vis=" << out_turns.back()["vis_hash"].get<std::string>()
                  << "  out: " << turn_out.substr(0, 64) << (turn_out.size() > 64 ? "..." : "") << "\n";
        ++step;
    };

    if (convo.contains("conversations")) {
        // interleaved multi-slot mode: named conversations + an `order` list; each
        // occurrence of a name advances that conversation's next turn. Exercises
        // select_kv_slot / activate / save / evict / LRU on one persistent instance.
        struct Conv { std::string system; std::vector<std::string> turns; std::vector<Content> hist; size_t next = 0; bool started = false; };
        std::map<std::string, Conv> convs;
        for (auto it = convo["conversations"].begin(); it != convo["conversations"].end(); ++it) {
            Conv c;
            c.system = it.value().value("system", system_prompt);
            for (const auto &t : it.value().at("turns")) c.turns.push_back(t.get<std::string>());
            convs[it.key()] = std::move(c);
        }
        for (const auto &o : convo.at("order")) {
            const std::string name = o.get<std::string>();
            auto ci = convs.find(name);
            if (ci == convs.end()) { std::cerr << "order references unknown conv: " << name << "\n"; continue; }
            Conv &c = ci->second;
            if (!c.started) { c.hist.push_back({SYSTEM, TEXT, c.system}); c.started = true; }
            if (c.next >= c.turns.size()) { std::cerr << "conv " << name << " out of turns, skipping\n"; continue; }
            run_step(c.hist, c.turns[c.next], json{{"conv", name}, {"turn", (int)c.next}}, std::string());
            c.next++;
        }
    } else {
        // linear single-conversation mode (original)
        std::vector<Content> history;
        history.push_back({SYSTEM, TEXT, system_prompt});
        for (size_t i = 0; i < turns.size(); ++i)
            run_step(history, turns[i], json{{"turn", (int)i}}, (i == 0 ? image_path : std::string()));
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
        if (r.contains("vis_hash") && r.value("vis_hash", std::string()) != std::string("off") &&
            r.value("vis_hash", std::string()) != o.value("vis_hash", std::string())) {
            std::cout << "FAIL turn " << i << ": VISION embed differs (golden " << r.value("vis_hash", std::string())
                      << " got " << o.value("vis_hash", std::string()) << ")\n";
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
