#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#ifdef USE_AXCL
#include <axcl.h>
#include "utils/axcl_manager.h"
#else
#include <ax_engine_api.h>
#include <ax_sys_api.h>
#endif

#include "robot/SmolVLARunner.hpp"
#include "utils/json.hpp"

static std::string dirname_of(const std::string& p)
{
    std::filesystem::path path(p);
    auto parent = path.parent_path();
    return parent.empty() ? "." : parent.string();
}

static std::string resolve_path(const std::string& base, const std::string& p)
{
    if (p.empty()) return p;
    std::filesystem::path path(p);
    if (path.is_absolute()) return path.lexically_normal().string();
    return (std::filesystem::path(base) / path).lexically_normal().string();
}

static bool read_f32_bin(const std::string& path, std::vector<float>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg(0, std::ios::end);
    const auto n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0 || ((std::uint64_t)n % sizeof(float)) != 0) return false;
    out.resize((size_t)n / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return true;
}

static void trace(const std::string& msg)
{
    const std::string line = "[smolvla_infer] " + msg + "\n";
    ::write(2, line.data(), line.size());
}

static void resolve_smolvla_config_paths(smolvla::Config& cfg, const std::string& base)
{
    cfg.image_encoder_axmodel = resolve_path(base, cfg.image_encoder_axmodel);
    cfg.state_proj_axmodel = resolve_path(base, cfg.state_proj_axmodel);
    cfg.action_embed_axmodel = resolve_path(base, cfg.action_embed_axmodel);
    cfg.action_out_axmodel = resolve_path(base, cfg.action_out_axmodel);
    cfg.image_encoder_onnx = resolve_path(base, cfg.image_encoder_onnx);
    cfg.state_proj_onnx = resolve_path(base, cfg.state_proj_onnx);
    cfg.action_embed_onnx = resolve_path(base, cfg.action_embed_onnx);
    cfg.action_out_onnx = resolve_path(base, cfg.action_out_onnx);
    cfg.llm_template_axmodel = resolve_path(base, cfg.llm_template_axmodel);
    cfg.llm_post_axmodel = resolve_path(base, cfg.llm_post_axmodel);
    cfg.tokens_embed = resolve_path(base, cfg.tokens_embed);
}

int main(int argc, char** argv)
{
    trace("main enter");
    if (argc < 3) {
        std::cerr << "Usage: smolvla_infer <smolvla_config.json> <input.json>\n";
        std::cerr << "input.json keys: image_bin, state, state_bin, language_tokens, language_mask, noise_bin\n";
        return 1;
    }

    const std::string config_path = argv[1];
    const std::string input_path = argv[2];
    trace("config_path=" + config_path);
    trace("input_path=" + input_path);

    smolvla::Config cfg;
    std::string err;
    if (!smolvla::LoadConfigJson(config_path, cfg, err)) {
        std::cerr << "load config failed: " << err << "\n";
        return 2;
    }
    const std::string config_dir = dirname_of(config_path);
    resolve_smolvla_config_paths(cfg, config_dir);
    trace("config loaded");

    std::ifstream fin(input_path);
    if (!fin.is_open()) {
        std::cerr << "open input json failed: " << input_path << "\n";
        return 2;
    }
    nlohmann::json j;
    fin >> j;
    const std::string input_dir = dirname_of(input_path);

    smolvla::Input in;
    if (j.contains("image_bin")) {
        if (!read_f32_bin(resolve_path(input_dir, j["image_bin"].get<std::string>()), in.images)) {
            std::cerr << "read image_bin failed\n";
            return 2;
        }
    }
    if (j.contains("state")) in.state = j["state"].get<std::vector<float>>();
    if (j.contains("state_bin")) {
        if (!read_f32_bin(resolve_path(input_dir, j["state_bin"].get<std::string>()), in.state)) {
            std::cerr << "read state_bin failed\n";
            return 2;
        }
    }
    if (j.contains("language_tokens")) in.language_tokens = j["language_tokens"].get<std::vector<int>>();
    if (j.contains("language_mask")) {
        auto mask_i = j["language_mask"].get<std::vector<int>>();
        in.language_mask.resize(mask_i.size());
        for (size_t i = 0; i < mask_i.size(); ++i) in.language_mask[i] = mask_i[i] ? 1 : 0;
    }
    if (j.contains("noise_bin")) {
        if (!read_f32_bin(resolve_path(input_dir, j["noise_bin"].get<std::string>()), in.noise)) {
            std::cerr << "read noise_bin failed\n";
            return 2;
        }
    }
    trace("input loaded images=" + std::to_string(in.images.size()) +
          " state=" + std::to_string(in.state.size()) +
          " lang=" + std::to_string(in.language_tokens.size()) +
          " mask=" + std::to_string(in.language_mask.size()) +
          " noise=" + std::to_string(in.noise.size()));

#ifdef USE_AXCL
    trace("axclInit begin");
    auto ret = axclInit(nullptr);
    if (ret != 0) {
        std::cerr << "axclInit failed: " << ret << "\n";
        return ret;
    }
    trace("axclInit ok");
    trace("axcl_Init begin");
    if (axcl_Init(0) != 0) {
        std::cerr << "axcl_Init(0) failed\n";
        axclFinalize();
        return 3;
    }
    trace("axcl_Init ok");
#else
    AX_ENGINE_NPU_ATTR_T npu_attr;
    std::memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    trace("AX_SYS_Init begin");
    AX_SYS_Init();
    trace("AX_SYS_Init done");
    trace("AX_ENGINE_Init begin");
    auto ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) {
        std::cerr << "AX_ENGINE_Init failed: " << ret << "\n";
        return ret;
    }
    trace("AX_ENGINE_Init ok");
#endif

    smolvla::Runner runner;
    trace("Runner.Init begin");
    if (!runner.Init(cfg,
#ifdef USE_AXCL
                     0
#else
                     -1
#endif
                     )) {
        std::cerr << "SmolVLA init failed: " << runner.LastError() << "\n";
#ifdef USE_AXCL
        axcl_Exit(0);
        axclFinalize();
#else
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
#endif
        return 3;
    }
    trace("Runner.Init ok");

    std::vector<float> actions;
    trace("Runner.Predict begin");
    if (!runner.Predict(in, actions)) {
        std::cerr << "SmolVLA predict failed: " << runner.LastError() << "\n";
        runner.Deinit();
#ifdef USE_AXCL
        axcl_Exit(0);
        axclFinalize();
#else
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
#endif
        return 4;
    }
    trace("Runner.Predict ok actions=" + std::to_string(actions.size()));

    std::cerr << "SmolVLA predict ok: actions=" << actions.size() << "\n";
    std::cout << "{";
    std::cout << "\"chunk_size\":" << cfg.chunk_size << ",";
    std::cout << "\"action_dim\":" << (cfg.output_action_dim > 0 ? cfg.output_action_dim : cfg.action_dim) << ",";
    std::cout << "\"actions\":[";
    for (size_t i = 0; i < actions.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << actions[i];
    }
    std::cout << "]}" << std::endl;
    std::cout.flush();
    std::fflush(stdout);

    runner.Deinit();
#ifdef USE_AXCL
    axcl_Exit(0);
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif
    return 0;
}
