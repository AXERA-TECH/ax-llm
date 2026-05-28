#include "robot/SmolVLARunner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>

#include "LLMEmbedSelector.hpp"
#include "utils/bfloat16.hpp"
#include "utils/json.hpp"
#include "utils/sample_log.h"

#ifdef USE_AXCL
#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "utils/axcl_manager.h"
using ax_runner_t = ax_runner_axcl;
#else
#include "ax_model_runner/ax_model_runner_ax650.hpp"
using ax_runner_t = ax_runner_ax650;
#endif

#ifdef USE_AXCL
static inline void v_h2d(void* phy_dst, const void* src, size_t n, int devid) { axcl_Memcpy(phy_dst, src, n, AXCL_MEMCPY_HOST_TO_DEVICE, devid); }
static inline void v_d2h(void* dst, const void* phy_src, size_t n, int devid) { axcl_Memcpy(dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_HOST, devid); }
#define V_WADDR(t) ((void*)(t).phyAddr)
#define V_RADDR(t) ((const void*)(t).phyAddr)
#else
static inline void v_h2d(void* vir_dst, const void* src, size_t n, int /*devid*/) { std::memcpy(vir_dst, src, n); }
static inline void v_d2h(void* dst, const void* vir_src, size_t n, int /*devid*/) { std::memcpy(dst, vir_src, n); }
#define V_WADDR(t) ((t).pVirAddr)
#define V_RADDR(t) ((const void*)(t).pVirAddr)
#endif

namespace smolvla {
namespace {

static constexpr float kMaskNeg = -65536.0f;

const ax_runner_tensor_t& input_by_name_or_index(ax_runner_t& r, const std::string& name, int idx)
{
    try {
        return r.get_input(name);
    } catch (...) {
        return r.get_input(idx);
    }
}

const ax_runner_tensor_t& output_by_name_or_index(ax_runner_t& r, const std::string& name, int idx)
{
    try {
        return r.get_output(name);
    } catch (...) {
        return r.get_output(idx);
    }
}

const ax_runner_tensor_t& group_input(ax_runner_t& r, int gid, const std::string& name)
{
    return r.get_input(gid, name);
}

const ax_runner_tensor_t& group_output(ax_runner_t& r, int gid, const std::string& name)
{
    return r.get_output(gid, name);
}

bool copy_float_input(ax_runner_t& r, const ax_runner_tensor_t& t, const std::vector<float>& data, std::string& err)
{
    const size_t fp32_bytes = data.size() * sizeof(float);
    const int devid = r.get_devid();
    if ((size_t)t.nSize == fp32_bytes) {
        v_h2d(V_WADDR(t), data.data(), fp32_bytes, devid);
        return true;
    }

    const size_t bf16_bytes = data.size() * sizeof(uint16_t);
    if ((size_t)t.nSize == bf16_bytes) {
        std::vector<uint16_t> bf16(data.size());
        for (size_t i = 0; i < data.size(); ++i) bf16[i] = fp32_to_bfloat16_rne(data[i]);
        v_h2d(V_WADDR(t), bf16.data(), bf16_bytes, devid);
        return true;
    }

    if ((size_t)t.nSize > fp32_bytes) {
        std::vector<uint8_t> tmp((size_t)t.nSize, 0);
        std::memcpy(tmp.data(), data.data(), fp32_bytes);
        v_h2d(V_WADDR(t), tmp.data(), tmp.size(), devid);
        return true;
    }

    err = "input tensor is smaller than supplied fp32 data: " + t.sName;
    return false;
}

bool copy_u32_input(ax_runner_t& r, const ax_runner_tensor_t& t, const std::vector<uint32_t>& data, std::string& err)
{
    const size_t bytes = data.size() * sizeof(uint32_t);
    if ((size_t)t.nSize < bytes) {
        err = "u32 input tensor is smaller than supplied data: " + t.sName;
        return false;
    }
    if ((size_t)t.nSize == bytes) {
        v_h2d(V_WADDR(t), data.data(), bytes, r.get_devid());
    } else {
        std::vector<uint8_t> tmp((size_t)t.nSize, 0);
        std::memcpy(tmp.data(), data.data(), bytes);
        v_h2d(V_WADDR(t), tmp.data(), tmp.size(), r.get_devid());
    }
    return true;
}

std::vector<uint16_t> read_tensor_bf16(ax_runner_t& r, const ax_runner_tensor_t& t, size_t expected_elems, std::string& err)
{
    std::vector<uint16_t> out(expected_elems);
    if ((size_t)t.nSize == expected_elems * sizeof(uint16_t)) {
        v_d2h(out.data(), V_RADDR(t), t.nSize, r.get_devid());
        return out;
    }
    if ((size_t)t.nSize == expected_elems * sizeof(float)) {
        std::vector<float> tmp(expected_elems);
        v_d2h(tmp.data(), V_RADDR(t), t.nSize, r.get_devid());
        for (size_t i = 0; i < expected_elems; ++i) out[i] = fp32_to_bfloat16_rne(tmp[i]);
        return out;
    }
    err = "output tensor size does not match expected bf16/fp32 elements: " + t.sName;
    out.clear();
    return out;
}

std::vector<float> read_tensor_fp32(ax_runner_t& r, const ax_runner_tensor_t& t, size_t expected_elems, std::string& err)
{
    std::vector<float> out(expected_elems);
    if ((size_t)t.nSize == expected_elems * sizeof(float)) {
        v_d2h(out.data(), V_RADDR(t), t.nSize, r.get_devid());
        return out;
    }
    if ((size_t)t.nSize == expected_elems * sizeof(uint16_t)) {
        std::vector<uint16_t> tmp(expected_elems);
        v_d2h(tmp.data(), V_RADDR(t), t.nSize, r.get_devid());
        for (size_t i = 0; i < expected_elems; ++i) out[i] = bfloat16(tmp[i]).fp32();
        return out;
    }
    err = "output tensor size does not match expected fp32/bf16 elements: " + t.sName;
    out.clear();
    return out;
}

void append_scaled_token(std::vector<uint16_t>& dst, const uint16_t* src, int hidden)
{
    const float scale = std::sqrt((float)hidden);
    for (int i = 0; i < hidden; ++i) {
        dst.push_back(fp32_to_bfloat16_rne(bfloat16(src[i]).fp32() * scale));
    }
}

std::vector<uint16_t> build_mask(const std::vector<uint8_t>& allow, int rows, int cols)
{
    std::vector<uint16_t> out((size_t)rows * (size_t)cols);
    const uint16_t zero = bfloat16(0.0f).data;
    const uint16_t neg = bfloat16(kMaskNeg).data;
    for (size_t i = 0; i < out.size(); ++i) out[i] = allow[i] ? zero : neg;
    return out;
}

std::vector<uint8_t> prefix_allow_mask(const std::vector<uint8_t>& pad, const std::vector<uint8_t>& att)
{
    const int n = (int)pad.size();
    std::vector<int> cumsum((size_t)n, 0);
    int cur = 0;
    for (int i = 0; i < n; ++i) {
        cur += att[(size_t)i] ? 1 : 0;
        cumsum[(size_t)i] = cur;
    }
    std::vector<uint8_t> allow((size_t)n * (size_t)n, 0);
    for (int q = 0; q < n; ++q) {
        for (int k = 0; k < n; ++k) {
            allow[(size_t)q * (size_t)n + (size_t)k] =
                (pad[(size_t)q] && pad[(size_t)k] && cumsum[(size_t)k] <= cumsum[(size_t)q]) ? 1 : 0;
        }
    }
    return allow;
}

std::vector<uint8_t> suffix_allow_mask(const std::vector<uint8_t>& prefix_pad, int cache_len, int chunk, int cols)
{
    std::vector<uint8_t> cache_pad((size_t)std::max(0, cache_len), 0);
    const int copy_prefix = std::min<int>((int)prefix_pad.size(), cache_len);
    for (int i = 0; i < copy_prefix; ++i) cache_pad[(size_t)i] = prefix_pad[(size_t)i];

    std::vector<uint8_t> allow((size_t)chunk * (size_t)cols, 0);
    for (int q = 0; q < chunk; ++q) {
        const int cache_cols = std::min(cache_len, cols);
        for (int k = 0; k < cache_cols; ++k) {
            allow[(size_t)q * (size_t)cols + (size_t)k] = cache_pad[(size_t)k] ? 1 : 0;
        }
        if (cols > cache_len) {
            const int suffix_cols = std::min(chunk, cols - cache_len);
            for (int k = 0; k < suffix_cols; ++k) {
                allow[(size_t)q * (size_t)cols + (size_t)cache_len + (size_t)k] = (k <= q) ? 1 : 0;
            }
        }
    }
    return allow;
}

std::vector<uint32_t> prefix_indices(const std::vector<uint8_t>& pad)
{
    std::vector<uint32_t> out(pad.size(), 0);
    int cur = -1;
    for (size_t i = 0; i < pad.size(); ++i) {
        if (pad[i]) ++cur;
        out[i] = cur < 0 ? 0u : (uint32_t)cur;
    }
    return out;
}

std::vector<uint32_t> suffix_indices(int prefix_valid, int chunk)
{
    std::vector<uint32_t> out((size_t)chunk);
    for (int i = 0; i < chunk; ++i) out[(size_t)i] = (uint32_t)(prefix_valid + i);
    return out;
}

} // namespace

struct Runner::Impl {
    Config cfg;
    LLaMaEmbedSelector embed_selector;
    ax_runner_t image_encoder;
    ax_runner_t state_proj;
    ax_runner_t action_embed;
    ax_runner_t action_out;
    ax_runner_t post;
    std::vector<std::unique_ptr<ax_runner_t>> layers;

    int prefix_gid = 1;
    int decode_gid = 0;

    bool init_models(int devid, std::string& err)
    {
        auto init_one = [&](ax_runner_t& r, const std::string& path, const char* name) -> bool {
            if (path.empty()) {
                err = std::string(name) + " path is empty";
                return false;
            }
            if (r.init(path.c_str(), devid) != 0) {
                err = std::string("init failed for ") + name + ": " + path;
                return false;
            }
#ifdef USE_AXCL
            r.set_auto_sync_before_inference(true);
            r.set_auto_sync_after_inference(true);
#endif
            return true;
        };

        if (!embed_selector.Init(cfg.tokens_embed, (unsigned int)cfg.tokens_embed_num, (unsigned int)cfg.vlm_hidden_size, cfg.use_mmap_embed)) {
            err = "token embedding init failed: " + cfg.tokens_embed;
            return false;
        }
        if (!init_one(image_encoder, cfg.image_encoder_axmodel, "image_encoder")) return false;
        if (!init_one(state_proj, cfg.state_proj_axmodel, "state_proj")) return false;
        if (!init_one(action_embed, cfg.action_embed_axmodel, "action_embed")) return false;
        if (!init_one(action_out, cfg.action_out_axmodel, "action_out")) return false;
        if (!init_one(post, cfg.llm_post_axmodel, "llm_post")) return false;

        layers.resize((size_t)cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; ++i) {
            char path[2048];
            std::snprintf(path, sizeof(path), cfg.llm_template_axmodel.c_str(), i);
            layers[(size_t)i].reset(new ax_runner_t());
            if (!init_one(*layers[(size_t)i], path, "llm_layer")) return false;
        }
        prefix_gid = layers.empty() ? 0 : (layers[0]->get_num_input_groups() > 1 ? 1 : 0);
        decode_gid = 0;
        return true;
    }

    void deinit()
    {
        for (auto& l : layers) {
            if (l) l->deinit();
        }
        layers.clear();
        post.deinit();
        action_out.deinit();
        action_embed.deinit();
        state_proj.deinit();
        image_encoder.deinit();
        embed_selector.Deinit();
    }
};

Runner::Runner() = default;
Runner::~Runner() { Deinit(); }

bool Runner::Init(const Config& cfg, int devid)
{
    Deinit();
    impl_ = new Impl();
    impl_->cfg = cfg;
    if (!impl_->init_models(devid, last_error_)) {
        Deinit();
        return false;
    }
    return true;
}

void Runner::Deinit()
{
    if (!impl_) return;
    impl_->deinit();
    delete impl_;
    impl_ = nullptr;
}

bool Runner::Predict(const Input& input, std::vector<float>& out_actions)
{
    if (!impl_) {
        last_error_ = "runner is not initialized";
        return false;
    }
    auto& I = *impl_;
    const Config& cfg = I.cfg;

    const size_t image_elems = (size_t)cfg.num_images * 3u * (size_t)cfg.image_height * (size_t)cfg.image_width;
    std::vector<float> images = input.images;
    images.resize(image_elems, 0.0f);

    std::vector<int> lang = input.language_tokens;
    lang.resize((size_t)cfg.language_tokens, 0);
    std::vector<uint8_t> lang_mask = input.language_mask;
    if (lang_mask.empty()) {
        lang_mask.assign((size_t)cfg.language_tokens, 0);
        for (size_t i = 0; i < input.language_tokens.size() && i < lang_mask.size(); ++i) lang_mask[i] = 1;
    } else {
        lang_mask.resize((size_t)cfg.language_tokens, 0);
    }

    std::vector<float> state = input.state;
    state.resize((size_t)cfg.state_dim, 0.0f);

    std::vector<uint16_t> prefix_embed;
    prefix_embed.reserve((size_t)cfg.prefix_len * (size_t)cfg.vlm_hidden_size);
    std::vector<uint8_t> prefix_pad;
    prefix_pad.reserve((size_t)cfg.prefix_len);
    std::vector<uint8_t> prefix_att_ar;
    prefix_att_ar.reserve((size_t)cfg.prefix_len);

    const size_t one_image_elems = (size_t)3 * (size_t)cfg.image_height * (size_t)cfg.image_width;
    for (int img_i = 0; img_i < cfg.num_images; ++img_i) {
        std::vector<float> image(images.begin() + (size_t)img_i * one_image_elems,
                                 images.begin() + (size_t)(img_i + 1) * one_image_elems);
        const auto& t_in = input_by_name_or_index(I.image_encoder, "image", 0);
        if (!copy_float_input(I.image_encoder, t_in, image, last_error_)) return false;
        I.image_encoder.inference();
        const size_t out_elems = (size_t)cfg.image_tokens * (size_t)cfg.vlm_hidden_size;
        const auto& t_out = output_by_name_or_index(I.image_encoder, "image_embeds", 0);
        auto image_embeds = read_tensor_bf16(I.image_encoder, t_out, out_elems, last_error_);
        if (image_embeds.empty()) return false;
        prefix_embed.insert(prefix_embed.end(), image_embeds.begin(), image_embeds.end());
        prefix_pad.insert(prefix_pad.end(), (size_t)cfg.image_tokens, 1);
        prefix_att_ar.insert(prefix_att_ar.end(), (size_t)cfg.image_tokens, 0);
    }

    std::vector<uint16_t> tok_embed((size_t)cfg.vlm_hidden_size);
    for (int i = 0; i < cfg.language_tokens; ++i) {
        if (lang_mask[(size_t)i]) {
            I.embed_selector.getByIndex((unsigned int)std::max(0, lang[(size_t)i]), tok_embed.data());
            append_scaled_token(prefix_embed, tok_embed.data(), cfg.vlm_hidden_size);
        } else {
            prefix_embed.insert(prefix_embed.end(), (size_t)cfg.vlm_hidden_size, 0);
        }
        prefix_pad.push_back(lang_mask[(size_t)i] ? 1 : 0);
        prefix_att_ar.push_back(0);
    }

    const auto& state_in = input_by_name_or_index(I.state_proj, "state", 0);
    if (!copy_float_input(I.state_proj, state_in, state, last_error_)) return false;
    I.state_proj.inference();
    const auto& state_out = output_by_name_or_index(I.state_proj, "state_embeds", 0);
    auto state_embed = read_tensor_bf16(I.state_proj, state_out, (size_t)cfg.vlm_hidden_size, last_error_);
    if (state_embed.empty()) return false;
    prefix_embed.insert(prefix_embed.end(), state_embed.begin(), state_embed.end());
    prefix_pad.push_back(1);
    prefix_att_ar.push_back(1);

    if ((int)prefix_pad.size() > cfg.prefix_len) {
        last_error_ = "constructed prefix is longer than compiled prefix_len";
        return false;
    }
    while ((int)prefix_pad.size() < cfg.prefix_len) {
        prefix_embed.insert(prefix_embed.end(), (size_t)cfg.vlm_hidden_size, 0);
        prefix_pad.push_back(0);
        prefix_att_ar.push_back(0);
    }

    const int prefix_valid = std::accumulate(prefix_pad.begin(), prefix_pad.end(), 0);
    auto p_allow = prefix_allow_mask(prefix_pad, prefix_att_ar);
    auto p_mask = build_mask(p_allow, cfg.prefix_len, cfg.prefix_len);
    auto p_indices = prefix_indices(prefix_pad);

    std::vector<std::vector<uint16_t>> k_cache((size_t)cfg.num_layers);
    std::vector<std::vector<uint16_t>> v_cache((size_t)cfg.num_layers);
    std::vector<uint16_t> layer_in = prefix_embed;
    for (int layer_i = 0; layer_i < cfg.num_layers; ++layer_i) {
        auto& lyr = *I.layers[(size_t)layer_i];
        const int gid = I.prefix_gid;
        const auto& t_input = group_input(lyr, gid, "input");
        const auto& t_mask = group_input(lyr, gid, "mask");
        const auto& t_indices = group_input(lyr, gid, "indices");
        v_h2d(V_WADDR(t_input), layer_in.data(), std::min((size_t)t_input.nSize, layer_in.size() * sizeof(uint16_t)), lyr.get_devid());
        v_h2d(V_WADDR(t_mask), p_mask.data(), std::min((size_t)t_mask.nSize, p_mask.size() * sizeof(uint16_t)), lyr.get_devid());
        if (!copy_u32_input(lyr, t_indices, p_indices, last_error_)) return false;
        lyr.inference(gid);

        const auto& out_k = group_output(lyr, gid, "K_cache_out");
        const auto& out_v = group_output(lyr, gid, "V_cache_out");
        k_cache[(size_t)layer_i] = read_tensor_bf16(lyr, out_k, (size_t)cfg.prefix_len * (size_t)cfg.kv_cache_size, last_error_);
        v_cache[(size_t)layer_i] = read_tensor_bf16(lyr, out_v, (size_t)cfg.prefix_len * (size_t)cfg.kv_cache_size, last_error_);
        if (k_cache[(size_t)layer_i].empty() || v_cache[(size_t)layer_i].empty()) return false;

        const auto& out = group_output(lyr, gid, "output");
        layer_in = read_tensor_bf16(lyr, out, (size_t)cfg.prefix_len * (size_t)cfg.vlm_hidden_size, last_error_);
        if (layer_in.empty()) return false;
    }

    std::vector<float> x_t = input.noise;
    x_t.resize((size_t)cfg.chunk_size * (size_t)cfg.action_dim);
    if (input.noise.empty()) {
        std::mt19937 gen((uint32_t)cfg.seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float& v : x_t) v = dist(gen);
    }

    const float dt = -1.0f / (float)std::max(1, cfg.num_steps);
    const auto s_indices = suffix_indices(prefix_valid, cfg.chunk_size);

    for (int step = 0; step < cfg.num_steps; ++step) {
        const float t = 1.0f + (float)step * dt;
        const auto& a_in = input_by_name_or_index(I.action_embed, "noisy_actions", 0);
        if (!copy_float_input(I.action_embed, a_in, x_t, last_error_)) return false;
        const auto& t_in = input_by_name_or_index(I.action_embed, "timestep", 1);
        std::vector<float> timestep{t};
        if (!copy_float_input(I.action_embed, t_in, timestep, last_error_)) return false;
        I.action_embed.inference();
        const auto& a_out = output_by_name_or_index(I.action_embed, "suffix_embeds", 0);
        std::vector<uint16_t> suffix_hidden = read_tensor_bf16(
            I.action_embed,
            a_out,
            (size_t)cfg.chunk_size * (size_t)cfg.expert_hidden_size,
            last_error_);
        if (suffix_hidden.empty()) return false;

        for (int layer_i = 0; layer_i < cfg.num_layers; ++layer_i) {
            auto& lyr = *I.layers[(size_t)layer_i];
            const int gid = I.decode_gid;
            const auto& t_layer_input = group_input(lyr, gid, "input");
            const auto& t_layer_mask = group_input(lyr, gid, "mask");
            const auto& t_layer_indices = group_input(lyr, gid, "indices");
            const auto& t_layer_k = group_input(lyr, gid, "K_cache");
            const auto& t_layer_v = group_input(lyr, gid, "V_cache");
            const size_t cache_elems = (size_t)t_layer_k.nSize / sizeof(uint16_t);
            if (cfg.kv_cache_size <= 0 || cache_elems % (size_t)cfg.kv_cache_size != 0) {
                last_error_ = "decode K_cache tensor size is not divisible by kv_cache_size";
                return false;
            }
            const int cache_len = (int)(cache_elems / (size_t)cfg.kv_cache_size);
            if (cache_len < cfg.prefix_len) {
                last_error_ = "decode K_cache length is shorter than prefix_len";
                return false;
            }
            if ((size_t)t_layer_v.nSize != cache_elems * sizeof(uint16_t)) {
                last_error_ = "decode K_cache/V_cache tensor sizes differ";
                return false;
            }
            const size_t mask_elems = (size_t)t_layer_mask.nSize / sizeof(uint16_t);
            if (mask_elems % (size_t)cfg.chunk_size != 0) {
                last_error_ = "decode mask tensor size is not divisible by chunk_size";
                return false;
            }
            const int mask_cols = (int)(mask_elems / (size_t)cfg.chunk_size);
            auto s_allow = suffix_allow_mask(prefix_pad, cache_len, cfg.chunk_size, mask_cols);
            auto s_mask = build_mask(s_allow, cfg.chunk_size, mask_cols);
            std::vector<uint16_t> k_in(cache_elems, 0);
            std::vector<uint16_t> v_in(cache_elems, 0);
            std::copy_n(k_cache[(size_t)layer_i].begin(),
                        std::min(k_cache[(size_t)layer_i].size(), k_in.size()),
                        k_in.begin());
            std::copy_n(v_cache[(size_t)layer_i].begin(),
                        std::min(v_cache[(size_t)layer_i].size(), v_in.size()),
                        v_in.begin());
            v_h2d(V_WADDR(t_layer_input), suffix_hidden.data(), std::min((size_t)t_layer_input.nSize, suffix_hidden.size() * sizeof(uint16_t)), lyr.get_devid());
            v_h2d(V_WADDR(t_layer_mask), s_mask.data(), std::min((size_t)t_layer_mask.nSize, s_mask.size() * sizeof(uint16_t)), lyr.get_devid());
            v_h2d(V_WADDR(t_layer_k), k_in.data(), std::min((size_t)t_layer_k.nSize, k_in.size() * sizeof(uint16_t)), lyr.get_devid());
            v_h2d(V_WADDR(t_layer_v), v_in.data(), std::min((size_t)t_layer_v.nSize, v_in.size() * sizeof(uint16_t)), lyr.get_devid());
            if (!copy_u32_input(lyr, t_layer_indices, s_indices, last_error_)) return false;
            lyr.inference(gid);
            const auto& t_layer_out = group_output(lyr, gid, "output");
            suffix_hidden = read_tensor_bf16(
                lyr,
                t_layer_out,
                (size_t)cfg.chunk_size * (size_t)cfg.expert_hidden_size,
                last_error_);
            if (suffix_hidden.empty()) return false;
        }

        const auto& post_in = input_by_name_or_index(I.post, "input", 0);
        v_h2d(V_WADDR(post_in), suffix_hidden.data(), std::min((size_t)post_in.nSize, suffix_hidden.size() * sizeof(uint16_t)), I.post.get_devid());
        I.post.inference();
        const auto& post_out = output_by_name_or_index(I.post, "output", 0);
        suffix_hidden = read_tensor_bf16(I.post, post_out, (size_t)cfg.chunk_size * (size_t)cfg.expert_hidden_size, last_error_);
        if (suffix_hidden.empty()) return false;

        std::vector<float> suffix_hidden_fp32((size_t)cfg.chunk_size * (size_t)cfg.expert_hidden_size);
        for (size_t i = 0; i < suffix_hidden.size(); ++i) suffix_hidden_fp32[i] = bfloat16(suffix_hidden[i]).fp32();

        const auto& out_in = input_by_name_or_index(I.action_out, "suffix_hidden", 0);
        if (!copy_float_input(I.action_out, out_in, suffix_hidden_fp32, last_error_)) return false;
        I.action_out.inference();
        const auto& out_v = output_by_name_or_index(I.action_out, "velocity", 0);
        auto v_t = read_tensor_fp32(I.action_out, out_v, x_t.size(), last_error_);
        if (v_t.empty()) return false;
        for (size_t i = 0; i < x_t.size(); ++i) x_t[i] += dt * v_t[i];
    }

    out_actions = std::move(x_t);
    const int norm_dim = std::min<int>(
        cfg.action_dim,
        std::min(cfg.action_mean.size(), cfg.action_std.size()));
    if (norm_dim > 0) {
        for (int t = 0; t < cfg.chunk_size; ++t) {
            for (int d = 0; d < norm_dim; ++d) {
                const size_t idx = (size_t)t * (size_t)cfg.action_dim + (size_t)d;
                out_actions[idx] = out_actions[idx] * cfg.action_std[(size_t)d] + cfg.action_mean[(size_t)d];
            }
        }
    }
    if (cfg.output_action_dim > 0 && cfg.output_action_dim < cfg.action_dim) {
        std::vector<float> truncated((size_t)cfg.chunk_size * (size_t)cfg.output_action_dim);
        for (int t = 0; t < cfg.chunk_size; ++t) {
            std::copy_n(out_actions.begin() + (size_t)t * (size_t)cfg.action_dim,
                        cfg.output_action_dim,
                        truncated.begin() + (size_t)t * (size_t)cfg.output_action_dim);
        }
        out_actions = std::move(truncated);
    }
    return true;
}

bool LoadConfigJson(const std::string& path, Config& cfg, std::string& err)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        err = "failed to open config: " + path;
        return false;
    }
    nlohmann::json j;
    try {
        f >> j;
        auto get_s = [&](const char* k, std::string& v) { if (j.contains(k)) v = j[k].get<std::string>(); };
        auto get_i = [&](const char* k, int& v) { if (j.contains(k)) v = j[k].get<int>(); };
        get_s("image_encoder_axmodel", cfg.image_encoder_axmodel);
        get_s("state_proj_axmodel", cfg.state_proj_axmodel);
        get_s("action_embed_axmodel", cfg.action_embed_axmodel);
        get_s("action_out_axmodel", cfg.action_out_axmodel);
        get_s("llm_template_axmodel", cfg.llm_template_axmodel);
        get_s("llm_post_axmodel", cfg.llm_post_axmodel);
        get_s("tokens_embed", cfg.tokens_embed);
        get_i("tokens_embed_num", cfg.tokens_embed_num);
        get_i("vlm_hidden_size", cfg.vlm_hidden_size);
        get_i("expert_hidden_size", cfg.expert_hidden_size);
        get_i("kv_cache_size", cfg.kv_cache_size);
        get_i("num_layers", cfg.num_layers);
        get_i("num_images", cfg.num_images);
        get_i("image_width", cfg.image_width);
        get_i("image_height", cfg.image_height);
        get_i("image_tokens", cfg.image_tokens);
        get_i("language_tokens", cfg.language_tokens);
        get_i("prefix_len", cfg.prefix_len);
        get_i("chunk_size", cfg.chunk_size);
        get_i("state_dim", cfg.state_dim);
        get_i("action_dim", cfg.action_dim);
        get_i("output_action_dim", cfg.output_action_dim);
        get_i("num_steps", cfg.num_steps);
        get_i("seed", cfg.seed);
        if (j.contains("use_mmap_embed")) cfg.use_mmap_embed = j["use_mmap_embed"].get<bool>();
        if (j.contains("action_mean")) cfg.action_mean = j["action_mean"].get<std::vector<float>>();
        if (j.contains("action_std")) cfg.action_std = j["action_std"].get<std::vector<float>>();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

} // namespace smolvla
