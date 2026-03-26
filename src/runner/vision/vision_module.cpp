#include "vision_module.hpp"

#include <cctype>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>

#include "bfloat16.hpp"
#include "sample_log.h"
#include "utils/files.hpp"
#include "utils/ax_cv.hpp"
#include "utils/image_processor.hpp"
#include "utils/mrope.hpp"

#ifdef USE_AXCL
#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "utils/axcl_manager.h"
using ax_runner_t = ax_runner_axcl;
static inline void v_h2d(void *phy_dst, const void *src, size_t n, int devid) { axcl_Memcpy(phy_dst, src, n, AXCL_MEMCPY_HOST_TO_DEVICE, devid); }
static inline void v_d2h(void *dst, const void *phy_src, size_t n, int devid) { axcl_Memcpy(dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_HOST, devid); }
#define V_WADDR(t) ((void *)(t).phyAddr)
#define V_RADDR(t) ((const void *)(t).phyAddr)
#else
#include "ax_model_runner/ax_model_runner_ax650.hpp"
using ax_runner_t = ax_runner_ax650;
static inline void v_h2d(void *vir_dst, const void *src, size_t n, int /*devid*/) { memcpy(vir_dst, src, n); }
static inline void v_d2h(void *dst, const void *vir_src, size_t n, int /*devid*/) { memcpy(dst, vir_src, n); }
#define V_WADDR(t) ((t).pVirAddr)
#define V_RADDR(t) ((const void *)(t).pVirAddr)
#endif

namespace vision {

struct VisionModule::Impl {
    ax_runner_t encoder;
    bool encoder_inited = false;
    int encoder_output_is_bf16 = -1;

    // For "classic image encoder" layout detection.
    int input_is_nchw = -1; // 1=NCHW float32, 0=NHWC u8, -1=unknown
};

static bool env_flag_false(const char* v)
{
    if (!v) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return (s == "0" || s == "false" || s == "off" || s == "no");
}

static bool try_infer_qwen_hw_from_input_bytes(size_t input_nbytes,
                                               int temporal_patch_size,
                                               int channels,
                                               int patch_size,
                                               int& io_h,
                                               int& io_w,
                                               std::string& note)
{
    if (temporal_patch_size <= 0 || channels <= 0) return false;
    const size_t denom = (size_t)temporal_patch_size * (size_t)channels;
    if (denom == 0 || (input_nbytes % denom) != 0) return false;

    const size_t hw = input_nbytes / denom; // H*W
    if (hw == 0) return false;

    // Prefer square if possible.
    const int side = (int)(std::sqrt((double)hw) + 0.5);
    if ((size_t)side * (size_t)side == hw) {
        if (patch_size > 0 && (side % patch_size) != 0) {
            note = "perfect square but not divisible by patch_size";
            return false;
        }
        io_h = side;
        io_w = side;
        note = "square";
        return true;
    }

    // Try keep configured height and solve width.
    if (io_h > 0 && (hw % (size_t)io_h) == 0) {
        int w = (int)(hw / (size_t)io_h);
        if (patch_size <= 0 || ((io_h % patch_size) == 0 && (w % patch_size) == 0)) {
            io_w = w;
            note = "matched config height";
            return true;
        }
    }

    // Fallback: search a reasonable factor pair close to sqrt, honoring patch_size divisibility.
    size_t best_diff = (size_t)-1;
    int best_h = -1, best_w = -1;
    for (size_t h = 1; h * h <= hw; ++h) {
        if (hw % h) continue;
        size_t w = hw / h;
        if (patch_size > 0) {
            if (((int)h % patch_size) != 0) continue;
            if (((int)w % patch_size) != 0) continue;
        }
        size_t diff = (w > h) ? (w - h) : (h - w);
        if (diff < best_diff) {
            best_diff = diff;
            best_h = (int)h;
            best_w = (int)w;
        }
    }
    if (best_h > 0 && best_w > 0) {
        io_h = best_h;
        io_w = best_w;
        note = "factor-search";
        return true;
    }

    return false;
}

template <typename ShapeVec>
static bool try_infer_hw_from_4d_shape_with_c3(const ShapeVec& shape, int& out_h, int& out_w, int* out_is_nchw = nullptr)
{
    if (shape.size() != 4) return false;
    // NCHW
    if ((int)shape[1] == 3) {
        out_h = (int)shape[2];
        out_w = (int)shape[3];
        if (out_is_nchw) *out_is_nchw = 1;
        return (out_h > 0 && out_w > 0);
    }
    // NHWC
    if ((int)shape[3] == 3) {
        out_h = (int)shape[1];
        out_w = (int)shape[2];
        if (out_is_nchw) *out_is_nchw = 0;
        return (out_h > 0 && out_w > 0);
    }
    return false;
}

static bool get_single_token_id(const std::shared_ptr<BaseTokenizer>& tok, const std::string& s, int& out_id, std::string& err)
{
    auto ids = tok->encode(s);
    if (ids.size() != 1) {
        err = "special token is not a single id: '" + s + "' size=" + std::to_string(ids.size());
        return false;
    }
    out_id = ids[0];
    return true;
}

static bool file_sig(const std::string& path, uint64_t& size_out, uint64_t& mtime_ns_out)
{
    std::error_code ec;
    const auto file_size = std::filesystem::file_size(path, ec);
    if (ec) return false;

    const auto write_time = std::filesystem::last_write_time(path, ec);
    if (ec) return false;

    size_out = static_cast<uint64_t>(file_size);
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(write_time.time_since_epoch()).count();
    mtime_ns_out = static_cast<uint64_t>(ns);
    return true;
}

static std::string normalize_path_for_key(const std::string& path)
{
    try {
        return std::filesystem::absolute(path).string();
    } catch (...) {
        return path;
    }
}

static std::string to_hex_u64(uint64_t v)
{
    static const char* kHex = "0123456789abcdef";
    char buf[16];
    for (int i = 15; i >= 0; --i) { buf[i] = kHex[v & 0xF]; v >>= 4; }
    return std::string(buf, 16);
}

static uint64_t fnv1a64(const void* data, size_t n, uint64_t seed = 14695981039346656037ull)
{
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = seed;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t fnv1a64_str(const std::string& s, uint64_t seed = 14695981039346656037ull)
{
    return fnv1a64(s.data(), s.size(), seed);
}

static std::string make_image_cache_key(const std::string& prefix, const std::string& path)
{
    uint64_t sz = 0, mt = 0;
    if (!file_sig(path, sz, mt)) return prefix + "|missing:" + normalize_path_for_key(path);
    return prefix + "|file:" + normalize_path_for_key(path) + "|sz:" + std::to_string(sz) + "|mt_ns:" + std::to_string(mt);
}

struct DiskHeaderV1 {
    uint32_t magic = 0x41585631; // "AXV1"
    uint32_t version = 1;
    uint32_t tokens_embed_size = 0;
    uint32_t tokens_per_block = 0;
    uint32_t num_blocks = 0;
    uint32_t key_len = 0;
};

struct DiskHeaderV2 {
    uint32_t magic = 0x41585632; // "AXV2"
    uint32_t version = 2;
    uint32_t tokens_embed_size = 0;
    uint32_t tokens_per_block = 0;
    uint32_t num_blocks = 0;
    uint32_t key_len = 0;
    uint32_t deepstack_layers = 0;
    uint32_t deepstack_elem_count = 0; // per layer, float count
};

static bool disk_cache_load(const std::string& cache_dir,
                            const std::string& key,
                            int tokens_embed_size,
                            int tokens_per_block,
                            int deepstack_layers,
                            std::vector<std::vector<unsigned short>>& blocks_out,
                            std::vector<std::vector<float>>& deepstack_out)
{
    if (cache_dir.empty()) return false;
    std::filesystem::path dir(cache_dir);
    std::filesystem::path file = dir / (to_hex_u64(fnv1a64_str(key)) + ".bin");
    std::FILE* fp = std::fopen(file.string().c_str(), "rb");
    if (!fp) return false;

    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(uint32_t), 1, fp) != 1) { std::fclose(fp); return false; }
    std::fseek(fp, 0, SEEK_SET);

    bool is_v2 = (magic == 0x41585632);

    uint32_t num_blocks = 0;
    uint32_t key_len = 0;
    uint32_t ds_layers = 0;
    uint32_t ds_elem_count = 0;

    if (is_v2) {
        DiskHeaderV2 hdr{};
        if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1) { std::fclose(fp); return false; }
        if (hdr.magic != 0x41585632 || hdr.version != 2) { std::fclose(fp); return false; }
        if ((int)hdr.tokens_embed_size != tokens_embed_size) { std::fclose(fp); return false; }
        if ((int)hdr.tokens_per_block != tokens_per_block) { std::fclose(fp); return false; }
        if ((int)hdr.deepstack_layers != deepstack_layers) { std::fclose(fp); return false; }
        num_blocks = hdr.num_blocks;
        key_len = hdr.key_len;
        ds_layers = hdr.deepstack_layers;
        ds_elem_count = hdr.deepstack_elem_count;
    } else {
        DiskHeaderV1 hdr{};
        if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1) { std::fclose(fp); return false; }
        if (hdr.magic != 0x41585631 || hdr.version != 1) { std::fclose(fp); return false; }
        if (deepstack_layers != 0) { std::fclose(fp); return false; } // old cache has no deepstack
        if ((int)hdr.tokens_embed_size != tokens_embed_size) { std::fclose(fp); return false; }
        if ((int)hdr.tokens_per_block != tokens_per_block) { std::fclose(fp); return false; }
        num_blocks = hdr.num_blocks;
        key_len = hdr.key_len;
        ds_layers = 0;
        ds_elem_count = 0;
    }

    std::string key_read;
    key_read.resize(key_len);
    if (key_len > 0) {
        if (std::fread(key_read.data(), 1, key_len, fp) != key_len) { std::fclose(fp); return false; }
        if (key_read != key) { std::fclose(fp); return false; }
    }

    std::vector<uint32_t> elem_counts(num_blocks);
    if (num_blocks > 0) {
        if (std::fread(elem_counts.data(), sizeof(uint32_t), num_blocks, fp) != num_blocks) { std::fclose(fp); return false; }
    }

    blocks_out.clear();
    blocks_out.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t n = elem_counts[i];
        std::vector<unsigned short> b;
        b.resize(n);
        if (n > 0) {
            if (std::fread(b.data(), sizeof(unsigned short), n, fp) != n) { std::fclose(fp); return false; }
        }
        blocks_out.push_back(std::move(b));
    }

    deepstack_out.clear();
    if (ds_layers > 0) {
        if (ds_elem_count == 0) { std::fclose(fp); return false; }
        deepstack_out.resize(ds_layers);
        for (uint32_t li = 0; li < ds_layers; ++li) {
            auto& v = deepstack_out[li];
            v.resize(ds_elem_count);
            if (std::fread(v.data(), sizeof(float), ds_elem_count, fp) != ds_elem_count) { std::fclose(fp); return false; }
        }
    }

    std::fclose(fp);
    return true;
}

static void disk_cache_save(const std::string& cache_dir,
                            const std::string& key,
                            int tokens_embed_size,
                            int tokens_per_block,
                            const std::vector<std::vector<unsigned short>>& blocks,
                            const std::vector<std::vector<float>>& deepstack)
{
    if (cache_dir.empty()) return;
    try { std::filesystem::create_directories(cache_dir); } catch (...) { return; }

    std::filesystem::path dir(cache_dir);
    std::filesystem::path file = dir / (to_hex_u64(fnv1a64_str(key)) + ".bin");

    std::FILE* fp = std::fopen(file.string().c_str(), "wb");
    if (!fp) return;

    DiskHeaderV2 hdr{};
    hdr.tokens_embed_size = (uint32_t)tokens_embed_size;
    hdr.tokens_per_block = (uint32_t)tokens_per_block;
    hdr.num_blocks = (uint32_t)blocks.size();
    hdr.key_len = (uint32_t)key.size();
    hdr.deepstack_layers = (uint32_t)deepstack.size();
    uint32_t ds_elem_count = 0;
    if (!deepstack.empty()) ds_elem_count = (uint32_t)deepstack[0].size();
    hdr.deepstack_elem_count = ds_elem_count;
    (void)std::fwrite(&hdr, sizeof(hdr), 1, fp);
    if (!key.empty()) (void)std::fwrite(key.data(), 1, key.size(), fp);

    std::vector<uint32_t> elem_counts;
    elem_counts.reserve(blocks.size());
    for (const auto& b : blocks) elem_counts.push_back((uint32_t)b.size());
    if (!elem_counts.empty()) (void)std::fwrite(elem_counts.data(), sizeof(uint32_t), elem_counts.size(), fp);

    for (const auto& b : blocks) {
        if (!b.empty()) (void)std::fwrite(b.data(), sizeof(unsigned short), b.size(), fp);
    }

    if (!deepstack.empty()) {
        // Require consistent sizes.
        for (const auto& v : deepstack) {
            if (v.size() != ds_elem_count) { std::fclose(fp); return; }
        }
        for (const auto& v : deepstack) {
            (void)std::fwrite(v.data(), sizeof(float), v.size(), fp);
        }
    }

    std::fclose(fp);
}

VisionModule::VisionModule() = default;
VisionModule::~VisionModule() { Deinit(); }
VisionModule::VisionModule(VisionModule&&) noexcept = default;
VisionModule& VisionModule::operator=(VisionModule&&) noexcept = default;

void VisionModule::Deinit()
{
    if (impl_) {
        if (impl_->encoder_inited) impl_->encoder.deinit();
        impl_.reset();
    }
    enabled_ = false;
    cache_enabled_ = true;
    type_ = VLMType::None;
    tokens_per_block_ = 0;
    deepstack_layers_ = 0;
    image_cache_.clear();
}

bool VisionModule::Init(VLMType type,
                        const std::string& encoder_axmodel,
                        const std::string& cache_dir,
                        int tokens_embed_size,
                        int devid,
                        const std::shared_ptr<BaseTokenizer>& tokenizer,
                        int vision_width,
                        int vision_height,
                        int temporal_patch_size,
                        int spatial_merge_size,
                        int patch_size,
                        int fps,
                        int tokens_per_second,
                        std::string& err)
{
    Deinit();

    tokens_embed_size_ = tokens_embed_size;
    tokenizer_ = tokenizer;
    cache_dir_ = cache_dir;
    type_ = type;

    // Debug switch: disable both disk+memory vision cache.
    cache_enabled_ = !env_flag_false(std::getenv("AXLLM_VISION_CACHE"));
    if (!cache_enabled_) {
        ALOGW("Vision cache disabled by env AXLLM_VISION_CACHE=0 (disk+mem cache bypassed).");
        cache_dir_.clear();
    }

    if (type_ == VLMType::None) {
        enabled_ = false;
        return true;
    }

    impl_.reset(new Impl());

    vision_width_ = vision_width;
    vision_height_ = vision_height;
    temporal_patch_size_ = temporal_patch_size;
    spatial_merge_size_ = spatial_merge_size;
    patch_size_ = patch_size;
    fps_ = fps;
    tokens_per_second_ = tokens_per_second;

    // Load encoder axmodel (all supported VLM types in this repo need it).
    if (encoder_axmodel.empty()) {
        err = "filename_image_encoder_axmodel is empty";
        return false;
    }
    if (impl_->encoder.init(encoder_axmodel.c_str(), devid) != 0) {
        err = "init vision encoder axmodel failed: " + encoder_axmodel;
        return false;
    }
    impl_->encoder_inited = true;

#ifdef USE_AXCL
    impl_->encoder.set_auto_sync_before_inference(true);
    impl_->encoder.set_auto_sync_after_inference(true);
#endif

    const auto& in0 = impl_->encoder.get_input(0);

    // Auto-resolve vision width/height from encoder input shape/size, so users don't need to manually set them.
    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) {
        const int old_w = vision_width_;
        const int old_h = vision_height_;

        // PaddleOCRVL VIT takes float32 input (not uint8 like Qwen-VL);
        // divide nSize by sizeof(float) to get the effective element count.
        size_t eff_nSize = (size_t)in0.nSize;
        if (type_ == VLMType::PaddleOCRVL && (eff_nSize % sizeof(float)) == 0) {
            eff_nSize /= sizeof(float);
            ALOGI("PaddleOCRVL: encoder input nSize=%zu -> eff_nSize=%zu (float32 input)",
                  (size_t)in0.nSize, eff_nSize);
        }

        const size_t cfg_bytes = (size_t)std::max(0, old_h) * (size_t)std::max(0, old_w) *
                                 (size_t)std::max(1, temporal_patch_size_) * (size_t)3;

        int h = old_h;
        int w = old_w;
        std::string note;
        if (try_infer_qwen_hw_from_input_bytes(eff_nSize,
                                               std::max(1, temporal_patch_size_),
                                               3,
                                               std::max(1, patch_size_),
                                               h, w, note)) {
            if (w != old_w || h != old_h) {
                ALOGW("Qwen-VL vision size override: cfg=%dx%d bytes=%zu, model_input_bytes=%zu -> %dx%d (%s).",
                      old_w, old_h, cfg_bytes, (size_t)in0.nSize, w, h, note.c_str());
            }
            vision_width_ = w;
            vision_height_ = h;
        } else {
            if (vision_width_ <= 0 || vision_height_ <= 0) {
                err = "failed to infer Qwen-VL vision_width/vision_height from encoder input";
                return false;
            }
            if (cfg_bytes != (size_t)in0.nSize) {
                ALOGW("Qwen-VL vision size mismatch (cfg=%dx%d bytes=%zu, model_input_bytes=%zu). Will pad/zero input tail.",
                      vision_width_, vision_height_, cfg_bytes, (size_t)in0.nSize);
            }
        }
    } else if (type_ == VLMType::InternVL3 || type_ == VLMType::FastVLM) {
        // Classic image encoder: detect NCHW/NHWC layout and image size from model input shape.
        impl_->input_is_nchw = -1;
        int h = 0, w = 0, is_nchw = -1;
        const bool got_shape = try_infer_hw_from_4d_shape_with_c3(in0.vShape, h, w, &is_nchw);
        if (got_shape) {
            impl_->input_is_nchw = is_nchw;
            if (vision_width_ > 0 && vision_height_ > 0 && (vision_width_ != w || vision_height_ != h)) {
                ALOGW("classic vision size override: cfg=%dx%d -> model=%dx%d (from input shape)",
                      vision_width_, vision_height_, w, h);
            }
            vision_width_ = w;
            vision_height_ = h;
        } else {
            // Fallback: if shape cannot be parsed, try layout by config+nSize.
            if (vision_width_ <= 0 || vision_height_ <= 0) {
                err = "classic vision encoder input shape missing; please provide valid vision_width/vision_height";
                return false;
            }
            const size_t need_nhwc_u8 = (size_t)vision_width_ * (size_t)vision_height_ * (size_t)3;
            const size_t need_nchw_f32 = need_nhwc_u8 * sizeof(float);
            if ((size_t)in0.nSize == need_nchw_f32) impl_->input_is_nchw = 1;
            else if ((size_t)in0.nSize == need_nhwc_u8) impl_->input_is_nchw = 0;
            else {
                err = "classic vision encoder layout not detected from input shape/nSize";
                return false;
            }
            ALOGW("classic vision input shape unavailable; fallback to cfg size %dx%d by nSize=%zu (layout=%s)",
                  vision_width_, vision_height_, (size_t)in0.nSize,
                  (impl_->input_is_nchw == 1 ? "NCHW-fp32" : "NHWC-u8"));
        }
    } else if (type_ == VLMType::SmolVLM2) {
        // SmolVLM2 encoder is usually NHWC u8; try shape first, then infer from nSize.
        const int old_w = vision_width_;
        const int old_h = vision_height_;
        int h = 0, w = 0, tmp_layout = -1;
        bool inferred = try_infer_hw_from_4d_shape_with_c3(in0.vShape, h, w, &tmp_layout);
        if (!inferred && ((size_t)in0.nSize % 3u) == 0u) {
            const size_t hw = (size_t)in0.nSize / 3u;
            const int side = (int)(std::sqrt((double)hw) + 0.5);
            if ((size_t)side * (size_t)side == hw) {
                h = side;
                w = side;
                inferred = true;
            }
        }
        if (!inferred) {
            if (vision_width_ <= 0 || vision_height_ <= 0) {
                err = "failed to infer SmolVLM2 vision_width/vision_height from encoder input";
                return false;
            }
            ALOGW("SmolVLM2 vision size inference failed; keep cfg=%dx%d", vision_width_, vision_height_);
        } else {
            if (old_w != w || old_h != h) {
                ALOGW("SmolVLM2 vision size override: cfg=%dx%d -> model=%dx%d", old_w, old_h, w, h);
            }
            vision_width_ = w;
            vision_height_ = h;
        }
    }

    // Detect encoder output dtype + tokens_per_block.
    // Some AX* runners/models report unreliable vShape for Qwen-VL vision encoders; prefer nSize + config sanity.
    {
        const auto& out0 = impl_->encoder.get_output(0);

        auto try_pick_by_bytes = [&](int bytes_per_elem, int& out_is_bf16, int& out_tokens_per_block) -> bool {
            if (bytes_per_elem <= 0) return false;
            if ((out0.nSize % (size_t)bytes_per_elem) != 0) return false;
            const size_t elem = (size_t)out0.nSize / (size_t)bytes_per_elem;
            if (elem % (size_t)tokens_embed_size_ != 0) return false;
            out_tokens_per_block = (int)(elem / (size_t)tokens_embed_size_);
            out_is_bf16 = (bytes_per_elem == 2) ? 1 : 0;
            return true;
        };

        bool picked = false;

        if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) {
            const int grid_h = vision_height_ / std::max(1, patch_size_);
            const int grid_w = vision_width_ / std::max(1, patch_size_);
            const int llm_grid_h = grid_h / std::max(1, spatial_merge_size_);
            const int llm_grid_w = grid_w / std::max(1, spatial_merge_size_);
            const int expected_tokens = std::max(1, llm_grid_h) * std::max(1, llm_grid_w);

            // Prefer fp32 if it matches the expected token count (Qwen3-VL reference branches use fp32 outputs).
            int out_is_bf16 = -1, tpb = 0;
            if (try_pick_by_bytes(4, out_is_bf16, tpb) && tpb == expected_tokens) {
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            } else if (try_pick_by_bytes(2, out_is_bf16, tpb) && tpb == expected_tokens) {
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            } else if (try_pick_by_bytes(4, out_is_bf16, tpb)) {
                // Fallback: still prefer fp32 if valid, but warn about unexpected token count.
                ALOGW("vision encoder tokens_per_block=%d (expected=%d). Using fp32 by nSize inference (out0.nSize=%zu).",
                      tpb, expected_tokens, (size_t)out0.nSize);
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            } else if (try_pick_by_bytes(2, out_is_bf16, tpb)) {
                ALOGW("vision encoder tokens_per_block=%d (expected=%d). Using bf16 by nSize inference (out0.nSize=%zu).",
                      tpb, expected_tokens, (size_t)out0.nSize);
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            }
        }

        if (!picked) {
            // Generic path: rely on vShape + nSize.
            int elem_count = 1;
            for (auto d : out0.vShape) elem_count *= (int)d;
            if (elem_count * 2 == out0.nSize) impl_->encoder_output_is_bf16 = 1;
            else if (elem_count * 4 == out0.nSize) impl_->encoder_output_is_bf16 = 0;
            else {
                err = "vision encoder output dtype not supported (nSize mismatch)";
                return false;
            }
            if (elem_count % tokens_embed_size_ != 0) {
                err = "vision encoder output element count not divisible by tokens_embed_size";
                return false;
            }
            tokens_per_block_ = elem_count / tokens_embed_size_;
        }
    }

    // Optional deepstack (Qwen3VL family from older branches): encoder provides extra float outputs.
    deepstack_layers_ = 0;
    if (type_ == VLMType::Qwen3VL) {
        const int nout = impl_->encoder.get_num_outputs();
        if (nout > 1) {
            deepstack_layers_ = std::min(3, nout - 1);
        }
    }

    cache_key_prefix_ = "vlm=" + std::string(VLMTypeName(type_)) + "|enc=" + normalize_path_for_key(encoder_axmodel) +
                        "|e=" + std::to_string(tokens_embed_size_) +
                        "|t=" + std::to_string(tokens_per_block_) +
                        "|ds=" + std::to_string(deepstack_layers_) +
                        "|vw=" + std::to_string(vision_width_) +
                        "|vh=" + std::to_string(vision_height_) +
                        "|tp=" + std::to_string(temporal_patch_size_) +
                        "|sm=" + std::to_string(spatial_merge_size_) +
                        "|ps=" + std::to_string(patch_size_) +
                        "|fps=" + std::to_string(fps_) +
                        "|tps=" + std::to_string(tokens_per_second_);

    // Sanity checks after auto inference.
    if (vision_width_ <= 0 || vision_height_ <= 0) {
        err = "invalid vision size after inference: " + std::to_string(vision_width_) + "x" + std::to_string(vision_height_);
        return false;
    }
    if (type_ == VLMType::InternVL3 || type_ == VLMType::FastVLM) {
        if (impl_->input_is_nchw != 0 && impl_->input_is_nchw != 1) {
            err = "classic vision encoder input layout (NCHW/NHWC) not detected";
            return false;
        }
    }

    // Token ids for placeholder locating.
    switch (type_) {
    case VLMType::Qwen2_5VL:
    case VLMType::Qwen3VL:
        if (!get_single_token_id(tokenizer_, "<|image_pad|>", image_pad_id_, err)) return false;
        if (!get_single_token_id(tokenizer_, "<|video_pad|>", video_pad_id_, err)) return false;
        if (!get_single_token_id(tokenizer_, "<|vision_start|>", vision_start_id_, err)) return false;
        ALOGI("Qwen-VL token ids: vision_start=%d image_pad=%d video_pad=%d", vision_start_id_, image_pad_id_, video_pad_id_);
        break;
    case VLMType::PaddleOCRVL:
        if (!get_single_token_id(tokenizer_, "<|IMAGE_PLACEHOLDER|>", image_pad_id_, err)) return false;
        if (!get_single_token_id(tokenizer_, "<|video_pad|>", video_pad_id_, err)) return false;
        if (!get_single_token_id(tokenizer_, "<|IMAGE_START|>", vision_start_id_, err)) return false;
        ALOGI("PaddleOCR-VL token ids: vision_start=%d image_pad=%d video_pad=%d", vision_start_id_, image_pad_id_, video_pad_id_);
        break;
    case VLMType::InternVL3:
        if (!get_single_token_id(tokenizer_, "<IMG_CONTEXT>", image_pad_id_, err)) return false;
        video_pad_id_ = image_pad_id_;
        if (!get_single_token_id(tokenizer_, "<img>", vision_start_id_, err)) {
            // Not required for injection; mRoPE not used.
            vision_start_id_ = -1;
        }
        break;
    case VLMType::FastVLM:
        if (!get_single_token_id(tokenizer_, "<image>", image_pad_id_, err)) return false;
        video_pad_id_ = image_pad_id_;
        vision_start_id_ = -1;
        break;
    case VLMType::SmolVLM2:
        if (!get_single_token_id(tokenizer_, "<image>", image_pad_id_, err)) return false;
        video_pad_id_ = image_pad_id_;
        vision_start_id_ = -1;
        break;
    default:
        break;
    }

    enabled_ = true;
    ALOGI("VisionModule init ok: type=%s, tokens_per_block=%d, embed_size=%d, out_dtype=%s",
          std::string(VLMTypeName(type_)).c_str(),
          tokens_per_block_,
          tokens_embed_size_,
          (impl_->encoder_output_is_bf16 ? "bf16" : "fp32"));
    if (deepstack_layers_ > 0) {
        ALOGI("VisionModule deepstack enabled: layers=%d", deepstack_layers_);
    }
#if !defined(AXLLM_USE_OPENCV)
    ALOGW("Vision preprocess backend: SimpleCV (OpenCV not found at build time; minor differences vs OpenCV are possible)");
#endif
    return true;
}

// PaddleOCRVL: convert uint8 patches to normalized float32 before feeding to the encoder.
// Normalization: (pixel / 255.0 - mean) / std, with mean=0.5, std=0.5 => pixel / 127.5 - 1.0
static bool encode_block_normalized_float(ax_runner_t& enc, int devid, int out_is_bf16,
                                          const std::vector<unsigned char>& bytes,
                                          std::vector<unsigned short>& out_bf16,
                                          float img_mean, float img_std,
                                          int deepstack_layers,
                                          std::vector<std::vector<float>>* deepstack_out,
                                          std::string& err)
{
    const auto& in0 = enc.get_input(0);
    const size_t expected_float_elems = bytes.size();
    const size_t expected_float_bytes = expected_float_elems * sizeof(float);
    if ((size_t)in0.nSize < expected_float_bytes) {
        err = "encoder input tensor too small for float32 conversion";
        return false;
    }

    // Convert uint8 -> normalized float32
    const float scale = 1.0f / (255.0f * img_std);
    const float shift = -img_mean / img_std;
    std::vector<float> fp32(expected_float_elems);
    for (size_t i = 0; i < expected_float_elems; ++i) {
        fp32[i] = (float)bytes[i] * scale + shift;
    }

    // Copy float32 data to encoder input; zero-pad tail if needed.
    if (expected_float_bytes == (size_t)in0.nSize) {
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, fp32.data(), expected_float_bytes);
        } else {
            v_h2d(V_WADDR(in0), fp32.data(), expected_float_bytes, devid);
        }
    } else {
        std::vector<unsigned char> tmp((size_t)in0.nSize, 0);
        std::memcpy(tmp.data(), fp32.data(), expected_float_bytes);
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, tmp.data(), tmp.size());
        } else {
            v_h2d(V_WADDR(in0), tmp.data(), tmp.size(), devid);
        }
    }
    fp32.clear();
    fp32.shrink_to_fit();

    enc.inference();

    // Read output - reuse the same logic as encode_block_u8.
    const auto& out0 = enc.get_output(0);
    int elem_count = 0;
    if (out_is_bf16) {
        elem_count = (int)((size_t)out0.nSize / sizeof(unsigned short));
    } else {
        elem_count = (int)((size_t)out0.nSize / sizeof(float));
    }
    if (elem_count <= 0) { err = "vision encoder output elem_count invalid"; return false; }
    out_bf16.resize(elem_count);
    if (out_is_bf16) {
        if (out0.pVirAddr) std::memcpy(out_bf16.data(), out0.pVirAddr, (size_t)elem_count * sizeof(unsigned short));
        else v_d2h(out_bf16.data(), V_RADDR(out0), (size_t)elem_count * sizeof(unsigned short), devid);
    } else {
        std::vector<float> tmp(elem_count);
        if (out0.pVirAddr) std::memcpy(tmp.data(), out0.pVirAddr, (size_t)elem_count * sizeof(float));
        else v_d2h(tmp.data(), V_RADDR(out0), (size_t)elem_count * sizeof(float), devid);
        for (int i = 0; i < elem_count; ++i) out_bf16[i] = bfloat16(tmp[i]).data;
    }
    return true;
}

static bool encode_block_u8(ax_runner_t& enc, int devid, int out_is_bf16,
                            const std::vector<unsigned char>& bytes,
                            std::vector<unsigned short>& out_bf16,
                            int deepstack_layers,
                            std::vector<std::vector<float>>* deepstack_out,
                            std::string& err)
{
    const auto& in0 = enc.get_input(0);
    if ((size_t)in0.nSize < bytes.size()) {
        err = "encoder input tensor too small";
        return false;
    }
    // If the model expects more bytes than the patchifier produced (config mismatch),
    // we must zero-pad the tail; otherwise leftover bytes from previous runs can dominate.
    if (bytes.size() == (size_t)in0.nSize) {
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, bytes.data(), bytes.size());
        } else {
            v_h2d(V_WADDR(in0), bytes.data(), bytes.size(), devid);
        }
    } else {
        static bool warned = false;
        if (!warned) {
            ALOGW("vision encoder input size mismatch: write=%zu < tensor=%zu (zero-pad tail).", bytes.size(), (size_t)in0.nSize);
            warned = true;
        }
        std::vector<unsigned char> tmp;
        tmp.assign((size_t)in0.nSize, 0);
        if (!bytes.empty()) std::memcpy(tmp.data(), bytes.data(), bytes.size());
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, tmp.data(), tmp.size());
        } else {
            v_h2d(V_WADDR(in0), tmp.data(), tmp.size(), devid);
        }
    }
    enc.inference();

    const auto& out0 = enc.get_output(0);
    int elem_count = 0;
    if (out_is_bf16) {
        elem_count = (int)((size_t)out0.nSize / sizeof(unsigned short));
    } else {
        elem_count = (int)((size_t)out0.nSize / sizeof(float));
    }
    if (elem_count <= 0) {
        err = "vision encoder output elem_count invalid";
        return false;
    }
    out_bf16.resize(elem_count);

    if (out_is_bf16) {
        if (out0.pVirAddr) {
            std::memcpy(out_bf16.data(), out0.pVirAddr, (size_t)elem_count * sizeof(unsigned short));
        } else {
            v_d2h(out_bf16.data(), V_RADDR(out0), (size_t)elem_count * sizeof(unsigned short), devid);
        }
    } else {
        std::vector<float> tmp(elem_count);
        if (out0.pVirAddr) {
            std::memcpy(tmp.data(), out0.pVirAddr, (size_t)elem_count * sizeof(float));
        } else {
            v_d2h(tmp.data(), V_RADDR(out0), (size_t)elem_count * sizeof(float), devid);
        }
        for (int i = 0; i < elem_count; ++i) out_bf16[i] = bfloat16(tmp[i]).data;
    }

    if (deepstack_out && deepstack_layers > 0) {
        if ((int)deepstack_out->size() != deepstack_layers) deepstack_out->assign((size_t)deepstack_layers, {});
        for (int li = 0; li < deepstack_layers; ++li) {
            const auto& o = enc.get_output(li + 1);
            int o_elems = (int)((size_t)o.nSize / sizeof(float));
            int o_is_fp32 = 1;
            if ((size_t)o_elems * sizeof(float) != (size_t)o.nSize) {
                // fallback: bf16 -> fp32
                o_elems = (int)((size_t)o.nSize / sizeof(unsigned short));
                o_is_fp32 = 0;
                if ((size_t)o_elems * sizeof(unsigned short) != (size_t)o.nSize) {
                    err = "deepstack output dtype not supported (nSize mismatch)";
                    return false;
                }
            }
            std::vector<float> feat;
            feat.resize(o_elems);

            if (o_is_fp32) {
                if (o.pVirAddr) {
                    std::memcpy(feat.data(), o.pVirAddr, (size_t)o_elems * sizeof(float));
                } else {
                    v_d2h(feat.data(), V_RADDR(o), (size_t)o_elems * sizeof(float), devid);
                }
            } else {
                std::vector<unsigned short> tmp_bf16(o_elems);
                if (o.pVirAddr) {
                    std::memcpy(tmp_bf16.data(), o.pVirAddr, (size_t)o_elems * sizeof(unsigned short));
                } else {
                    v_d2h(tmp_bf16.data(), V_RADDR(o), (size_t)o_elems * sizeof(unsigned short), devid);
                }
                for (int i = 0; i < o_elems; ++i) feat[i] = bfloat16(tmp_bf16[i]).fp32();
            }

            (*deepstack_out)[li].insert((*deepstack_out)[li].end(), feat.begin(), feat.end());
        }
    }

    return true;
}

static bool encode_classic_image(ax_runner_t& enc, int devid, int out_is_bf16, int input_is_nchw,
                                 int tgt_w, int tgt_h,
                                 const axcv::Mat& img_bgr,
                                 std::vector<unsigned short>& out_bf16,
                                 std::string& err)
{
    if (axcv::empty(img_bgr)) { err = "empty image"; return false; }

    axcv::Mat dst_rs;
    axcv::resize(img_bgr, dst_rs, tgt_w, tgt_h);
    axcv::Mat dst;
    axcv::cvtColorBGR2RGB(dst_rs, dst);

    const auto& in0 = enc.get_input(0);

    if (input_is_nchw) {
        // float32 NCHW with imagenet mean/std
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float stdv[3] = {0.229f, 0.224f, 0.225f};

        std::vector<float> tmp((size_t)3 * tgt_h * tgt_w);
        for (int h = 0; h < tgt_h; h++) {
            const uint8_t* row = axcv::row_ptr(dst, h);
            for (int w = 0; w < tgt_w; w++) {
                for (int c = 0; c < 3; c++) {
                    int in_index = w * 3 + c;
                    int out_index = c * tgt_h * tgt_w + h * tgt_w + w;
                    tmp[out_index] = (float(row[in_index]) / 255.0f - mean[c]) / stdv[c];
                }
            }
        }
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, tmp.data(), tmp.size() * sizeof(float));
        } else {
            v_h2d(V_WADDR(in0), tmp.data(), tmp.size() * sizeof(float), devid);
        }
    } else {
        // u8 NHWC
        const size_t need = (size_t)tgt_h * tgt_w * 3;
        if ((size_t)in0.nSize < need) { err = "encoder input tensor too small"; return false; }
        // Pack tightly row-major.
        std::vector<uint8_t> packed;
        packed.resize(need);
        for (int r = 0; r < tgt_h; ++r) {
            const uint8_t* sp = axcv::row_ptr(dst, r);
            std::memcpy(packed.data() + (size_t)r * (size_t)tgt_w * 3, sp, (size_t)tgt_w * 3);
        }
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, packed.data(), packed.size());
        } else {
            v_h2d(V_WADDR(in0), packed.data(), packed.size(), devid);
        }
    }

    enc.inference();

    const auto& out0 = enc.get_output(0);
    int elem_count = 0;
    if (out_is_bf16) {
        elem_count = (int)((size_t)out0.nSize / sizeof(unsigned short));
    } else {
        elem_count = (int)((size_t)out0.nSize / sizeof(float));
    }
    if (elem_count <= 0) {
        err = "vision encoder output elem_count invalid";
        return false;
    }
    out_bf16.resize(elem_count);

    if (out_is_bf16) {
        v_d2h(out_bf16.data(), V_RADDR(out0), (size_t)elem_count * sizeof(unsigned short), devid);
        return true;
    }

    std::vector<float> tmp(elem_count);
    v_d2h(tmp.data(), V_RADDR(out0), (size_t)elem_count * sizeof(float), devid);
    for (int i = 0; i < elem_count; ++i) out_bf16[i] = bfloat16(tmp[i]).data;
    return true;
}

bool VisionModule::EncodeForContent(const Content& content,
                                    const MediaInputs& media,
                                    int& out_num_media_for_tokenizer,
                                    std::vector<std::vector<unsigned short>>& out_blocks,
                                    std::vector<std::vector<float>>* out_deepstack_append,
                                    std::vector<std::vector<int>>& out_image_grid_thw,
                                    std::vector<std::vector<int>>& out_video_grid_thw,
                                    std::string& err)
{
    if (!enabled_ || !impl_ || !impl_->encoder_inited) { err = "vision module not initialized"; return false; }
    if (content.type != IMAGE && content.type != VIDEO) { err = "content is not image/video"; return false; }
    if (media.uris.empty()) { err = "media.uris empty"; return false; }

    out_blocks.clear();
    out_image_grid_thw.clear();
    out_video_grid_thw.clear();

    const int devid = impl_->encoder.get_devid();

    if (content.type == IMAGE) {
        // Expand all uris into a flat image file list (directory is treated as multiple images).
        std::vector<std::string> image_files;
        for (const auto& uri : media.uris) {
            if (is_file(uri)) {
                image_files.push_back(uri);
            } else if (is_directory(uri)) {
                auto files = list_files(uri);
                image_files.insert(image_files.end(), files.begin(), files.end());
            } else {
                err = "invalid image uri: " + uri;
                return false;
            }
        }
        if (image_files.empty()) { err = "no images found"; return false; }

        const int grid_h = vision_height_ / patch_size_;
        const int grid_w = vision_width_ / patch_size_;

        out_num_media_for_tokenizer = (int)image_files.size();
        out_image_grid_thw.reserve(image_files.size());

        for (const auto& file : image_files) {
            const std::string key = make_image_cache_key(cache_key_prefix_, file);

            if (cache_enabled_) {
                auto it = image_cache_.find(key);
                if (it == image_cache_.end()) {
                    std::vector<std::vector<unsigned short>> cached_blocks;
                    std::vector<std::vector<float>> cached_deepstack;
                    if (disk_cache_load(cache_dir_, key, tokens_embed_size_, tokens_per_block_, deepstack_layers_, cached_blocks, cached_deepstack)) {
                        ALOGI("vision cache hit (disk): %s", file.c_str());
                        CachedImage ci;
                        ci.blocks_bf16 = std::move(cached_blocks);
                        ci.deepstack_features = std::move(cached_deepstack);
                        it = image_cache_.emplace(key, std::move(ci)).first;
                    }
                }

                if (it != image_cache_.end()) {
                    ALOGI("vision cache hit (mem): %s", file.c_str());
                    for (const auto& b : it->second.blocks_bf16) out_blocks.push_back(b);
                    if (out_deepstack_append && !it->second.deepstack_features.empty()) {
                        if (out_deepstack_append->size() != it->second.deepstack_features.size()) {
                            err = "deepstack cache layer count mismatch";
                            return false;
                        }
                        for (size_t li = 0; li < it->second.deepstack_features.size(); ++li) {
                            const auto& v = it->second.deepstack_features[li];
                            (*out_deepstack_append)[li].insert((*out_deepstack_append)[li].end(), v.begin(), v.end());
                        }
                    }
                    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) out_image_grid_thw.push_back({1, grid_h, grid_w});
                    continue;
                }
            }

            axcv::Mat img = axcv::imread(file, axcv::IMREAD_COLOR);
            if (axcv::empty(img)) { err = "failed to read image: " + file; return false; }

            std::vector<std::vector<unsigned short>> blocks_for_one;
            std::vector<std::vector<float>> deepstack_for_one;
            if (out_deepstack_append && deepstack_layers_ > 0) deepstack_for_one.resize((size_t)deepstack_layers_);

            if (type_ == VLMType::SmolVLM2) {
                std::vector<axcv::Mat> one{img};
                std::vector<std::vector<unsigned char>> pixel_values;
                Smolvlm2ImageProcessor(one, pixel_values, vision_width_, vision_height_);
                blocks_for_one.reserve(pixel_values.size()); // expected 5
                for (auto& pv : pixel_values) {
                    std::vector<unsigned short> emb;
                    if (!encode_block_u8(impl_->encoder, devid, impl_->encoder_output_is_bf16, pv, emb, 0, nullptr, err)) return false;
                    blocks_for_one.push_back(std::move(emb));
                }
            }
            else if (type_ == VLMType::PaddleOCRVL) {
                // PaddleOCR-VL VIT expects patches in [N, C, pH, pW] format (channel-first per patch,
                // no spatial merge in preprocessing — merge happens inside the VIT model).
                std::vector<unsigned char> pv;
                PaddleOCRVLImageProcessor(img, pv, vision_height_, vision_width_, patch_size_);
                {
                    unsigned char mn = 255, mx = 0;
                    for (unsigned char b : pv) { if (b < mn) mn = b; if (b > mx) mx = b; }
                    ALOGI("PaddleOCRVL pixel_values bytes=%zu min=%u max=%u (w=%d h=%d ps=%d)",
                          pv.size(), (unsigned)mn, (unsigned)mx,
                          vision_width_, vision_height_, patch_size_);
                }
                std::vector<unsigned short> emb;
                if (!encode_block_normalized_float(impl_->encoder, devid, impl_->encoder_output_is_bf16,
                                                  pv, emb, 0.5f, 0.5f,
                                                  0, nullptr, err))
                    return false;
                blocks_for_one.push_back(std::move(emb));
            }
            else if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL) {
                std::vector<axcv::Mat> one{img};
                std::vector<std::vector<unsigned char>> pixel_values;
                Qwen2VideoProcessor(one, pixel_values, vision_height_, vision_width_, temporal_patch_size_, spatial_merge_size_, patch_size_);
                if (pixel_values.size() != 1) { err = "Qwen2VideoProcessor(image) returned != 1 block"; return false; }
                {
                    // Quick sanity: if preprocessing is broken, pixel_values often becomes all zeros.
                    const auto& pv = pixel_values[0];
                    unsigned char mn = 255, mx = 0;
                    for (unsigned char b : pv) { if (b < mn) mn = b; if (b > mx) mx = b; }
                    ALOGI("Qwen-VL pixel_values[0] bytes=%zu min=%u max=%u (w=%d h=%d tp=%d ps=%d sm=%d)",
                          pv.size(), (unsigned)mn, (unsigned)mx,
                          vision_width_, vision_height_, temporal_patch_size_, patch_size_, spatial_merge_size_);
                }
                std::vector<unsigned short> emb;
                if (!encode_block_u8(impl_->encoder, devid, impl_->encoder_output_is_bf16, pixel_values[0], emb,
                                     deepstack_layers_, (out_deepstack_append ? &deepstack_for_one : nullptr), err))
                    return false;
                blocks_for_one.push_back(std::move(emb));
            }
            else if (type_ == VLMType::InternVL3 || type_ == VLMType::FastVLM) {
                std::vector<unsigned short> emb;
                if (!encode_classic_image(impl_->encoder, devid, impl_->encoder_output_is_bf16,
                                          impl_->input_is_nchw, vision_width_, vision_height_, img, emb, err))
                    return false;
                blocks_for_one.push_back(std::move(emb));
            }
            else {
                err = "IMAGE not supported for this vlm_type";
                return false;
            }

            if (cache_enabled_) {
                ALOGI("vision cache store: %s", file.c_str());
                disk_cache_save(cache_dir_, key, tokens_embed_size_, tokens_per_block_, blocks_for_one, deepstack_for_one);
                CachedImage ci;
                ci.blocks_bf16 = blocks_for_one;
                ci.deepstack_features = deepstack_for_one;
                image_cache_[key] = std::move(ci);
            }

            for (auto& b : blocks_for_one) out_blocks.push_back(std::move(b));
            if (out_deepstack_append && !deepstack_for_one.empty()) {
                if (out_deepstack_append->size() != deepstack_for_one.size()) {
                    err = "deepstack layer count mismatch";
                    return false;
                }
                for (size_t li = 0; li < deepstack_for_one.size(); ++li) {
                    const auto& v = deepstack_for_one[li];
                    (*out_deepstack_append)[li].insert((*out_deepstack_append)[li].end(), v.begin(), v.end());
                }
            }
            if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) out_image_grid_thw.push_back({1, grid_h, grid_w});
        }

        return true;

        err = "IMAGE not supported for this vlm_type";
        return false;
    }

    // VIDEO
    if (media.uris.size() != 1) {
        err = "VIDEO currently supports exactly 1 uri (directory of frames)";
        return false;
    }
    auto frames = ReadImages(media.uris[0]);
    if (frames.empty()) { err = "no video frames loaded"; return false; }

    if (type_ == VLMType::SmolVLM2) {
        std::vector<std::vector<unsigned char>> pixel_values;
        Smolvlm2VideoProcessor(frames, pixel_values, vision_width_, vision_height_);
        out_num_media_for_tokenizer = (int)pixel_values.size();
        out_blocks.reserve(pixel_values.size());
        for (auto& pv : pixel_values) {
            std::vector<unsigned short> emb;
            if (!encode_block_u8(impl_->encoder, devid, impl_->encoder_output_is_bf16, pv, emb, 0, nullptr, err)) return false;
            out_blocks.push_back(std::move(emb));
        }
        return true;
    }

    if (type_ == VLMType::PaddleOCRVL) {
        // PaddleOCR-VL video: process each frame independently with the same VIT as images.
        const int grid_h = vision_height_ / patch_size_;
        const int grid_w = vision_width_ / patch_size_;

        out_num_media_for_tokenizer = (int)frames.size();
        out_video_grid_thw.push_back({(int)frames.size(), grid_h, grid_w});
        out_blocks.reserve(frames.size());
        for (auto& frame : frames) {
            std::vector<unsigned char> pv;
            PaddleOCRVLImageProcessor(frame, pv, vision_height_, vision_width_, patch_size_);
            std::vector<unsigned short> emb;
            if (!encode_block_normalized_float(impl_->encoder, devid, impl_->encoder_output_is_bf16,
                                               pv, emb, 0.5f, 0.5f, 0, nullptr, err))
                return false;
            out_blocks.push_back(std::move(emb));
        }
        return true;
    }

    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL) {
        const int grid_h = vision_height_ / patch_size_;
        const int grid_w = vision_width_ / patch_size_;

        std::vector<std::vector<unsigned char>> pixel_values;
        Qwen2VideoProcessor(frames, pixel_values, vision_height_, vision_width_, temporal_patch_size_, spatial_merge_size_, patch_size_);
        out_num_media_for_tokenizer = (int)pixel_values.size();
        out_video_grid_thw.push_back({(int)pixel_values.size(), grid_h, grid_w});
        out_blocks.reserve(pixel_values.size());
        for (auto& pv : pixel_values) {
            std::vector<unsigned short> emb;
            if (!encode_block_u8(impl_->encoder, devid, impl_->encoder_output_is_bf16, pv, emb,
                                 deepstack_layers_, out_deepstack_append, err))
                return false;
            out_blocks.push_back(std::move(emb));
        }
        return true;
    }

    err = "VIDEO not supported for this vlm_type";
    return false;
}

bool VisionModule::BuildInjectionState(const std::vector<int>& input_ids,
                                       const std::vector<std::vector<unsigned short>>& blocks,
                                       const std::vector<std::vector<float>>& deepstack,
                                       RunState& state_out,
                                       std::string& err)
{
    state_out = {};
    state_out.pos2vision.assign(input_ids.size(), -1);

    // Collect placeholder positions in order.
    std::vector<int> placeholder_pos;
    placeholder_pos.reserve(input_ids.size());
    for (size_t i = 0; i < input_ids.size(); ++i) {
        const int id = input_ids[i];
        if (id == image_pad_id_ || id == video_pad_id_) placeholder_pos.push_back((int)i);
    }

    // Flatten blocks to vision tokens (bf16).
    size_t total_elems = 0;
    for (const auto& b : blocks) total_elems += b.size();
    if (total_elems % (size_t)tokens_embed_size_ != 0) {
        err = "vision blocks total size not divisible by tokens_embed_size";
        return false;
    }
    const size_t vision_token_count = total_elems / (size_t)tokens_embed_size_;

    if (placeholder_pos.size() != vision_token_count) {
        err = "placeholder token count mismatch: placeholder=" + std::to_string(placeholder_pos.size()) +
              " vision_tokens=" + std::to_string(vision_token_count);
        return false;
    }

    state_out.vision_embed.resize(total_elems);
    size_t off = 0;
    for (const auto& b : blocks) {
        memcpy(state_out.vision_embed.data() + off, b.data(), b.size() * sizeof(unsigned short));
        off += b.size();
    }

    state_out.deepstack_features.clear();
    if (!deepstack.empty()) {
        // Expect one entry per layer, each flattened as [total_elems].
        state_out.deepstack_features = deepstack;
        for (const auto& v : state_out.deepstack_features) {
            if (v.size() != total_elems) {
                err = "deepstack feature size mismatch";
                return false;
            }
        }
    }

    for (size_t i = 0; i < placeholder_pos.size(); ++i) state_out.pos2vision[placeholder_pos[i]] = (int)i;
    return true;
}

bool VisionModule::Prepare(const std::vector<Content>& history_in,
                           const std::vector<MediaInputs>& media_inputs,
                           std::vector<Content>& history_out,
                           std::vector<int>& input_ids_out,
                           RunState& state_out,
                           std::string& err)
{
    if (!enabled_) { err = "vision module disabled"; return false; }
    if (!tokenizer_) { err = "tokenizer not set"; return false; }

    history_out = history_in;

    // Map content_index -> MediaInputs (at most one entry per content index).
    std::unordered_map<size_t, MediaInputs> media_map;
    media_map.reserve(media_inputs.size());
    for (const auto& m : media_inputs) media_map[m.content_index] = m;

    std::vector<std::vector<unsigned short>> all_blocks;
    std::vector<std::vector<float>> all_deepstack;
    if (deepstack_layers_ > 0) all_deepstack.resize((size_t)deepstack_layers_);
    std::vector<std::vector<int>> image_grid_thw;
    std::vector<std::vector<int>> video_grid_thw;

    for (size_t i = 0; i < history_out.size(); ++i) {
        auto& c = history_out[i];
        if (c.type != IMAGE && c.type != VIDEO) continue;

        auto it = media_map.find(i);
        if (it == media_map.end()) {
            err = "missing media_inputs for history index " + std::to_string(i);
            return false;
        }

        int num_media_for_tokenizer = 0;
        std::vector<std::vector<unsigned short>> blocks;
        std::vector<std::vector<int>> img_grid, vid_grid;
        if (!EncodeForContent(c, it->second, num_media_for_tokenizer, blocks,
                              (deepstack_layers_ > 0 ? &all_deepstack : nullptr),
                              img_grid, vid_grid, err))
            return false;

        c.num_media = num_media_for_tokenizer;
        c.num_media_tokens = tokens_per_block_;

        for (auto& b : blocks) all_blocks.push_back(std::move(b));
        image_grid_thw.insert(image_grid_thw.end(), img_grid.begin(), img_grid.end());
        video_grid_thw.insert(video_grid_thw.end(), vid_grid.begin(), vid_grid.end());
    }

    input_ids_out = tokenizer_->encode(history_out);
    if (!BuildInjectionState(input_ids_out, all_blocks, all_deepstack, state_out, err)) return false;

    // Optional: mRoPE (Qwen-VL)
    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) {
        mrope::Config cfg;
        cfg.vision_config.temporal_patch_size = temporal_patch_size_;
        cfg.vision_config.tokens_per_second = tokens_per_second_;
        cfg.vision_config.spatial_merge_size = spatial_merge_size_;
        cfg.vision_config.patch_size = patch_size_;
        cfg.vision_config.width = vision_width_;
        cfg.vision_config.height = vision_height_;
        cfg.vision_config.fps = fps_;
        cfg.image_token_id = image_pad_id_;
        cfg.video_token_id = video_pad_id_;
        cfg.vision_start_token_id = vision_start_id_;

        if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::PaddleOCRVL) {
            std::vector<double> second_per_grid_ts;
            second_per_grid_ts.reserve(video_grid_thw.size());
            for (size_t i = 0; i < video_grid_thw.size(); ++i) {
                second_per_grid_ts.push_back(double(temporal_patch_size_) / double(std::max(1, fps_)));
            }
            state_out.position_ids = mrope::get_rope_index_qwen2_5(cfg, input_ids_out, image_grid_thw, video_grid_thw, second_per_grid_ts);
        } else {
            state_out.position_ids = mrope::get_rope_index_qwen3(cfg, input_ids_out, image_grid_thw, video_grid_thw);
        }

        int max_pos = -1;
        for (const auto& row : state_out.position_ids) {
            for (int v : row) max_pos = std::max(max_pos, v);
        }
        if (max_pos >= 0) state_out.decode_start = max_pos + 1;
    }

    return true;
}

} // namespace vision
