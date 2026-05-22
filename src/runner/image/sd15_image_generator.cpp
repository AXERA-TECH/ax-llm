#include "image/sd15_image_generator.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwctype>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openai_api/core/output_chunk.hpp"
#include "runner/ax_model_runner/ax_model_runner_ax650.hpp"
#include "runner/utils/bfloat16.hpp"
#include "runner/utils/ax_cv.hpp"
#include "runner/utils/files.hpp"
#include "runner/utils/image_processor.hpp"
#include "runner/utils/json.hpp"
#include "runner/utils/sample_log.h"

#if defined(AXLLM_USE_OPENCV)
#include <opencv2/imgcodecs.hpp>
#else
#include "SimpleCV.hpp"
#endif

#ifdef USE_AXCL
#include "runner/ax_model_runner/ax_model_runner_axcl.hpp"
#include <axcl.h>
using sd15_runner_t = ax_runner_axcl;
#else
using sd15_runner_t = ax_runner_ax650;
#endif

namespace sd15 {
namespace {

constexpr const char* kTimeEmbedKey = "/down_blocks.0/resnets.0/act_1/Mul_output_0";
constexpr float kLatentScale = 0.18215f;
constexpr int kPromptMaxLength = 77;
constexpr int kBosTokenId = 49406;
constexpr int kEosTokenId = 49407;
constexpr double kPi = 3.14159265358979323846;

static const std::array<int, 4> kTxt2ImgTimesteps = {999, 759, 499, 259};
static const std::array<int, 2> kImg2ImgTimesteps = {499, 259};
static const std::array<int, 4> kImg2ImgSelfTimesteps = {999, 759, 499, 259};
static const std::array<int, 2> kImg2ImgStepIndex = {2, 3};

struct VariantCandidate {
    std::string fallback_openai_size;
    std::string dir_name;
};

static const VariantCandidate kVariantCandidates[] = {
    {"512x512", "models"},
    {"768x1024", "models_1024x768"},
    {"256x256", "ax620e_models"},
    {"320x320", "ax620e_320x320_models"},
    {"320x320", "models_320x320"},
};

static inline std::string trim_copy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(start, end - start);
}

static std::string ret_hex_string(int ret) {
    std::ostringstream oss;
    oss << "0x" << std::hex << static_cast<uint32_t>(ret);
    return oss.str();
}

enum class TensorFloatType {
    FP32,
    FP16,
    BF16,
};

static const char* tensor_float_type_name(TensorFloatType type) {
    switch (type) {
        case TensorFloatType::FP32: return "fp32";
        case TensorFloatType::FP16: return "fp16";
        case TensorFloatType::BF16: return "bf16";
    }
    return "unknown";
}

static size_t tensor_float_type_bytes(TensorFloatType type) {
    switch (type) {
        case TensorFloatType::FP32: return 4;
        case TensorFloatType::FP16:
        case TensorFloatType::BF16:
            return 2;
    }
    return 0;
}

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;

    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t mantissa = mant;
            int32_t exp32 = 127 - 15 + 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exp32;
            }
            mantissa &= 0x03FFu;
            bits = sign | ((uint32_t)exp32 << 23) | (mantissa << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t exp32 = exp + (127 - 15);
        bits = sign | (exp32 << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t fp32_to_fp16_rne(float float_val) {
    std::uint32_t bits;
    std::memcpy(&bits, &float_val, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::uint32_t mant = bits & 0x007FFFFFu;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(1 - exp);
        std::uint32_t half_mant = mant >> (shift + 13);
        const std::uint32_t round_bit = (mant >> (shift + 12)) & 1u;
        half_mant += round_bit;
        return static_cast<uint16_t>(sign | half_mant);
    }

    if (exp >= 31) {
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13));
    }

    mant += 0x00001000u;
    if (mant & 0x00800000u) {
        mant = 0;
        ++exp;
        if (exp >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}

static TensorFloatType parse_tensor_float_type(const std::filesystem::path& config_path) {
    if (!std::filesystem::exists(config_path)) {
        return TensorFloatType::FP16;
    }

    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        return TensorFloatType::FP16;
    }

    nlohmann::json j;
    ifs >> j;
    std::string dtype;
    if (j.contains("torch_dtype") && j["torch_dtype"].is_string()) {
        dtype = j["torch_dtype"].get<std::string>();
    } else if (j.contains("weight_dtype") && j["weight_dtype"].is_string()) {
        dtype = j["weight_dtype"].get<std::string>();
    }

    std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (dtype.find("bfloat16") != std::string::npos || dtype == "bf16") {
        return TensorFloatType::BF16;
    }
    if (dtype.find("float32") != std::string::npos || dtype == "fp32" || dtype == "f32") {
        return TensorFloatType::FP32;
    }
    if (dtype.find("float16") != std::string::npos || dtype == "fp16" || dtype == "f16") {
        return TensorFloatType::FP16;
    }
    return TensorFloatType::FP16;
}

static bool parse_openai_image_size(const std::string& size, int& width, int& height) {
    const std::string s = trim_copy(size);
    const size_t pos = s.find('x');
    if (pos == std::string::npos) return false;
    try {
        width = std::stoi(s.substr(0, pos));
        height = std::stoi(s.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return width > 0 && height > 0;
}

static std::string normalize_openai_size(const std::string& size, const std::string& fallback = {}) {
    int w = 0;
    int h = 0;
    if (!parse_openai_image_size(size, w, h)) {
        if (!fallback.empty() && parse_openai_image_size(fallback, w, h)) {
            return std::to_string(w) + "x" + std::to_string(h);
        }
        return {};
    }
    return std::to_string(w) + "x" + std::to_string(h);
}

static std::string make_model_id(const std::string& base, const std::string& size) {
    return base + "-" + size;
}

static bool parse_variant_candidates_from_config(const std::filesystem::path& config_path,
                                                 std::vector<VariantCandidate>& candidates,
                                                 std::string& err) {
    if (!std::filesystem::exists(config_path)) {
        return true;
    }

    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        return true;
    }

    nlohmann::json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        err = std::string("failed to parse config.json: ") + e.what();
        return false;
    }

    if (!j.contains("image_variants")) {
        return true;
    }
    if (!j["image_variants"].is_array()) {
        err = "config.json image_variants must be an array";
        return false;
    }

    for (const auto& item : j["image_variants"]) {
        if (!item.is_object()) {
            err = "config.json image_variants item must be an object";
            return false;
        }
        if (!item.contains("dir") || !item["dir"].is_string()) {
            err = "config.json image_variants item is missing string dir";
            return false;
        }

        std::string size;
        if (item.contains("size") && item["size"].is_string()) {
            size = item["size"].get<std::string>();
        } else if (item.contains("openai_size") && item["openai_size"].is_string()) {
            size = item["openai_size"].get<std::string>();
        }

        size = normalize_openai_size(size);
        if (size.empty()) {
            err = "config.json image_variants item has invalid size";
            return false;
        }

        VariantCandidate candidate;
        candidate.fallback_openai_size = size;
        candidate.dir_name = item["dir"].get<std::string>();
        candidates.push_back(std::move(candidate));
    }

    return true;
}

static std::string base64_encode(const std::string& input) {
    static const char base64_chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string encoded;
    encoded.reserve(((input.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < input.size()) {
        const uint32_t chunk =
            (static_cast<uint8_t>(input[i]) << 16) |
            (static_cast<uint8_t>(input[i + 1]) << 8) |
            (static_cast<uint8_t>(input[i + 2]));
        encoded.push_back(base64_chars[(chunk >> 18) & 0x3f]);
        encoded.push_back(base64_chars[(chunk >> 12) & 0x3f]);
        encoded.push_back(base64_chars[(chunk >> 6) & 0x3f]);
        encoded.push_back(base64_chars[chunk & 0x3f]);
        i += 3;
    }

    const size_t rem = input.size() - i;
    if (rem == 1) {
        const uint32_t chunk = static_cast<uint8_t>(input[i]) << 16;
        encoded.push_back(base64_chars[(chunk >> 18) & 0x3f]);
        encoded.push_back(base64_chars[(chunk >> 12) & 0x3f]);
        encoded.push_back('=');
        encoded.push_back('=');
    } else if (rem == 2) {
        const uint32_t chunk =
            (static_cast<uint8_t>(input[i]) << 16) |
            (static_cast<uint8_t>(input[i + 1]) << 8);
        encoded.push_back(base64_chars[(chunk >> 18) & 0x3f]);
        encoded.push_back(base64_chars[(chunk >> 12) & 0x3f]);
        encoded.push_back(base64_chars[(chunk >> 6) & 0x3f]);
        encoded.push_back('=');
    }

    return encoded;
}

static bool read_binary_file(const std::filesystem::path& path, std::vector<uint8_t>& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    ifs.seekg(0, std::ios::end);
    const auto size = ifs.tellg();
    if (size < 0) return false;
    ifs.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (size > 0) {
        ifs.read(reinterpret_cast<char*>(out.data()), size);
    }
    return ifs.good() || ifs.eof();
}

static std::vector<uint8_t> base64_decode_bytes(const std::string& encoded) {
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::vector<uint8_t> out;
    int val = 0;
    int valb = -8;
    for (unsigned char c : encoded) {
        if (c == '=') break;
        const auto pos = chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(pos);
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

static bool parse_base64_data_uri(const std::string& uri, std::vector<uint8_t>& bytes) {
    const std::string prefix = "data:";
    if (uri.rfind(prefix, 0) != 0) return false;

    const auto semi = uri.find(';', prefix.size());
    if (semi == std::string::npos) return false;
    const std::string b64tag = "base64,";
    if (uri.compare(semi + 1, b64tag.size(), b64tag) != 0) return false;

    bytes = base64_decode_bytes(uri.substr(semi + 1 + b64tag.size()));
    return !bytes.empty();
}

static bool load_request_image_bytes(const std::vector<uint8_t>& inline_bytes,
                                     const std::string& image_ref,
                                     std::vector<uint8_t>& bytes,
                                     std::string& err) {
    if (!inline_bytes.empty()) {
        bytes = inline_bytes;
        return true;
    }
    if (image_ref.empty()) {
        err = "init image bytes are required for img2img";
        return false;
    }
    if (parse_base64_data_uri(image_ref, bytes)) {
        return true;
    }
    if (read_binary_file(image_ref, bytes)) {
        return true;
    }
    err = "failed to load init image: " + image_ref;
    return false;
}

static bool load_npy_float32(const std::filesystem::path& path,
                             std::vector<float>& values,
                             std::vector<size_t>& shape) {
    values.clear();
    shape.clear();

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;

    char magic[6] = {};
    ifs.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) return false;

    uint8_t ver_major = 0;
    uint8_t ver_minor = 0;
    ifs.read(reinterpret_cast<char*>(&ver_major), 1);
    ifs.read(reinterpret_cast<char*>(&ver_minor), 1);

    uint32_t header_len = 0;
    if (ver_major == 1) {
        uint16_t len16 = 0;
        ifs.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        ifs.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    ifs.read(header.data(), header.size());

    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        return false;
    }

    const auto shape_pos = header.find("shape");
    if (shape_pos == std::string::npos) return false;
    const auto lparen = header.find('(', shape_pos);
    const auto rparen = header.find(')', lparen);
    if (lparen == std::string::npos || rparen == std::string::npos) return false;

    std::string dims = header.substr(lparen + 1, rparen - lparen - 1);
    std::stringstream ss(dims);
    std::string token;
    size_t total = 1;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) continue;
        const size_t dim = static_cast<size_t>(std::stoul(token));
        shape.push_back(dim);
        total *= dim;
    }

    if (shape.empty()) return false;
    values.resize(total);
    ifs.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(total * sizeof(float)));
    return ifs.good() || ifs.eof();
}

static float float_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static float get_alpha_cumprod_exact(int timestep) {
    switch (timestep) {
        case 0: return float_from_bits(0x3f7fc84b);   // 0.9991499781608582
        case 259: return float_from_bits(0x3f28b29a); // 0.6589752435684204
        case 499: return float_from_bits(0x3e8e2ab0); // 0.27766942977905273
        case 759: return float_from_bits(0x3d55dd2f); // 0.05221289023756981
        case 999: return float_from_bits(0x3b98b3b6); // 0.00466009508818388
        default: break;
    }
    return std::numeric_limits<float>::quiet_NaN();
}

constexpr int kMersenneStateN = 624;
constexpr int kMersenneStateM = 397;
constexpr uint32_t kMersenneMatrixA = 0x9908b0dfu;
constexpr uint32_t kMersenneUMask = 0x80000000u;
constexpr uint32_t kMersenneLMask = 0x7fffffffu;

class TorchMt19937 {
public:
    explicit TorchMt19937(uint64_t seed = 5489) {
        init(seed);
    }

    uint32_t random() {
        if (--left_ == 0) {
            next_state();
        }
        uint32_t y = state_[next_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

private:
    void init(uint64_t seed) {
        state_[0] = static_cast<uint32_t>(seed & 0xffffffffu);
        for (int j = 1; j < kMersenneStateN; ++j) {
            state_[j] = 1812433253u * (state_[j - 1] ^ (state_[j - 1] >> 30)) + static_cast<uint32_t>(j);
        }
        left_ = 1;
        next_ = 0;
    }

    static uint32_t mix_bits(uint32_t u, uint32_t v) {
        return (u & kMersenneUMask) | (v & kMersenneLMask);
    }

    static uint32_t twist(uint32_t u, uint32_t v) {
        return (mix_bits(u, v) >> 1) ^ ((v & 1u) ? kMersenneMatrixA : 0u);
    }

    void next_state() {
        uint32_t* p = state_.data();
        left_ = kMersenneStateN;
        next_ = 0;

        for (int j = kMersenneStateN - kMersenneStateM + 1; --j; ++p) {
            *p = p[kMersenneStateM] ^ twist(p[0], p[1]);
        }
        for (int j = kMersenneStateM; --j; ++p) {
            *p = p[kMersenneStateM - kMersenneStateN] ^ twist(p[0], p[1]);
        }
        *p = p[kMersenneStateM - kMersenneStateN] ^ twist(p[0], state_[0]);
    }

    std::array<uint32_t, kMersenneStateN> state_{};
    int left_ = 1;
    uint32_t next_ = 0;
};

class TorchNormalRng {
public:
    explicit TorchNormalRng(uint64_t seed) : engine_(seed) {}

    float next_normal_scalar() {
        if (cached_sample_.has_value()) {
            const float value = *cached_sample_;
            cached_sample_.reset();
            return value;
        }

        const float u1 = uniform_real_f32();
        const float u2 = uniform_real_f32();
        const float radius = std::sqrt(-2.0f * std::log1pf(-u2));
        const float theta = static_cast<float>(2.0 * kPi) * u1;
        cached_sample_ = radius * std::sin(theta);
        return radius * std::cos(theta);
    }

    void fill_randn(std::vector<float>& values, bool quantize_fp16 = false) {
        if (quantize_fp16) {
            fill_randn_half(values);
            return;
        }

        const size_t size = values.size();
        if (size >= 16) {
            for (float& value : values) value = uniform_real_f32();

            for (size_t i = 0; i + 15 < size; i += 16) {
                box_muller_block16(values.data() + i);
            }

            if (size % 16 != 0) {
                const size_t offset = size - 16;
                for (size_t j = 0; j < 16; ++j) values[offset + j] = uniform_real_f32();
                box_muller_block16(values.data() + offset);
            }
        } else {
            for (float& value : values) value = next_normal_scalar();
        }

    }

private:
    float uniform_real_f32() {
        constexpr uint32_t mask = (1u << std::numeric_limits<float>::digits) - 1u;
        constexpr float divisor = 1.0f / static_cast<float>(1u << std::numeric_limits<float>::digits);
        return static_cast<float>(engine_.random() & mask) * divisor;
    }

    float uniform_real_f16() {
        constexpr uint32_t mask = (1u << 11) - 1u;
        constexpr float divisor = 1.0f / static_cast<float>(1u << 11);
        return fp16_to_fp32(fp32_to_fp16_rne(static_cast<float>(engine_.random() & mask) * divisor));
    }

    void fill_randn_half(std::vector<float>& values) {
        const size_t size = values.size();
        if (size >= 16) {
            for (float& value : values) value = uniform_real_f16();

            for (size_t i = 0; i + 15 < size; i += 16) {
                box_muller_half_block16(values.data() + i);
            }

            if (size % 16 != 0) {
                const size_t offset = size - 16;
                for (size_t j = 0; j < 16; ++j) values[offset + j] = uniform_real_f16();
                box_muller_half_block16(values.data() + offset);
            }
        } else {
            for (float& value : values) value = fp16_to_fp32(fp32_to_fp16_rne(next_normal_scalar()));
        }
    }

    static void box_muller_block16(float* data) {
        for (size_t j = 0; j < 8; ++j) {
            const float u1 = 1.0f - data[j];
            const float u2 = data[j + 8];
            const float radius = std::sqrt(-2.0f * std::log(u1));
            const float theta = static_cast<float>(2.0f * kPi * static_cast<double>(u2));
            data[j] = std::fma(radius * std::cos(theta), 1.0f, 0.0f);
            data[j + 8] = std::fma(radius * std::sin(theta), 1.0f, 0.0f);
        }
    }

    static void box_muller_half_block16(float* data) {
        for (size_t j = 0; j < 8; ++j) {
            const float u1 = 1.0f - data[j];
            const float u2 = data[j + 8];
            const float radius = fp16_to_fp32(fp32_to_fp16_rne(std::sqrt(-2.0f * std::log(u1))));
            const float theta = fp16_to_fp32(fp32_to_fp16_rne(static_cast<float>(2.0f * kPi * static_cast<double>(u2))));
            data[j] = fp16_to_fp32(fp32_to_fp16_rne(radius * std::cos(theta)));
            data[j + 8] = fp16_to_fp32(fp32_to_fp16_rne(radius * std::sin(theta)));
        }
    }

    TorchMt19937 engine_;
    std::optional<float> cached_sample_;
};

class ClipBPETokenizer {
public:
    bool load(const std::filesystem::path& tokenizer_dir, std::string& err) {
        const auto vocab_path = tokenizer_dir / "vocab.json";
        const auto merges_path = tokenizer_dir / "merges.txt";
        if (!std::filesystem::exists(vocab_path) || !std::filesystem::exists(merges_path)) {
            err = "clip tokenizer vocab.json or merges.txt missing under " + tokenizer_dir.string();
            return false;
        }

        std::ifstream vocab_ifs(vocab_path);
        if (!vocab_ifs.is_open()) {
            err = "failed to open " + vocab_path.string();
            return false;
        }
        nlohmann::json vocab_json;
        vocab_ifs >> vocab_json;
        if (!vocab_json.is_object()) {
            err = "clip tokenizer vocab.json is not an object";
            return false;
        }

        encoder_.clear();
        size_t max_id = 0;
        for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
            const int id = it.value().get<int>();
            encoder_[it.key()] = id;
            max_id = std::max(max_id, static_cast<size_t>(id));
        }
        decoder_.assign(max_id + 1, "");
        for (const auto& kv : encoder_) {
            decoder_[static_cast<size_t>(kv.second)] = kv.first;
        }

        std::ifstream merges_ifs(merges_path);
        if (!merges_ifs.is_open()) {
            err = "failed to open " + merges_path.string();
            return false;
        }
        bpe_ranks_.clear();
        std::string line;
        int rank = 0;
        while (std::getline(merges_ifs, line)) {
            line = trim_copy(line);
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string left;
            std::string right;
            ss >> left >> right;
            if (left.empty() || right.empty()) continue;
            bpe_ranks_[{utf8_to_wstring(left), utf8_to_wstring(right)}] = rank++;
        }

        const auto config_path = tokenizer_dir / "tokenizer_config.json";
        do_lower_case_ = true;
        if (std::filesystem::exists(config_path)) {
            std::ifstream cfg_ifs(config_path);
            if (cfg_ifs.is_open()) {
                nlohmann::json cfg_json;
                cfg_ifs >> cfg_json;
                if (cfg_json.contains("do_lower_case") && cfg_json["do_lower_case"].is_boolean()) {
                    do_lower_case_ = cfg_json["do_lower_case"].get<bool>();
                }
            }
        }

        build_byte_maps();
        return true;
    }

    std::vector<int> encode(const std::string& text) const {
        const std::wstring normalized = normalize_text(text);
        std::vector<std::wstring> tokens = split_tokens(normalized);
        std::vector<int> ids;
        ids.reserve(tokens.size() * 2);
        for (const auto& token : tokens) {
            append_bpe_ids(token, ids);
        }
        return ids;
    }

private:
    struct PairHash {
        size_t operator()(const std::pair<std::wstring, std::wstring>& value) const {
            return std::hash<std::wstring>{}(value.first) ^ (std::hash<std::wstring>{}(value.second) << 1);
        }
    };

    static std::wstring utf8_to_wstring(const std::string& text) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        return conv.from_bytes(text);
    }

    static std::string wstring_to_utf8(const std::wstring& text) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        return conv.to_bytes(text);
    }

    static bool is_letter(wchar_t ch) {
        return std::iswalpha(ch) != 0;
    }

    static bool is_digit(wchar_t ch) {
        return std::iswdigit(ch) != 0;
    }

    static bool is_space(wchar_t ch) {
        return std::iswspace(ch) != 0;
    }

    static bool match_contraction(const std::wstring& text, size_t pos, size_t& len) {
        static const std::wstring kContractions[] = {
            L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d",
        };
        for (const auto& item : kContractions) {
            if (pos + item.size() <= text.size() && text.compare(pos, item.size(), item) == 0) {
                len = item.size();
                return true;
            }
        }
        return false;
    }

    void build_byte_maps() {
        for (size_t i = 0; i < b2u_.size(); ++i) {
            b2u_[i] = 0;
        }
        auto fill_range = [&](int start, int end) {
            for (int c = start; c <= end; ++c) {
                b2u_[static_cast<size_t>(static_cast<uint8_t>(c))] = static_cast<wchar_t>(c);
            }
        };
        fill_range('!', '~');
        fill_range(0xA1, 0xAC);
        fill_range(0xAE, 0xFF);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (b2u_[static_cast<size_t>(b)] == 0) {
                b2u_[static_cast<size_t>(b)] = static_cast<wchar_t>(256 + n);
                ++n;
            }
        }
    }

    std::wstring normalize_text(const std::string& text) const {
        std::wstring ws = utf8_to_wstring(text);
        if (do_lower_case_) {
            std::transform(ws.begin(), ws.end(), ws.begin(), [](wchar_t ch) {
                return static_cast<wchar_t>(std::towlower(ch));
            });
        }

        std::wstring out;
        out.reserve(ws.size());
        bool last_was_space = true;
        for (wchar_t ch : ws) {
            if (is_space(ch)) {
                if (!last_was_space) out.push_back(L' ');
                last_was_space = true;
            } else {
                out.push_back(ch);
                last_was_space = false;
            }
        }
        if (!out.empty() && out.back() == L' ') out.pop_back();
        return out;
    }

    std::vector<std::wstring> split_tokens(const std::wstring& text) const {
        std::vector<std::wstring> tokens;
        size_t i = 0;
        while (i < text.size()) {
            if (is_space(text[i])) {
                ++i;
                continue;
            }

            size_t contraction_len = 0;
            if (text[i] == L'\'' && match_contraction(text, i, contraction_len)) {
                tokens.push_back(text.substr(i, contraction_len));
                i += contraction_len;
                continue;
            }

            if (is_letter(text[i])) {
                size_t j = i + 1;
                while (j < text.size() && is_letter(text[j])) ++j;
                tokens.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }

            if (is_digit(text[i])) {
                tokens.push_back(text.substr(i, 1));
                ++i;
                continue;
            }

            size_t j = i + 1;
            while (j < text.size() && !is_space(text[j]) && !is_letter(text[j]) && !is_digit(text[j])) {
                if (text[j] == L'\'') {
                    size_t len = 0;
                    if (match_contraction(text, j, len)) break;
                }
                ++j;
            }
            tokens.push_back(text.substr(i, j - i));
            i = j;
        }
        return tokens;
    }

    std::wstring byte_encode_token(const std::wstring& token) const {
        const std::string utf8 = wstring_to_utf8(token);
        std::wstring out;
        out.reserve(utf8.size());
        for (unsigned char c : utf8) {
            out.push_back(b2u_[static_cast<size_t>(c)]);
        }
        return out;
    }

    void append_bpe_ids(const std::wstring& token, std::vector<int>& ids) const {
        if (token.empty()) return;

        std::wstring encoded = byte_encode_token(token);
        if (encoded.empty()) return;

        std::vector<std::wstring> word;
        word.reserve(encoded.size());
        for (size_t i = 0; i + 1 < encoded.size(); ++i) {
            word.emplace_back(1, encoded[i]);
        }
        word.push_back(std::wstring(1, encoded.back()) + L"</w>");

        while (word.size() > 1) {
            auto best_it = bpe_ranks_.end();
            std::pair<std::wstring, std::wstring> best_pair;
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                const auto it = bpe_ranks_.find({word[i], word[i + 1]});
                if (it != bpe_ranks_.end() && (best_it == bpe_ranks_.end() || it->second < best_it->second)) {
                    best_it = it;
                    best_pair = it->first;
                }
            }
            if (best_it == bpe_ranks_.end()) break;

            std::vector<std::wstring> merged;
            merged.reserve(word.size());
            for (size_t i = 0; i < word.size();) {
                if (i + 1 < word.size() && word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                    merged.push_back(word[i] + word[i + 1]);
                    i += 2;
                } else {
                    merged.push_back(word[i]);
                    ++i;
                }
            }
            word.swap(merged);
        }

        for (const auto& piece : word) {
            const std::string utf8_piece = wstring_to_utf8(piece);
            const auto it = encoder_.find(utf8_piece);
            ids.push_back(it != encoder_.end() ? it->second : kEosTokenId);
        }
    }

    bool do_lower_case_ = true;
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
    std::unordered_map<std::pair<std::wstring, std::wstring>, int, PairHash> bpe_ranks_;
    std::array<wchar_t, 256> b2u_{};
};

static std::vector<int> encode_clip_prompt(ClipBPETokenizer& tokenizer, const std::string& prompt) {
    std::vector<int> ids;
    ids.reserve(kPromptMaxLength);
    ids.push_back(kBosTokenId);

    std::vector<int> body = tokenizer.encode(prompt);
    for (int id : body) {
        if (static_cast<int>(ids.size()) >= kPromptMaxLength - 1) break;
        ids.push_back(id);
    }
    ids.push_back(kEosTokenId);
    while (static_cast<int>(ids.size()) < kPromptMaxLength) {
        ids.push_back(kEosTokenId);
    }
    return ids;
}

static axcv::Mat decode_image_from_bytes(const std::vector<uint8_t>& bytes) {
#if defined(AXLLM_USE_OPENCV)
    if (bytes.empty()) return axcv::Mat();
    cv::Mat encoded(1, static_cast<int>(bytes.size()), CV_8UC1, const_cast<uint8_t*>(bytes.data()));
    return cv::imdecode(encoded, cv::IMREAD_COLOR);
#else
    return SimpleCV::imdecode(bytes, SimpleCV::ColorSpace::BGR);
#endif
}

static void convert_image_to_nchw_float(const axcv::Mat& image,
                                        int target_w,
                                        int target_h,
                                        std::vector<float>& out) {
    axcv::Mat resized;
    if (axcv::width(image) != target_w || axcv::height(image) != target_h) {
        axcv::resize(image, resized, target_w, target_h);
    } else {
        resized = image;
    }

    axcv::Mat rgb;
    axcv::cvtColorBGR2RGB(resized, rgb);

    out.assign(static_cast<size_t>(3 * target_h * target_w), 0.0f);
    for (int y = 0; y < target_h; ++y) {
        const uint8_t* row = axcv::row_ptr(rgb, y);
        for (int x = 0; x < target_w; ++x) {
            const uint8_t* px = row + x * 3;
            out[static_cast<size_t>(0 * target_h * target_w + y * target_w + x)] = (static_cast<float>(px[0]) / 255.0f) * 2.0f - 1.0f;
            out[static_cast<size_t>(1 * target_h * target_w + y * target_w + x)] = (static_cast<float>(px[1]) / 255.0f) * 2.0f - 1.0f;
            out[static_cast<size_t>(2 * target_h * target_w + y * target_w + x)] = (static_cast<float>(px[2]) / 255.0f) * 2.0f - 1.0f;
        }
    }
}

static bool encode_png_rgb(const std::vector<uint8_t>& rgb,
                           int width,
                           int height,
                           std::vector<uint8_t>& out_png) {
    if (rgb.size() != static_cast<size_t>(width * height * 3)) return false;
#if defined(AXLLM_USE_OPENCV)
    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
    cv::Mat bgr;
    cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
    return cv::imencode(".png", bgr, out_png);
#else
    SimpleCV::Mat image(height, width, 3);
    std::memcpy(image.data, rgb.data(), rgb.size());
    return SimpleCV::imencode(image, out_png);
#endif
}

class TokenizerExporter {
public:
    static bool ensure_tokenizer_txt(const std::filesystem::path& tokenizer_dir,
                                     const std::filesystem::path& dst_path,
                                     std::string& err) {
        if (std::filesystem::exists(dst_path)) return true;

        const auto vocab_path = tokenizer_dir / "vocab.json";
        const auto merges_path = tokenizer_dir / "merges.txt";
        if (!std::filesystem::exists(vocab_path) || !std::filesystem::exists(merges_path)) {
            err = "tokenizer vocab.json or merges.txt missing under " + tokenizer_dir.string();
            return false;
        }

        std::ifstream vocab_ifs(vocab_path);
        if (!vocab_ifs.is_open()) {
            err = "failed to open " + vocab_path.string();
            return false;
        }
        nlohmann::json vocab_json;
        vocab_ifs >> vocab_json;
        if (!vocab_json.is_object()) {
            err = "tokenizer vocab.json is not an object";
            return false;
        }

        size_t max_id = 0;
        for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
            max_id = std::max(max_id, static_cast<size_t>(it.value().get<int>()));
        }
        std::vector<std::string> vocab(max_id + 1, "<unk>");
        for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
            vocab[static_cast<size_t>(it.value().get<int>())] = it.key();
        }

        std::vector<std::string> merge_lines;
        {
            std::ifstream merges_ifs(merges_path);
            if (!merges_ifs.is_open()) {
                err = "failed to open " + merges_path.string();
                return false;
            }
            std::string line;
            while (std::getline(merges_ifs, line)) {
                line = trim_copy(line);
                if (line.empty() || line[0] == '#') continue;
                merge_lines.push_back(line);
            }
        }

        std::filesystem::create_directories(dst_path.parent_path());
        std::ofstream ofs(dst_path);
        if (!ofs.is_open()) {
            err = "failed to create " + dst_path.string();
            return false;
        }

        ofs << "430 3\n";
        ofs << "0 0 0\n";
        ofs << "\n";
        ofs << vocab.size() << " " << merge_lines.size() << "\n";
        for (const auto& token : vocab) {
            ofs << "b64:" << base64_encode(token) << "\n";
        }
        for (const auto& line : merge_lines) {
            std::stringstream ls(line);
            std::string lhs;
            std::string rhs;
            ls >> lhs >> rhs;
            if (lhs.empty() || rhs.empty()) {
                err = "invalid BPE merge line in " + merges_path.string() + ": " + line;
                return false;
            }
            ofs << "b64:" << base64_encode(lhs) << " b64:" << base64_encode(rhs) << "\n";
        }
        return true;
    }
};

struct VariantRuntime {
    ImageModelVariant meta;
    std::unique_ptr<ClipBPETokenizer> clip_tokenizer;
    TensorFloatType default_two_byte_type = TensorFloatType::FP16;
    TensorFloatType text_encoder_output_type = TensorFloatType::FP32;
    TensorFloatType unet_input_sample_type = TensorFloatType::FP16;
    TensorFloatType unet_input_time_type = TensorFloatType::FP16;
    TensorFloatType unet_input_hidden_type = TensorFloatType::FP16;
    TensorFloatType unet_output_type = TensorFloatType::FP16;
    TensorFloatType vae_decoder_input_type = TensorFloatType::FP16;
    TensorFloatType vae_decoder_output_type = TensorFloatType::FP16;
    TensorFloatType vae_encoder_input_type = TensorFloatType::FP16;
    TensorFloatType vae_encoder_output_type = TensorFloatType::FP16;

    std::unique_ptr<sd15_runner_t> text_encoder;
    std::unique_ptr<sd15_runner_t> unet;
    std::unique_ptr<sd15_runner_t> vae_decoder;
    std::unique_ptr<sd15_runner_t> vae_encoder;

    std::vector<float> txt2img_time_inputs;
    std::vector<size_t> txt2img_time_shape;
    std::vector<float> img2img_time_inputs;
    std::vector<size_t> img2img_time_shape;

};

static bool infer_tensor_float_type(const ax_runner_tensor_t& tensor,
                                    size_t expected_float_elems,
                                    TensorFloatType two_byte_type,
                                    TensorFloatType& out_type,
                                    std::string& err,
                                    const char* tensor_name);

static bool init_variant_runtime(const std::string& base_model_name,
                                 const std::filesystem::path& dir,
                                 const char* fallback_openai_size,
                                 std::unique_ptr<VariantRuntime>& runtime,
                                 std::string& err) {
    runtime = std::make_unique<VariantRuntime>();
    runtime->meta.model_dir = dir.string();
    runtime->meta.supports_img2img = std::filesystem::exists(dir / "vae_encoder.axmodel");
    runtime->default_two_byte_type = parse_tensor_float_type(dir / "text_encoder" / "config.json");

    runtime->clip_tokenizer = std::make_unique<ClipBPETokenizer>();
    if (!runtime->clip_tokenizer || !runtime->clip_tokenizer->load(dir / "tokenizer", err)) {
        err = "failed to load clip tokenizer: " + err;
        return false;
    }

    runtime->text_encoder = std::make_unique<sd15_runner_t>();
    runtime->unet = std::make_unique<sd15_runner_t>();
    runtime->vae_decoder = std::make_unique<sd15_runner_t>();

    int init_ret = runtime->text_encoder->init((dir / "text_encoder" / "sd15_text_encoder_sim.axmodel").string().c_str(), -1);
    if (init_ret != 0) {
        err = "failed to init text encoder ret=" + ret_hex_string(init_ret);
        return false;
    }
    init_ret = runtime->unet->init((dir / "unet.axmodel").string().c_str(), -1);
    if (init_ret != 0) {
        err = "failed to init unet ret=" + ret_hex_string(init_ret);
        return false;
    }
    init_ret = runtime->vae_decoder->init((dir / "vae_decoder.axmodel").string().c_str(), -1);
    if (init_ret != 0) {
        err = "failed to init vae_decoder ret=" + ret_hex_string(init_ret);
        return false;
    }
    if (runtime->meta.supports_img2img) {
        runtime->vae_encoder = std::make_unique<sd15_runner_t>();
        init_ret = runtime->vae_encoder->init((dir / "vae_encoder.axmodel").string().c_str(), -1);
        if (init_ret != 0) {
            err = "failed to init vae_encoder ret=" + ret_hex_string(init_ret);
            return false;
        }
    }

    runtime->text_encoder->set_auto_sync_before_inference(true);
    runtime->text_encoder->set_auto_sync_after_inference(true);
    runtime->unet->set_auto_sync_before_inference(true);
    runtime->unet->set_auto_sync_after_inference(true);
    runtime->vae_decoder->set_auto_sync_before_inference(true);
    runtime->vae_decoder->set_auto_sync_after_inference(true);
    if (runtime->vae_encoder) {
        runtime->vae_encoder->set_auto_sync_before_inference(true);
        runtime->vae_encoder->set_auto_sync_after_inference(true);
    }

    if (!load_npy_float32(dir / "time_input_txt2img.npy", runtime->txt2img_time_inputs, runtime->txt2img_time_shape)) {
        err = "failed to load time_input_txt2img.npy";
        return false;
    }
    if (runtime->meta.supports_img2img &&
        !load_npy_float32(dir / "time_input_img2img.npy", runtime->img2img_time_inputs, runtime->img2img_time_shape)) {
        err = "failed to load time_input_img2img.npy";
        return false;
    }

    int inferred_width = 0;
    int inferred_height = 0;
    const auto& vae_out_shape = runtime->vae_decoder->get_output(0).vShape;
    if (vae_out_shape.size() >= 4 &&
        vae_out_shape[1] == 3 &&
        vae_out_shape[2] > 0 &&
        vae_out_shape[3] > 0) {
        inferred_height = vae_out_shape[2];
        inferred_width = vae_out_shape[3];
    } else if (!parse_openai_image_size(fallback_openai_size, inferred_width, inferred_height)) {
        err = "unable to infer image size";
        return false;
    }
    runtime->meta.width = inferred_width;
    runtime->meta.height = inferred_height;
    runtime->meta.openai_size = std::to_string(inferred_width) + "x" + std::to_string(inferred_height);
    runtime->meta.model_id = make_model_id(base_model_name, runtime->meta.openai_size);

    const auto& te_out = runtime->text_encoder->get_output(0);
    if (!infer_tensor_float_type(te_out,
                                 77u * 768u,
                                 runtime->default_two_byte_type,
                                 runtime->text_encoder_output_type,
                                 err,
                                 "text encoder output")) {
        return false;
    }

    const auto& unet_sample = runtime->unet->get_input("sample");
    const auto& unet_time = runtime->unet->get_input(kTimeEmbedKey);
    const auto& unet_hidden = runtime->unet->get_input("encoder_hidden_states");
    const auto& unet_out = runtime->unet->get_output(0);
    const size_t latent_elems = static_cast<size_t>(4 * (runtime->meta.height / 8) * (runtime->meta.width / 8));
    if (!infer_tensor_float_type(unet_sample,
                                 latent_elems,
                                 runtime->default_two_byte_type,
                                 runtime->unet_input_sample_type,
                                 err,
                                 "unet sample") ||
        (!runtime->txt2img_time_shape.empty() &&
         !infer_tensor_float_type(unet_time,
                                  runtime->txt2img_time_shape.back(),
                                  runtime->default_two_byte_type,
                                  runtime->unet_input_time_type,
                                  err,
                                  "unet time input")) ||
        !infer_tensor_float_type(unet_hidden,
                                 77u * 768u,
                                 runtime->default_two_byte_type,
                                 runtime->unet_input_hidden_type,
                                 err,
                                 "unet encoder hidden states") ||
        !infer_tensor_float_type(unet_out,
                                 latent_elems,
                                 runtime->default_two_byte_type,
                                 runtime->unet_output_type,
                                 err,
                                 "unet output")) {
        return false;
    }

    const auto& vae_dec_in = runtime->vae_decoder->get_input(0);
    const auto& vae_dec_out = runtime->vae_decoder->get_output(0);
    if (!infer_tensor_float_type(vae_dec_in,
                                 latent_elems,
                                 runtime->default_two_byte_type,
                                 runtime->vae_decoder_input_type,
                                 err,
                                 "vae decoder input") ||
        !infer_tensor_float_type(vae_dec_out,
                                 static_cast<size_t>(3 * runtime->meta.height * runtime->meta.width),
                                 runtime->default_two_byte_type,
                                 runtime->vae_decoder_output_type,
                                 err,
                                 "vae decoder output")) {
        return false;
    }

    if (runtime->vae_encoder) {
        const auto& vae_enc_in = runtime->vae_encoder->get_input(0);
        const auto& vae_enc_out = runtime->vae_encoder->get_output(0);
        if (!infer_tensor_float_type(vae_enc_in,
                                     static_cast<size_t>(3 * runtime->meta.height * runtime->meta.width),
                                     runtime->default_two_byte_type,
                                     runtime->vae_encoder_input_type,
                                     err,
                                     "vae encoder input") ||
            !infer_tensor_float_type(vae_enc_out,
                                     latent_elems * 2,
                                     runtime->default_two_byte_type,
                                     runtime->vae_encoder_output_type,
                                     err,
                                     "vae encoder output")) {
            return false;
        }
    }

    ALOGI("sd15 variant %s tensor dtypes: text_out=%s unet[in_sample=%s in_time=%s in_hidden=%s out=%s] vae[in=%s out=%s]%s",
          runtime->meta.model_id.c_str(),
          tensor_float_type_name(runtime->text_encoder_output_type),
          tensor_float_type_name(runtime->unet_input_sample_type),
          tensor_float_type_name(runtime->unet_input_time_type),
          tensor_float_type_name(runtime->unet_input_hidden_type),
          tensor_float_type_name(runtime->unet_output_type),
          tensor_float_type_name(runtime->vae_decoder_input_type),
          tensor_float_type_name(runtime->vae_decoder_output_type),
          runtime->vae_encoder ? " vae_enc=enabled" : "");

    return true;
}

static bool tensor_matches_float_type(const ax_runner_tensor_t& tensor,
                                      size_t expected_float_elems,
                                      TensorFloatType type) {
    return expected_float_elems > 0 &&
           static_cast<size_t>(tensor.nSize) == expected_float_elems * tensor_float_type_bytes(type);
}

static bool infer_tensor_float_type(const ax_runner_tensor_t& tensor,
                                    size_t expected_float_elems,
                                    TensorFloatType two_byte_type,
                                    TensorFloatType& out_type,
                                    std::string& err,
                                    const char* tensor_name) {
    if (expected_float_elems == 0) {
        err = std::string("invalid expected element count for ") + tensor_name;
        return false;
    }

    const size_t fp32_bytes = expected_float_elems * tensor_float_type_bytes(TensorFloatType::FP32);
    const size_t two_byte_bytes = expected_float_elems * tensor_float_type_bytes(two_byte_type);
    if (static_cast<size_t>(tensor.nSize) == fp32_bytes) {
        out_type = TensorFloatType::FP32;
        return true;
    }
    if (static_cast<size_t>(tensor.nSize) == two_byte_bytes) {
        out_type = two_byte_type;
        return true;
    }

    err = std::string("unexpected ") + tensor_name + " size " + std::to_string(tensor.nSize) +
          ", expected " + std::to_string(fp32_bytes) + " (fp32) or " +
          std::to_string(two_byte_bytes) + " (" + tensor_float_type_name(two_byte_type) + ")";
    return false;
}

static bool copy_fp32_to_tensor(const std::vector<float>& src,
                                const ax_runner_tensor_t& tensor,
                                TensorFloatType tensor_type,
                                std::string& err) {
    const size_t elem_count = src.size();
    const size_t expected_bytes = elem_count * tensor_float_type_bytes(tensor_type);
    if (static_cast<size_t>(tensor.nSize) != expected_bytes) {
        err = "tensor size mismatch while writing fp32 data";
        return false;
    }
    switch (tensor_type) {
        case TensorFloatType::FP32:
            std::memcpy(tensor.pVirAddr, src.data(), expected_bytes);
            break;
        case TensorFloatType::FP16: {
            auto* dst = reinterpret_cast<uint16_t*>(tensor.pVirAddr);
            for (size_t i = 0; i < elem_count; ++i) dst[i] = fp32_to_fp16_rne(src[i]);
            break;
        }
        case TensorFloatType::BF16: {
            auto* dst = reinterpret_cast<uint16_t*>(tensor.pVirAddr);
            for (size_t i = 0; i < elem_count; ++i) dst[i] = fp32_to_bfloat16_rne(src[i]);
            break;
        }
    }
    return true;
}

static bool copy_tensor_to_fp32(const ax_runner_tensor_t& tensor,
                                TensorFloatType tensor_type,
                                std::vector<float>& dst,
                                std::string& err) {
    const size_t elem_count = static_cast<size_t>(tensor.nSize) / tensor_float_type_bytes(tensor_type);
    if (elem_count == 0) {
        err = "tensor has zero elements";
        return false;
    }
    dst.resize(elem_count);
    switch (tensor_type) {
        case TensorFloatType::FP32:
            std::memcpy(dst.data(), tensor.pVirAddr, elem_count * sizeof(float));
            break;
        case TensorFloatType::FP16: {
            const auto* src = reinterpret_cast<const uint16_t*>(tensor.pVirAddr);
            for (size_t i = 0; i < elem_count; ++i) dst[i] = fp16_to_fp32(src[i]);
            break;
        }
        case TensorFloatType::BF16: {
            const auto* src = reinterpret_cast<const uint16_t*>(tensor.pVirAddr);
            for (size_t i = 0; i < elem_count; ++i) dst[i] = bfloat16(src[i]).fp32();
            break;
        }
    }
    return true;
}

class Sd15ImageGenerator final : public ImageGenerator {
public:
    bool init(const std::string& model_root, std::string& err) override {
        variants_.clear();
        variant_dirs_.clear();
        variant_fallback_sizes_.clear();
        active_runtime_.reset();
        active_model_id_.clear();

        std::filesystem::path root(model_root);
        if (!std::filesystem::exists(root)) {
            err = "image model root not found: " + model_root;
            return false;
        }

        const auto config_path = root / "config.json";
        std::string base_model_name = "lcm-lora-sdv1-5";
        {
            if (std::filesystem::exists(config_path)) {
                std::ifstream ifs(config_path);
                if (ifs.is_open()) {
                    nlohmann::json j;
                    ifs >> j;
                    if (j.contains("model_name") && j["model_name"].is_string()) {
                        base_model_name = j["model_name"].get<std::string>();
                    }
                }
            }
        }

        base_model_name_ = base_model_name;

        std::vector<VariantCandidate> candidates;
        if (!parse_variant_candidates_from_config(config_path, candidates, err)) {
            return false;
        }
        if (candidates.empty()) {
            candidates.assign(std::begin(kVariantCandidates), std::end(kVariantCandidates));
        }

        for (const auto& candidate : candidates) {
            const auto dir = root / candidate.dir_name;
            if (!std::filesystem::exists(dir)) continue;

            const auto text_encoder_path = dir / "text_encoder" / "sd15_text_encoder_sim.axmodel";
            const auto unet_path = dir / "unet.axmodel";
            const auto vae_decoder_path = dir / "vae_decoder.axmodel";
            if (!std::filesystem::exists(text_encoder_path) ||
                !std::filesystem::exists(unet_path) ||
                !std::filesystem::exists(vae_decoder_path)) {
                ALOGW("skip image variant %s: required model files are missing", candidate.dir_name.c_str());
                continue;
            }

            int width = 0;
            int height = 0;
            if (!parse_openai_image_size(candidate.fallback_openai_size, width, height)) {
                ALOGW("skip image variant %s: invalid fallback size %s",
                      candidate.dir_name.c_str(),
                      candidate.fallback_openai_size.c_str());
                continue;
            }

            ImageModelVariant meta;
            meta.model_dir = dir.string();
            meta.openai_size = candidate.fallback_openai_size;
            meta.width = width;
            meta.height = height;
            meta.supports_img2img = std::filesystem::exists(dir / "vae_encoder.axmodel");
            meta.model_id = make_model_id(base_model_name_, meta.openai_size);
            variants_.push_back(meta);
            variant_dirs_.emplace(meta.model_id, dir);
            variant_fallback_sizes_.emplace(meta.model_id, candidate.fallback_openai_size);
        }

        if (variants_.empty()) {
            err = "no sd1.5 axmodel variants found under " + model_root;
            return false;
        }

        std::sort(variants_.begin(), variants_.end(), [](const auto& a, const auto& b) {
            if (a.width != b.width) return a.width < b.width;
            return a.height < b.height;
        });
        return true;
    }

    std::vector<ImageModelVariant> variants() const override {
        return variants_;
    }

    bool generate(const openai_api::ImageGenRequest& req,
                  std::vector<std::vector<uint8_t>>& png_images,
                  std::string& err) override {
        png_images.clear();

        const std::string model_id = select_model_id(req, err);
        if (model_id.empty()) return false;

        VariantRuntime* runtime = ensure_runtime_loaded(model_id, err);
        if (!runtime) return false;

        const int image_count = std::max(1, req.n);
        png_images.reserve(static_cast<size_t>(image_count));

        for (int i = 0; i < image_count; ++i) {
            std::vector<uint8_t> one_png;
            if (!generate_single(*runtime, req, req.seed >= 0 ? req.seed + i : -1, one_png, err)) {
                return false;
            }
            png_images.push_back(std::move(one_png));
        }
        return true;
    }

private:
    std::string select_model_id(const openai_api::ImageGenRequest& req, std::string& err) {
        if (!req.model.empty()) {
            auto it = variant_dirs_.find(req.model);
            if (it != variant_dirs_.end()) return it->first;
        }

        const std::string normalized_size = normalize_openai_size(req.size);
        if (!normalized_size.empty()) {
            for (auto& variant : variants_) {
                if (variant.openai_size == normalized_size) {
                    return variant.model_id;
                }
            }
        }

        if (!variants_.empty()) {
            return variants_.front().model_id;
        }

        err = "no matching image model variant found";
        return {};
    }

    VariantRuntime* ensure_runtime_loaded(const std::string& model_id, std::string& err) {
        if (active_runtime_ && active_model_id_ == model_id) {
            return active_runtime_.get();
        }

        auto dir_it = variant_dirs_.find(model_id);
        auto fallback_it = variant_fallback_sizes_.find(model_id);
        if (dir_it == variant_dirs_.end() || fallback_it == variant_fallback_sizes_.end()) {
            err = "image model variant metadata is missing for " + model_id;
            return nullptr;
        }

        std::unique_ptr<VariantRuntime> runtime;
        if (!init_variant_runtime(base_model_name_, dir_it->second, fallback_it->second.c_str(), runtime, err)) {
            err = "failed to load image runtime for " + model_id + ": " + err;
            return nullptr;
        }

        active_model_id_ = model_id;
        active_runtime_ = std::move(runtime);
        return active_runtime_.get();
    }

    bool run_text_encoder(VariantRuntime& runtime,
                          const std::string& prompt,
                          std::vector<float>& prompt_embeds,
                          std::string& err) {
        if (!runtime.clip_tokenizer) {
            err = "clip tokenizer is not initialized";
            return false;
        }
        std::vector<int> input_ids = encode_clip_prompt(*runtime.clip_tokenizer, prompt);

        const auto& input = runtime.text_encoder->get_input("input_ids");
        if (input.nSize < static_cast<int>(input_ids.size() * sizeof(int32_t))) {
            err = "text encoder input buffer too small";
            return false;
        }

        std::memset(input.pVirAddr, 0, static_cast<size_t>(input.nSize));
        auto* dst = reinterpret_cast<int32_t*>(input.pVirAddr);
        for (size_t i = 0; i < input_ids.size(); ++i) {
            dst[i] = input_ids[i];
        }

        int ret = runtime.text_encoder->inference();
        if (ret != 0) {
            err = "text encoder inference failed: ret=" + ret_hex_string(ret);
            return false;
        }

        const auto& output = runtime.text_encoder->get_output(0);
        if (!copy_tensor_to_fp32(output, runtime.text_encoder_output_type, prompt_embeds, err)) {
            return false;
        }
        return true;
    }

    bool run_unet(VariantRuntime& runtime,
                  const std::vector<float>& latent,
                  const float* time_embed,
                  size_t time_embed_elems,
                  const std::vector<float>& prompt_embeds,
                  std::vector<float>& noise_pred,
                  std::string& err) {
        const auto& sample = runtime.unet->get_input("sample");
        const auto& time_input = runtime.unet->get_input(kTimeEmbedKey);
        const auto& hidden = runtime.unet->get_input("encoder_hidden_states");

        std::vector<float> time_embed_vec(time_embed, time_embed + time_embed_elems);
        if (!copy_fp32_to_tensor(latent, sample, runtime.unet_input_sample_type, err)) {
            err = "unet sample shape mismatch";
            return false;
        }
        if (!copy_fp32_to_tensor(time_embed_vec, time_input, runtime.unet_input_time_type, err)) {
            err = "unet time_input shape mismatch";
            return false;
        }
        if (!copy_fp32_to_tensor(prompt_embeds, hidden, runtime.unet_input_hidden_type, err)) {
            err = "unet encoder_hidden_states shape mismatch";
            return false;
        }

        int ret = runtime.unet->inference();
        if (ret != 0) {
            err = "unet inference failed: ret=" + ret_hex_string(ret);
            return false;
        }

        const auto& output = runtime.unet->get_output(0);
        if (!copy_tensor_to_fp32(output, runtime.unet_output_type, noise_pred, err)) {
            return false;
        }
        return true;
    }

    bool run_vae_decoder(VariantRuntime& runtime,
                         const std::vector<float>& latent,
                         std::vector<uint8_t>& png,
                         std::string& err) {
        const auto& input = runtime.vae_decoder->get_input(0);
        if (!copy_fp32_to_tensor(latent, input, runtime.vae_decoder_input_type, err)) {
            err = "vae decoder input shape mismatch";
            return false;
        }

        int ret = runtime.vae_decoder->inference();
        if (ret != 0) {
            err = "vae decoder inference failed: ret=" + ret_hex_string(ret);
            return false;
        }

        const auto& output = runtime.vae_decoder->get_output(0);
        std::vector<float> image;
        if (!copy_tensor_to_fp32(output, runtime.vae_decoder_output_type, image, err)) {
            err = "vae decoder output read failed";
            return false;
        }

        const int width = runtime.meta.width;
        const int height = runtime.meta.height;
        std::vector<uint8_t> rgb(static_cast<size_t>(width * height * 3), 0);
        const size_t plane = static_cast<size_t>(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t idx = static_cast<size_t>(y * width + x);
                for (int c = 0; c < 3; ++c) {
                    const float value = std::clamp(image[c * plane + idx] / 2.0f + 0.5f, 0.0f, 1.0f);
                    rgb[idx * 3 + c] = static_cast<uint8_t>(std::round(value * 255.0f));
                }
            }
        }

        if (!encode_png_rgb(rgb, width, height, png)) {
            err = "png encoding failed";
            return false;
        }
        return true;
    }

    bool run_vae_encoder(VariantRuntime& runtime,
                         const std::vector<uint8_t>& image_bytes,
                         std::vector<float>& latent,
                         TorchNormalRng& rng,
                         std::string& err) {
        if (!runtime.vae_encoder) {
            err = "img2img is not supported for this model variant";
            return false;
        }

        axcv::Mat image = decode_image_from_bytes(image_bytes);
        if (axcv::empty(image)) {
            err = "failed to decode init image";
            return false;
        }

        std::vector<float> nchw;
        convert_image_to_nchw_float(image, runtime.meta.width, runtime.meta.height, nchw);

        const auto& input = runtime.vae_encoder->get_input(0);
        if (!copy_fp32_to_tensor(nchw, input, runtime.vae_encoder_input_type, err)) {
            err = "vae encoder input shape mismatch";
            return false;
        }

        int ret = runtime.vae_encoder->inference();
        if (ret != 0) {
            err = "vae encoder inference failed: ret=" + ret_hex_string(ret);
            return false;
        }

        const auto& output = runtime.vae_encoder->get_output(0);
        std::vector<float> stats;
        if (!copy_tensor_to_fp32(output, runtime.vae_encoder_output_type, stats, err)) {
            err = "vae encoder output read failed";
            return false;
        }
        const size_t elems = stats.size();

        if (elems % 2 != 0) {
            err = "vae encoder output should contain mean/logvar pairs";
            return false;
        }

        const size_t half = elems / 2;
        latent.resize(half, 0.0f);
        std::vector<float> noise(half, 0.0f);
        rng.fill_randn(noise);
        for (size_t i = 0; i < half; ++i) {
            const float mean = stats[i];
            const float logvar = stats[half + i];
            const float stddev = std::exp(0.5f * logvar);
            latent[i] = (mean + stddev * noise[i]) * kLatentScale;
        }
        return true;
    }

    std::vector<float> make_latent_noise(size_t elems, TorchNormalRng& rng, bool quantize_fp16 = false) {
        std::vector<float> values(elems, 0.0f);
        rng.fill_randn(values, quantize_fp16);
        return values;
    }

    static void add_noise(std::vector<float>& sample,
                          const std::vector<float>& noise,
                          float alpha_prod_t) {
        const float sqrt_alpha = std::sqrt(alpha_prod_t);
        const float sqrt_one_minus_alpha = std::sqrt(1.0f - alpha_prod_t);
        for (size_t i = 0; i < sample.size(); ++i) {
            sample[i] = sqrt_alpha * sample[i] + sqrt_one_minus_alpha * noise[i];
        }
    }

    bool denoise_loop(VariantRuntime& runtime,
                      const openai_api::ImageGenRequest& req,
                      std::vector<float>& latent,
                      const std::vector<float>& prompt_embeds,
                      bool img2img,
                      TorchNormalRng& rng,
                      std::string& err) {
        const std::vector<float>& time_inputs = img2img ? runtime.img2img_time_inputs : runtime.txt2img_time_inputs;
        const size_t time_embed_size = img2img ? runtime.img2img_time_shape.back() : runtime.txt2img_time_shape.back();
        std::vector<int> timesteps;
        if (img2img) {
            timesteps.assign(kImg2ImgTimesteps.begin(), kImg2ImgTimesteps.end());
        } else {
            timesteps.assign(kTxt2ImgTimesteps.begin(), kTxt2ImgTimesteps.end());
        }

        if (time_inputs.size() < time_embed_size * timesteps.size()) {
            err = "time_input npy data is incomplete";
            return false;
        }

        const float final_alphas_cumprod = get_alpha_cumprod_exact(0);

        for (size_t i = 0; i < timesteps.size(); ++i) {
            std::vector<float> noise_pred;
            if (!run_unet(runtime,
                          latent,
                          time_inputs.data() + i * time_embed_size,
                          time_embed_size,
                          prompt_embeds,
                          noise_pred,
                          err)) {
                return false;
            }

            const int timestep = timesteps[i];
            int prev_timestep = timestep;
            if (img2img) {
                const int prev_step_index = kImg2ImgStepIndex[i] + 1;
                if (prev_step_index < static_cast<int>(kImg2ImgSelfTimesteps.size())) {
                    prev_timestep = kImg2ImgSelfTimesteps[prev_step_index];
                }
            } else if (i + 1 < timesteps.size()) {
                prev_timestep = timesteps[i + 1];
            }

            const float alpha_prod_t = get_alpha_cumprod_exact(timestep);
            const float alpha_prod_t_prev =
                prev_timestep >= 0 ? get_alpha_cumprod_exact(prev_timestep) : final_alphas_cumprod;
            if (!std::isfinite(alpha_prod_t) || !std::isfinite(alpha_prod_t_prev)) {
                err = "unsupported scheduler timestep encountered";
                return false;
            }
            const float beta_prod_t = 1.0f - alpha_prod_t;
            const float beta_prod_t_prev = 1.0f - alpha_prod_t_prev;

            const double scaled_timestep = static_cast<double>(timestep * 10);
            const float c_skip = static_cast<float>((0.5 * 0.5) / (scaled_timestep * scaled_timestep + 0.5 * 0.5));
            const float c_out = static_cast<float>(scaled_timestep / std::sqrt(scaled_timestep * scaled_timestep + 0.5 * 0.5));
            const float sqrt_alpha_prod_t = static_cast<float>(std::sqrt(static_cast<double>(alpha_prod_t)));
            const float sqrt_beta_prod_t = static_cast<float>(std::sqrt(static_cast<double>(beta_prod_t)));
            const float sqrt_alpha_prod_t_prev = static_cast<float>(std::sqrt(static_cast<double>(alpha_prod_t_prev)));
            const float sqrt_beta_prod_t_prev = static_cast<float>(std::sqrt(static_cast<double>(beta_prod_t_prev)));

            std::vector<float> denoised(latent.size(), 0.0f);
            for (size_t j = 0; j < latent.size(); ++j) {
                const volatile float noise_term = sqrt_beta_prod_t * noise_pred[j];
                const volatile float numerator = latent[j] - noise_term;
                const volatile float predicted_original = numerator / sqrt_alpha_prod_t;
                const volatile float denoised_main = c_out * predicted_original;
                const volatile float denoised_skip = c_skip * latent[j];
                denoised[j] = denoised_main + denoised_skip;
            }
            if (i + 1 < timesteps.size()) {
                std::vector<float> noise = make_latent_noise(latent.size(), rng, img2img);
                for (size_t j = 0; j < latent.size(); ++j) {
                    const volatile float prev_main = sqrt_alpha_prod_t_prev * denoised[j];
                    const volatile float prev_noise = sqrt_beta_prod_t_prev * noise[j];
                    latent[j] = prev_main + prev_noise;
                }
            } else {
                latent.swap(denoised);
            }
        }
        return true;
    }

    bool generate_single(VariantRuntime& runtime,
                         const openai_api::ImageGenRequest& req,
                         int seed,
                         std::vector<uint8_t>& png,
                         std::string& err) {
        const bool img2img = req.is_edit || req.has_image_input() || !req.init_image.empty();
        std::vector<float> prompt_embeds;
        if (!run_text_encoder(runtime, req.prompt, prompt_embeds, err)) {
            return false;
        }

        const size_t latent_h = static_cast<size_t>(runtime.meta.height / 8);
        const size_t latent_w = static_cast<size_t>(runtime.meta.width / 8);
        std::vector<float> latent(4 * latent_h * latent_w, 0.0f);
        const uint32_t rng_seed = img2img
            ? static_cast<uint32_t>(seed >= 0 ? seed : 0)
            : (seed >= 0 ? static_cast<uint32_t>(seed) : std::random_device{}());
        TorchNormalRng rng(rng_seed);

        if (img2img) {
            std::vector<uint8_t> image_bytes;
            if (!load_request_image_bytes(req.image_data, req.init_image, image_bytes, err)) {
                return false;
            }
            if (!run_vae_encoder(runtime, image_bytes, latent, rng, err)) {
                return false;
            }
            std::vector<float> noise = make_latent_noise(latent.size(), rng, true);
            add_noise(latent, noise, get_alpha_cumprod_exact(kImg2ImgTimesteps[0]));
        } else {
            latent = make_latent_noise(latent.size(), rng);
        }

        if (!denoise_loop(runtime, req, latent, prompt_embeds, img2img, rng, err)) {
            return false;
        }
        for (float& value : latent) {
            value /= kLatentScale;
        }
        return run_vae_decoder(runtime, latent, png, err);
    }

private:
    std::string base_model_name_;
    std::vector<ImageModelVariant> variants_;
    std::map<std::string, std::filesystem::path> variant_dirs_;
    std::map<std::string, std::string> variant_fallback_sizes_;
    std::unique_ptr<VariantRuntime> active_runtime_;
    std::string active_model_id_;
};

} // namespace

std::unique_ptr<ImageGenerator> create_image_generator() {
    return std::make_unique<Sd15ImageGenerator>();
}

} // namespace sd15
