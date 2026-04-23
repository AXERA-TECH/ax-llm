#pragma once

#include <optional>
#include <cctype>
#include <string>
#include <string_view>

#include "magic_enum/magic_enum.hpp"

// VLM type is a "wiring/profile" selector for how to:
// - preprocess image/video
// - run the vision encoder
// - inject visual embeddings into the LLM embedding stream
// - (optional) compute mRoPE position ids
//
// It is intentionally independent from `ModelType` (tokenizer model family).
enum class VLMType : int {
    None = 0,
    Qwen2_5VL = 1,
    Qwen3VL = 2,
    InternVL3 = 3,
    FastVLM = 4,
    SmolVLM2 = 5,
    PaddleOCRVL = 6,
    Gemma4VL = 7,
};

inline constexpr std::string_view VLMTypeName(VLMType t) {
    return magic_enum::enum_name(t);
}

inline std::optional<VLMType> VLMTypeFromInt(int v) {
    return magic_enum::enum_cast<VLMType>(v);
}

inline std::string VLMTypeKey(std::string_view s) {
    // Lowercase and keep only [A-Za-z0-9]. This makes matching tolerant to:
    // - case
    // - separators like '_' '-' '.'
    // Example: "QWEN3_VL" "qwen3-vl" "Qwen3VL" all -> "qwen3vl"
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        const unsigned char uc = (unsigned char)c;
        if (std::isalnum(uc)) out.push_back((char)std::tolower(uc));
    }
    return out;
}

inline std::optional<VLMType> VLMTypeFromString(std::string_view s) {
    const std::string key = VLMTypeKey(s);
    for (auto t : magic_enum::enum_values<VLMType>()) {
        const auto name = magic_enum::enum_name(t);
        if (VLMTypeKey(name) == key) return t;
    }
    return std::nullopt;
}

inline std::string VLMTypeChoices() {
    std::string out;
    for (auto t : magic_enum::enum_values<VLMType>()) {
        if (!out.empty()) out.append(", ");
        out.append(std::string(magic_enum::enum_name(t)));
        out.push_back('(');
        out.append(std::to_string((int)t));
        out.push_back(')');
    }
    return out;
}
