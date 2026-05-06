#include <iostream>
#include <string>
#include <vector>

#include "BaseTokenizer.hpp"

static std::vector<std::string> required_tokenizers()
{
    // Keep this list in sync with the models supported by axllm.
    // When adding a new model family/tokenizer_type, add its name here so CI
    // fails if the tokenizer submodule pointer is not updated.
    return {
        "Qwen2_5",
        "Qwen3",
        "Qwen3VL",
        "InternVL3",
        "HunYuan",
        "SmolVLM2",
        "FastVLM",
        "MiniCPM4",
        "MiniCPMV4",
        "Gemma3",
        "Gemma3VL",
        "SmolLM2",
        "SmolLM3",
        "GLM5",
        "GLM5VL",
        "KimiK25",
        "KimiK25VL",
        "Qwen3Omni",
        "Qwen3_5",
        "Qwen3_5VL",
        "MiniMaxM2",
        "MiniMaxM2VL",
        "MiniCPMO4_5",
        "InternVL3_5",
        "PaddleOCRVL",
        "Gemma4",
        "Gemma4VL",
    };
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    int failed = 0;
    const auto names = required_tokenizers();
    for (const auto &name : names)
    {
        auto tokenizer = create_tokenizer(name);
        if (!tokenizer)
        {
            std::cerr << "[FAIL] create_tokenizer(\"" << name << "\") returned null\n";
            failed++;
        }
    }

    if (failed)
    {
        std::cerr << "[FAIL] Missing tokenizers: " << failed << " / " << names.size() << "\n";
        std::cerr << "Hint: did you forget to update/commit the tokenizer submodule pointer?\n";
        return 1;
    }

    std::cout << "[OK] Tokenizer create smoke passed (" << names.size() << " types)\n";
    return 0;
}

