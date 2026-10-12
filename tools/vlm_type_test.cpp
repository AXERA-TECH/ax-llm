#include "VLMType.hpp"

#include <cassert>
#include <string>

int main() {
    assert(VLMTypeFromInt(11).value() == VLMType::Qwen3_5VL);
    assert(VLMTypeFromString("Qwen3_5VL").value() == VLMType::Qwen3_5VL);
    assert(VLMTypeFromString("qwen3_5vl").value() == VLMType::Qwen3_5VL);
    assert(VLMTypeChoices().find("Qwen3_5VL(11)") != std::string::npos);
    return 0;
}
