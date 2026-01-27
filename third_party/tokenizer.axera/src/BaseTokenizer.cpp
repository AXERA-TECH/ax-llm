#include "BaseTokenizer.hpp"
#include <memory>
#include <algorithm>

#include "utils/object_register.hpp"

#include "Qwen3Tokenizer.hpp"
#include "InternVL3Tokenizer.hpp"
#include "SmolVLM2Tokenizer.hpp"
#include "FastVLMTokenizer.hpp"

std::shared_ptr<BaseTokenizer> create_tokenizer(ModelType type)
{
    delete_fun destroy_fun = nullptr;
    BaseTokenizer *tokenizer = (BaseTokenizer *)OBJFactory::getInstance().getObjectByID(type, destroy_fun);
    if (!tokenizer)
    {
        fprintf(stderr, "create tokenizer failed");
        return nullptr;
    }
    return std::shared_ptr<BaseTokenizer>(tokenizer, destroy_fun);
}