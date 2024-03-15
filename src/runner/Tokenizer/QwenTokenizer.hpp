#pragma once
#include <string>
#include <vector>
#include "tiktoken.h"

struct QwenConfig
{
    // int pad_token_id;
    // for tokenizer
    int eos_token_id = 151643;
    int im_start_id = 151644;
    int im_end_id = 151645;
};

class QwenTokenizer
{
public:
    QwenTokenizer(const std::string &tiktoken_path, const QwenConfig &config);

    auto encode(const std::string &text, int max_length) const -> std::vector<int>;

    auto decode(const std::vector<int> &ids) const -> std::string;

    auto encode_history(const std::vector<std::string> &history, int max_length) const -> std::vector<int>;

    auto build_prompt(const std::vector<std::string> &history) const -> std::string;

    auto is_special_id(int id) const -> bool;

    tiktoken::tiktoken tokenizer;
    int eos_token_id;
    int im_start_id;
    int im_end_id;
};