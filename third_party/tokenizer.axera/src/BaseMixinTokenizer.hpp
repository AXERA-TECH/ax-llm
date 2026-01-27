#pragma once
#include "BaseTokenizer.hpp"

#include "tokenizer/tokenizer_optimized.hpp"

template <ContentType... Types>
class BaseMixinTokenizer : public BaseTokenizer
{
protected:
    std::shared_ptr<MNN::Transformer::Tokenizer> tokenizer;
    static constexpr ContentType support_types[] = {Types...};

    std::vector<int> addition_stop_tokens = {};

    static inline void trim_inplace(std::string &s)
    {
        // 去掉前面空白
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
                                        { return !std::isspace(ch); }));
        // 去掉后面空白
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch)
                             { return !std::isspace(ch); })
                    .base(),
                s.end());
    }

    // 从文本中移除 end_think_token 之前的内容,并 trim 空格和换行符
    static std::string remove_thinking(const std::string &text, std::string end_think_token = "</think>", bool trim = true)
    {
        // 查找 token
        size_t pos = text.find(end_think_token);

        // 找不到 token：返回原文本（按需可更改）
        if (pos == std::string::npos)
        {
            std::string result = text;
            if (trim)
                trim_inplace(result);
            return result;
        }

        // 提取 token 之后的内容
        std::string result = text.substr(pos + end_think_token.size());

        // 可选 trim
        if (trim)
            trim_inplace(result);

        return result;
    }

public:
    bool load(const std::string tokenizer_path) override
    {
        tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));
        return tokenizer != nullptr;
    }

    bool support(ContentType type) override
    {
        return std::find(std::begin(support_types), std::end(support_types), type) != std::end(support_types);
    }

    bool is_stop(int token) override
    {
        return tokenizer->is_stop(token) || std::find(addition_stop_tokens.begin(), addition_stop_tokens.end(), token) != addition_stop_tokens.end();
    }

    void add_stop_token(int token) override
    {
        addition_stop_tokens.push_back(token);
    }
    bool add_stop_token(std::string stop_token) override
    {
        auto tokens = tokenizer->encode(stop_token);
        if (tokens.size() != 1)
        {
            return false;
        }
        addition_stop_tokens.push_back(tokens[0]);
        return true;
    }
    void clear_addition_stop_tokens() override
    {
        addition_stop_tokens.clear();
    }
    std::vector<int> get_stop_tokens() override
    {
        std::vector<int> stop_tokens = tokenizer->get_stop_tokens();
        stop_tokens.insert(stop_tokens.end(), addition_stop_tokens.begin(), addition_stop_tokens.end());
        return stop_tokens;
    }

    std::vector<int> encode(const std::string &text) override
    {
        return tokenizer->encode(text);
    }

    std::string decode(const std::vector<int> &ids) override
    {
        std::stringstream text;
        for (auto id : ids)
        {
            text << tokenizer->decode(id);
        }
        return text.str();
    }

    std::string decode(int id) override
    {
        return tokenizer->decode(id);
    }
};