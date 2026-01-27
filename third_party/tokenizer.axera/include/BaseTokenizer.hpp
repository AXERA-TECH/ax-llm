#pragma once
#include <string>
#include <vector>
#include <memory>

enum ModelType
{
    Qwen2_5,
    Qwen3,
    Qwen3VL,

    InternVL3,

    SmolVLM2,
    FastVLM,
};

enum RoleType
{
    SYSTEM,
    ASSISTANT,
    USER,
};

enum ContentType
{
    TEXT,
    AUDIO,
    IMAGE,
    VIDEO,
};

struct Content
{
    RoleType role;
    ContentType type;
    std::string data;
    int num_media = 0;        // 多模态数量
    int num_media_tokens = 0; // 多模态token pad数量
};

class BaseTokenizer
{
protected:
    // 是否在上下文中保留thinking内容, 默认为false
    bool think_in_prompt = false;

    
public:
    virtual bool load(const std::string tokenizer_path) = 0;
    virtual bool support(ContentType type) = 0;
    virtual void set_think_in_prompt(bool think_in_prompt)
    {
        this->think_in_prompt = think_in_prompt;
    }

    virtual bool is_stop(int token) = 0;
    virtual void add_stop_token(int token) = 0;
    virtual bool add_stop_token(std::string stop_token) = 0;
    virtual void clear_addition_stop_tokens() = 0;
    virtual std::vector<int> get_stop_tokens() = 0;

    virtual std::vector<int> encode(const std::vector<Content> &contents)
    {
        return encode(apply_chat_template(contents));
    }

    virtual std::string apply_chat_template(const std::vector<Content> &contents) = 0;
    virtual std::vector<int> encode(const std::string &text) = 0;
    virtual std::string decode(const std::vector<int> &ids) = 0;
    virtual std::string decode(int id) = 0;
};

std::shared_ptr<BaseTokenizer> create_tokenizer(ModelType type);
