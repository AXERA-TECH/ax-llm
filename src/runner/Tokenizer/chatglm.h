#pragma once

#include <cmath>
#include <iomanip>
#include <sentencepiece_processor.h>
#include <sstream>
#include <unordered_map>

namespace chatglm
{

    struct FunctionMessage
    {
        std::string name;
        std::string arguments;

        FunctionMessage() = default;
        FunctionMessage(std::string name, std::string arguments) : name(std::move(name)), arguments(std::move(arguments)) {}

        friend std::ostream &operator<<(std::ostream &os, const FunctionMessage &self)
        {
            return os << "FunctionMessage(name=" << std::quoted(self.name) << ", arguments=" << std::quoted(self.arguments)
                      << ")";
        }
    };

    struct CodeMessage
    {
        std::string input;

        CodeMessage() = default;
        CodeMessage(std::string input) : input(std::move(input)) {}

        friend std::ostream &operator<<(std::ostream &os, const CodeMessage &self)
        {
            return os << "CodeMessage(input=" << std::quoted(self.input) << ")";
        }
    };

    struct ToolCallMessage
    {
        std::string type;
        FunctionMessage function;
        CodeMessage code;

        static const std::string TYPE_FUNCTION;
        static const std::string TYPE_CODE;

        ToolCallMessage(FunctionMessage function) : type(TYPE_FUNCTION), function(std::move(function)) {}

        ToolCallMessage(CodeMessage code) : type(TYPE_CODE), code(std::move(code)) {}

        friend std::ostream &operator<<(std::ostream &os, const ToolCallMessage &self)
        {
            return os << "ToolCallMessage(type=" << std::quoted(self.type) << ", function=" << self.function
                      << ", code=" << self.code << ")";
        }
    };

    struct ChatMessage
    {
        std::string role;
        std::string content;
        std::vector<ToolCallMessage> tool_calls;

        static const std::string ROLE_USER;
        static const std::string ROLE_ASSISTANT;
        static const std::string ROLE_SYSTEM;
        static const std::string ROLE_OBSERVATION;

        ChatMessage() = default;
        ChatMessage(std::string role, std::string content, std::vector<ToolCallMessage> tool_calls = {})
            : role(std::move(role)), content(std::move(content)), tool_calls(std::move(tool_calls)) {}

        friend std::ostream &operator<<(std::ostream &os, const ChatMessage &self)
        {
            os << "ChatMessage(role=" << std::quoted(self.role) << ", content=" << std::quoted(self.content)
               << ", tool_calls=[";
            for (size_t i = 0; i < self.tool_calls.size(); i++)
            {
                os << (i > 0 ? ", " : "") << self.tool_calls[i];
            }
            return os << "])";
        }
    };

    class BaseTokenizer
    {
    public:
        virtual ~BaseTokenizer() = default;

        virtual std::vector<int> encode(const std::string &text, int max_length) const = 0;

        virtual std::string decode(const std::vector<int> &ids) const = 0;

        virtual std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const = 0;

        virtual ChatMessage decode_message(const std::vector<int> &ids) const
        {
            return {ChatMessage::ROLE_ASSISTANT, decode(ids)};
        }

    protected:
        static void check_chat_messages(const std::vector<ChatMessage> &messages);
    };

    // ===== ChatGLM-6B =====

    class ChatGLMTokenizer : public BaseTokenizer
    {
    public:
        ChatGLMTokenizer(std::string_view serialized_model_proto);

        std::vector<int> encode(const std::string &text, int max_length) const override;

        std::string decode(const std::vector<int> &ids) const override;

        std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

        static std::string build_prompt(const std::vector<ChatMessage> &messages);

    private:
        static std::string preprocess(const std::string &text);

        static std::string postprocess(const std::string &text);

    public:
        sentencepiece::SentencePieceProcessor sp;
        int bos_token_id;
        int eos_token_id;
        int mask_token_id;
        int gmask_token_id;
        int pad_token_id;
    };

    // ===== ChatGLM2-6B =====

    class ChatGLM2Tokenizer : public BaseTokenizer
    {
    public:
        ChatGLM2Tokenizer(std::string_view serialized_model_proto);

        std::vector<int> encode(const std::string &text, int max_length) const override;

        std::string decode(const std::vector<int> &ids) const override;

        std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

        static std::string build_prompt(const std::vector<ChatMessage> &messages);

    private:
        bool is_special_id(int id) const;

    public:
        sentencepiece::SentencePieceProcessor sp;
        int mask_token_id;
        int gmask_token_id;
        int smask_token_id;
        int sop_token_id;
        int eop_token_id;
    };

    // ===== ChatGLM3-6B =====

    class ChatGLM3Tokenizer : public BaseTokenizer
    {
    public:
        ChatGLM3Tokenizer(std::string_view serialized_model_proto);

        std::vector<int> encode(const std::string &text, int max_length) const override;

        std::string decode(const std::vector<int> &ids) const override;

        std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

        ChatMessage decode_message(const std::vector<int> &ids) const override;

    private:
        std::vector<int> encode_single_message(const std::string &role, const std::string &content) const;

        std::string decode_with_special_tokens(const std::vector<int> &ids) const;

        static std::string remove_special_tokens(const std::string &text);

        int get_command(const std::string &token) const;

        bool is_special_id(int id) const;

        static void truncate(std::vector<int> &ids, int max_length);

    public:
        sentencepiece::SentencePieceProcessor sp;
        int mask_token_id;
        int gmask_token_id;
        int smask_token_id;
        int sop_token_id;
        int eop_token_id;
        int system_token_id;
        int user_token_id;
        int assistant_token_id;
        int observation_token_id;
        std::unordered_map<std::string, int> special_tokens;
        std::unordered_map<int, std::string> index_special_tokens;
    };

    // ===== Baichuan =====

    class BaichuanTokenizer : public BaseTokenizer
    {
    public:
        BaichuanTokenizer(std::string_view serialized_model_proto);

        std::vector<int> encode(const std::string &text, int max_length) const override;

        std::string decode(const std::vector<int> &ids) const override;

        std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

    private:
        bool is_special_id(int id) const;

        static void truncate(std::vector<int> &ids, int max_length);

    public:
        static constexpr int USER_TOKEN_ID = 195;
        static constexpr int ASSISTANT_TOKEN_ID = 196;

        sentencepiece::SentencePieceProcessor sp;
        int bos_token_id;
        int eos_token_id;
        int pad_token_id;
    };

    // ===== InternLM =====

    class InternLMTokenizer : public BaseTokenizer
    {
    public:
        InternLMTokenizer(std::string_view serialized_model_proto);

        std::vector<int> encode(const std::string &text, int max_length) const override;

        std::string decode(const std::vector<int> &ids) const override;

        std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

        static std::string build_prompt(const std::vector<ChatMessage> &messages);

    private:
        bool is_special_id(int id) const { return id == unk_token_id || id == bos_token_id || id == eos_token_id; }

    public:
        sentencepiece::SentencePieceProcessor sp;
        static constexpr int unk_token_id = 0;
        static constexpr int bos_token_id = 1;
        static constexpr int eos_token_id = 2;
    };
} // namespace chatglm