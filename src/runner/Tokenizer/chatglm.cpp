#include "chatglm.h"
#include "../utils/sample_log.h"

#include <regex>
#include <codecvt>
namespace chatglm
{

    class LogMessageFatal
    {
    public:
        LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
        [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
        std::ostringstream &stream() { return oss_; }

    private:
        std::ostringstream oss_;
    };

#define CHATGLM_THROW chatglm::LogMessageFatal(__FILE__, __LINE__).stream()
#define CHATGLM_CHECK(cond) \
    if (!(cond))            \
    CHATGLM_THROW << "check failed (" #cond ") "

    const std::string ToolCallMessage::TYPE_FUNCTION = "function";
    const std::string ToolCallMessage::TYPE_CODE = "code";

    const std::string ChatMessage::ROLE_USER = "user";
    const std::string ChatMessage::ROLE_ASSISTANT = "assistant";
    const std::string ChatMessage::ROLE_SYSTEM = "system";
    const std::string ChatMessage::ROLE_OBSERVATION = "observation";

    // trim from start (in place)
    static inline void ltrim(std::string &s)
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
                                        { return !std::isspace(ch); }));
    }

    // trim from end (in place)
    static inline void rtrim(std::string &s)
    {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch)
                             { return !std::isspace(ch); })
                    .base(),
                s.end());
    }

    // trim from both ends (in place)
    static inline void trim(std::string &s)
    {
        rtrim(s);
        ltrim(s);
    }

    void BaseTokenizer::check_chat_messages(const std::vector<ChatMessage> &messages)
    {
        CHATGLM_CHECK(messages.size() % 2 == 1) << "invalid chat messages size " << messages.size();
        for (size_t i = 0; i < messages.size(); i++)
        {
            const std::string &target_role = (i % 2 == 0) ? ChatMessage::ROLE_USER : ChatMessage::ROLE_ASSISTANT;
            CHATGLM_CHECK(messages[i].role == target_role)
                << "expect messages[" << i << "].role to be " << target_role << ", but got " << messages[i].role;
        }
    }

    // ===== ChatGLM-6B =====

    ChatGLMTokenizer::ChatGLMTokenizer(std::string_view serialized_model_proto)
    {
        const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
        CHATGLM_CHECK(status.ok()) << status.ToString();

        bos_token_id = sp.PieceToId("<sop>");
        eos_token_id = sp.PieceToId("<eop>");
        mask_token_id = sp.PieceToId("[MASK]");
        gmask_token_id = sp.PieceToId("[gMASK]");
        pad_token_id = sp.PieceToId("<pad>");
    }

    std::vector<int> ChatGLMTokenizer::encode(const std::string &text, int max_length) const
    {
        std::string input = preprocess(text);
        std::vector<int> ids;
        sp.Encode(input, &ids);
        ids.insert(ids.end(), {gmask_token_id, bos_token_id});
        if ((int)ids.size() > max_length)
        {
            // sliding window: always take the last max_length tokens
            ids.erase(ids.begin(), ids.end() - max_length);
        }
        return ids;
    }

    std::vector<int> ChatGLMTokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const
    {
        std::string prompt = build_prompt(messages);
        std::vector<int> input_ids = encode(prompt, max_length);
        return input_ids;
    }

    std::string ChatGLMTokenizer::build_prompt(const std::vector<ChatMessage> &messages)
    {
        check_chat_messages(messages);

        std::ostringstream oss_prompt;
        if (messages.size() == 1)
        {
            oss_prompt << messages.front().content;
        }
        else
        {
            for (size_t i = 0; i < messages.size(); i += 2)
            {
                oss_prompt << "[Round " << i / 2 << "]\n问：" << messages[i].content << "\n答：";
                if (i + 1 < messages.size())
                {
                    oss_prompt << messages[i + 1].content << "\n";
                }
            }
        }
        return oss_prompt.str();
    }

    std::string ChatGLMTokenizer::decode(const std::vector<int> &ids) const
    {
        std::string text;
        sp.Decode(ids, &text);
        text = postprocess(text);
        return text;
    }

    static std::string regex_replace(const std::string &input, const std::regex &regex,
                                     std::function<std::string(const std::smatch &)> format)
    {
        std::ostringstream oss;
        int last_index = 0;
        for (auto it = std::sregex_iterator(input.begin(), input.end(), regex); it != std::sregex_iterator(); it++)
        {
            oss << it->prefix() << format(*it);
            last_index = it->position() + it->length();
        }
        oss << input.substr(last_index);
        return oss.str();
    }

    std::string ChatGLMTokenizer::preprocess(const std::string &text)
    {
        std::string output;

        // newline token
        {
            static const std::regex newline_regex("\n");
            output = std::regex_replace(text, newline_regex, "<n>");
        }
        // tab token
        {
            static const std::regex tab_regex("\t");
            output = std::regex_replace(output, tab_regex, "<|tab|>");
        }
        // blank tokens
        {
            static const std::regex pattern(R"([ ]{2,80})");
            output = regex_replace(output, pattern, [](const std::smatch &sm)
                                   {
            std::ostringstream oss;
            oss << "<|blank_" << sm.str().size() << "|>";
            return oss.str(); });
        }

        return output;
    }

    static inline std::string replace_punctuations(const std::string &text)
    {
        // reference: https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
        static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        static const std::vector<std::pair<std::wregex, std::wstring>> punct_map{
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]),)")), converter.from_bytes("$1，")},
            {std::wregex(converter.from_bytes(R"(,([\u4e00-\u9fff]))")), converter.from_bytes("，$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])!)")), converter.from_bytes("$1！")},
            {std::wregex(converter.from_bytes(R"(!([\u4e00-\u9fff]))")), converter.from_bytes("！$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]):)")), converter.from_bytes("$1：")},
            {std::wregex(converter.from_bytes(R"(:([\u4e00-\u9fff]))")), converter.from_bytes("：$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]);)")), converter.from_bytes("$1；")},
            {std::wregex(converter.from_bytes(R"(;([\u4e00-\u9fff]))")), converter.from_bytes("；$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])\?)")), converter.from_bytes("$1？")},
            {std::wregex(converter.from_bytes(R"(\?([\u4e00-\u9fff]))")), converter.from_bytes("？$1")},
        };
        std::wstring w_output = converter.from_bytes(text);
        for (const auto &punct_pair : punct_map)
        {
            w_output = std::regex_replace(w_output, punct_pair.first, punct_pair.second);
        }
        std::string output = converter.to_bytes(w_output);
        return output;
    }

    std::string ChatGLMTokenizer::postprocess(const std::string &text)
    {
        std::string output;

        // newline token
        {
            static const std::regex pattern(R"(<n>)");
            output = std::regex_replace(text, pattern, "\n");
        }
        // tab token
        {
            static const std::regex pattern(R"(<\|tab\|>)");
            output = std::regex_replace(output, pattern, "\t");
        }
        // blank tokens
        {
            static const std::regex pattern(R"(<\|blank_(\d+)\|>)");
            output = regex_replace(output, pattern,
                                   [](const std::smatch &sm)
                                   { return std::string(std::stoi(sm[1].str()), ' '); });
        }
        // punctuations
        output = replace_punctuations(output);

        return output;
    }

    // ===== ChatGLM2-6B =====

    ChatGLM2Tokenizer::ChatGLM2Tokenizer(std::string_view serialized_model_proto)
    {
        const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
        CHATGLM_CHECK(status.ok()) << status.ToString();

        int special_id = sp.GetPieceSize();
        mask_token_id = special_id++;
        gmask_token_id = special_id++;
        smask_token_id = special_id++;
        sop_token_id = special_id++;
        eop_token_id = special_id++;
    }

    std::vector<int> ChatGLM2Tokenizer::encode(const std::string &text, int max_length) const
    {
        std::vector<int> ids;
        sp.Encode(text, &ids);
        ids.insert(ids.begin(), {gmask_token_id, sop_token_id}); // special prefix
        if ((int)ids.size() > max_length)
        {
            // sliding window: drop the least recent history while keeping the two special prefix tokens
            int num_drop = (int)ids.size() - max_length;
            ids.erase(ids.begin() + 2, ids.begin() + 2 + num_drop);
        }
        return ids;
    }

    std::string ChatGLM2Tokenizer::decode(const std::vector<int> &ids) const
    {
        // filter out special tokens
        std::vector<int> normal_ids(ids);
        normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id)
                                        { return is_special_id(id); }),
                         normal_ids.end());

        std::string text;
        sp.Decode(normal_ids, &text);
        text = replace_punctuations(text);
        return text;
    }

    std::vector<int> ChatGLM2Tokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const
    {
        std::string prompt = build_prompt(messages);
        std::vector<int> input_ids = encode(prompt, max_length);
        return input_ids;
    }

    std::string ChatGLM2Tokenizer::build_prompt(const std::vector<ChatMessage> &messages)
    {
        check_chat_messages(messages);

        std::ostringstream oss_prompt;
        for (size_t i = 0; i < messages.size(); i += 2)
        {
            oss_prompt << "[Round " << i / 2 + 1 << "]\n\n问：" << messages[i].content << "\n\n答：";
            if (i < messages.size() - 1)
            {
                oss_prompt << messages[i + 1].content << "\n\n";
            }
        }
        return oss_prompt.str();
    }

    bool ChatGLM2Tokenizer::is_special_id(int id) const
    {
        return id == mask_token_id || id == gmask_token_id || id == smask_token_id || id == sop_token_id ||
               id == eop_token_id;
    }

    // ===== ChatGLM3-6B =====

    ChatGLM3Tokenizer::ChatGLM3Tokenizer(std::string_view serialized_model_proto)
    {
        const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
        CHATGLM_CHECK(status.ok()) << status.ToString();

        int special_id = sp.GetPieceSize();
        mask_token_id = special_id++;
        gmask_token_id = special_id++;
        smask_token_id = special_id++;
        sop_token_id = special_id++;
        eop_token_id = special_id++;
        system_token_id = special_id++;
        user_token_id = special_id++;
        assistant_token_id = special_id++;
        observation_token_id = special_id++;

        special_tokens = {
            {"[MASK]", mask_token_id},
            {"[gMASK]", gmask_token_id},
            {"[sMASK]", smask_token_id},
            {"sop", sop_token_id},
            {"eop", eop_token_id},
            {"<|system|>", system_token_id},
            {"<|user|>", user_token_id},
            {"<|assistant|>", assistant_token_id},
            {"<|observation|>", observation_token_id},
        };

        for (const auto &item : special_tokens)
        {
            index_special_tokens[item.second] = item.first;
        }
    }

    std::vector<int> ChatGLM3Tokenizer::encode(const std::string &text, int max_length) const
    {
        std::vector<int> ids;
        sp.Encode(text, &ids);
        ids.insert(ids.begin(), {gmask_token_id, sop_token_id}); // special prefix
        truncate(ids, max_length);
        return ids;
    }

    std::string ChatGLM3Tokenizer::decode(const std::vector<int> &ids) const
    {
        std::string text = decode_with_special_tokens(ids);
        text = remove_special_tokens(text);
        return text;
    }

    std::string ChatGLM3Tokenizer::decode_with_special_tokens(const std::vector<int> &ids) const
    {
        std::vector<std::string> pieces;
        for (int id : ids)
        {
            auto pos = index_special_tokens.find(id);
            if (pos != index_special_tokens.end())
            {
                // special tokens
                pieces.emplace_back(pos->second);
            }
            else
            {
                // normal tokens
                pieces.emplace_back(sp.IdToPiece(id));
            }
        }

        std::string text = sp.DecodePieces(pieces);
        return text;
    }

    std::string ChatGLM3Tokenizer::remove_special_tokens(const std::string &text)
    {
        std::string output = text;
        static const std::vector<std::regex> special_token_regex{
            // std::regex(R"(<\|assistant\|> interpreter)"),
            // std::regex(R"(<\|assistant\|> interpre)"),
            std::regex(R"(<\|assistant\|>)"),
            std::regex(R"(<\|user\|>)"),
            std::regex(R"(<\|observation\|>)"),
        };
        for (const auto &re : special_token_regex)
        {
            output = std::regex_replace(output, re, "");
        }
        return output;
    }

    std::vector<int> ChatGLM3Tokenizer::encode_single_message(const std::string &role, const std::string &content) const
    {
        std::vector<int> input_ids;
        input_ids.emplace_back(get_command("<|" + role + "|>"));
        // TODO: support metadata
        std::vector<int> newline_ids;
        sp.Encode("\n", &newline_ids);
        input_ids.insert(input_ids.end(), newline_ids.begin(), newline_ids.end());
        std::vector<int> content_ids;
        sp.Encode(content, &content_ids);
        input_ids.insert(input_ids.end(), content_ids.begin(), content_ids.end());
        return input_ids;
    }

    std::vector<int> ChatGLM3Tokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const
    {
        std::vector<int> input_ids{gmask_token_id, sop_token_id};
        for (const auto &msg : messages)
        {
            auto msg_ids = encode_single_message(msg.role, msg.content);
            input_ids.insert(input_ids.end(), msg_ids.begin(), msg_ids.end());

            // encode code block into a separate message
            if (!msg.tool_calls.empty() && msg.tool_calls.front().type == ToolCallMessage::TYPE_CODE)
            {
                auto code_ids = encode_single_message(msg.role, msg.tool_calls.front().code.input);
                input_ids.insert(input_ids.end(), code_ids.begin(), code_ids.end());
            }
        }
        input_ids.emplace_back(assistant_token_id);
        truncate(input_ids, max_length);
        return input_ids;
    }

    ChatMessage ChatGLM3Tokenizer::decode_message(const std::vector<int> &ids) const
    {
        ChatMessage message;
        if (!ids.empty() && ids.back() == observation_token_id)
        {
            // insert an <|assistant|> token before content to match possible interpreter delimiter
            std::vector<int> full_ids{assistant_token_id};
            full_ids.insert(full_ids.end(), ids.begin(), ids.end());

            std::string output = decode_with_special_tokens(full_ids);
            const std::string ci_delim = "<|assistant|> interpreter";
            size_t ci_pos = output.find(ci_delim);
            if (ci_pos != std::string::npos)
            {
                // code interpreter
                std::string chat_output = output.substr(0, ci_pos);
                chat_output = remove_special_tokens(chat_output);
                trim(chat_output);
                std::string code_output = output.substr(ci_pos + ci_delim.size());
                code_output = remove_special_tokens(code_output);
                trim(code_output);
                message = ChatMessage(ChatMessage::ROLE_ASSISTANT, std::move(chat_output),
                                      {ToolCallMessage(CodeMessage(std::move(code_output)))});
            }
            else
            {
                // tool call
                output = remove_special_tokens(output);

                // parse tool name
                std::string tool_name = "PARSE_ERROR";
                size_t pos = output.find('\n');
                if (pos != std::string::npos)
                {
                    // split tool name and args by 1st linebreak
                    tool_name = output.substr(0, pos);
                    trim(tool_name);
                    output.erase(0, pos + 1);
                }

                // post process output
                trim(output);

                // extract args
                std::string tool_args = "PARSE_ERROR";
                static const std::regex args_regex(R"(```.*?\n(.*?)\n```)");
                std::smatch sm;
                if (std::regex_search(output, sm, args_regex))
                {
                    CHATGLM_CHECK(sm.size() == 2) << "unexpected regex match results";
                    tool_args = sm[1];
                }

                message = ChatMessage(ChatMessage::ROLE_ASSISTANT, std::move(output),
                                      {ToolCallMessage(FunctionMessage(std::move(tool_name), std::move(tool_args)))});
            }
        }
        else
        {
            // conversation
            message = BaseTokenizer::decode_message(ids);
            trim(message.content); // strip leading linebreak in conversation mode
        }
        return message;
    }

    int ChatGLM3Tokenizer::get_command(const std::string &token) const
    {
        auto pos = special_tokens.find(token);
        CHATGLM_CHECK(pos != special_tokens.end()) << token << " is not a special token";
        return pos->second;
    }

    bool ChatGLM3Tokenizer::is_special_id(int id) const { return index_special_tokens.count(id) > 0; }

    void ChatGLM3Tokenizer::truncate(std::vector<int> &ids, int max_length)
    {
        if ((int)ids.size() > max_length)
        {
            // sliding window: drop the least recent history while keeping the two special prefix tokens
            int num_drop = (int)ids.size() - max_length;
            ids.erase(ids.begin() + 2, ids.begin() + 2 + num_drop);
        }
    }

    // ===== Baichuan =====

    BaichuanTokenizer::BaichuanTokenizer(std::string_view serialized_model_proto)
    {
        const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
        CHATGLM_CHECK(status.ok()) << status.ToString();
    }

    std::vector<int> BaichuanTokenizer::encode(const std::string &text, int max_length) const
    {
        std::vector<int> ids;
        sp.Encode(text, &ids);
        truncate(ids, max_length);
        return ids;
    }

    std::string BaichuanTokenizer::decode(const std::vector<int> &ids) const
    {
        std::vector<int> normal_ids(ids);
        normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id)
                                        { return is_special_id(id); }),
                         normal_ids.end());

        std::string text;
        sp.Decode(normal_ids, &text);
        return text;
    }

    std::vector<int> BaichuanTokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const
    {
        check_chat_messages(messages);

        std::vector<int> ids;
        ids.reserve(max_length);
        for (const auto &msg : messages)
        {
            ids.push_back((msg.role == ChatMessage::ROLE_USER) ? USER_TOKEN_ID : ASSISTANT_TOKEN_ID);
            std::vector<int> content_ids = encode(msg.content, max_length);
            ids.insert(ids.end(), content_ids.begin(), content_ids.end());
        }
        ids.push_back(ASSISTANT_TOKEN_ID);

        truncate(ids, max_length);
        return ids;
    }

    bool BaichuanTokenizer::is_special_id(int id) const
    {
        return id == bos_token_id || id == eos_token_id || id == pad_token_id;
    }

    void BaichuanTokenizer::truncate(std::vector<int> &ids, int max_length)
    {
        if ((int)ids.size() > max_length)
        {
            ids.erase(ids.begin(), ids.end() - max_length);
        }
    }

    // ===== InternLM =====

    InternLMTokenizer::InternLMTokenizer(std::string_view serialized_model_proto)
    {
        const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
        CHATGLM_CHECK(status.ok()) << status.ToString();
    }

    std::vector<int> InternLMTokenizer::encode(const std::string &text, int max_length) const
    {
        std::vector<int> ids;
        sp.Encode(text, &ids);
        ids.insert(ids.begin(), {bos_token_id}); // special prefix
        if ((int)ids.size() > max_length)
        {
            // sliding window: drop the least recent history while keeping the special prefix
            int num_drop = (int)ids.size() - max_length;
            ids.erase(ids.begin() + 1, ids.begin() + 1 + num_drop);
        }
        return ids;
    }

    std::string InternLMTokenizer::decode(const std::vector<int> &ids) const
    {
        // filter out special tokens
        std::vector<int> normal_ids(ids);
        normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id)
                                        { return is_special_id(id); }),
                         normal_ids.end());

        std::string text;
        sp.Decode(normal_ids, &text);
        // remove <eoa> and its following
        size_t eoa_pos = text.find("<eoa>");
        if (eoa_pos != std::string::npos)
        {
            text.erase(eoa_pos);
        }
        return text;
    }

    std::vector<int> InternLMTokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const
    {
        std::string prompt = build_prompt(messages);
        std::vector<int> input_ids = encode(prompt, max_length);
        return input_ids;
    }

    std::string InternLMTokenizer::build_prompt(const std::vector<ChatMessage> &messages)
    {
        check_chat_messages(messages);

        std::ostringstream oss_prompt;
        for (const auto &msg : messages)
        {
            if (msg.role == ChatMessage::ROLE_USER)
            {
                oss_prompt << "<|User|>:" << msg.content << "<eoh>\n<|Bot|>:";
            }
            else
            {
                oss_prompt << msg.content << "<eoa>\n";
            }
        }
        return oss_prompt.str();
    }
}