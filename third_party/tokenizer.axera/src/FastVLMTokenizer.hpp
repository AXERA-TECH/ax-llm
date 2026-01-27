#include "Qwen3Tokenizer.hpp"

class FastVLMTokenizer : public Qwen3Tokenizer<TEXT, IMAGE>
{
public:
    FastVLMTokenizer()
    {
        image_pad_token = "<image>";
        img_start_token = "<img>";
        img_end_token = "</img>";
    }

    std::string apply_chat_template(const std::vector<Content> &contents) override
    {
        // check contents type
        for (const auto &content : contents)
        {
            if (!support(content.type))
            {
                ALOGE("unsupport content type: %d", content.type);
                return {};
            }
        }

        std::stringstream text;
        for (const auto &content : contents)
        {
            if (content.role == SYSTEM)
            {
                text << "<|im_start|>system\n"
                     << content.data << "<|im_end|>\n";
            }
            else if (content.role == USER)
            {
                switch (content.type)
                {
                case TEXT:
                    text << "<|im_start|>user\n"
                         << content.data << "<|im_end|>\n";
                    break;
                case IMAGE:
                    text << "<|im_start|>user\n";
                    text << content.data;
                    for (int i = 0; i < content.num_media; i++)
                    {
                        text << "\n";
                        //     << img_start_token;
                        for (int j = 0; j < content.num_media_tokens; j++)
                        {
                        text << image_pad_token;
                        }
                        
                        text << "\n";
                        // text << img_end_token << "\n";
                    }
                    text << "<|im_end|>\n";
                    break;
                case VIDEO:
                    break;
                default:
                    break;
                }
            }
            else if (content.role == ASSISTANT)
            {
                if (!think_in_prompt)
                {
                    auto cleaned_data = remove_thinking(content.data);
                    text << "<|im_start|>assistant\n"
                         << cleaned_data << "<|im_end|>\n";
                }
                else
                {
                    text << "<|im_start|>assistant\n"
                         << content.data << "<|im_end|>\n";
                }
            }
        }

        if (contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_start|>assistant\n";
        }
        
        return text.str();
        // ALOGD("text: \n%s", text.str().c_str());
        // return tokenizer->encode(text.str());
    }
};
REGISTER(FastVLM, FastVLMTokenizer)
