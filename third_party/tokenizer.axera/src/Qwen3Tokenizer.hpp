#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class Qwen3Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    bool add_generation_prompt = true;
    std::string video_pad_token = "<|video_pad|>";
    std::string image_pad_token = "<|image_pad|>";

    std::string img_start_token = "<|vision_start|>";
    std::string img_end_token = "<|vision_end|>";

public:
    std::string apply_chat_template(const std::vector<Content> &contents) override
    {
        // check contents type
        for (const auto &content : contents)
        {
            if (!this->support(content.type))
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
                    text << "<|im_start|>user\n" << img_start_token;
                    for (int i = 0; i < content.num_media * content.num_media_tokens; i++)
                    {
                        text << image_pad_token;
                    }
                    text << img_end_token;
                    text << content.data << "<|im_end|>\n";
                    break;
                case VIDEO:
                    text << "<|im_start|>user\n" << img_start_token;
                    for (int i = 0; i < content.num_media * content.num_media_tokens; i++)
                    {
                        text << video_pad_token;
                    }
                    text << img_end_token;
                    text << content.data << "<|im_end|>\n";
                    break;

                default:
                    break;
                }
            }
            else if (content.role == ASSISTANT)
            {
                if (!this->think_in_prompt)
                {
                    auto cleaned_data = this->remove_thinking(content.data);
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

        // ALOGD("text: \n%s", text.str().c_str());
        return text.str();
    }
};
using qwen3vl_tokenizer = Qwen3Tokenizer<TEXT, IMAGE, VIDEO>;
REGISTER(Qwen3VL, qwen3vl_tokenizer)
using qwen3_tokenizer = Qwen3Tokenizer<TEXT>;
REGISTER(Qwen3, qwen3_tokenizer)
using qwen2_5_tokenizer = Qwen3Tokenizer<TEXT>;
REGISTER(Qwen2_5, qwen2_5_tokenizer)
