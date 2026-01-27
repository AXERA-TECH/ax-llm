#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"
#include "utils/num2words_en.hpp"

class SmolVLM2Tokenizer : public BaseMixinTokenizer<TEXT, IMAGE, VIDEO>
{
protected:
    bool add_generation_prompt = true;
    std::string image_pad_token = "<image>";

    Num2Word_EN num2word_en;

    std::vector<std::vector<std::string>> image_head_tokens =
        {
            {"<row_1_col_1>", "<row_1_col_2>"},
            {"<row_2_col_1>", "<row_2_col_2>"}};

    std::string get_image_tokens(int num_media_tokens)
    {
        std::stringstream ss;
        for (size_t i = 0; i < image_head_tokens.size(); i++)
        {
            for (const auto &token : image_head_tokens[i])
            {
                ss << "<fake_token_around_image>" << token;

                for (int j = 0; j < num_media_tokens; j++)
                {
                    ss << image_pad_token;
                }
            }
            ss << "\n";
        }
        ss << "\n";
        ss << "<fake_token_around_image><global-img>";

        for (int i = 0; i < num_media_tokens; i++)
        {
            ss << image_pad_token;
        }
        ss << "<fake_token_around_image>";
        return ss.str();
    }

    std::string get_video_tokens(int num_media_tokens)
    {
        std::stringstream ss;
        ss << "Frame from 00:00:<fake_token_around_image><global-img>";
        for (int j = 0; j < num_media_tokens; j++)
        {
            ss << image_pad_token;
        }
        ss << "<fake_token_around_image>\n";
        return ss.str();
    }

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
                ALOGW("system content is not supported");
                continue;
            }
            else if (content.role == USER)
            {
                switch (content.type)
                {
                case TEXT:
                    text << "<|im_start|>User:"
                         << content.data << "<end_of_utterance>\n";
                    break;
                case IMAGE:
                    text << "<|im_start|>User:";
                    for (int i = 0; i < content.num_media; i++)
                    {
                        text << get_image_tokens(content.num_media_tokens);
                    }
                    text << content.data << "<end_of_utterance>\n";
                    break;
                case VIDEO:
                    text << "<|im_start|>User:";
                    text << "You are provided the following series of " << num2word_en[content.num_media] << " frames from a 0:00:00 [H:MM:SS] video.\n\n";
                    for (int i = 0; i < content.num_media; i++)
                    {
                        text << get_video_tokens(content.num_media_tokens);
                    }
                    text << "\n";
                    text << content.data << "<end_of_utterance>\n";
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
                    text << "Assistant:"
                         << cleaned_data << "<end_of_utterance>\n";
                }
                else
                {
                    text << "Assistant:"
                         << content.data << "<end_of_utterance>\n";
                }
            }
        }

        if (contents.back().role == USER && add_generation_prompt)
        {
            text << "Assistant:";
        }
        return text.str();
        // ALOGD("text: \n%s", text.str().c_str());
        // return this->tokenizer->encode(text.str());
    }
};
REGISTER(SmolVLM2, SmolVLM2Tokenizer)
