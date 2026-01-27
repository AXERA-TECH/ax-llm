#include "Qwen3Tokenizer.hpp"

// class InternVL2_5_Tokenizer : public Qwen3Tokenizer<TEXT, IMAGE, VIDEO>
// {
// public:
//     InternVL2_5_Tokenizer()
//     {
//         image_pad_token = "<IMG_CONTEXT>";
//         img_start_token = "<img>";
//         img_end_token = "</img>";
//     }

//     std::vector<int> encode(const std::vector<Content> &contents) override
//     {
//         // check contents type
//         for (const auto &content : contents)
//         {
//             if (!support(content.type))
//             {
//                 ALOGE("unsupport content type: %d", content.type);
//                 return {};
//             }
//         }

//         std::stringstream text;
//         for (const auto &content : contents)
//         {
//             if (content.role == SYSTEM)
//             {
//                 text << "<|im_start|>system\n"
//                      << content.data << "<|im_end|>\n";
//             }
//             else if (content.role == USER)
//             {
//                 switch (content.type)
//                 {
//                 case TEXT:
//                     text << "<|im_start|>user\n"
//                          << content.data << "<|im_end|>\n";
//                     break;
//                 case IMAGE:
//                 case VIDEO:
//                     text << "<|im_start|>user\n";
//                     for (int i = 0; i < content.num_media; i++)
//                     {
//                         text << "\n"
//                              << img_start_token;
//                         for (int j = 0; j < content.num_media_tokens; j++)
//                         {
//                             text << image_pad_token;
//                         }
//                         text << img_end_token << "\n";
//                     }
//                     text << content.data;
//                     text << "<|im_end|>\n";
//                     break;
//                 default:
//                     break;
//                 }
//             }
//             else if (content.role == ASSISTANT)
//             {
//                 if (!think_in_prompt)
//                 {
//                     auto cleaned_data = remove_thinking(content.data);
//                     text << "<|im_start|>assistant\n"
//                          << cleaned_data << "<|im_end|>\n";
//                 }
//                 else
//                 {
//                     text << "<|im_start|>assistant\n"
//                          << content.data << "<|im_end|>\n";
//                 }
//             }
//         }

//         if (contents.back().role == USER && add_generation_prompt)
//         {
//             text << "<|im_start|>assistant\n";
//         }

//         ALOGD("text: \n%s", text.str().c_str());
//         return tokenizer->encode(text.str());
//     }
// };


class InternVL3Tokenizer : public Qwen3Tokenizer<TEXT, IMAGE, VIDEO>
{
public:
    InternVL3Tokenizer()
    {
        image_pad_token = "<IMG_CONTEXT>";
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
                case VIDEO:
                    text << "<|im_start|>user\n";
                    text << content.data;
                    for (int i = 0; i < content.num_media; i++)
                    {
                        text << "\n"
                             << img_start_token;
                        for (int j = 0; j < content.num_media_tokens; j++)
                        {
                            text << image_pad_token;
                        }
                        text << img_end_token << "\n";
                    }
                    text << "<|im_end|>\n";
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
REGISTER(InternVL3, InternVL3Tokenizer)
// using InternVL2_5Tokenizer = InternVL3Tokenizer;
// REGISTER(InternVL2_5, InternVL2_5_Tokenizer)
