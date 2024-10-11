#include "signal.h"

#include "runner/LLM.hpp"

#include "cmdline.hpp"

#include <opencv2/opencv.hpp>

static LLM lLaMa;

void __sigExit(int iSigNo)
{
    lLaMa.Stop();
    return;
}

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    fprintf(stdout, "%s", p_str);
    fflush(stdout);
}

std::string prompt_complete(std::string prompt, TokenizerType tokenizer_type)
{
    std::ostringstream oss_prompt;
    switch (tokenizer_type)
    {
    case TKT_LLaMa:
        oss_prompt << "<|user|>\n"
                   << prompt << "</s><|assistant|>\n";
        break;
    case TKT_MINICPM:
        oss_prompt << "<用户><image></image>\n";
        oss_prompt << prompt << "<AI>";
        break;
    case TKT_Phi3:
        oss_prompt << prompt << " ";
        break;
    case TKT_Qwen:
        oss_prompt << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
        oss_prompt << "\n<|im_start|>user\n"
                   << prompt << "<|im_end|>\n<|im_start|>assistant\n";
        break;
    case TKT_HTTP:
    default:
        oss_prompt << prompt;
        break;
    }

    return oss_prompt.str();
}
int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    LLMAttrType attr;
    std::string prompt = "Hi";
    bool b_continue = false;

    cmdline::parser cmd;
    cmd.add<std::string>("prompt", 'p', "prompt", true, prompt);
    cmd.add<std::string>("image", 'i', "image", true);
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<int>("tokenizer_type", 0, "tokenizer type 0:LLaMa 1:Qwen 2:HTTP 3:Phi3 4:MINICPM", false, attr.tokenizer_type);
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<std::string>("filename_vpm_encoder_axmodedl", 0, "vpm encoder axmodel path", false, attr.filename_vpm_encoder_axmodedl);
    cmd.add<std::string>("filename_vpm_resampler_axmodedl", 0, "vpm resampler axmodel path", true, attr.filename_vpm_resampler_axmodedl);
    cmd.add<bool>("vpm_two_stage", 0, "", false, attr.b_vpm_two_stage);

    cmd.add<bool>("bos", 0, "", false, attr.b_bos);
    cmd.add<bool>("eos", 0, "", false, attr.b_eos);
    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    // cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.add<bool>("use_topk", 0, "", false, attr.b_use_topk);
    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);
    cmd.add<bool>("dynamic_load_axmodel_layer", 0, "it can save cmm memory", false, attr.b_dynamic_load_axmodel_layer);

    cmd.add<bool>("live_print", 0, "print in live if set true, else print in end", false);

    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);

    cmd.parse_check(argc, argv);

    prompt = cmd.get<std::string>("prompt");
    auto image_prompt = cmd.get<std::string>("image");
    attr.tokenizer_type = (TokenizerType)cmd.get<int>("tokenizer_type");
    attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");
    // attr.template_prefill_filename_axmodel = cmd.get<std::string>("template_prefill_filename_axmodel");
    // attr.prefill_axmodel_num = cmd.get<int>("prefill_axmodel_num");

    attr.filename_vpm_encoder_axmodedl = cmd.get<std::string>("filename_vpm_encoder_axmodedl");
    attr.filename_vpm_resampler_axmodedl = cmd.get<std::string>("filename_vpm_resampler_axmodedl");
    attr.b_vpm_two_stage = cmd.get<bool>("vpm_two_stage");
    attr.b_bos = cmd.get<bool>("bos");
    attr.b_eos = cmd.get<bool>("eos");
    attr.b_use_topk = cmd.get<bool>("use_topk");
    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");

    attr.b_use_mmap_load_embed = cmd.get<bool>("use_mmap_load_embed");
    attr.b_dynamic_load_axmodel_layer = cmd.get<bool>("dynamic_load_axmodel_layer");

    bool b_live_print = cmd.get<bool>("live_print");
    if (b_live_print)
    {
        attr.runing_callback = llm_running_callback;
        attr.reserve = 0;
    }

    b_continue = cmd.get<bool>("continue");

    if (!lLaMa.Init(attr))
    {
        return -1;
    }

    std::vector<unsigned short> prompt_data;
    std::vector<unsigned short> img_embed;
    //     std::vector<unsigned short> _tmp_data;
    //     lLaMa.RunVpm(src, _tmp_data);
    //     // printf("%d \n", _tmp_data.size());
    //     memcpy(prompt_data.data() + 5 * attr.tokens_embed_size, _tmp_data.data(), _tmp_data.size() * sizeof(unsigned short));
    // }

    if (prompt != "")
    {
        std::string output;
        cv::Mat src = cv::imread(image_prompt, cv::IMREAD_COLOR);
        if (src.empty())
        {
            // output = lLaMa.Run(prompt);
            ALOGE("image_prompt can't be empty");
        }
        else
        {
            lLaMa.Encode(src, img_embed);
            lLaMa.Encode(img_embed, prompt_data, prompt_complete(prompt, attr.tokenizer_type));
            output = lLaMa.Run(prompt_data);
        }

        if (!b_live_print && !output.empty())
            printf("%s\n", output.c_str());
    }

    //
    if (b_continue)
    {
        printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
        // lLaMa.Reset();
    }

    while (b_continue)
    {
        printf("prompt >> ");
        fflush(stdout);
        std::getline(std::cin, prompt);
        if (prompt == "q")
        {
            break;
        }
        if (prompt == "")
        {
            continue;
        }

        printf("image >> ");
        fflush(stdout);
        std::getline(std::cin, image_prompt);
        std::string output;
        if (image_prompt == "")
        {
            lLaMa.Encode(prompt_data, prompt_complete(prompt, attr.tokenizer_type));
            output = lLaMa.Run(prompt_data);
        }
        else
        {
            cv::Mat src = cv::imread(image_prompt, cv::IMREAD_COLOR);
            if (src.empty())
            {
                // output = lLaMa.Run(prompt);
                ALOGE("image prompt(%s) not found", image_prompt.c_str());
                // continue;
                lLaMa.Encode(prompt_data, prompt_complete(prompt, attr.tokenizer_type));
                output = lLaMa.Run(prompt_data);
            }
            else
            {
                lLaMa.Encode(src, img_embed);
                lLaMa.Encode(img_embed, prompt_data, prompt_complete(prompt, attr.tokenizer_type));
                output = lLaMa.Run(prompt_data);
            }
        }

        if (!b_live_print)
            printf("%s\n", output.c_str());
    }

    lLaMa.Deinit();

    return 0;
}