#include <signal.h>
#include <opencv2/opencv.hpp>

#include "runner/LLM.hpp"
#include "cmdline.hpp"
#include "string_utility.hpp"

#define IS_AXCL 1

#if IS_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

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

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    LLMAttrType attr;
    std::string prompt = "Hi";
    bool b_continue = true;

    cmdline::parser cmd;
    // cmd.add<std::string>("prompt", 'p', "prompt", true, prompt);
    // cmd.add<std::string>("image", 'i', "single image file or .txt file for images list", true);
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("filename_tokenizer_txt", 0, "tokenizer txt path", false, attr.filename_tokenizer_txt);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<std::string>("filename_image_encoder_axmodel", 0, "vpm encoder axmodel path", false, attr.filename_image_encoder_axmodel);

    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    // cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);

    // cmd.add<int>("image_context", 0, "image context, 151667 for InternVL 2.5/3, 92546 for InternVL 2.5-8B-MPO", false, attr.IMAGE_CONTEXT);
    // cmd.add<int>("image_start_context", 0, "image start context, 151665 for InternVL 2.5/3, 92544 for InternVL 2.5-8B-MPO", false, attr.IMAGE_START_CONTEXT);

#if IS_AXCL
    cmd.add<std::string>("devices", 0, "devices id,for example: \"0,1,2,3\" ", true, "0,1,2,3");
#endif

    cmd.add<bool>("live_print", 0, "print in live if set true, else print in end", false);
    cmd.add<int>("img_width", 'w', "image width", true);
    cmd.add<int>("img_height", 'h', "image height", true);

    // cmd.add<int>("img_token_id", 0, "image token id", false, 151655); 
    // cmd.add<int>("video_token_id", 0, "video token id", false, 151656);
    // cmd.add<int>("vision_start_token_id", 0, "vision_start_token_id", false, 151652);
    
    // cmd.add<int>("temporal_patch_size", 0, "temporal_patch_size", false, 2);
    // cmd.add<int>("tokens_per_second", 0, "tokens_per_second", false, 2);
    // cmd.add<int>("spatial_merge_size", 0, "spatial_merge_size", false, 2);
    // cmd.add<int>("patch_size", 0, "patch size", false, 16);
    // cmd.add<int>("fps", 0, "fps", false, 1);

    cmd.add<std::string>("post_config_path", 0, "post config path", false, attr.post_config_path);

    cmd.parse_check(argc, argv);

    // prompt = cmd.get<std::string>("prompt");
    // auto image_prompt = cmd.get<std::string>("image");
    attr.filename_tokenizer_txt = cmd.get<std::string>("filename_tokenizer_txt");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");
    // attr.template_prefill_filename_axmodel = cmd.get<std::string>("template_prefill_filename_axmodel");
    // attr.prefill_axmodel_num = cmd.get<int>("prefill_axmodel_num");

    attr.filename_image_encoder_axmodel = cmd.get<std::string>("filename_image_encoder_axmodel");
    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");
    // attr.IMAGE_CONTEXT = cmd.get<int>("image_context");
    // attr.IMAGE_START_CONTEXT = cmd.get<int>("image_start_context");

    attr.b_use_mmap_load_embed = cmd.get<bool>("use_mmap_load_embed");

#if IS_AXCL
    auto devices_str = cmd.get<std::string>("devices");
    std::vector<int> devices;
    std::stringstream ss(devices_str);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        devices.push_back(std::stoi(item));
    }

    attr.dev_ids = devices;

    auto ret = axclInit(nullptr);
    if (0 != ret)
    {
        return ret;
    }
#else
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    auto ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret)
    {
        return ret;
    }
#endif

    bool b_live_print = cmd.get<bool>("live_print");
    if (b_live_print)
    {
        attr.runing_callback = llm_running_callback;
        attr.reserve = 0;
    }

    if (!lLaMa.Init(attr))
    {
        ALOGE("lLaMa.Init failed");
#if IS_AXCL
        axclFinalize();
#else
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
#endif
        return -1;
    }

    std::vector<unsigned short> prompt_data;
    
    std::vector<std::vector<unsigned short>> img_embed;
    std::vector<int> visual_pos_mask;

    Config config;
    config.vision_config.width = cmd.get<int>("img_width");
    config.vision_config.height = cmd.get<int>("img_height");
    //
    if (b_continue)
    {
        printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
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

        {
            printf("image >> ");
        }
        
        fflush(stdout);
        std::string image_prompt;
        std::getline(std::cin, image_prompt);
        std::string output;
        if (image_prompt == "")
        {
            lLaMa.Encode(prompt_data, prompt);
            output = lLaMa.Run(prompt_data);
        }
        else
        {
            if (string_utility<std::string>::ends_with(image_prompt, ".txt"))
            {
                std::vector<std::string> lines;
                std::ifstream ifs(image_prompt);
                while (std::getline(ifs, image_prompt))
                {
                    lines.push_back(image_prompt);
                }
                ifs.close();

                std::vector<cv::Mat> imgs;
                for (auto &line : lines)
                {
                    cv::Mat src = cv::imread(line);
                    if (src.empty())
                    {
                        ALOGE("image prompt(%s) not found", line.c_str());
                        continue;
                    }
                    imgs.push_back(src);
                }
                std::vector<std::vector<unsigned short>> imgs_embed;
                if (auto ret = lLaMa.Encode(imgs, imgs_embed); ret != 0)
                {
                    ALOGE("lLaMa.Encode failed");
                    continue;
                }
                if (auto ret = lLaMa.Encode(imgs_embed, prompt_data, prompt); ret != 0)
                {
                    ALOGE("lLaMa.Encode failed");
                    continue;
                }
                output = lLaMa.Run(prompt_data);
            }
            else
            {
                cv::Mat src = cv::imread(image_prompt);
                if (src.empty())
                {
                    ALOGE("image prompt(%s) not found", image_prompt.c_str());
                    continue;
                }
                else
                {
                    std::vector<unsigned short> img_embed;
                    if (auto ret = lLaMa.Encode(src, img_embed); ret != 0)
                    {
                        ALOGE("lLaMa.Encode failed");
                        continue;
                    }
                    if (auto ret = lLaMa.Encode(img_embed, prompt_data, prompt); ret != 0)
                    {
                        ALOGE("lLaMa.Encode failed");
                        continue;
                    }
                    output = lLaMa.Run(prompt_data);
                }
            }
        }

        if (!b_live_print)
            printf("%s\n", output.c_str());

        std::vector<unsigned short>().swap(prompt_data);

        for (auto& inner_vec : img_embed) {
            std::vector<unsigned short>().swap(inner_vec); 
        }
        std::vector<std::vector<unsigned short>>().swap(img_embed);

        std::vector<int>().swap(visual_pos_mask);
    }

    lLaMa.Deinit();

#if IS_AXCL
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif

    return 0;
}