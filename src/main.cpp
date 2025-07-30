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

std::string prompt_complete(std::string prompt, TokenizerType tokenizer_type)
{
    std::ostringstream oss_prompt;
    switch (tokenizer_type)
    {
    case TKT_HTTP:
        oss_prompt << prompt;
        break;
    default:
        ALOGE("tokenizer type %d not support", tokenizer_type);
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
    bool b_continue = true;

    cmdline::parser cmd;
    // cmd.add<std::string>("prompt", 'p', "prompt", true, prompt);
    // cmd.add<std::string>("image", 'i', "single image file or .txt file for images list", true);
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<std::string>("filename_image_encoder_axmodedl", 0, "vpm encoder axmodel path", false, attr.filename_image_encoder_axmodedl);

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
    cmd.add<bool>("video", 0, "inputs are video", false);
    cmd.add<int>("img_width", 'w', "image width", true);
    cmd.add<int>("img_height", 'h', "image height", true);
    cmd.add<int>("img_token_id", 0, "image token id", false, 151655); 
    cmd.add<int>("video_token_id", 0, "video token id", false, 151656);
    cmd.add<int>("vision_start_token_id", 0, "vision_start_token_id", false, 151652);
    
    cmd.add<int>("temporal_patch_size", 0, "temporal_patch_size", false, 2);
    cmd.add<int>("tokens_per_second", 0, "tokens_per_second", false, 2);
    cmd.add<int>("spatial_merge_size", 0, "spatial_merge_size", false, 2);
    cmd.add<int>("patch_size", 0, "patch size", false, 14);
    cmd.add<int>("fps", 0, "fps", false, 1);

    cmd.add<std::string>("post_config_path", 0, "post config path", false, attr.post_config_path);

    cmd.parse_check(argc, argv);

    // prompt = cmd.get<std::string>("prompt");
    // auto image_prompt = cmd.get<std::string>("image");
    attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");
    // attr.template_prefill_filename_axmodel = cmd.get<std::string>("template_prefill_filename_axmodel");
    // attr.prefill_axmodel_num = cmd.get<int>("prefill_axmodel_num");

    attr.filename_image_encoder_axmodedl = cmd.get<std::string>("filename_image_encoder_axmodedl");
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
    std::vector<std::vector<int>> position_ids;

    Config config;    
    config.vision_config.temporal_patch_size = cmd.get<int>("temporal_patch_size");
    config.vision_config.tokens_per_second = cmd.get<int>("tokens_per_second");
    config.vision_config.spatial_merge_size = cmd.get<int>("spatial_merge_size");
    config.vision_config.patch_size = cmd.get<int>("patch_size");
    config.vision_config.width = cmd.get<int>("img_width");
    config.vision_config.height = cmd.get<int>("img_height");
    config.vision_config.fps = cmd.get<int>("fps");

    config.image_token_id =  cmd.get<int>("img_token_id");
    config.video_token_id = cmd.get<int>("video_token_id");
    config.vision_start_token_id = cmd.get<int>("vision_start_token_id");

    bool b_video = cmd.get<bool>("video");
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

        printf("image >> ");
        fflush(stdout);
        std::string image_prompt;
        std::getline(std::cin, image_prompt);
        std::string output;
        if (image_prompt == "")
        {
            lLaMa.Encode(prompt_data, position_ids, config, prompt_complete(prompt, attr.tokenizer_type));
            output = lLaMa.Run(prompt_data, position_ids);
        }
        else
        {
            auto src = ReadImages(image_prompt);
            if (src.empty())
            {
                // output = lLaMa.Run(prompt);
                ALOGE("image prompt(%s) not found", image_prompt.c_str());
                // continue;
                lLaMa.Encode(prompt_data, position_ids, config, prompt_complete(prompt, attr.tokenizer_type));
                output = lLaMa.Run(prompt_data, position_ids);
            }
            else
            {
                lLaMa.Encode(src, b_video, img_embed, config);
                lLaMa.Encode(img_embed, prompt_data, position_ids, config, prompt_complete(prompt, attr.tokenizer_type));
                output = lLaMa.Run(prompt_data, position_ids);
            }
        }

        if (!b_live_print)
            printf("%s\n", output.c_str());
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