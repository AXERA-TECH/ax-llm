#include "signal.h"

#include "runner/LLM.hpp"

#include "cmdline.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

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
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<int>("tokenizer_type", 0, "tokenizer type 0:LLaMa 1:Qwen 2:HTTP 3:Phi3 4:MINICPM", false, attr.tokenizer_type);
    cmd.add<std::string>("url_tokenizer_model", 0, "tokenizer model path", false, attr.url_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    // cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);

    cmd.add<bool>("live_print", 0, "print in live if set true, else print in end", false);

    cmd.parse_check(argc, argv);

    attr.tokenizer_type = (TokenizerType)cmd.get<int>("tokenizer_type");
    attr.url_tokenizer_model = cmd.get<std::string>("url_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");
    // attr.template_prefill_filename_axmodel = cmd.get<std::string>("template_prefill_filename_axmodel");
    // attr.prefill_axmodel_num = cmd.get<int>("prefill_axmodel_num");

    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");

    attr.b_use_mmap_load_embed = cmd.get<bool>("use_mmap_load_embed");

    bool b_live_print = cmd.get<bool>("live_print");
    if (b_live_print)
    {
        attr.runing_callback = llm_running_callback;
        attr.reserve = 0;
    }

    // 1. init engine
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    auto ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret)
    {
        return ret;
    }

    if (!lLaMa.Init(attr))
    {
        ALOGE("lLaMa.Init failed");
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }

    //
    if (b_continue)
    {
        printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
        // lLaMa.Reset();
    }
    std::vector<unsigned short> prompt_data;
    std::string last_reply;
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    int precompute_len = 0;
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
        std::vector<int> tokens_ids, tokens_diff;
        lLaMa.Encode(prompt_data, prompt_complete(prompt, attr.tokenizer_type), last_reply, tokens_ids, tokens_diff);
        lLaMa.SetKVCache(k_caches, v_caches, precompute_len, tokens_diff.size());
        last_reply = lLaMa.Run(prompt_data);
        lLaMa.GetKVCache(k_caches, v_caches, precompute_len);

        if (!b_live_print)
            printf("%s\n", last_reply.c_str());
    }

    lLaMa.Deinit();
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}