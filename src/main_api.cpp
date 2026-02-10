#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <queue>
#include <signal.h>
#include <filesystem> // C++17

#include "runner/LLM.hpp"

#include "openai_api/server.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <glob.h>
#endif
#include <future>
#include <cmdline.hpp>

#define IS_AXCL 0

#if IS_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

openai_api::Server server;

void __sigExit(int iSigNo)
{
    server.stop();
    return;
}

bool handle_body(const nlohmann::json &messages, std::vector<Content> &history)
{
    for (auto &item : messages)
    {
        Content content;
        content.type = TEXT;
        if (item.contains("role") && item["role"] == "system")
        {
            content.role = SYSTEM;
        }
        else if (item.contains("role") && item["role"] == "user")
        {
            content.role = USER;
        }
        else if (item.contains("role") && item["role"] == "assistant")
        {
            content.role = ASSISTANT;
        }
        else
        {
            ALOGE("content type not support");
            return false;
        }

        if (item.contains("content") && item["content"].is_string())
        {
            content.data = item["content"];
        }
        else if (item.contains("content") && item["content"].is_array())
        {
            for (auto &item : item["content"])
            {
                if (item.contains("type") && item["type"] == "text")
                {
                    content.data += item["text"];
                }
            }
        }
        else
        {
            ALOGE("content type not support");
            return false;
        }
        history.push_back(content);
    }

    for (auto &content : history)
    {
        switch (content.role)
        {
        case SYSTEM:
            printf("\33[33msystem:%s\33[0m\n", content.data.c_str());
            break;
        case USER:
            printf("\33[32muser:%s\33[0m\n", content.data.c_str());
            break;
        case ASSISTANT:
            printf("\33[34massistant:%s\33[0m\n", content.data.c_str());
            break;
        default:
            break;
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);

    std::string kModelName = "AXERA-TECH/Qwen3-1.7B";
    int port = 8080;
    LLM llm;

    LLMAttrType attr;
    cmdline::parser cmd;
    cmd.add<int>("port", 0, "port", false, port);
    cmd.add<std::string>("model_name", 0, "model name", false, kModelName);
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("url_tokenizer_model", 0, "tokenizer model path", false, attr.url_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    // cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);

#if IS_AXCL
    cmd.add<std::string>("devices", 0, "devices id,for example: \"0,1,2,3\" ", true, "0,1,2,3");
#endif
    cmd.parse_check(argc, argv);
    port = cmd.get<int>("port");
    attr.url_tokenizer_model = cmd.get<std::string>("url_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");

    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");

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

    if (!llm.Init(attr))
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

    server.setMaxConcurrency(1);
    UTF8Filter g_utf8_filter;

    server.registerChat(kModelName, [&llm](const openai_api::ChatRequest &req, std::shared_ptr<openai_api::BaseDataProvider> provider)
                        {
                if (!provider->is_writable()) {
                    ALOGE("provider not writable");
                        return;
                    }

                std::vector<Content> history;
                if (!handle_body(req.messages, history))
                {
                    ALOGE("handle_body failed");
                } 

                //void llm_running_callback(const char *p_str, float token_per_sec, void *reserve)
                auto callback = [provider](const char *p_str, float token_per_sec, void *reserve)
                {
                    if (!provider->is_writable())
                    {
                        ALOGE("provider not writable");
                        return;
                    }
                    openai_api::OutputChunk chunk;
                    chunk.type = openai_api::OutputChunkType::TextDelta;
                    chunk.text = p_str;
                    provider->push(chunk);
                };
                llm.getAttr()->runing_callback = callback;
                llm.Run(history, req.max_tokens);
                provider->end(); });
    server.run(port);

    llm.Deinit();

#if IS_AXCL
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif
    return 0;
}