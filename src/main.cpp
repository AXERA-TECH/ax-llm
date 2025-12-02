#include "signal.h"

#include "runner/LLM.hpp"

#include "cmdline.hpp"

#define IS_AXCL 0

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

bool save_kvcache(std::string target_path, std::string system_prompt, int precompute_len, std::vector<std::vector<unsigned short>> &k_caches, std::vector<std::vector<unsigned short>> &v_caches)
{
    for (size_t i = 0; i < k_caches.size(); i++)
    {
        std::string k_cache_path = target_path + "/k_cache_" + std::to_string(i) + ".bin";
        std::string v_cache_path = target_path + "/v_cache_" + std::to_string(i) + ".bin";
        std::ofstream k_cache_file(k_cache_path);
        std::ofstream v_cache_file(v_cache_path);
        if (!k_cache_file.is_open() || !v_cache_file.is_open())
        {
            ALOGE("save kvcache failed");
            return false;
        }
        k_cache_file.write((char *)k_caches[i].data(), k_caches[i].size() * sizeof(unsigned short));
        v_cache_file.write((char *)v_caches[i].data(), v_caches[i].size() * sizeof(unsigned short));
        k_cache_file.close();
        v_cache_file.close();
    }
    nlohmann::json j;
    j["system_prompt"] = system_prompt;
    j["precompute_len"] = precompute_len;
    std::string config_path = target_path + "/config.json";
    std::ofstream config_file(config_path);
    config_file << j.dump();
    config_file.close();
    return true;
}

bool load_kvcache(std::string target_path, int axmodel_num, std::vector<std::vector<unsigned short>> &k_caches, std::vector<std::vector<unsigned short>> &v_caches, std::string &system_prompt, int &precompute_len)
{
    k_caches.resize(axmodel_num);
    v_caches.resize(axmodel_num);
    for (size_t i = 0; i < k_caches.size(); i++)
    {
        std::string k_cache_path = target_path + "/k_cache_" + std::to_string(i) + ".bin";
        std::string v_cache_path = target_path + "/v_cache_" + std::to_string(i) + ".bin";
        if (file_exist(k_cache_path) && file_exist(v_cache_path))
        {
            std::vector<unsigned short> k_cache;
            std::vector<unsigned short> v_cache;
            std::ifstream k_cache_file(k_cache_path);
            std::ifstream v_cache_file(v_cache_path);

            k_cache_file.seekg(0, std::ios::end);
            k_cache.resize(k_cache_file.tellg() / sizeof(unsigned short));
            k_cache_file.seekg(0, std::ios::beg);

            v_cache_file.seekg(0, std::ios::end);
            v_cache.resize(v_cache_file.tellg() / sizeof(unsigned short));
            v_cache_file.seekg(0, std::ios::beg);

            k_cache_file.read((char *)k_cache.data(), k_cache.size() * sizeof(unsigned short));
            v_cache_file.read((char *)v_cache.data(), v_cache.size() * sizeof(unsigned short));

            k_cache_file.close();
            v_cache_file.close();
            k_caches[i] = k_cache;
            v_caches[i] = v_cache;
        }
        else
        {
            ALOGE("k_cache %s or v_cache %s not exist", k_cache_path.c_str(), v_cache_path.c_str());
            return false;
        }
    }

    std::string config_path = target_path + "/config.json";
    if (file_exist(config_path))
    {
        std::ifstream config_file(config_path);
        nlohmann::json j;
        config_file >> j;
        system_prompt = j["system_prompt"].get<std::string>();
        precompute_len = j["precompute_len"].get<int>();
        config_file.close();
    }
    else
    {
        ALOGE("config %s not exist", config_path.c_str());
        return false;
    }
    return true;
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    LLMAttrType attr;
    std::string prompt = "Hi";
    bool b_continue = true;
    std::string kvcache_path;

    cmdline::parser cmd;
    cmd.add<std::string>("system_prompt", 0, "system prompt", false, attr.system_prompt);
    cmd.add<std::string>("kvcache_path", 0, "kvcache path", false, kvcache_path);
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("url_tokenizer_model", 0, "tokenizer model path", false, attr.url_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    // cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);

    cmd.add<bool>("live_print", 0, "print in live if set true, else print in end", false);
#if IS_AXCL
    cmd.add<std::string>("devices", 0, "devices id,for example: \"0,1,2,3\" ", true, "0,1,2,3");
#endif
    cmd.parse_check(argc, argv);

    attr.system_prompt = cmd.get<std::string>("system_prompt");
    kvcache_path = cmd.get<std::string>("kvcache_path");
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

    //
    if (b_continue)
    {
        printf("Type \"q\" to exit\nCtrl+c to stop current running\n\"reset\" to reset kvcache\n\"dd\" to remove last conversation.\n");
        // lLaMa.Reset();
    }

    std::vector<Content> history = {{SYSTEM, TEXT, attr.system_prompt}};

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
        if (prompt == "reset")
        {
            ALOGI("reset kvcache");
            lLaMa.ResetKVCache();
            history = {{SYSTEM, TEXT, attr.system_prompt}};
            continue;
        }
        if (prompt == "dd")
        {
            ALOGI("remove last conversation");
            if (history.size() >= 3) // system, user, assistant, user, assistant, ...
            {
                history.pop_back();
                history.pop_back();
            }

            continue;
        }

        history.push_back({USER, TEXT, prompt});
        history = lLaMa.Run(history);

        if (!b_live_print)
            printf("%s\n", history.back().data.c_str());
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