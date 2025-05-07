#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <queue>
#include <signal.h>

#include "runner/utils/httplib.h"
#include "runner/utils/json.hpp"
#include "runner/utils/string_utility.hpp"
#include "runner/LLM.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <glob.h>
#endif
#include <future>
#include <cmdline.hpp>

#define IS_AXCL 0

#if !IS_AXCL
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

std::vector<std::string> glob(const std::string &pattern)
{
    std::vector<std::string> results;

#ifdef _WIN32
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &findFileData);

    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                results.push_back(findFileData.cFileName);
            }
        } while (FindNextFileA(hFind, &findFileData) != 0);
        FindClose(hFind);
    }
#else
    glob_t glob_result;
    if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0)
    {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        {
            results.emplace_back(glob_result.gl_pathv[i]);
        }
        globfree(&glob_result);
    }
#endif

    return results;
}

httplib::Server svr;
const int PORT = 8000;
const std::string UPLOAD_DIR = "uploads/";
const std::string MODEL_DIR = "models/";

static std::queue<std::string> g_msg_queue;
static std::mutex g_msg_locker;

void __sigExit(int iSigNo)
{
    svr.stop();
    return;
}

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    fprintf(stdout, "%s", p_str);
    fflush(stdout);
    std::lock_guard<std::mutex> tmp_locker(g_msg_locker);
    g_msg_queue.push(p_str);
}

template <typename T>
T getAttr(nlohmann::json &json, const std::string &key, T default_value)
{
    if (json.contains(key))
    {
        return json[key].get<T>();
    }
    return default_value;
}

class Worker
{
public:
    LLM gllm;
    std::atomic<bool> gllm_runing = false;
    bool gllm_init = false;
    bool gllm_initing = false;

    std::vector<int> _token_ids;
    std::vector<unsigned short> prompt_data;
    std::string last_reply;
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    int precompute_len = 0;

private:
    using Task = std::function<void()>;
    std::thread worker_thread;
    std::queue<Task> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop_flag;

    void run()
    {
        while (true)
        {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this]
                               { return stop_flag || !tasks.empty(); });
                if (stop_flag && tasks.empty())
                {
                    break;
                }
                task = std::move(tasks.front());
                tasks.pop();
            }
            task(); // 执行任务
        }
    }

    // **支持无参数任务**
    void addTask(Task task)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.push(std::move(task));
        }
        condition.notify_one();
    }

    // 模板接口：添加任务并返回 std::future 用于获取返回值
    template <typename F, typename... Args>
    auto addTaskWithResult(F &&f, Args &&...args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using result_type = typename std::result_of<F(Args...)>::type;
        // 将函数及其参数绑定成一个无参函数
        auto task = std::make_shared<std::packaged_task<result_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<result_type> res = task->get_future();
        // 将任务封装为 lambda，确保在工作线程中执行
        addTask([task]()
                { (*task)(); });
        return res;
    }

public:
    Worker() : stop_flag(false) {}

    ~Worker()
    {
        Stop();
    }

    bool Run()
    {
        worker_thread = std::thread(&Worker::run, this);
        return true;
    }

    void Stop()
    {
        if (!stop_flag)
        {
            stop_flag = true;
            condition.notify_one();
            if (worker_thread.joinable())
            {
                worker_thread.join();
            }
        }
    }

    // **支持带参数的任务**
    void RunAsync(std::string prompt)
    {

        addTask([this, prompt]()
                { RunSync(prompt, llm_running_callback); });
    }

    void ResetASync(std::string system_prompt)
    {
        addTask([this, system_prompt]()
                { ResetSync(system_prompt); });
    }

    void ResetSync(std::string system_prompt)
    {
        if (gllm_runing)
        {
            return;
        }
        gllm_runing = true;
        gllm.SetSystemPrompt(system_prompt, _token_ids);
        gllm.GenerateKVCachePrefill(_token_ids, k_caches, v_caches, precompute_len);
        gllm_runing = false;
    }

    std::string RunSync(std::string prompt, LLMRuningCallback cb)
    {
        if (gllm_runing)
        {
            return "";
        }
        gllm_runing = true;
        gllm.getAttr()->runing_callback = cb;
        // auto output = gllm.Run(prompt);
        std::vector<int> tokens_ids, tokens_diff;
        gllm.Encode(prompt_data, prompt, last_reply, tokens_ids, tokens_diff);
        if (auto ret = gllm.SetKVCache(k_caches, v_caches, precompute_len, tokens_diff.size()); ret != 0)
        {
            return "error: SetKVCache failed: %d,the context may be full,please reset";
        }
        last_reply = gllm.Run(prompt_data);
        gllm.GetKVCache(k_caches, v_caches, precompute_len);

        gllm_runing = false;
        // std::cout << "Chat result: " << output << std::endl;
        return last_reply;
    }
};

Worker worker;

bool check_model_available(const httplib::Request &req, httplib::Response &res)
{
    if (!worker.gllm_init)
    {
        if (worker.gllm_initing)
        {
            ALOGE("model initing");
            res.status = 400;
            res.set_content("{\"error\": \"Model initing\"}", "application/json");
            return false;
        }
        else
        {
            ALOGE("model not init");
            res.status = 400;
            res.set_content("{\"error\": \"Model not init\"}", "application/json");
            return false;
        }
    }

    if (worker.gllm_runing)
    {
        res.status = 400;
        res.set_content("{\"error\": \"llm is running\"}", "application/json");
        return false;
    }

    return true;
}

void set_llm_config(nlohmann::json &body)
{
    if (body.contains("temperature"))
    {
        float temperature = body["temperature"];
        ALOGI("temperature: %f", temperature);
        if (temperature > 0)
        {
            worker.gllm.getPostprocess()->set_temperature(true, temperature);
        }
        else
        {
            ALOGE("temperature: %f is invalid", temperature);
            worker.gllm.getPostprocess()->set_temperature(false, temperature);
        }
    }
    if (body.contains("repetition_penalty"))
    {
        float repetition_penalty = body["repetition_penalty"];
        ALOGI("repetition_penalty: %f", repetition_penalty);
        if (repetition_penalty - 1 < 0.0001)
        {
            ALOGI("repetition_penalty: %f is skip", repetition_penalty);
        }
        else
        {
            worker.gllm.getPostprocess()->set_repetition_penalty(true, repetition_penalty);
        }
    }
    if (body.contains("top-p"))
    {
        float top_p = body["top-p"];
        ALOGI("top-p: %f", top_p);
        if (top_p > 0 && top_p < 1)
        {
            worker.gllm.getPostprocess()->set_top_p_sampling(true, top_p);
        }
        else
        {
            ALOGE("top-p: %f is invalid", top_p);
            worker.gllm.getPostprocess()->set_top_p_sampling(false, top_p);
        }
    }
    if (body.contains("top-k"))
    {
        int top_k = body["top-k"];
        ALOGI("top-k: %d", top_k);
        if (top_k > 0)
        {
            worker.gllm.getPostprocess()->set_top_k_sampling(true, top_k);
        }
        else
        {
            ALOGE("top-k: %d is invalid", top_k);
            worker.gllm.getPostprocess()->set_top_k_sampling(false, top_k);
        }
    }
}

void content_provider(const httplib::Request &req, httplib::Response &res)
{
    std::lock_guard<std::mutex> tmp_locker(g_msg_locker);
    std::string bot_response;

    while (!g_msg_queue.empty())
    {
        auto str = g_msg_queue.front(); // 用 front 取出
        g_msg_queue.pop();

        bot_response += str;
    }

    // 最后发送 done=true
    nlohmann::json chunk;
    chunk["response"] = bot_response; // 最后不再补发内容，避免重复
    if (worker.gllm_runing)
        chunk["done"] = false;
    else
        chunk["done"] = true;

    res.status = 200;
    res.set_content(chunk.dump(), "application/json");
}

void handle_generate(const httplib::Request &req, httplib::Response &res)
{
    auto body = nlohmann::json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.contains("prompt"))
    {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }

    if (!check_model_available(req, res))
    {
        ALOGE("model not available");
        return;
    }

    set_llm_config(body);

    std::string prompt = body["prompt"];
    ALOGI("%s", prompt.c_str());
    worker.RunAsync(prompt);
    res.status = 200;
    res.set_content("{\"status\": \"ok\"}", "application/json");
}

void handle_stop(const httplib::Request &req, httplib::Response &res)
{
    worker.gllm.Stop();

    res.status = 200;
    res.set_content("{\"status\": \"ok\"}", "application/json");
    return;
}

void handle_reset(const httplib::Request &req, httplib::Response &res)
{
    auto body = nlohmann::json::parse(req.body, nullptr, false);
    if (body.is_discarded())
    {
        ALOGE("Invalid request format, body is discarded %s", req.body.c_str());
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }
    std::string system_prompt;
    if (body.contains("system_prompt"))
    {
        system_prompt = body["system_prompt"];
    }

    if (!check_model_available(req, res))
    {
        ALOGE("model not available");
        return;
    }

    worker.ResetASync(system_prompt);

    res.status = 200;
    res.set_content("{\"status\": \"ok\"}", "application/json");
    return;
}

void handle_chat(const httplib::Request &req, httplib::Response &res)
{
    auto body = nlohmann::json::parse(req.body, nullptr, false);

    if (body.is_discarded() || !body.contains("messages"))
    {
        ALOGE("Invalid request format");
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }
    if (!check_model_available(req, res))
    {
        ALOGE("model not available");
        return;
    }

    set_llm_config(body);

    std::vector<nlohmann::json> messages = body["messages"];

    for (auto &message : messages)
    {
        if (message.contains("role") && message.contains("content"))
        {
            auto output = worker.RunSync(message["content"].get<std::string>(), nullptr);

            nlohmann::json response;
            response["message"] = output;
            response["done"] = true;

            res.status = 200;
            res.set_content(response.dump(), "application/json");

            return;
        }
    }

    res.status = 400;
    res.set_content("{\"error\": \"Invalid message format\"}", "application/json");
    return;
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);

    LLMAttrType attr;
    cmdline::parser cmd;
    cmd.add<std::string>("system_prompt", 0, "system prompt", false, attr.system_prompt);
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

#if IS_AXCL
    cmd.add<std::string>("devices", 0, "devices id,for example: \"0,1,2,3\" ", true, "0,1,2,3");
#endif
    cmd.parse_check(argc, argv);

    attr.system_prompt = cmd.get<std::string>("system_prompt");
    attr.tokenizer_type = (TokenizerType)cmd.get<int>("tokenizer_type");
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

    if (!worker.gllm.Init(attr))
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
    worker.gllm_init = true;

    worker.Run();
    svr.Get("/api/stop", handle_stop);
    svr.Post("/api/reset", handle_reset);
    svr.Post("/api/generate", handle_generate);
    svr.Get("/api/generate_provider", content_provider);
    svr.Post("/api/chat", handle_chat);

    svr.set_pre_routing_handler([](const httplib::Request &req, httplib::Response &res) -> httplib::Server::HandlerResponse
                                {
                                    res.set_header("Access-Control-Allow-Origin", "*");
                                    if (req.method == "OPTIONS")
                                    {
                                        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                                        res.set_header("Access-Control-Allow-Headers", "Content-Type");
                                        res.status = 200;
                                        return httplib::Server::HandlerResponse::Handled; // 表示已处理，不再继续
                                    }
                                    return httplib::Server::HandlerResponse::Unhandled; // 继续处理请求
                                });

    std::cout << "Server running on port " << PORT << "..." << std::endl;
    svr.listen("0.0.0.0", PORT);
    worker.Stop();
    worker.gllm.Deinit();
#if IS_AXCL
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif
    return 0;
}