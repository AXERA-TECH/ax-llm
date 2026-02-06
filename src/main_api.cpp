#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <queue>
#include <signal.h>
#include <filesystem> // C++17

#include "runner/utils/httplib.h"
#include "runner/utils/json.hpp"
#include "runner/utils/string_utility.hpp"
#include "runner/LLM.hpp"
#include "UTF8Filter.hpp"

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

static const char *kModelName = "AXERA-TECH/Qwen3-1.7B"; // 保持和聊天里返回的 model 一致

httplib::Server svr;
const int PORT = 8000;

static std::queue<std::string> g_msg_queue;
static std::condition_variable g_msg_cv;
static std::mutex g_msg_locker;

static UTF8Filter g_utf8_filter;

void __sigExit(int iSigNo)
{
    svr.stop();
    return;
}

void llm_running_callback(const char *p_str, float token_per_sec, void *reserve)
{
    // fprintf(stdout, "%s", p_str);
    // fflush(stdout);

    if (!p_str || *p_str == '\0')
        return;

    {
        std::lock_guard<std::mutex> queue_lk(g_msg_locker);
        g_msg_queue.push(g_utf8_filter.filter(p_str));
    }
    g_msg_cv.notify_one();
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
    void RunAsync(std::vector<Content> history)
    {
        addTask([this, history]()
                { RunSync(history, llm_running_callback); });
    }

    std::string RunSync(std::vector<Content> history, LLMRuningCallback cb)
    {
        gllm_runing = true;
        gllm.getAttr()->runing_callback = cb;

        history = gllm.Run(history);

        gllm_runing = false;
        g_msg_cv.notify_all(); // 唤醒 content_provider_stream 的等待
        std::cout << "Chat result: " << history.back().data << std::endl;
        return history.back().data;
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

// ====== 处理函数 ======
void handle_models_list(const httplib::Request &req, httplib::Response &res)
{
    nlohmann::json model;
    model["id"] = kModelName;
    model["object"] = "model";
    model["created"] = std::time(nullptr);
    model["owned_by"] = "owner";

    nlohmann::json resp;
    resp["object"] = "list";
    resp["data"] = nlohmann::json::array({model});

    res.status = 200;
    res.set_content(resp.dump(), "application/json");
}

void handle_model_get(const httplib::Request &req, httplib::Response &res)
{
    // 路由里捕获的 {id}
    auto id = req.matches.size() > 1 ? req.matches[1].str() : "";

    if (id == kModelName)
    {
        nlohmann::json model;
        model["id"] = kModelName;
        model["object"] = "model";
        model["created"] = std::time(nullptr);
        model["owned_by"] = "owner";
        res.status = 200;
        res.set_content(model.dump(), "application/json");
    }
    else
    {
        nlohmann::json err;
        err["error"] = {
            {"message", "Model not found"},
            {"type", "invalid_request_error"},
            {"param", "model"},
            {"code", "model_not_found"}};
        res.status = 404;
        res.set_content(err.dump(), "application/json");
    }
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

bool content_provider_stream(size_t /*offset*/, httplib::DataSink &sink)
{
    auto send_sse = [&](const std::string &json_str)
    {
        if (sink.is_writable())
        {
            sink.write("data: ", 6);
            sink.write(json_str.data(), json_str.size());
            sink.write("\n\n", 2);
        }
    };

    const std::string id = "cmpl-" + std::to_string(std::time(nullptr));
    const long created = std::time(nullptr);
    const std::string model_name = kModelName;

    // 先发一帧角色（可选，兼容性更好）
    {
        nlohmann::json first;
        first["id"] = id;
        first["object"] = "chat.completion.chunk";
        first["created"] = created;
        first["model"] = model_name;
        nlohmann::json ch;
        ch["index"] = 0;
        ch["delta"] = {{"role", "assistant"}};
        ch["finish_reason"] = nullptr;
        first["choices"] = nlohmann::json::array({ch});
        send_sse(first.dump());
    }

    for (;;)
    {
        std::string str;

        // 阻塞等待：直到队列非空或推理结束
        {
            std::unique_lock<std::mutex> lk(g_msg_locker);
            g_msg_cv.wait(lk, [&]
                          { return !g_msg_queue.empty() || !worker.gllm_runing.load(); });

            if (!g_msg_queue.empty())
            {
                str = std::move(g_msg_queue.front());
                g_msg_queue.pop();
            }
            else
            {
                // 队列空且不再运行 -> 退出发送循环
                break;
            }
        }

        // 发送一个 OpenAI 样式 chunk
        nlohmann::json chunk;
        chunk["id"] = id;
        chunk["object"] = "chat.completion.chunk";
        chunk["created"] = created;
        chunk["model"] = model_name;

        nlohmann::json choice;
        choice["index"] = 0;
        choice["delta"] = {{"content", str}};
        choice["finish_reason"] = nullptr;
        chunk["choices"] = nlohmann::json::array({choice});

        send_sse(chunk.dump());
    }

    // 收尾：空 delta + finish_reason=stop
    {
        nlohmann::json final_chunk;
        final_chunk["id"] = id;
        final_chunk["object"] = "chat.completion.chunk";
        final_chunk["created"] = created;
        final_chunk["model"] = model_name;

        nlohmann::json choice;
        choice["index"] = 0;
        choice["delta"] = nlohmann::json::object();
        choice["finish_reason"] = "stop";
        final_chunk["choices"] = nlohmann::json::array({choice});

        send_sse(final_chunk.dump());
    }

    // 最后一条 [DONE]
    sink.write("data: [DONE]\n\n", 14);
    sink.done();
    return true;
}

bool handle_body(const nlohmann::json &body, std::vector<Content> &history, bool &stream)
{
    if (body.contains("stream") && body["stream"].is_boolean())
    {
        stream = body["stream"];
    }
    std::string model = body["model"];
    ALOGI("model:%s\n", model.c_str());

    nlohmann::json messages = body["messages"];

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

void handle_generate(const httplib::Request &req, httplib::Response &res)
{
    auto body = nlohmann::json::parse(req.body, nullptr, false);
    if (body.is_discarded())
    {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }

    if (worker.gllm_runing.load())
    {
        res.status = 400;
        res.set_content("{\"error\": \"LLM is running\"}", "application/json");
        return;
    }

    // printf("body:%s\n", body.dump(4).c_str());

    if (!check_model_available(req, res))
    {
        ALOGE("model not available");
        return;
    }

    set_llm_config(body);

    printf("body:\n%s\n", body.dump(4).c_str());

    std::vector<Content> history;
    bool stream = false;
    if (!handle_body(body, history, stream))
    {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }

    if (stream)
    {
        worker.gllm_runing = true;
        worker.RunAsync(history);

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        // httplib 会在 set_chunked_content_provider 时自动设置 Transfer-Encoding: chunked
        res.set_chunked_content_provider(
            "text/event-stream", // ✅ SSE
            content_provider_stream);
        return; // ✅ 这里不要再 set_content，否则会覆盖
    }
    else
    {
        auto output = worker.RunSync(history, nullptr);

        nlohmann::json response;
        response["id"] = "cmpl-" + std::to_string(std::time(nullptr));
        response["object"] = "chat.completion";
        response["created"] = std::time(nullptr);
        response["model"] = kModelName;

        nlohmann::json choice;
        choice["index"] = 0;
        choice["message"] = {
            {"role", "assistant"},
            {"content", output}};
        choice["finish_reason"] = "stop";

        response["choices"] = nlohmann::json::array({choice});
        // // 可选：统计用
        // response["usage"] = {
        //     {"prompt_tokens", prompt.size()},
        //     {"completion_tokens", output.size()},
        //     {"total_tokens", prompt.size() + output.size()}};

        res.status = 200;
        res.set_content(response.dump(), "application/json");
    }
}

void handle_stop(const httplib::Request &req, httplib::Response &res)
{
    worker.gllm.Stop();

    res.status = 200;
    res.set_content("{\"status\": \"ok\"}", "application/json");
    return;
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);

    LLMAttrType attr;
    cmdline::parser cmd;
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

    svr.set_read_timeout(300);
    svr.set_write_timeout(300);
    svr.set_keep_alive_timeout(300);

    svr.Get("/v1/stop", handle_stop);
    svr.Post("/v1/chat/completions", handle_generate);
    // svr.Get("/v1/generate_provider", content_provider);
    // svr.Post("/v1/chat", handle_chat);
    // svr.Post("/v1/upload", handle_upload);

    // 列表
    svr.Get("/v1/models", handle_models_list);

    // 单个
    svr.Get(R"(/v1/models/(.+))", handle_model_get);

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
                                    printf("req.method:%s req.path:%s \n", req.method.c_str(), req.path.c_str());
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