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

void __sigExit(int iSigNo)
{
    svr.stop();
    return;
}

// 放在函数外部，或者作为类的成员变量
// 用于缓存上一次回调遗留下来的“半个字符”
static std::string g_utf8_buffer = ""; 
// 保护 buffer 的锁（如果 callback 是单线程调用的，这个锁可以去掉，但保留更安全）
static std::mutex g_buffer_mutex; 

// 辅助函数：计算字符串中“有效完整 UTF-8”部分的长度
// 返回值是“可以安全发送的字节数”
size_t get_valid_utf8_len(const std::string &str) {
    size_t len = str.length();
    if (len == 0) return 0;

    // 从字符串末尾开始回溯，检查最后一个字符是否完整
    // UTF-8 最大长度为 4 字节，所以最多回溯 4 步
    for (int i = 0; i < 4; ++i) {
        if (len <= i) break; // 已经回溯到头了

        unsigned char byte = static_cast<unsigned char>(str[len - 1 - i]);

        // 1. 如果是 ASCII (0xxxxxxx)，那就是完整的，结束在它后面
        if ((byte & 0x80) == 0) {
            // 如果回溯了 (i > 0)，说明后面跟着的字节是不合法的孤立后缀，
            // 但在流式场景下，我们通常假设数据流是合法的，只是还没发完。
            // 这里为了简单：如果最后一位是 ASCII，那它就是完整的边界。
            if (i == 0) return len; 
            // 如果 i > 0，说明后面有 i 个 continuation byte 找不到头，这属于流还没到齐
            // 继续找头
        }

        // 2. 检查是否是 Header Byte (11xxxxxx)
        if ((byte & 0xC0) == 0xC0) {
            int needed_extra = 0;
            if ((byte & 0xE0) == 0xC0) needed_extra = 1;      // 2-byte char
            else if ((byte & 0xF0) == 0xE0) needed_extra = 2; // 3-byte char
            else if ((byte & 0xF8) == 0xF0) needed_extra = 3; // 4-byte char
            
            // i 是当前 Header 后面实际跟的字节数
            if (i >= needed_extra) {
                return len; // 完整了
            } else {
                // 不完整，这个 Header 以及后面的 i 个字节都要留给下一次
                return len - 1 - i;
            }
        }
        
        // 3. 如果是 Continuation Byte (10xxxxxx)，继续回溯找 Header
    }
    
    // 如果回溯 4 步都没找到 Header 或 ASCII，说明数据可能有问题
    // 为了防止死锁，我们假设全部发送（让 JSON 库去处理或报错，或者丢弃）
    // 但在流式拼接中，通常返回 0 等待更多数据
    return 0; 
}

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    // 1. 打印日志 (可选)
    // fprintf(stdout, "%s", p_str);
    // fflush(stdout);

    if (!p_str || *p_str == '\0') return;

    std::lock_guard<std::mutex> buffer_lock(g_buffer_mutex);

    // 2. 将新数据拼接到缓存中
    g_utf8_buffer += p_str;

    // 3. 计算缓存中有多长的数据是“UTF-8 安全”的
    size_t send_len = get_valid_utf8_len(g_utf8_buffer);

    if (send_len > 0) {
        // 4. 切割出完整部分
        std::string part_to_send = g_utf8_buffer.substr(0, send_len);
        
        // 5. 将完整部分推入消息队列
        {
            std::lock_guard<std::mutex> queue_lk(g_msg_locker);
            g_msg_queue.push(std::move(part_to_send));
        }
        g_msg_cv.notify_one();

        // 6. 保留剩下的残缺部分（如果有）到下一次
        if (send_len < g_utf8_buffer.size()) {
            g_utf8_buffer = g_utf8_buffer.substr(send_len);
        } else {
            g_utf8_buffer.clear();
        }
    }
    // 如果 send_len == 0，说明当前 buffer 里只有半个汉字，什么都不做，等下一次回调
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

    std::vector<Content> history;

private:
    using Task = std::function<void()>;
    std::thread worker_thread;
    std::queue<Task> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop_flag;

    void reset()
    {
        // std::vector<unsigned short>().swap(prompt_data);
    }

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

    void ResetSync(std::string system_prompt)
    {
        if (gllm_runing)
        {
            return;
        }
        gllm_runing = true;

        gllm.ResetKVCache();
        history = {{SYSTEM, TEXT, system_prompt}};
        gllm_runing = false;
    }

    void ResetASync(std::string system_prompt)
    {
        addTask([this, system_prompt]()
                { ResetSync(system_prompt); });
    }

    // **支持带参数的任务**
    void RunAsync(std::string prompt)
    {
        addTask([this, prompt]()
                { RunSync(prompt, llm_running_callback); });
    }

    std::string RunSync(std::string prompt, LLMRuningCallback cb)
    {
        gllm_runing = true;
        gllm.getAttr()->runing_callback = cb;

        history.push_back({USER, TEXT, prompt});
        history = gllm.Run(history);

        gllm_runing = false;
        g_msg_cv.notify_all(); // 唤醒 content_provider_stream 的等待
        std::cout << "Chat result: " << history.back().data << std::endl;
        reset();
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

bool content_provider_stream(size_t /*offset*/, httplib::DataSink &sink)
{
    auto send_sse = [&](const std::string &json_str)
    {
        sink.write("data: ", 6);
        sink.write(json_str.data(), json_str.size());
        sink.write("\n\n", 2);
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

bool handle_body(const nlohmann::json &body, std::string &prompt, std::vector<std::string> &image_paths, bool &b_video, bool &stream)
{
    if (body.contains("stream") && body["stream"].is_boolean())
    {
        stream = body["stream"];
    }
    std::string model = body["model"];
    ALOGI("model:%s\n", model.c_str());

    nlohmann::json messages = body["messages"];

    if (messages.contains("role") &&
        messages["role"] == "user" &&
        messages.contains("content"))
    {
        if (messages["content"].is_array())
        {
            for (auto &item : messages["content"])
            {
                if (item.contains("type") && item["type"] == "text")
                {
                    prompt = item["text"];
                }
                else if (item.contains("type") && item["type"] == "image_url")
                {
                    if (item["image_url"].is_array())
                    {
                        if (item.contains("is_video") && item["is_video"].is_boolean())
                        {
                            b_video = item["is_video"];
                        }
                        for (auto &img : item["image_url"])
                        {
                            image_paths.push_back(img);
                        }
                    }
                    else
                    {
                        image_paths.push_back(item["image_url"]);
                    }
                }
            }
        }
        else if (messages["content"].is_string())
        {
            prompt = messages["content"];
        }
        else
        {
            ALOGE("content type not support");
            return false;
        }
    }
    else
    {
        ALOGE("content type not support");
        return false;
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

    std::string prompt;
    std::vector<std::string> image_paths;
    bool b_video = false;
    bool stream = false;
    if (!handle_body(body, prompt, image_paths, b_video, stream))
    {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }

    ALOGI("prompt:%s  stream:%d", prompt.c_str(), stream);

    if (stream)
    {
        worker.gllm_runing = true;
        worker.RunAsync(prompt);

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
        auto output = worker.RunSync(prompt, nullptr);

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
        // 可选：统计用
        response["usage"] = {
            {"prompt_tokens", prompt.size()},
            {"completion_tokens", output.size()},
            {"total_tokens", prompt.size() + output.size()}};

        res.status = 200;
        res.set_content(response.dump(), "application/json");
    }
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
    cmd.add<std::string>("system_prompt", 0, "system prompt", false, attr.system_prompt);
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

    attr.system_prompt = cmd.get<std::string>("system_prompt");
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

    svr.Post("/v1/reset", handle_reset);

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