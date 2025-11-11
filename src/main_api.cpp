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
#include "utils/mrope.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <glob.h>
#endif
#include <future>
#include <cmdline.hpp>

#define IS_AXCL 1

#if IS_AXCL
#include <axcl.h>
#else
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

static const char *kModelName = "AXERA-TECH/Qwen3-VL-2B-Instruct-GPTQ-Int4"; // 保持和聊天里返回的 model 一致

httplib::Server svr;
const int PORT = 8000;
const std::string UPLOAD_DIR = "uploads/";

static std::queue<std::string> g_msg_queue;
static std::condition_variable g_msg_cv;
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

    const size_t CHUNK = 256;
    std::string s = p_str ? std::string(p_str) : "";
    for (size_t i = 0; i < s.size(); i += CHUNK)
    {
        std::string part = s.substr(i, CHUNK);
        {
            std::lock_guard<std::mutex> lk(g_msg_locker);
            g_msg_queue.push(std::move(part));
        }
        g_msg_cv.notify_one();
    }
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
    Config config;
    std::atomic<bool> gllm_runing = false;
    bool gllm_init = false;
    bool gllm_initing = false;

    std::vector<unsigned short> prompt_data;
    std::vector<std::vector<unsigned short>> img_embed;
    std::vector<std::vector<float>> deepstack_features;
    std::vector<int> visual_pos_mask;
    std::vector<std::vector<int>> position_ids;

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

    bool is_file(const std::string &path)
    {
        struct stat st;
        if (stat(path.c_str(), &st) != 0)
            return false;
        return S_ISREG(st.st_mode);
    }

    bool is_base64(const std::string &str)
    {
        // Base64 characters
        static const std::string base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";

        // Empty string is not valid base64
        if (str.empty())
            return false;

        // All characters must be base64 or padding '='
        size_t padding_count = 0;
        for (size_t i = 0; i < str.size(); ++i)
        {
            char c = str[i];
            if (base64_chars.find(c) != std::string::npos)
            {
                continue;
            }
            else if (c == '=')
            {
                // Padding can only be at the end
                padding_count++;
                // '=' characters must be at the end and there can be at most two
                if (i < str.size() - 2)
                    return false;
            }
            else
            {
                // Invalid character
                return false;
            }
        }

        // Valid base64 has a length divisible by 4
        if (str.size() % 4 != 0)
            return false;

        // Padding must be 0, 1, or 2 for valid base64
        if (padding_count > 2)
            return false;

        // All checks passed
        return true;
    }

    std::string base64_decode(const std::string &encoded_string)
    {
        // 一个最简 base64 解码器实现
        static const std::string base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";

        int in_len = encoded_string.size();
        int i = 0, j = 0, in_ = 0;
        unsigned char char_array_4[4], char_array_3[3];
        std::string ret;

        while (in_len-- && (encoded_string[in_] != '=') &&
                   isalnum(encoded_string[in_]) ||
               (encoded_string[in_] == '+') || (encoded_string[in_] == '/'))
        {
            char_array_4[i++] = encoded_string[in_];
            in_++;
            if (i == 4)
            {
                for (i = 0; i < 4; i++)
                    char_array_4[i] = base64_chars.find(char_array_4[i]);

                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                for (i = 0; (i < 3); i++)
                    ret += char_array_3[i];
                i = 0;
            }
        }

        if (i)
        {
            for (j = i; j < 4; j++)
                char_array_4[j] = 0;

            for (j = 0; j < 4; j++)
                char_array_4[j] = base64_chars.find(char_array_4[j]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; (j < i - 1); j++)
                ret += char_array_3[j];
        }

        return ret;
    }

    cv::Mat dataUrlToMat(const std::string &dataUrl)
    {
        // 找到 base64 部分
        size_t commaPos = dataUrl.find(',');
        if (commaPos == std::string::npos)
        {
            throw std::runtime_error("Invalid data URL");
        }
        std::string base64Part = dataUrl.substr(commaPos + 1);

        // 解码 base64 -> 二进制数据
        std::string decoded = base64_decode(base64Part);
        std::vector<uchar> data(decoded.begin(), decoded.end());

        // 用 OpenCV 解码 JPEG 数据
        cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
        if (img.empty())
        {
            throw std::runtime_error("Failed to decode image");
        }
        return img;
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
    void RunAsync(std::string prompt, std::vector<std::string> image_paths, bool b_video)
    {
        addTask([this, prompt, image_paths, b_video]()
                { RunSync(prompt, image_paths, b_video, llm_running_callback); });
    }

    std::string RunSync(std::string prompt, std::vector<std::string> image_paths, bool b_video, LLMRuningCallback cb)
    {
        gllm_runing = true;
        gllm.getAttr()->runing_callback = cb;

        std::string output;

        std::vector<cv::Mat> srcs;
        for (auto &image_path : image_paths)
        {
            cv::Mat src;

            if (image_path.find("data:image") != std::string::npos)
            {
                src = dataUrlToMat(image_path);
            }
            else if (is_file(image_path))
            {
                src = cv::imread(image_path);
            }

            if (src.empty())
            {
                ALOGE("read image %s failed", image_path.c_str());
                continue;
            }

            ALOGI("width:%d, height:%d", src.cols, src.rows);

            // else if (is_file(image_path))
            // {

            //     ALOGI("width:%d, height:%d", src.cols, src.rows);
            // }

            srcs.push_back(src);
        }

        if (srcs.empty())
        {
            ALOGE("image prompt not found");
            // if (auto ret = gllm.Encode(prompt_data, prompt); ret != 0)
            // {
            //     ALOGE("lLaMa.Encode failed");
            //     return "";
            // }
            // output = gllm.Run(prompt_data);

            gllm.Encode(prompt_data, position_ids, config, prompt);
            output = gllm.Run(prompt_data, position_ids, deepstack_features, visual_pos_mask);
        }
        else
        {
            if (auto ret = gllm.EncodeImage(srcs, b_video, config, img_embed, deepstack_features); ret != 0)
            {
                ALOGE("lLaMa.Encode failed");
                return "";
            }
            if (auto ret = gllm.Encode(img_embed, b_video, prompt_data, position_ids, visual_pos_mask, config, prompt); ret != 0)
            {
                ALOGE("lLaMa.Encode failed");
                return "";
            }
            output = gllm.Run(prompt_data, position_ids, deepstack_features, visual_pos_mask);
        }

        gllm_runing = false;
        g_msg_cv.notify_all(); // 唤醒 content_provider_stream 的等待
        std::cout << "Chat result: " << output << std::endl;
        return output;
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

    ALOGI("prompt:%s  image_paths:%ld  b_video:%d  stream:%d", prompt.c_str(), image_paths.size(), b_video, stream);

    if (stream)
    {
        worker.gllm_runing = true;
        worker.RunAsync(prompt, image_paths, b_video);

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
        auto output = worker.RunSync(prompt, image_paths, b_video, nullptr);

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
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<std::string>("filename_image_encoder_axmodedl", 0, "vpm encoder axmodel path", false, attr.filename_image_encoder_axmodedl);

    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    // cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);

    cmd.add<std::string>("post_config_path", 0, "post config path", false, attr.post_config_path);

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

#if IS_AXCL
    cmd.add<std::string>("devices", 0, "devices id,for example: \"0,1,2,3\" ", true, "0,1,2,3");
#endif

    cmd.parse_check(argc, argv);

    cmd.parse_check(argc, argv);

    attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");

    attr.filename_image_encoder_axmodedl = cmd.get<std::string>("filename_image_encoder_axmodedl");
    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");

    attr.b_use_mmap_load_embed = cmd.get<bool>("use_mmap_load_embed");

    attr.post_config_path = cmd.get<std::string>("post_config_path");

    worker.config.vision_config.temporal_patch_size = cmd.get<int>("temporal_patch_size");
    worker.config.vision_config.tokens_per_second = cmd.get<int>("tokens_per_second");
    worker.config.vision_config.spatial_merge_size = cmd.get<int>("spatial_merge_size");
    worker.config.vision_config.patch_size = cmd.get<int>("patch_size");
    worker.config.vision_config.width = cmd.get<int>("img_width");
    worker.config.vision_config.height = cmd.get<int>("img_height");
    worker.config.vision_config.fps = cmd.get<int>("fps");

    worker.config.image_token_id = cmd.get<int>("img_token_id");
    worker.config.video_token_id = cmd.get<int>("video_token_id");
    worker.config.vision_start_token_id = cmd.get<int>("vision_start_token_id");

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