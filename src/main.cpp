#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <queue>
#include <signal.h>
#include <filesystem>
#include <sstream>
#include <termios.h>
#include <unistd.h>

#include "runner/LLM.hpp"
#include "openai_api/server.hpp"
#include "runner/utils/memory_utils.hpp"
#include "runner/utils/sample_log.h"

#ifdef USE_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

// Global variables
static LLM g_llm;
static openai_api::Server g_server;
static bool g_running = true;

// Terminal settings for handling UTF-8 backspace
struct termios g_orig_termios;
static bool g_terminal_modified = false;

void save_terminal_settings()
{
    tcgetattr(STDIN_FILENO, &g_orig_termios);
}

void restore_terminal_settings()
{
    if (g_terminal_modified)
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &g_orig_termios);
        g_terminal_modified = false;
    }
}

void setup_terminal_for_utf8()
{
    struct termios new_termios;
    tcgetattr(STDIN_FILENO, &new_termios);
    // Disable canonical mode and echo for custom input handling
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    g_terminal_modified = true;
}

void __sigExit(int iSigNo)
{
    g_running = false;
    g_llm.Stop();
    g_server.stop();
    restore_terminal_settings();
    return;
}

void llm_running_callback(std::string str, float token_per_sec, void *reserve)
{
    fprintf(stdout, "%s", str.c_str());
    fflush(stdout);
}

// Config structure for JSON configuration
struct ModelConfig
{
    // Model paths
    std::string model_name = "AXERA-TECH/Qwen3-1.7B";
    LLMAttrType attr;
    int port = 8000;

    bool load_from_json(const std::string &config_path)
    {
        if (!file_exist(config_path))
        {
            ALOGE("Config file not found: %s", config_path.c_str());
            return false;
        }

        try
        {
            std::ifstream f(config_path);
            nlohmann::json j;
            f >> j;
#define check_key(key)                   \
    if (!j.contains(key))                \
    {                                    \
        ALOGE("Key not found: %s", key); \
        return false;                    \
    }

            check_key("template_filename_axmodel");
            attr.template_filename_axmodel = j["template_filename_axmodel"].get<std::string>();

            check_key("filename_post_axmodel");
            attr.filename_post_axmodel = j["filename_post_axmodel"].get<std::string>();

            check_key("url_tokenizer_model");
            attr.url_tokenizer_model = j["url_tokenizer_model"].get<std::string>();

            check_key("tokenizer_type");
            attr.tokenizer_type = j["tokenizer_type"].get<std::string>();

            check_key("filename_tokens_embed");
            attr.filename_tokens_embed = j["filename_tokens_embed"].get<std::string>();

            check_key("post_config_path");
            attr.post_config_path = j["post_config_path"].get<std::string>();

            check_key("axmodel_num");
            attr.axmodel_num = j["axmodel_num"].get<int>();

            check_key("tokens_embed_num");
            attr.tokens_embed_num = j["tokens_embed_num"].get<int>();

            check_key("tokens_embed_size");
            attr.tokens_embed_size = j["tokens_embed_size"].get<int>();

            // Load options
            if (j.contains("b_use_mmap_load_embed"))
            {
                attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();
            }
            else if (j.contains("use_mmap_load_embed"))
            {
                attr.b_use_mmap_load_embed = j["use_mmap_load_embed"].get<bool>();
            }

            // Optional VLM switch
            if (j.contains("vlm_type") || j.contains("VLM_TYPE"))
            {
                const auto &v = j.contains("vlm_type") ? j["vlm_type"] : j["VLM_TYPE"];
                std::optional<VLMType> parsed;
                if (v.is_number_integer())
                {
                    parsed = VLMTypeFromInt(v.get<int>());
                }
                else if (v.is_string())
                {
                    parsed = VLMTypeFromString(v.get<std::string>());
                }
                else
                {
                    ALOGE("vlm_type must be int or string. choices: %s", VLMTypeChoices().c_str());
                    return false;
                }

                if (!parsed.has_value())
                {
                    ALOGE("invalid vlm_type value. choices: %s", VLMTypeChoices().c_str());
                    return false;
                }
                attr.vlm_type = *parsed;
            }

            if (j.contains("filename_image_encoder_axmodel"))
            {
                attr.filename_image_encoder_axmodel = j["filename_image_encoder_axmodel"].get<std::string>();
            }
            else if (j.contains("filename_image_encoder_axmodedl"))
            {
                // Backward compatible with older branches.
                attr.filename_image_encoder_axmodel = j["filename_image_encoder_axmodedl"].get<std::string>();
            }

            if (j.contains("vision_cache_dir"))
            {
                attr.vision_cache_dir = j["vision_cache_dir"].get<std::string>();
            }

            if (j.contains("vision_width")) attr.vision_width = j["vision_width"].get<int>();
            if (j.contains("vision_height")) attr.vision_height = j["vision_height"].get<int>();
            if (j.contains("vision_temporal_patch_size")) attr.vision_temporal_patch_size = j["vision_temporal_patch_size"].get<int>();
            if (j.contains("vision_spatial_merge_size")) attr.vision_spatial_merge_size = j["vision_spatial_merge_size"].get<int>();
            if (j.contains("vision_patch_size")) attr.vision_patch_size = j["vision_patch_size"].get<int>();
            if (j.contains("vision_fps")) attr.vision_fps = j["vision_fps"].get<int>();
            if (j.contains("vision_tokens_per_second")) attr.vision_tokens_per_second = j["vision_tokens_per_second"].get<int>();

#if USE_AXCL
            check_key("devices");
            attr.dev_ids = j["devices"].get<std::vector<int>>();

#endif
            // Load prompt
            if (j.contains("system_prompt"))
            {
                attr.system_prompt = j["system_prompt"].get<std::string>();
            }

            // Load server settings
            check_key("model_name");
            model_name = j["model_name"].get<std::string>();

            if (j.contains("port"))
            {
                port = j["port"].get<int>();
            }

            return true;
        }
        catch (const std::exception &e)
        {
            ALOGE("Failed to parse config file: %s", e.what());
            return false;
        }
    }
};

// Helper function to resolve relative paths
std::string resolve_path(const std::string &base_path, const std::string &relative_path)
{
    if (relative_path.empty())
        return relative_path;
    if (relative_path[0] == '/' || relative_path.substr(0, 2) == "./")
    {
        return relative_path; // Already absolute or explicit relative
    }
    return base_path + "/" + relative_path;
}

// Helper function to make paths absolute in config
static inline bool is_url(const std::string &p)
{
    auto pos = p.find("://");
    return pos != std::string::npos;
}

void resolve_config_paths(ModelConfig &config, const std::string &model_path)
{
    config.attr.template_filename_axmodel = resolve_path(model_path, config.attr.template_filename_axmodel);
    config.attr.filename_post_axmodel = resolve_path(model_path, config.attr.filename_post_axmodel);
    if (!is_url(config.attr.url_tokenizer_model))
        config.attr.url_tokenizer_model = resolve_path(model_path, config.attr.url_tokenizer_model);
    config.attr.filename_tokens_embed = resolve_path(model_path, config.attr.filename_tokens_embed);
    config.attr.post_config_path = resolve_path(model_path, config.attr.post_config_path);
    config.attr.filename_image_encoder_axmodel = resolve_path(model_path, config.attr.filename_image_encoder_axmodel);
    config.attr.vision_cache_dir = resolve_path(model_path, config.attr.vision_cache_dir);
}

// Read UTF-8 character length
size_t utf8_char_len(unsigned char c)
{
    if (c < 0x80)
        return 1;
    if ((c & 0xE0) == 0xC0)
        return 2;
    if ((c & 0xF0) == 0xE0)
        return 3;
    if ((c & 0xF8) == 0xF0)
        return 4;
    return 1; // Invalid UTF-8, treat as single byte
}

// Custom input handling for proper UTF-8 backspace support
std::string read_line_with_utf8_support()
{
    std::string line;
    char c;

    while (read(STDIN_FILENO, &c, 1) == 1)
    {
        if (c == '\n' || c == '\r')
        {
            printf("\n");
            fflush(stdout);
            break;
        }
        else if (c == 0x7F || c == '\b')
        { // Backspace or DEL
            if (!line.empty())
            {
                // Calculate how many bytes to remove for the last UTF-8 character
                size_t remove_len = 0;
                size_t pos = line.length();

                // Find the start of the last UTF-8 character
                while (pos > 0 && ((unsigned char)line[pos - 1] & 0x80) && !((unsigned char)line[pos - 1] & 0x40))
                {
                    pos--;
                }
                if (pos > 0)
                {
                    remove_len = line.length() - pos;
                    if (remove_len == 0)
                        remove_len = 1;

                    // Erase the last character
                    line.erase(line.length() - remove_len);

                    // Move cursor back and clear to end of line
                    for (size_t i = 0; i < remove_len; i++)
                    {
                        printf("\b \b");
                    }
                    fflush(stdout);
                }
            }
        }
        else if (c == 0x03)
        { // Ctrl+C
            printf("\n");
            fflush(stdout);
            raise(SIGINT);
            return "";
        }
        else if (c == 0x04)
        { // Ctrl+D
            if (line.empty())
            {
                printf("\n");
                fflush(stdout);
                return "q"; // Exit on Ctrl+D at empty line
            }
        }
        else
        {
            line.push_back(c);
            printf("%c", c);
            fflush(stdout);
        }
    }

    return line;
}

// Run interactive mode
int run_interactive_mode(ModelConfig &config)
{
    config.attr.runing_callback = llm_running_callback;
    config.attr.reserve = nullptr;

    // Initialize engine
#if USE_AXCL
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

    if (!g_llm.Init(config.attr))
    {
        ALOGE("LLM.Init failed");
#if USE_AXCL
        axclFinalize();
#else
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
#endif
        return -1;
    }

    printf("Type \"q\" to exit\n");
    printf("Ctrl+c to stop current running\n");
    printf("\"reset\" to reset kvcache\n");
    printf("\"dd\" to remove last conversation.\n");
    printf("\"pp\" to print history.\n");
    if (config.attr.vlm_type != VLMType::None)
    {
        printf("VLM enabled: after each prompt, input image path (empty = text-only). Use \"video:<frames_dir>\" for video.\n");
    }
    printf("----------------------------------------\n");

    std::vector<Content> history = {{SYSTEM, TEXT, config.attr.system_prompt}};
    std::vector<MediaInputs> media_inputs; // keep for the whole session (indices refer to `history`)

    // Setup terminal for UTF-8 input handling
    save_terminal_settings();
    setup_terminal_for_utf8();

    while (g_running)
    {
        printf("prompt >> ");
        fflush(stdout);

        std::string prompt = read_line_with_utf8_support();

        if (prompt == "q")
        {
            break;
        }
        if (prompt.empty())
        {
            continue;
        }
        if (prompt == "reset")
        {
            ALOGI("reset kvcache");
            g_llm.ResetKVCache();
            history = {{SYSTEM, TEXT, config.attr.system_prompt}};
            media_inputs.clear();
            continue;
        }
        if (prompt == "dd")
        {
            if (history.size() >= 3)
            {
                ALOGI("remove last conversation \nQ:%s \nA:%s",
                      history[history.size() - 2].data.c_str(),
                      history[history.size() - 1].data.c_str());
                history.pop_back();
                history.pop_back();

                // Drop any media mappings that refer to removed tail entries.
                while (!media_inputs.empty() && media_inputs.back().content_index >= history.size())
                {
                    media_inputs.pop_back();
                }
            }
            continue;
        }
        if (prompt == "pp")
        {
            ALOGI("history size: %zu", history.size());
            for (auto &item : history)
            {
                switch (item.role)
                {
                case SYSTEM:
                    printf("system: %s\n", item.data.c_str());
                    break;
                case USER:
                    if (item.type == IMAGE) printf("user(image): %s\n", item.data.c_str());
                    else if (item.type == VIDEO) printf("user(video): %s\n", item.data.c_str());
                    else printf("user: %s\n", item.data.c_str());
                    break;
                case ASSISTANT:
                    printf("assistant: %s\n", item.data.c_str());
                    break;
                default:
                    break;
                }
            }
            continue;
        }

        // Optional media input (VLM interactive workflow).
        bool has_media = false;
        bool is_video = false;
        std::vector<std::string> uris;
        if (config.attr.vlm_type != VLMType::None)
        {
            printf("image >> ");
            fflush(stdout);
            std::string media_line = read_line_with_utf8_support();
            if (!media_line.empty())
            {
                // Trim leading spaces.
                size_t p0 = 0;
                while (p0 < media_line.size() && (media_line[p0] == ' ' || media_line[p0] == '\t')) p0++;
                media_line = media_line.substr(p0);

                if (media_line.rfind("video:", 0) == 0 || media_line.rfind("VIDEO:", 0) == 0)
                {
                    is_video = true;
                    media_line = media_line.substr(6);
                    while (!media_line.empty() && (media_line[0] == ' ' || media_line[0] == '\t')) media_line.erase(media_line.begin());
                }

                // Split by whitespace for multiple image uris.
                std::istringstream iss(media_line);
                std::string tok;
                while (iss >> tok) uris.push_back(tok);
                has_media = !uris.empty();
            }
        }

        Content user;
        user.role = USER;
        user.data = prompt;
        user.type = (has_media ? (is_video ? VIDEO : IMAGE) : TEXT);

        const size_t idx = history.size();
        history.push_back(user);
        if (has_media)
        {
            media_inputs.push_back({idx, uris});
        }

        if (config.attr.vlm_type != VLMType::None && !media_inputs.empty())
        {
            history = g_llm.Run(history, media_inputs);
        }
        else
        {
            history = g_llm.Run(history);
        }
    }

    restore_terminal_settings();
    g_llm.Deinit();

#if USE_AXCL
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif

    return 0;
}

// Handle HTTP API messages
bool handle_api_messages(const nlohmann::json &messages, std::vector<Content> &history, std::vector<MediaInputs> *media_inputs = nullptr)
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

        std::vector<std::string> image_uris;

        if (item.contains("content") && item["content"].is_string())
        {
            content.data = item["content"];
        }
        else if (item.contains("content") && item["content"].is_array())
        {
            for (auto &c : item["content"])
            {
                if (c.contains("type") && c["type"] == "text")
                {
                    content.data += c["text"];
                }
                else if (c.contains("type") && c["type"] == "image_url")
                {
                    // OpenAI style: {type:"image_url", image_url:{url:"..."}}
                    if (c.contains("image_url") && c["image_url"].is_object() && c["image_url"].contains("url"))
                    {
                        image_uris.push_back(c["image_url"]["url"].get<std::string>());
                    }
                    else if (c.contains("image_url") && c["image_url"].is_string())
                    {
                        image_uris.push_back(c["image_url"].get<std::string>());
                    }
                }
            }
        }
        else
        {
            ALOGE("content type not support");
            return false;
        }

        if (!image_uris.empty() && content.role == USER)
        {
            content.type = IMAGE;
            if (media_inputs)
            {
                media_inputs->push_back({history.size(), image_uris});
            }
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

// Run server mode
int run_server_mode(const ModelConfig &config, int port)
{

    LLM llm;

    // Initialize engine
#if USE_AXCL
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

    if (!llm.Init(config.attr))
    {
        ALOGE("LLM.Init failed");
#if USE_AXCL
        axclFinalize();
#else
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
#endif
        return -1;
    }

    g_server.setMaxConcurrency(1);

    std::string model_name = config.model_name;

    g_server.registerChat(model_name, [&llm](const openai_api::ChatRequest &req,
                                             std::shared_ptr<openai_api::BaseDataProvider> provider)
                          {
        if (!provider->is_writable()) {
            ALOGE("provider not writable");
            return;
        }

        std::vector<Content> history;
        std::vector<MediaInputs> media_inputs;
        if (!handle_api_messages(req.messages, history, &media_inputs)) {
            ALOGE("handle_body failed");
            provider->end();
            return;
        }

        if (req.stream) {
            auto callback = [provider, model_id = req.model](std::string str, float token_per_sec, void *reserve) {
                if (!provider->is_writable()) {
                    ALOGE("provider not writable");
                    return;
                }
                auto chunk = openai_api::OutputChunk::TextDelta(str, model_id);
                provider->push(chunk);
                fprintf(stdout, "%s", str.c_str());
                fflush(stdout);
            };

            llm.getAttr()->runing_callback = callback;
            if (!media_inputs.empty()) llm.Run(history, media_inputs, req.max_tokens);
            else llm.Run(history, req.max_tokens);
        } else {
            llm.getAttr()->runing_callback = nullptr;
            auto out_history = (!media_inputs.empty()) ? llm.Run(history, media_inputs, req.max_tokens) : llm.Run(history, req.max_tokens);
            std::string final_text;
            if (!out_history.empty() && out_history.back().role == ASSISTANT) {
                final_text = out_history.back().data;
            }
            auto chunk = openai_api::OutputChunk::FinalText(final_text, req.model);
            fprintf(stdout, "%s", final_text.c_str());
            fflush(stdout);
            provider->push(chunk);
        }

        provider->end(); });

    printf("Starting server on port %d with model '%s'...\n", port, model_name.c_str());
    g_server.run(port);

    llm.Deinit();

#if USE_AXCL
    axclFinalize();
#else
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif

    return 0;
}

// Print usage
void print_usage(const char *program_name)
{
    printf("Usage:\n");
    printf("  %s run <model_path> [options]    Run interactive chat mode\n", program_name);
    printf("  %s serve <model_path> [options]  Run HTTP API server mode\n", program_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  model_path    Path to model directory containing config.json and model files\n");
    printf("\n");
    printf("Serve options:\n");
    printf("  --port <port> Server port (default: 8080)\n");
    printf("\n");
    printf("Model directory structure:\n");
    printf("  model_path/\n");
    printf("    ├── config.json          # Model configuration\n");
    printf("    ├── tokenizer.txt        # Tokenizer model\n");
    printf("    ├── *.axmodel            # AXera model files\n");
    printf("    └── post_config.json     # Post-processing config (optional)\n");
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);

    if (argc < 3)
    {
        print_usage(argv[0]);
        return -1;
    }

    std::string mode = argv[1];
    std::string model_path = argv[2];

    // Check if model path exists
    if (!std::filesystem::exists(model_path))
    {
        ALOGE("Model path does not exist: %s", model_path.c_str());
        return -1;
    }

    // Load config from model directory
    std::string config_path = model_path + "/config.json";
    ModelConfig config;

    if (!config.load_from_json(config_path))
    {
        ALOGE("Failed to load config from: %s", config_path.c_str());
        // Try to use default config and resolve paths
        ALOGE("Using default configuration");
    }

    // Resolve relative paths to absolute paths based on model_path
    resolve_config_paths(config, model_path);

    if (mode == "run")
    {
        for (int i = 3; i < argc; i++)
        {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h")
            {
                print_usage(argv[0]);
                return 0;
            }
        }
        return run_interactive_mode(config);
    }
    else if (mode == "serve")
    {
        // Parse serve mode options
        int port = config.port;
        for (int i = 3; i < argc; i++)
        {
            std::string arg = argv[i];
            if (arg == "--port" && i + 1 < argc)
            {
                port = std::stoi(argv[++i]);
            }
            else if (arg == "--help" || arg == "-h")
            {
                print_usage(argv[0]);
                return 0;
            }
        }
        return run_server_mode(config, port);
    }
    else
    {
        ALOGE("Unknown mode: %s", mode.c_str());
        print_usage(argv[0]);
        return -1;
    }

    return 0;
}
