#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>
#include <queue>
#include <signal.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <cctype>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

#include "runner/LLM.hpp"
#include "openai_api/server.hpp"
#include "runner/utils/memory_utils.hpp"
#include "runner/utils/net_utils.hpp"
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
static std::atomic<bool> g_running{true};
// In interactive mode: SIGINT/Ctrl+C stops current generation and returns to prompt.
// In server mode: SIGINT/Ctrl+C exits the process (by stopping server loop).
static std::atomic<bool> g_exit_on_sigint{true};

// Terminal settings for handling UTF-8 backspace
#ifndef _WIN32
struct termios g_orig_termios;
#endif
static bool g_terminal_modified = false;

void save_terminal_settings()
{
#ifndef _WIN32
    tcgetattr(STDIN_FILENO, &g_orig_termios);
#endif
}

void restore_terminal_settings()
{
#ifndef _WIN32
    if (g_terminal_modified)
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &g_orig_termios);
        g_terminal_modified = false;
    }
#endif
}

void setup_terminal_for_utf8()
{
#ifndef _WIN32
    struct termios new_termios;
    tcgetattr(STDIN_FILENO, &new_termios);
    // Disable canonical mode and echo for custom input handling
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    g_terminal_modified = true;
#endif
}

void __sigExit(int iSigNo)
{
    (void)iSigNo;
    g_llm.Stop();

    if (g_exit_on_sigint.load(std::memory_order_relaxed))
    {
        g_running.store(false, std::memory_order_relaxed);
        g_server.stop();
        restore_terminal_settings();
    }
    return;
}

#ifdef _WIN32
static BOOL WINAPI __ConsoleCtrlHandler(DWORD ctrl_type)
{
    switch (ctrl_type)
    {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        __sigExit(SIGINT);
        return TRUE;
    default:
        return FALSE;
    }
}
#endif

void llm_running_callback(std::string str, float token_per_sec, void *reserve)
{
    fprintf(stdout, "%s", str.c_str());
    fflush(stdout);
}

static inline std::string get_effective_system_prompt(const LLMAttrType &attr)
{
    return attr.system_prompt;
}

static inline std::vector<Content> make_initial_history(const LLMAttrType &attr)
{
    std::vector<Content> history;
    const std::string system_prompt = get_effective_system_prompt(attr);
    if (!system_prompt.empty())
    {
        history.push_back({SYSTEM, TEXT, system_prompt});
    }
    return history;
}

// Config structure for JSON configuration
struct ModelConfig
{
    // Model paths
    std::string model_name = "AXERA-TECH/Qwen3-1.7B";
    LLMAttrType attr;
    int port = 8000;
    bool is_embedding = false;

    bool is_embedding_model() const { return is_embedding; }

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

            const bool has_post_config_path = j.contains("post_config_path");
            if (has_post_config_path)
            {
                attr.post_config_path = j["post_config_path"].get<std::string>();
            }
            else
            {
                // Optional for embedding-only configs.
                attr.post_config_path = "post_config.json";
            }

            check_key("axmodel_num");
            attr.axmodel_num = j["axmodel_num"].get<int>();

            // Optional: models with mixed linear/full attention layers (e.g., Qwen3.5).
            // Keep config minimal: only an interval is needed (layer count comes from `axmodel_num`).
            attr.full_attention_interval = 0;
            if (j.contains("full_attention_interval"))
            {
                attr.full_attention_interval = j["full_attention_interval"].get<int>();
            }
            else if (j.contains("text_config") && j["text_config"].contains("full_attention_interval"))
            {
                attr.full_attention_interval = j["text_config"]["full_attention_interval"].get<int>();
            }
            attr.num_kv_shared_layers = 0;
            if (j.contains("num_kv_shared_layers"))
            {
                attr.num_kv_shared_layers = j["num_kv_shared_layers"].get<int>();
            }
            else if (j.contains("text_config") && j["text_config"].contains("num_kv_shared_layers"))
            {
                attr.num_kv_shared_layers = j["text_config"]["num_kv_shared_layers"].get<int>();
            }
            attr.layer_types.clear();
            if (j.contains("layer_types"))
            {
                attr.layer_types = j["layer_types"].get<std::vector<std::string>>();
            }
            else if (j.contains("text_config") && j["text_config"].contains("layer_types"))
            {
                attr.layer_types = j["text_config"]["layer_types"].get<std::vector<std::string>>();
            }

            check_key("tokens_embed_num");
            attr.tokens_embed_num = j["tokens_embed_num"].get<int>();

            check_key("tokens_embed_size");
            attr.tokens_embed_size = j["tokens_embed_size"].get<int>();
            if (j.contains("pad_token_id")) attr.pad_token_id = j["pad_token_id"].get<int>();
            if (j.contains("hidden_size_per_layer_input")) attr.hidden_size_per_layer_input = j["hidden_size_per_layer_input"].get<int>();
            if (j.contains("rms_norm_eps")) attr.rms_norm_eps = j["rms_norm_eps"].get<float>();
            if (j.contains("filename_tokens_embed_per_layer"))
            {
                attr.filename_tokens_embed_per_layer = j["filename_tokens_embed_per_layer"].get<std::string>();
            }
            if (j.contains("filename_per_layer_model_projection"))
            {
                attr.filename_per_layer_model_projection = j["filename_per_layer_model_projection"].get<std::string>();
            }
            if (j.contains("filename_per_layer_projection_norm"))
            {
                attr.filename_per_layer_projection_norm = j["filename_per_layer_projection_norm"].get<std::string>();
            }

            // Load options
            if (j.contains("b_use_mmap_load_embed"))
            {
                attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();
            }
            else if (j.contains("use_mmap_load_embed"))
            {
                attr.b_use_mmap_load_embed = j["use_mmap_load_embed"].get<bool>();
            }

            // Optional: embedding mode switch (serve mode only).
            // Prefer a simple bool; model family is specified via `tokenizer_type`.
            is_embedding = false;
            if (j.contains("is_embedding"))
            {
                is_embedding = j["is_embedding"].get<bool>();
            }
            else if (j.contains("embedding"))
            {
                is_embedding = j["embedding"].get<bool>();
            }
            else if (j.contains("embedding_type") || j.contains("EMBEDDING_TYPE"))
            {
                // Backward compatibility for older configs; the actual embedding behavior is specialized by `tokenizer_type`.
                is_embedding = true;
                ALOGW("config key `embedding_type` is deprecated; please use `is_embedding`: true");
            }
            if (is_embedding && !has_post_config_path)
            {
                // Embedding models don't require postprocess config; keep logs clean by skipping load.
                attr.post_config_path.clear();
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
            if (j.contains("devices"))
            {
                attr.dev_ids = j["devices"].get<std::vector<int>>();
            }
            else
            {
                attr.dev_ids = {0};
            }

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
    {
        return relative_path;
    }

    const std::filesystem::path path(relative_path);
    if (path.is_absolute() || relative_path.rfind("./", 0) == 0 || relative_path.rfind(".\\", 0) == 0)
    {
        return path.lexically_normal().string();
    }

    return (std::filesystem::path(base_path) / path).lexically_normal().string();
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
    config.attr.filename_tokens_embed_per_layer = resolve_path(model_path, config.attr.filename_tokens_embed_per_layer);
    config.attr.filename_per_layer_model_projection = resolve_path(model_path, config.attr.filename_per_layer_model_projection);
    config.attr.filename_per_layer_projection_norm = resolve_path(model_path, config.attr.filename_per_layer_projection_norm);
    config.attr.post_config_path = resolve_path(model_path, config.attr.post_config_path);
    config.attr.filename_image_encoder_axmodel = resolve_path(model_path, config.attr.filename_image_encoder_axmodel);
    config.attr.vision_cache_dir = resolve_path(model_path, config.attr.vision_cache_dir);
}

static std::string trim_copy(const std::string &s)
{
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(start, end - start);
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

#ifdef _WIN32
    if (!std::getline(std::cin, line))
    {
        if (std::cin.eof())
        {
            return "/exit";
        }
        // Ctrl+C may interrupt console input and put `cin` into a failed state even if we handled the event.
        std::cin.clear();
        return "";
    }
    return line;
#else
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
                return "/exit"; // Exit on Ctrl+D at empty line
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
#endif
}

// Run interactive mode
int run_interactive_mode(ModelConfig &config)
{
    if (config.is_embedding_model())
    {
        ALOGE("Embedding models do not support interactive `run` mode. Please use `serve` mode with `/v1/embeddings`.");
        return -1;
    }

    g_exit_on_sigint.store(false, std::memory_order_relaxed);
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

    printf("Commands:\n");
    printf("  /q, /exit  退出\n");
    printf("  /reset     重置 kvcache\n");
    printf("  /dd        删除一轮对话\n");
    printf("  /pp        打印历史对话\n");
    printf("Ctrl+C: 停止当前生成\n");
    if (config.attr.vlm_type != VLMType::None)
    {
        printf("VLM enabled: after each prompt, input media path (empty = text-only). Use \"video:<frames_dir>\" for video, \"audio:<file>\" for reserved audio placeholder.\n");
    }
    printf("----------------------------------------\n");

    std::vector<Content> history = make_initial_history(config.attr);
    std::vector<MediaInputs> media_inputs; // keep for the whole session (indices refer to `history`)

    // Setup terminal for UTF-8 input handling
    save_terminal_settings();
    setup_terminal_for_utf8();

    while (g_running.load(std::memory_order_relaxed))
    {
        printf("prompt >> ");
        fflush(stdout);

        std::string prompt = read_line_with_utf8_support();
        const std::string cmd = trim_copy(prompt);

        if (cmd == "/q" || cmd == "/exit")
        {
            break;
        }
        if (cmd.empty())
        {
            continue;
        }
        if (cmd == "/reset")
        {
            ALOGI("reset kvcache");
            g_llm.ResetKVCache();
            history = make_initial_history(config.attr);
            media_inputs.clear();
            continue;
        }
        if (cmd == "/dd")
        {
            if (history.size() >= 2 &&
                history[history.size() - 2].role == USER &&
                history.back().role == ASSISTANT)
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
        if (cmd == "/pp")
        {
            ALOGI("history size: %zu", history.size());
            for (auto &item : history)
            {
                switch (item.role)
                {
                case SYSTEM:
                    axllm::Logger::print_chat_role("system", axllm::TextColor::Yellow, item.data);
                    break;
                case USER:
                    if (item.type == IMAGE) axllm::Logger::print_chat_role("user(image)", axllm::TextColor::Green, item.data);
                    else if (item.type == VIDEO) axllm::Logger::print_chat_role("user(video)", axllm::TextColor::Green, item.data);
                    else if (item.type == AUDIO) axllm::Logger::print_chat_role("user(audio)", axllm::TextColor::Green, item.data);
                    else axllm::Logger::print_chat_role("user", axllm::TextColor::Green, item.data);
                    break;
                case ASSISTANT:
                    axllm::Logger::print_chat_role("assistant", axllm::TextColor::Default, item.data);
                    break;
                default:
                    break;
                }
            }
            continue;
        }

        // Optional media input (VLM interactive workflow).
        bool has_media = false;
        ContentType media_type = IMAGE;
        std::vector<std::string> uris;
        if (config.attr.vlm_type != VLMType::None)
        {
            printf("media >> ");
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
                    media_type = VIDEO;
                    media_line = media_line.substr(6);
                    while (!media_line.empty() && (media_line[0] == ' ' || media_line[0] == '\t')) media_line.erase(media_line.begin());
                }
                else if (media_line.rfind("audio:", 0) == 0 || media_line.rfind("AUDIO:", 0) == 0)
                {
                    media_type = AUDIO;
                    media_line = media_line.substr(6);
                    while (!media_line.empty() && (media_line[0] == ' ' || media_line[0] == '\t')) media_line.erase(media_line.begin());
                }

                // Split by whitespace for multiple media uris.
                std::istringstream iss(media_line);
                std::string tok;
                while (iss >> tok) uris.push_back(tok);
                has_media = !uris.empty();
            }
        }

        Content user;
        user.role = USER;
        user.data = prompt;
        user.type = (has_media ? media_type : TEXT);

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

// ============ Base64 data-URI support ============
static const std::string g_base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static std::vector<uint8_t> base64_decode_bytes(const std::string &encoded)
{
    std::vector<uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    int val = 0, valb = -8;
    for (unsigned char c : encoded)
    {
        if (c == '=') break;
        auto pos = g_base64_chars.find(c);
        if (pos == std::string::npos) continue; // skip whitespace / invalid
        val = (val << 6) + (int)pos;
        valb += 6;
        if (valb >= 0)
        {
            out.push_back((uint8_t)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// Detect "data:image/<ext>;base64,<payload>" and return the extension + payload.
// Uses simple string operations instead of regex to avoid stack overflow on large payloads.
static bool parse_base64_data_uri(const std::string &uri, std::string &ext, std::string &payload)
{
    const std::string prefix = "data:image/";
    if (uri.compare(0, prefix.size(), prefix) != 0) return false;

    auto semi = uri.find(';', prefix.size());
    if (semi == std::string::npos) return false;

    ext = uri.substr(prefix.size(), semi - prefix.size());
    if (ext.empty()) return false;

    const std::string b64tag = "base64,";
    if (uri.compare(semi + 1, b64tag.size(), b64tag) != 0) return false;

    size_t data_start = semi + 1 + b64tag.size();
    if (data_start >= uri.size()) return false;

    payload = uri.substr(data_start);
    return true;
}

// Decode a base64 data-URI to a temporary file and return the path.
// Caller should eventually delete the file (or leave it in /tmp for OS cleanup).
static std::string save_base64_to_tempfile(const std::string &ext, const std::string &payload)
{
    auto bytes = base64_decode_bytes(payload);
    if (bytes.empty()) return {};

    std::filesystem::path tmpdir;
    try
    {
        tmpdir = std::filesystem::temp_directory_path() / "axllm_images";
    }
    catch (...)
    {
        tmpdir = std::filesystem::current_path() / "tmp" / "axllm_images";
    }
    std::filesystem::create_directories(tmpdir);

    // Generate a unique filename
    static std::atomic<uint64_t> g_tmp_seq{0};
    const auto now = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t seq = g_tmp_seq.fetch_add(1, std::memory_order_relaxed);
    const auto path = tmpdir / ("img_" + std::to_string(now) + "_" + std::to_string(seq) + "." + ext);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return {};
    ofs.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    ofs.close();
    return path.string();
}

// Resolve an image URI: if it's a base64 data-URI, decode to a temp file.
// Otherwise return as-is (file path / directory).
static std::string resolve_image_uri(const std::string &uri, std::vector<std::string> &temp_files)
{
    std::string ext, payload;
    if (parse_base64_data_uri(uri, ext, payload))
    {
        std::string path = save_base64_to_tempfile(ext, payload);
        if (!path.empty())
        {
            temp_files.push_back(path);
        }
        return path;
    }
    return uri; // plain file path
}

// Clean up temporary files created from base64 decoding.
static void cleanup_temp_files(std::vector<std::string> &files)
{
    for (auto &f : files)
    {
        std::filesystem::remove(f);
    }
    files.clear();
}

static std::string extract_media_url(const nlohmann::json &item, const char *primary_key, const char *fallback_key = nullptr)
{
    auto extract_from_key = [&item](const char *key) -> std::string
    {
        if (!key || !item.contains(key)) return {};
        const auto &field = item[key];
        if (field.is_object() && field.contains("url"))
        {
            return field["url"].get<std::string>();
        }
        if (field.is_string())
        {
            return field.get<std::string>();
        }
        return {};
    };

    std::string raw_url = extract_from_key(primary_key);
    if (!raw_url.empty()) return raw_url;
    return extract_from_key(fallback_key);
}

// Handle HTTP API messages
bool handle_api_messages(const nlohmann::json &messages, std::vector<Content> &history,
                         std::vector<MediaInputs> *media_inputs = nullptr,
                         std::vector<std::string> *temp_files = nullptr)
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

        std::vector<std::string> media_uris;
        ContentType media_type = TEXT;

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
                else if (c.contains("type") &&
                         (c["type"] == "image_url" || c["type"] == "image" ||
                          c["type"] == "video_url" || c["type"] == "video" ||
                          c["type"] == "audio_url" || c["type"] == "audio"))
                {
                    std::string raw_url;
                    ContentType part_type = TEXT;
                    if (c["type"] == "image_url" || c["type"] == "image")
                    {
                        part_type = IMAGE;
                        raw_url = extract_media_url(c, "image_url", "image");
                    }
                    else if (c["type"] == "video_url" || c["type"] == "video")
                    {
                        part_type = VIDEO;
                        raw_url = extract_media_url(c, "video_url", "video");
                    }
                    else if (c["type"] == "audio_url" || c["type"] == "audio")
                    {
                        part_type = AUDIO;
                        raw_url = extract_media_url(c, "audio_url", "audio");
                    }
                    if (!raw_url.empty())
                    {
                        if (media_type != TEXT && media_type != part_type)
                        {
                            ALOGE("mixed media types in a single message are not supported yet");
                            return false;
                        }
                        media_type = part_type;

                        if (part_type == IMAGE)
                        {
                            if (temp_files)
                            {
                                media_uris.push_back(resolve_image_uri(raw_url, *temp_files));
                            }
                            else
                            {
                                std::vector<std::string> dummy;
                                media_uris.push_back(resolve_image_uri(raw_url, dummy));
                            }
                        }
                        else
                        {
                            media_uris.push_back(raw_url);
                        }
                    }
                }
            }
        }
        else
        {
            ALOGE("content type not support");
            return false;
        }

        if (!media_uris.empty() && content.role == USER)
        {
            content.type = media_type;
            if (media_inputs)
            {
                media_inputs->push_back({history.size(), media_uris});
            }
        }
        history.push_back(content);
    }

        for (auto &content : history)
        {
            switch (content.role)
            {
            case SYSTEM:
                axllm::Logger::print_chat_role("system", axllm::TextColor::Yellow, content.data);
                break;
            case USER:
                axllm::Logger::print_chat_role("user", axllm::TextColor::Green, content.data);
                break;
            case ASSISTANT:
                axllm::Logger::print_chat_role("assistant", axllm::TextColor::Default, content.data);
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
    g_exit_on_sigint.store(true, std::memory_order_relaxed);

    // Check whether port is available.
    const char *port_error = "unknown";
    if (!axllm::is_port_available(port, &port_error))
    {
        ALOGE("Port %d is unavailable: %s", port, port_error);
        return -1;
    }

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

    if (config.is_embedding_model())
    {
        g_server.registerEmbedding(model_name, [&llm](const openai_api::EmbeddingRequest &req,
                                                           std::shared_ptr<openai_api::BaseDataProvider> provider)
                                   {
            if (!provider->is_writable()) {
                ALOGE("provider not writable");
                return;
            }

            if (!req.encoding_format.empty() && req.encoding_format != "float") {
                ALOGW("embedding encoding_format='%s' is not supported, using float", req.encoding_format.c_str());
            }

            std::vector<std::vector<float>> embeds;
            if (!llm.EmbedBatch(req.inputs, embeds)) {
                ALOGE("EmbedBatch failed");
                provider->end();
                return;
            }

            auto chunk = openai_api::OutputChunk::BatchEmbeddings(embeds, req.model);
            provider->push(chunk);
            provider->end(); });

        printf("Starting server on port %d with embedding model '%s'...\n", port, model_name.c_str());
        {
            std::vector<std::string> hosts = {"127.0.0.1"};
            for (const auto &ip : axllm::get_local_ipv4_addresses())
            {
                bool exists = false;
                for (const auto &h : hosts)
                {
                    if (h == ip)
                    {
                        exists = true;
                        break;
                    }
                }
                if (!exists) hosts.push_back(ip);
            }

            printf("API URLs:\n");
            for (const auto &host : hosts)
            {
                const std::string base = "http://" + host + ":" + std::to_string(port);
                printf("  GET  %s/health\n", base.c_str());
                printf("  GET  %s/v1/models\n", base.c_str());
                printf("  POST %s/v1/embeddings\n", base.c_str());
            }
            printf("Aliases:\n");
            for (const auto &host : hosts)
            {
                const std::string base = "http://" + host + ":" + std::to_string(port);
                printf("  GET  %s/models\n", base.c_str());
                printf("  POST %s/embeddings\n", base.c_str());
            }
        }
        g_server.run(port);

        llm.Deinit();
    }
    else
    {
        openai_api::ChatModelOptions options;
        options.supports_vision = config.attr.vlm_type != VLMType::None;
        options.extra_fields["prefill_max_token_num"] = llm.getAttr()->prefill_max_token_num;
        options.extra_fields["max_token_len"] = llm.getAttr()->max_token_len;

        g_server.registerChat(model_name, [&llm](const openai_api::ChatRequest &req,
                                                 std::shared_ptr<openai_api::BaseDataProvider> provider)
                              {
        if (!provider->is_writable()) {
            ALOGE("provider not writable");
            return;
        }

        std::vector<Content> history;
        std::vector<MediaInputs> media_inputs;
        std::vector<std::string> temp_files;
        if (!handle_api_messages(req.messages, history, &media_inputs, &temp_files)) {
            ALOGE("handle_body failed");
            cleanup_temp_files(temp_files);
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

        cleanup_temp_files(temp_files);
        provider->end(); });

        printf("Starting server on port %d with model '%s'...\n", port, model_name.c_str());
        {
            std::vector<std::string> hosts = {"127.0.0.1"};
            for (const auto &ip : axllm::get_local_ipv4_addresses())
            {
                bool exists = false;
                for (const auto &h : hosts)
                {
                    if (h == ip)
                    {
                        exists = true;
                        break;
                    }
                }
                if (!exists) hosts.push_back(ip);
            }

            printf("API URLs:\n");
            for (const auto &host : hosts)
            {
                const std::string base = "http://" + host + ":" + std::to_string(port);
                printf("  GET  %s/health\n", base.c_str());
                printf("  GET  %s/v1/models\n", base.c_str());
                printf("  POST %s/v1/chat/completions\n", base.c_str());
            }
            printf("Aliases:\n");
            for (const auto &host : hosts)
            {
                const std::string base = "http://" + host + ":" + std::to_string(port);
                printf("  GET  %s/models\n", base.c_str());
                printf("  POST %s/chat/completions\n", base.c_str());
            }
        }
        g_server.setTimeout(std::chrono::milliseconds(300000));  // 5 minutes timeout for long-running requests
        g_server.run(port);

        llm.Deinit();
    }

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
    printf("  --port <port> Server port (default: 8000)\n");
    printf("\n");
    printf("Embedding model config:\n");
    printf("  Set \"is_embedding\": true in config.json to enable /v1/embeddings.\n");
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
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif
    signal(SIGINT, __sigExit);
#ifdef _WIN32
    SetConsoleCtrlHandler(__ConsoleCtrlHandler, TRUE);
    // Initialize logger early so Windows console uses UTF-8 (prevents Chinese/box-drawing garbling).
    (void)axllm::Logger::level();
#endif

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
        if (config.is_embedding_model())
        {
            ALOGE("Embedding models do not support interactive `run` mode. Please use `serve` mode with `/v1/embeddings`.");
            return -1;
        }

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
