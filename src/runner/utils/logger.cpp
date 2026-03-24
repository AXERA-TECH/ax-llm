#include "logger.hpp"

#include <atomic>
#include <chrono>
#include <cctype>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace axllm {

namespace {

std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};
std::atomic<int> g_color_mode{static_cast<int>(ColorMode::Auto)};
// -1 means disabled, otherwise stores `TextColor` enum value.
std::atomic<int> g_stdout_sticky_color{-1};
std::mutex g_write_mutex;
std::once_flag g_init_once;

struct InplaceLineState {
    bool active = false;
    size_t width = 0; // visible chars on the current line
    FILE *stream = stdout;
};

InplaceLineState g_inplace;

#ifdef _WIN32
struct WinConsoleState {
    HANDLE stdout_handle = nullptr;
    HANDLE stderr_handle = nullptr;
    bool stdout_is_console = false;
    bool stderr_is_console = false;
    WORD stdout_default_attr = 0;
    WORD stderr_default_attr = 0;
};

WinConsoleState &win_state()
{
    static WinConsoleState s;
    return s;
}
#endif

static inline bool str_eq_nocase(std::string_view a, std::string_view b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const unsigned char ca = static_cast<unsigned char>(a[i]);
        const unsigned char cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) return false;
    }
    return true;
}

static inline bool env_is_set(const char *name)
{
    const char *v = std::getenv(name);
    return v && *v;
}

static inline std::string_view getenv_sv(const char *name)
{
    const char *v = std::getenv(name);
    return (v && *v) ? std::string_view(v) : std::string_view();
}

static LogLevel parse_level(std::string_view s, LogLevel fallback)
{
    if (s.empty()) return fallback;
    if (str_eq_nocase(s, "debug") || str_eq_nocase(s, "dbg") || str_eq_nocase(s, "d")) return LogLevel::Debug;
    if (str_eq_nocase(s, "info") || str_eq_nocase(s, "inf") || str_eq_nocase(s, "i")) return LogLevel::Info;
    if (str_eq_nocase(s, "notice") || str_eq_nocase(s, "note") || str_eq_nocase(s, "n")) return LogLevel::Notice;
    if (str_eq_nocase(s, "warn") || str_eq_nocase(s, "warning") || str_eq_nocase(s, "w")) return LogLevel::Warn;
    if (str_eq_nocase(s, "error") || str_eq_nocase(s, "err") || str_eq_nocase(s, "e")) return LogLevel::Error;
    if (str_eq_nocase(s, "critical") || str_eq_nocase(s, "crit") || str_eq_nocase(s, "c")) return LogLevel::Critical;
    if (str_eq_nocase(s, "alert") || str_eq_nocase(s, "a")) return LogLevel::Alert;
    if (str_eq_nocase(s, "emergency") || str_eq_nocase(s, "emerg")) return LogLevel::Emergency;

    // numeric: 0..7
    char *end = nullptr;
    const std::string tmp(s);
    const long v = std::strtol(tmp.c_str(), &end, 10);
    if (end && *end == '\0' && v >= 0 && v <= 7) return static_cast<LogLevel>(static_cast<int>(v));
    return fallback;
}

static ColorMode parse_color_mode(std::string_view s, ColorMode fallback)
{
    if (s.empty()) return fallback;
    if (str_eq_nocase(s, "auto")) return ColorMode::Auto;
    if (str_eq_nocase(s, "never") || str_eq_nocase(s, "off") || str_eq_nocase(s, "0") || str_eq_nocase(s, "false")) return ColorMode::Never;
    if (str_eq_nocase(s, "always") || str_eq_nocase(s, "on") || str_eq_nocase(s, "1") || str_eq_nocase(s, "true")) return ColorMode::Always;
    return fallback;
}

static void init_from_env()
{
    // Log level
    const auto lvl = getenv_sv("AXLLM_LOG_LEVEL");
    g_level.store(static_cast<int>(parse_level(lvl, LogLevel::Info)));

    // Color mode
    if (env_is_set("NO_COLOR") || env_is_set("AXLLM_NO_COLOR"))
    {
        g_color_mode.store(static_cast<int>(ColorMode::Never));
    }
    else
    {
        const auto cm = getenv_sv("AXLLM_COLOR");
        g_color_mode.store(static_cast<int>(parse_color_mode(cm, ColorMode::Auto)));
    }

#ifdef _WIN32
    auto &ws = win_state();
    ws.stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);
    ws.stderr_handle = GetStdHandle(STD_ERROR_HANDLE);

    DWORD mode = 0;
    ws.stdout_is_console = (ws.stdout_handle != nullptr) && GetConsoleMode(ws.stdout_handle, &mode);
    ws.stderr_is_console = (ws.stderr_handle != nullptr) && GetConsoleMode(ws.stderr_handle, &mode);

    if (ws.stdout_is_console)
    {
        CONSOLE_SCREEN_BUFFER_INFO info{};
        if (GetConsoleScreenBufferInfo(ws.stdout_handle, &info)) ws.stdout_default_attr = info.wAttributes;
    }
    if (ws.stderr_is_console)
    {
        CONSOLE_SCREEN_BUFFER_INFO info{};
        if (GetConsoleScreenBufferInfo(ws.stderr_handle, &info)) ws.stderr_default_attr = info.wAttributes;
    }
#endif
}

static inline void ensure_initialized()
{
    std::call_once(g_init_once, init_from_env);
}

static inline bool is_tty(FILE *stream)
{
#ifdef _WIN32
    const int fd = _fileno(stream);
    return fd >= 0 && _isatty(fd);
#else
    const int fd = fileno(stream);
    return fd >= 0 && isatty(fd);
#endif
}

static inline bool should_use_color(FILE *stream, bool assume_tty)
{
    const auto cm = static_cast<ColorMode>(g_color_mode.load());
    if (cm == ColorMode::Never) return false;
    if (cm == ColorMode::Always) return true;

    // Auto
    if (env_is_set("NO_COLOR") || env_is_set("AXLLM_NO_COLOR")) return false;
    if (!assume_tty) return false;
#ifndef _WIN32
    const auto term = getenv_sv("TERM");
    if (!term.empty() && str_eq_nocase(term, "dumb")) return false;
#endif
    return true;
}

static inline std::string format_timestamp()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t tt = system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms.count()));
    return std::string(buf);
}

static inline std::string_view level_tag(LogLevel level)
{
    switch (level)
    {
    case LogLevel::Emergency: return "EMG";
    case LogLevel::Alert: return "ALT";
    case LogLevel::Critical: return "CRT";
    case LogLevel::Error: return "ERR";
    case LogLevel::Warn: return "WRN";
    case LogLevel::Notice: return "NTC";
    case LogLevel::Info: return "INF";
    case LogLevel::Debug: return "DBG";
    default: return "UNK";
    }
}

static inline TextColor level_color(LogLevel level)
{
    switch (level)
    {
    case LogLevel::Emergency:
    case LogLevel::Alert:
    case LogLevel::Critical:
    case LogLevel::Error: return TextColor::Red;
    case LogLevel::Warn: return TextColor::Yellow;
    case LogLevel::Notice: return TextColor::Magenta;
    case LogLevel::Info: return TextColor::Green;
    case LogLevel::Debug: return TextColor::Gray;
    default: return TextColor::Default;
    }
}

static inline std::string_view ansi_code(TextColor c)
{
    switch (c)
    {
    case TextColor::Black: return "\x1b[30m";
    case TextColor::Red: return "\x1b[31m";
    case TextColor::Green: return "\x1b[32m";
    case TextColor::Yellow: return "\x1b[33m";
    case TextColor::Blue: return "\x1b[34m";
    case TextColor::Magenta: return "\x1b[35m";
    case TextColor::Cyan: return "\x1b[36m";
    case TextColor::White: return "\x1b[37m";
    case TextColor::Gray: return "\x1b[90m";
    case TextColor::Default:
    default: return "\x1b[0m";
    }
}

static inline void fwrite_sv(FILE *stream, std::string_view s)
{
    if (s.empty()) return;
    std::fwrite(s.data(), 1, s.size(), stream);
}

static inline void write_spaces(FILE *stream, size_t n)
{
    static const char kSpaces[] =
        "                                                                "; // 64 spaces
    constexpr size_t kChunk = sizeof(kSpaces) - 1;
    while (n > 0)
    {
        const size_t chunk = (n < kChunk) ? n : kChunk;
        std::fwrite(kSpaces, 1, chunk, stream);
        n -= chunk;
    }
}

static inline void clear_inplace_line_locked()
{
    if (!g_inplace.active) return;
    FILE *stream = g_inplace.stream ? g_inplace.stream : stdout;
    fwrite_sv(stream, "\r");
    if (g_inplace.width > 0)
    {
        write_spaces(stream, g_inplace.width);
        fwrite_sv(stream, "\r");
    }
    g_inplace.active = false;
    g_inplace.width = 0;
    g_inplace.stream = stdout;
}

static inline std::string strip_trailing_newlines(std::string_view s)
{
    size_t end = s.size();
    while (end > 0)
    {
        const char ch = s[end - 1];
        if (ch == '\n' || ch == '\r') end--;
        else break;
    }
    return std::string(s.substr(0, end));
}

static inline std::string vformat_printf(const char *fmt, std::va_list ap)
{
    if (!fmt) return {};
    std::va_list ap_copy;
    va_copy(ap_copy, ap);
    const int needed = std::vsnprintf(nullptr, 0, fmt, ap_copy);
    va_end(ap_copy);

    if (needed <= 0)
    {
        // Fallback: best-effort buffer
        char buf[1024];
        buf[0] = '\0';
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        return std::string(buf);
    }

    std::string out;
    out.resize(static_cast<size_t>(needed));
    std::vsnprintf(out.data(), out.size() + 1, fmt, ap);
    return out;
}

#ifdef _WIN32
static inline HANDLE stream_handle(FILE *stream)
{
    if (stream == stderr) return win_state().stderr_handle;
    return win_state().stdout_handle;
}

static inline bool stream_is_console(FILE *stream)
{
    if (stream == stderr) return win_state().stderr_is_console;
    return win_state().stdout_is_console;
}

static inline WORD stream_default_attr(FILE *stream)
{
    if (stream == stderr) return win_state().stderr_default_attr;
    return win_state().stdout_default_attr;
}

static inline WORD win_attr_for_color(TextColor c, WORD default_attr)
{
    const WORD bg = default_attr & 0xF0;
    WORD fg = 0;
    switch (c)
    {
    case TextColor::Black: fg = 0; break;
    case TextColor::Red: fg = FOREGROUND_RED | FOREGROUND_INTENSITY; break;
    case TextColor::Green: fg = FOREGROUND_GREEN | FOREGROUND_INTENSITY; break;
    case TextColor::Yellow: fg = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY; break;
    case TextColor::Blue: fg = FOREGROUND_BLUE | FOREGROUND_INTENSITY; break;
    case TextColor::Magenta: fg = FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY; break;
    case TextColor::Cyan: fg = FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY; break;
    case TextColor::White: fg = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY; break;
    case TextColor::Gray: fg = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; break;
    case TextColor::Default:
    default:
        return default_attr;
    }
    return bg | fg;
}
#endif

static inline void write_colored_piece(FILE *stream, TextColor color, std::string_view text, bool use_color)
{
#ifdef _WIN32
    if (use_color && stream_is_console(stream))
    {
        const HANDLE h = stream_handle(stream);
        WORD old_attr = stream_default_attr(stream);
        if (h)
        {
            CONSOLE_SCREEN_BUFFER_INFO info{};
            if (GetConsoleScreenBufferInfo(h, &info)) old_attr = info.wAttributes;
        }
        const WORD attr = win_attr_for_color(color, old_attr);
        if (h && attr != old_attr) SetConsoleTextAttribute(h, attr);
        fwrite_sv(stream, text);
        if (h) SetConsoleTextAttribute(h, old_attr);
        return;
    }
#endif

    if (use_color)
    {
        fwrite_sv(stream, ansi_code(color));
        fwrite_sv(stream, text);
        fwrite_sv(stream, "\x1b[0m");
        if (stream == stdout)
        {
            const int sticky = g_stdout_sticky_color.load(std::memory_order_relaxed);
            if (sticky >= 0) fwrite_sv(stream, ansi_code(static_cast<TextColor>(sticky)));
        }
        return;
    }
    fwrite_sv(stream, text);
}

static inline void write_log_line(FILE *stream,
                                  LogLevel level,
                                  std::string_view ts,
                                  const char *func,
                                  int line,
                                  std::string_view msg_line,
                                  bool use_color)
{
    // Format: "HH:MM:SS.mmm TAG func:line | message\n" (whole line colored by level)
    char head[256];
    std::snprintf(head, sizeof(head), "%.*s %.*s %s:%d | ",
                  (int)ts.size(), ts.data(),
                  (int)level_tag(level).size(), level_tag(level).data(),
                  (func && *func) ? func : "?",
                  line);
    std::string out;
    out.reserve(std::strlen(head) + msg_line.size() + 2);
    out.append(head);
    out.append(msg_line);
    out.push_back('\n');
    if (use_color)
    {
        write_colored_piece(stream, level_color(level), out, true);
    }
    else
    {
        fwrite_sv(stream, out);
    }
}

} // namespace

void Logger::set_level(LogLevel level)
{
    ensure_initialized();
    g_level.store(static_cast<int>(level));
}

LogLevel Logger::level()
{
    ensure_initialized();
    return static_cast<LogLevel>(g_level.load());
}

bool Logger::should_log(LogLevel message_level)
{
    ensure_initialized();
    return static_cast<int>(message_level) <= g_level.load();
}

void Logger::set_color_mode(ColorMode mode)
{
    ensure_initialized();
    g_color_mode.store(static_cast<int>(mode));
}

ColorMode Logger::color_mode()
{
    ensure_initialized();
    return static_cast<ColorMode>(g_color_mode.load());
}

void Logger::log(LogLevel lvl, const char * /*file*/, int line, const char *func, std::string_view message)
{
    ensure_initialized();
    if (!should_log(lvl)) return;

    FILE *stream = stdout;
    if (static_cast<int>(lvl) <= static_cast<int>(LogLevel::Warn)) stream = stderr;

    const bool tty = is_tty(stream);
    const bool use_color = should_use_color(stream, tty);

    const std::string ts = format_timestamp();
    const std::string msg = strip_trailing_newlines(message);

    std::lock_guard<std::mutex> lock(g_write_mutex);
    clear_inplace_line_locked();

    // Print each line with its own prefix for readability.
    size_t start = 0;
    while (true)
    {
        const size_t pos = msg.find('\n', start);
        const std::string_view line_sv = (pos == std::string::npos)
                                             ? std::string_view(msg).substr(start)
                                             : std::string_view(msg).substr(start, pos - start);
        write_log_line(stream, lvl, ts, func, line, line_sv, use_color);
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    std::fflush(stream);
}

void Logger::logf(LogLevel level, const char *file, int line, const char *func, const char *fmt, ...)
{
    std::va_list ap;
    va_start(ap, fmt);
    vlogf(level, file, line, func, fmt, ap);
    va_end(ap);
}

void Logger::vlogf(LogLevel level, const char *file, int line, const char *func, const char *fmt, std::va_list ap)
{
    ensure_initialized();
    if (!should_log(level)) return;

    const std::string msg = vformat_printf(fmt, ap);
    log(level, file, line, func, msg);
}

void Logger::print(TextColor color, std::string_view text, bool newline, bool to_stderr)
{
    ensure_initialized();

    FILE *stream = to_stderr ? stderr : stdout;
    const bool tty = is_tty(stream);
    const bool use_color = should_use_color(stream, tty);

    std::string msg = strip_trailing_newlines(text);
    if (newline) msg.push_back('\n');

    std::lock_guard<std::mutex> lock(g_write_mutex);
    if (newline) clear_inplace_line_locked();
    write_colored_piece(stream, color, msg, use_color);
    std::fflush(stream);
}

void Logger::print_parts(std::initializer_list<ColoredPart> parts, bool newline, bool to_stderr)
{
    ensure_initialized();

    FILE *stream = to_stderr ? stderr : stdout;
    const bool tty = is_tty(stream);
    bool use_color = should_use_color(stream, tty);

    std::lock_guard<std::mutex> lock(g_write_mutex);
    if (newline)
    {
        clear_inplace_line_locked();
    }
    else if (tty)
    {
        // In-place updates (progress bars): clear the previous in-place line first so other
        // logs never get appended to it and we won't leave trailing characters.
        clear_inplace_line_locked();
    }

    size_t visible_width = 0;
    if (!newline && tty)
    {
        for (const auto &p : parts)
        {
            for (const char ch : p.text)
            {
                if (ch == '\r') continue;
                visible_width++;
            }
        }
    }
    for (const auto &p : parts)
    {
        const TextColor c = use_color ? p.color : TextColor::Default;
        write_colored_piece(stream, c, p.text, use_color);
    }
    if (newline)
    {
        fwrite_sv(stream, "\n");
    }
    else if (tty)
    {
        g_inplace.active = true;
        g_inplace.width = visible_width;
        g_inplace.stream = stream;
    }
    std::fflush(stream);
}

void Logger::finish_inplace_line()
{
    ensure_initialized();
    std::lock_guard<std::mutex> lock(g_write_mutex);
    if (!g_inplace.active) return;
    FILE *stream = g_inplace.stream ? g_inplace.stream : stdout;
    fwrite_sv(stream, "\n");
    g_inplace.active = false;
    g_inplace.width = 0;
    g_inplace.stream = stdout;
    std::fflush(stream);
}

void Logger::print_chat_role(std::string_view role, TextColor role_color, std::string_view text)
{
    ensure_initialized();

    FILE *stream = stdout;
    const bool tty = is_tty(stream);
    const bool use_color = should_use_color(stream, tty);

    const std::string msg = strip_trailing_newlines(text);

    std::lock_guard<std::mutex> lock(g_write_mutex);
    clear_inplace_line_locked();
    std::string line;
    line.reserve(role.size() + 2 + msg.size() + 1);
    line.append(role);
    line.append(": ");
    line.append(msg);
    line.push_back('\n');
    write_colored_piece(stream, role_color, line, use_color);
    std::fflush(stream);
}

void Logger::set_stdout_sticky_color(TextColor color)
{
    ensure_initialized();
    FILE *stream = stdout;
    const bool tty = is_tty(stream);
    bool use_color = should_use_color(stream, tty);

    if (!use_color)
    {
        g_stdout_sticky_color.store(-1, std::memory_order_relaxed);
        return;
    }

    g_stdout_sticky_color.store(static_cast<int>(color), std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(g_write_mutex);
    clear_inplace_line_locked();
#ifdef _WIN32
    if (stream_is_console(stream))
    {
        const HANDLE h = stream_handle(stream);
        if (h)
        {
            WORD old_attr = stream_default_attr(stream);
            CONSOLE_SCREEN_BUFFER_INFO info{};
            if (GetConsoleScreenBufferInfo(h, &info)) old_attr = info.wAttributes;
            const WORD attr = win_attr_for_color(color, old_attr);
            SetConsoleTextAttribute(h, attr);
        }
    }
    else
    {
        fwrite_sv(stream, ansi_code(color));
        std::fflush(stream);
    }
#else
    fwrite_sv(stream, ansi_code(color));
    std::fflush(stream);
#endif
}

void Logger::clear_stdout_sticky_color()
{
    ensure_initialized();
    const int prev = g_stdout_sticky_color.exchange(-1, std::memory_order_relaxed);
    if (prev < 0) return;

    FILE *stream = stdout;
    std::lock_guard<std::mutex> lock(g_write_mutex);
    clear_inplace_line_locked();
#ifdef _WIN32
    if (stream_is_console(stream))
    {
        const HANDLE h = stream_handle(stream);
        if (h) SetConsoleTextAttribute(h, stream_default_attr(stream));
    }
    else
    {
        fwrite_sv(stream, "\x1b[0m");
        std::fflush(stream);
    }
#else
    fwrite_sv(stream, "\x1b[0m");
    std::fflush(stream);
#endif
}

Logger::ScopedStdoutColor::ScopedStdoutColor(TextColor color)
{
    ensure_initialized();
    prev_color_ = g_stdout_sticky_color.load(std::memory_order_relaxed);
    active_ = true;
    Logger::set_stdout_sticky_color(color);
}

Logger::ScopedStdoutColor::ScopedStdoutColor(ScopedStdoutColor &&other) noexcept
    : prev_color_(other.prev_color_), active_(other.active_)
{
    other.active_ = false;
}

Logger::ScopedStdoutColor &Logger::ScopedStdoutColor::operator=(ScopedStdoutColor &&other) noexcept
{
    if (this == &other) return *this;
    if (active_)
    {
        if (prev_color_ >= 0) Logger::set_stdout_sticky_color(static_cast<TextColor>(prev_color_));
        else Logger::clear_stdout_sticky_color();
    }
    prev_color_ = other.prev_color_;
    active_ = other.active_;
    other.active_ = false;
    return *this;
}

Logger::ScopedStdoutColor::~ScopedStdoutColor()
{
    if (!active_) return;
    if (prev_color_ >= 0)
    {
        Logger::set_stdout_sticky_color(static_cast<TextColor>(prev_color_));
    }
    else
    {
        Logger::clear_stdout_sticky_color();
    }
}

Logger::Line Logger::stream(LogLevel level, const char *file, int line, const char *func)
{
    return Line(level, file, line, func);
}

Logger::Line::Line(LogLevel level, const char *file, int line, const char *func)
    : enabled_(Logger::should_log(level)), level_(level), file_(file), line_(line), func_(func)
{
}

Logger::Line::Line(Line &&other) noexcept
    : enabled_(other.enabled_), level_(other.level_), file_(other.file_), line_(other.line_), func_(other.func_), stream_(std::move(other.stream_))
{
    other.enabled_ = false;
}

Logger::Line &Logger::Line::operator=(Line &&other) noexcept
{
    if (this == &other) return *this;
    enabled_ = other.enabled_;
    level_ = other.level_;
    file_ = other.file_;
    line_ = other.line_;
    func_ = other.func_;
    stream_ = std::move(other.stream_);
    other.enabled_ = false;
    return *this;
}

Logger::Line::~Line()
{
    if (!enabled_) return;
    try
    {
        Logger::log(level_, file_, line_, func_, stream_.str());
    }
    catch (...)
    {
        // Never throw from a destructor.
    }
}

} // namespace axllm
