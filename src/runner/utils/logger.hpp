#pragma once

#include <cstdarg>
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>

namespace axllm {

enum class LogLevel : int {
    Emergency = 0,
    Alert = 1,
    Critical = 2,
    Error = 3,
    Warn = 4,
    Notice = 5,
    Info = 6,
    Debug = 7,
};

enum class ColorMode : int {
    Auto = 0,
    Never = 1,
    Always = 2,
};

enum class TextColor : int {
    Default = 0,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    Gray,
};

class Logger {
public:
    static void set_level(LogLevel level);
    static LogLevel level();
    static bool should_log(LogLevel message_level);

    static void set_color_mode(ColorMode mode);
    static ColorMode color_mode();

    static void log(LogLevel level, const char *file, int line, const char *func, std::string_view message);
    static void logf(LogLevel level, const char *file, int line, const char *func, const char *fmt, ...);
    static void vlogf(LogLevel level, const char *file, int line, const char *func, const char *fmt, std::va_list ap);

    // Raw colored printing (no timestamp / level prefix).
    static void print(TextColor color, std::string_view text, bool newline = true, bool to_stderr = false);

    struct ColoredPart {
        TextColor color = TextColor::Default;
        std::string_view text;
    };
    static void print_parts(std::initializer_list<ColoredPart> parts, bool newline = true, bool to_stderr = false);
    // Finish the current in-place line (e.g. progress bar) by printing '\n'.
    // This keeps the last rendered progress text visible on screen.
    static void finish_inplace_line();

    // Keep stdout in a specific color so streaming `printf`/`fprintf` output (e.g. token callbacks)
    // remains readable. On POSIX terminals this uses ANSI; on Windows this sets console attributes.
    static void set_stdout_sticky_color(TextColor color);
    static void clear_stdout_sticky_color();

    class ScopedStdoutColor {
public:
        explicit ScopedStdoutColor(TextColor color);
        ScopedStdoutColor(const ScopedStdoutColor &) = delete;
        ScopedStdoutColor &operator=(const ScopedStdoutColor &) = delete;
        ScopedStdoutColor(ScopedStdoutColor &&other) noexcept;
        ScopedStdoutColor &operator=(ScopedStdoutColor &&other) noexcept;
        ~ScopedStdoutColor();

private:
        int prev_color_ = -1; // -1 means disabled
        bool active_ = false;
    };

    // Chat-style printing: color the whole "role: text" line.
    static void print_chat_role(std::string_view role, TextColor role_color, std::string_view text);

    class Line {
public:
        Line(LogLevel level, const char *file, int line, const char *func);
        Line(Line &&other) noexcept;
        Line &operator=(Line &&other) noexcept;
        ~Line();

        template <typename T>
        Line &operator<<(T &&value)
        {
            if (enabled_) stream_ << std::forward<T>(value);
            return *this;
        }

private:
        bool enabled_ = false;
        LogLevel level_ = LogLevel::Info;
        const char *file_ = nullptr;
        int line_ = 0;
        const char *func_ = nullptr;
        std::ostringstream stream_;
    };

    static Line stream(LogLevel level, const char *file, int line, const char *func);
};

} // namespace axllm

// Backward-compatible macros (printf-style).
// Note: `__VA_ARGS__` always includes the format string as its first argument.
#define ALOGE(...) do { axllm::Logger::logf(axllm::LogLevel::Error,   __FILE__, __LINE__, __func__, __VA_ARGS__); } while (0)
#define ALOGW(...) do { if (axllm::Logger::should_log(axllm::LogLevel::Warn))   axllm::Logger::logf(axllm::LogLevel::Warn,   __FILE__, __LINE__, __func__, __VA_ARGS__); } while (0)
#define ALOGN(...) do { if (axllm::Logger::should_log(axllm::LogLevel::Notice)) axllm::Logger::logf(axllm::LogLevel::Notice, __FILE__, __LINE__, __func__, __VA_ARGS__); } while (0)
#define ALOGI(...) do { if (axllm::Logger::should_log(axllm::LogLevel::Info))   axllm::Logger::logf(axllm::LogLevel::Info,   __FILE__, __LINE__, __func__, __VA_ARGS__); } while (0)
#define ALOGD(...) do { if (axllm::Logger::should_log(axllm::LogLevel::Debug))  axllm::Logger::logf(axllm::LogLevel::Debug,  __FILE__, __LINE__, __func__, __VA_ARGS__); } while (0)

// Stream-style logging (more modern C++):
//   AXLOGI() << "value=" << v;
#define AXLOGE() axllm::Logger::stream(axllm::LogLevel::Error,  __FILE__, __LINE__, __func__)
#define AXLOGW() axllm::Logger::stream(axllm::LogLevel::Warn,   __FILE__, __LINE__, __func__)
#define AXLOGN() axllm::Logger::stream(axllm::LogLevel::Notice, __FILE__, __LINE__, __func__)
#define AXLOGI() axllm::Logger::stream(axllm::LogLevel::Info,   __FILE__, __LINE__, __func__)
#define AXLOGD() axllm::Logger::stream(axllm::LogLevel::Debug,  __FILE__, __LINE__, __func__)
