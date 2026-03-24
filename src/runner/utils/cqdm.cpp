#include "cqdm.h"

#include <chrono>
#include <cstdio>
#include <string>

#include "logger.hpp"

// Returns the current time in milliseconds
long long get_msec_now()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

t_cqdm create_cqdm(int total, int size)
{
    t_cqdm cqdm;

    cqdm.total = total;
    cqdm.count = 0;
    cqdm.average_time = 0;
    cqdm.last_time = get_msec_now();
    cqdm.total_time = 0;
    cqdm.size = size;
    return (cqdm);
}

void update_cqdm(t_cqdm *cqdm, int x, const char *unit, const char *log_str)
{
    long long now;
    double temp;

    now = get_msec_now();

    cqdm->count++;
    cqdm->total_time += ((now - cqdm->last_time) / 1000.0);
    cqdm->average_time = cqdm->total_time / cqdm->count;
    cqdm->last_time = now;

    temp = ((double)(x + 1) / (double)cqdm->total);

    const int percent = static_cast<int>(temp * 100);
    const int filled = static_cast<int>(temp * cqdm->size);

    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "\r%3d%% | ", percent);

    std::string bar;
    bar.reserve(static_cast<size_t>(cqdm->size) + 4U);
    for (int i = 0; i < filled; ++i) bar += "#";
    for (int i = 0; i < (cqdm->size - filled); ++i) bar += " ";

    const double avg = (cqdm->average_time > 1e-9) ? cqdm->average_time : 1e-9;
    char stats[256];
    std::snprintf(stats, sizeof(stats), " | %3d / %3d [%2.2fs<%2.2fs, %2.2f %s/s] ",
                  x + 1, cqdm->total, cqdm->total_time, avg * cqdm->total, 1.0 / avg, unit ? unit : "it");

    axllm::Logger::print_parts(
        {
            {axllm::TextColor::Yellow, prefix},
            {axllm::TextColor::Default, bar},
            {axllm::TextColor::Default, stats},
            {axllm::TextColor::Green, log_str ? std::string_view(log_str) : std::string_view()},
        },
        /*newline=*/false);
}
