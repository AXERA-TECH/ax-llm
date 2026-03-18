#include "cqdm.h"

#include <chrono>

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
    cqdm->total_time += ((now - cqdm->last_time) / 1000);
    cqdm->average_time = cqdm->total_time / cqdm->count;
    cqdm->last_time = now;

    temp = ((double)(x + 1) / (double)cqdm->total);

    fprintf(stdout, "\033[1;30;33m\r%3d%% | ", (int)(temp * 100));
    for (int i = 0; i < (temp * cqdm->size); i++)
        fprintf(stdout, "█");
    for (int i = 0; i < (cqdm->size - temp * cqdm->size); i++)
        fprintf(stdout, " ");
    fprintf(stdout, " | %3d / %3d [%2.2fs<%2.2fs, %2.2f %s/s] \033[1;30;32m%s\033[0m", x + 1, cqdm->total, cqdm->total_time,
            cqdm->average_time * cqdm->total, 1 / cqdm->average_time, unit, log_str);
    fflush(stdout);
}
