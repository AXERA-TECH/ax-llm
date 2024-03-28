#ifndef CQDM_H
#define CQDM_H

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

// Structure to hold progress bar information
typedef struct s_cqdm
{
    int total;
    int count;
    int size;
    double average_time;
    double last_time;
    double total_time;
} t_cqdm;
#ifdef __cplusplus
extern "C"
{
#endif
    t_cqdm create_cqdm(int total, int size);
    void update_cqdm(t_cqdm *cqdm, int x, const char *unit, const char *log_str);
#ifdef __cplusplus
}
#endif

#endif