/**********************************************************************************
 *
 * Copyright (c) 2019-2022 Beijing AXera Technology Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Beijing AXera Technology Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Beijing AXera Technology Co., Ltd.
 *
 **********************************************************************************/
#ifndef _SAMPLE_LOG_H_
#define _SAMPLE_LOG_H_

#include <stdio.h>

typedef enum
{
    SAMPLE_LOG_MIN = -1,
    SAMPLE_LOG_EMERGENCY = 0,
    SAMPLE_LOG_ALERT = 1,
    SAMPLE_LOG_CRITICAL = 2,
    SAMPLE_LOG_ERROR = 3,
    SAMPLE_LOG_WARN = 4,
    SAMPLE_LOG_NOTICE = 5,
    SAMPLE_LOG_INFO = 6,
    SAMPLE_LOG_DEBUG = 7,
    SAMPLE_LOG_MAX
} SAMPLE_LOG_LEVEL_E;

static SAMPLE_LOG_LEVEL_E log_level = SAMPLE_LOG_INFO;

#if 1
#define MACRO_BLACK "\033[1;30;30m"
#define MACRO_RED "\033[1;30;31m"
#define MACRO_GREEN "\033[1;30;32m"
#define MACRO_YELLOW "\033[1;30;33m"
#define MACRO_BLUE "\033[1;30;34m"
#define MACRO_PURPLE "\033[1;30;35m"
#define MACRO_WHITE "\033[1;30;37m"
#define MACRO_END "\033[0m"
#else
#define MACRO_BLACK
#define MACRO_RED
#define MACRO_GREEN
#define MACRO_YELLOW
#define MACRO_BLUE
#define MACRO_PURPLE
#define MACRO_WHITE
#define MACRO_END
#endif

#define ALOGE(fmt, ...) printf(MACRO_RED "[E][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define ALOGW(fmt, ...)               \
    if (log_level >= SAMPLE_LOG_WARN) \
    printf(MACRO_YELLOW "[W][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define ALOGI(fmt, ...)               \
    if (log_level >= SAMPLE_LOG_INFO) \
    printf(MACRO_GREEN "[I][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define ALOGD(fmt, ...)                \
    if (log_level >= SAMPLE_LOG_DEBUG) \
    printf(MACRO_WHITE "[D][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define ALOGN(fmt, ...)                 \
    if (log_level >= SAMPLE_LOG_NOTICE) \
    printf(MACRO_PURPLE "[N][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)

#endif /* _SAMPLE_LOG_H_ */
