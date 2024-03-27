#pragma once
#include <string.h>
#include <vector>
#include <fstream>
#include <regex>

// #include "sample_log.h"

static std::string exec_cmd(std::string cmd)
{
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe)
    {
        return "";
    }
    char buffer[128];
    std::string result = "";
    while (!feof(pipe))
    {
        if (fgets(buffer, 128, pipe) != NULL)
        {
            result += buffer;
        }
    }
    pclose(pipe);
    return result;
}

static int get_remaining_cmm_size()
{
    std::string cmd = "cat /proc/ax_proc/mem_cmm_info |grep 'total size'";
    std::string result = exec_cmd(cmd);

    std::regex pattern("remain=(\\d+)KB\\((\\d+)MB \\+ (\\d+)KB\\)");
    std::smatch match;
    if (std::regex_search(result, match, pattern))
    {
        int remain_kb = std::stoi(match[1]);
        int remain_mb = std::stoi(match[2]);
        return remain_mb;
    }
    return -1;
}