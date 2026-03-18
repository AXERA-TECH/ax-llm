#pragma once
#include <cstdio>
#include <string.h>
#include <vector>
#include <fstream>
#include <regex>

// #include "sample_log.h"

static std::string exec_cmd(std::string cmd)
{
#ifdef _WIN32
    FILE *pipe = _popen(cmd.c_str(), "r");
#else
    FILE *pipe = popen(cmd.c_str(), "r");
#endif
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
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return result;
}

static int get_remaining_cmm_size()
{
    std::string cmd = "cat /proc/ax_proc/mem_cmm_info";
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

static int get_pcie_remaining_cmm_size(int devid)
{
    char command[256];
#ifdef _WIN32
    sprintf(command, "axcl-smi -d %d sh cat /proc/ax_proc/mem_cmm_info", devid);
#else
    sprintf(command, "axcl-smi -d %d sh cat /proc/ax_proc/mem_cmm_info", devid);
#endif
    std::string result = exec_cmd(command);

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
