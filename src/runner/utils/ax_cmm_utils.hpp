#pragma once
#include <string.h>
#include <vector>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <iostream>

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


static bool get_pcie_ids(std::vector<int> &ids)
{
    const char *command = "lspci | grep Axera | awk -F':' '{print $2+0}'";
    FILE *pipe = popen(command, "r");
    if (!pipe)
    {
        std::cerr << "Failed to run command!" << std::endl;
        return false;
    }
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        ids.push_back(std::stoi(buffer));
    }
    pclose(pipe);
    return true;
}

static int get_pcie_remaining_cmm_size()
{
    std::vector<int> ids;
    if (!get_pcie_ids(ids))
    {
        ALOGE("get_pcie_ids failed");
        return -1;
    }
    if (ids.size() == 0)
    {
        ALOGE("get_pcie_ids failed");
        return -1;
    }
    
    char command[128];
    sprintf(command, "axcl-smi -d %d sh cat /proc/ax_proc/mem_cmm_info |grep 'total size'", ids[0]);
    // printf("%s\n", command);
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