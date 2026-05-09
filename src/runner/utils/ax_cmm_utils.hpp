#pragma once
#include <cstdio>
#include <string.h>
#include <vector>
#include <fstream>
#include <regex>
#include <cstdlib>
#include <string>
#include <filesystem>

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

static std::string find_executable_in_path(const std::string &name)
{
#ifdef _WIN32
    // Best-effort: rely on shell PATH lookup on Windows.
    return name;
#else
    const char *path_env = std::getenv("PATH");
    if (!path_env) return "";
    std::string path_str(path_env);
    size_t start = 0;
    while (start <= path_str.size())
    {
        size_t end = path_str.find(':', start);
        if (end == std::string::npos) end = path_str.size();
        const std::string dir = path_str.substr(start, end - start);
        if (!dir.empty())
        {
            std::filesystem::path candidate = std::filesystem::path(dir) / name;
            std::error_code ec;
            auto st = std::filesystem::status(candidate, ec);
            if (!ec && std::filesystem::exists(st))
            {
                auto perms = st.permissions();
                if ((perms & std::filesystem::perms::owner_exec) != std::filesystem::perms::none ||
                    (perms & std::filesystem::perms::group_exec) != std::filesystem::perms::none ||
                    (perms & std::filesystem::perms::others_exec) != std::filesystem::perms::none)
                {
                    return candidate.string();
                }
            }
        }
        if (end == path_str.size()) break;
        start = end + 1;
    }
    return "";
#endif
}

static std::string get_axcl_smi_cmd()
{
    static std::string cached;
    if (!cached.empty()) return cached;

    if (const char *env = std::getenv("AXCL_SMI"))
    {
        if (env[0] != '\0')
        {
            cached = env;
            return cached;
        }
    }

    // Common install locations on Linux.
    const std::filesystem::path candidates[] = {
        "/usr/bin/axcl/axcl-smi",
        "/usr/local/bin/axcl-smi",
        "/usr/bin/axcl-smi",
    };
    for (const auto &p : candidates)
    {
        std::error_code ec;
        auto st = std::filesystem::status(p, ec);
        if (!ec && std::filesystem::exists(st))
        {
            cached = p.string();
            return cached;
        }
    }

    // Fallback to PATH lookup.
    if (auto found = find_executable_in_path("axcl-smi"); !found.empty())
    {
        cached = std::move(found);
        return cached;
    }

    cached = "axcl-smi";
    return cached;
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
    const std::string axcl_smi = get_axcl_smi_cmd();
    char command[512];
    // Redirect stderr to keep logs clean when axcl-smi isn't available.
#ifdef _WIN32
    sprintf(command, "%s -d %d sh cat /proc/ax_proc/mem_cmm_info 2>NUL", axcl_smi.c_str(), devid);
#else
    sprintf(command, "%s -d %d sh cat /proc/ax_proc/mem_cmm_info 2>/dev/null", axcl_smi.c_str(), devid);
#endif
    std::string result = exec_cmd(std::string(command));

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
