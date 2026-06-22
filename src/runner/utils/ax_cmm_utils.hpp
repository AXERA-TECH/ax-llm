#pragma once
#include <cstdio>
#include <string.h>
#include <vector>
#include <fstream>
#include <regex>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <iostream>
#include <cctype>
#ifndef _WIN32
#include <unistd.h>
#endif

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

// Truly-available host DDR in MB. Uses /proc/meminfo MemAvailable, which already
// accounts for reclaimable page cache / shared buffers, so it reflects what can
// actually be allocated (avoids false alarms from buff/cache). Returns -1 if it
// cannot be determined (e.g. non-Linux), so callers can skip the DDR check.
static int get_remaining_ddr_size()
{
#ifdef _WIN32
    return -1;
#else
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) return -1;
    std::string line;
    long mem_available_kb = -1, mem_free_kb = -1;
    while (std::getline(meminfo, line))
    {
        if (line.rfind("MemAvailable:", 0) == 0)
        {
            std::sscanf(line.c_str(), "MemAvailable: %ld kB", &mem_available_kb);
            break;
        }
        if (line.rfind("MemFree:", 0) == 0)
            std::sscanf(line.c_str(), "MemFree: %ld kB", &mem_free_kb);
    }
    // Older kernels may lack MemAvailable; fall back to MemFree.
    long kb = mem_available_kb >= 0 ? mem_available_kb : mem_free_kb;
    if (kb < 0) return -1;
    return (int)(kb / 1024);
#endif
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

// ---------------------------------------------------------------------------
// Pre-load memory guard
// ---------------------------------------------------------------------------
// Estimated footprint (MB) of a model file ~= its on-disk size (weights dominate
// device/host usage). 0 if the file can't be stat'd.
static int estimate_model_mb(const std::string &path)
{
    std::error_code ec;
    auto sz = std::filesystem::file_size(path, ec);
    if (ec) return 0;
    return (int)(sz / (1024 * 1024));
}

// Decide whether it is safe to load a piece of size `est_mb` given `remaining_mb`
// of the relevant memory pool. Returns true to proceed, false to abort.
// on_unsafe: "abort" | "warn" | "prompt" (prompt asks on a TTY, else aborts).
static bool mem_guard_allow_load(bool enable, int floor_mb, const std::string &on_unsafe,
                                 const std::string &what, int est_mb, int remaining_mb)
{
    if (!enable) return true;
    if (remaining_mb < 0) return true; // pool unknown (e.g. non-Linux / no proc) -> don't block
    const int after = remaining_mb - est_mb;
    if (after >= floor_mb) return true; // safe

    std::string mode = on_unsafe.empty() ? "prompt" : on_unsafe;
    for (auto &c : mode) c = (char)std::tolower((unsigned char)c);

    fprintf(stderr,
            "[mem-guard] UNSAFE: loading '%s' (~%d MB) would leave ~%d MB free "
            "(< floor %d MB); remaining=%d MB\n",
            what.c_str(), est_mb, after, floor_mb, remaining_mb);

    if (mode == "warn") return true;
    if (mode == "abort") return false;
    // prompt
#ifndef _WIN32
    if (isatty(STDIN_FILENO))
    {
        fprintf(stderr, "[mem-guard] Continue loading anyway? [y/N]: ");
        fflush(stderr);
        std::string line;
        if (!std::getline(std::cin, line)) return false;
        for (auto &c : line) c = (char)std::tolower((unsigned char)c);
        return line == "y" || line == "yes";
    }
#endif
    fprintf(stderr, "[mem-guard] non-interactive (no TTY): aborting load. "
                    "Set mem_guard_on_unsafe=warn to force, or mem_guard_enable=false to disable.\n");
    return false;
}
