#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ax_engine_api.h>
#include <ax_sys_api.h>

namespace {

constexpr AX_U32 kAlignSize = 128;
constexpr const char *kSessionName = "npu";

enum AllocStrategy {
    kDefault = 0,
    kCached = 1,
};

struct ModelCtx {
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_CONTEXT_T context = nullptr;
    std::vector<AX_ENGINE_IO_INFO_T *> io_info;
    std::vector<AX_ENGINE_IO_T> io_data;
};

bool read_file(const std::string &path, std::vector<char> &buf) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    std::streamsize size = ifs.tellg();
    if (size < 0) return false;
    ifs.seekg(0, std::ios::beg);
    buf.resize(static_cast<size_t>(size));
    return static_cast<bool>(ifs.read(buf.data(), size));
}

bool read_cmm_used_kb(int64_t &used_kb, int64_t &remain_kb) {
    std::ifstream ifs("/proc/ax_proc/mem_cmm_info");
    if (!ifs) return false;
    std::string line;
    std::regex pat(R"(used=(\d+)KB.*remain=(\d+)KB)");
    std::smatch m;
    while (std::getline(ifs, line)) {
        if (line.find("total size=") == std::string::npos) continue;
        if (std::regex_search(line, m, pat)) {
            used_kb = std::stoll(m[1].str());
            remain_kb = std::stoll(m[2].str());
            return true;
        }
    }
    return false;
}

int prepare_io_struct_only(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io) {
    std::memset(io, 0, sizeof(*io));
    io->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    io->nInputSize = info->nInputSize;
    std::memset(io->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);

    io->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    io->nOutputSize = info->nOutputSize;
    std::memset(io->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);

    for (AX_U32 i = 0; i < info->nInputSize; ++i) io->pInputs[i].nSize = info->pInputs[i].nSize;
    for (AX_U32 i = 0; i < info->nOutputSize; ++i) io->pOutputs[i].nSize = info->pOutputs[i].nSize;
    return 0;
}

bool should_skip(const std::vector<std::string> &skip_names, const char *name) {
    if (!name) return false;
    return std::find(skip_names.begin(), skip_names.end(), name) != skip_names.end();
}

int alloc_buf(AX_ENGINE_IO_BUFFER_T &buf, AllocStrategy strategy) {
    AX_S32 ret = 0;
    if (strategy == kCached) {
        ret = AX_SYS_MemAllocCached(reinterpret_cast<AX_U64 *>(&buf.phyAddr),
                                    &buf.pVirAddr,
                                    buf.nSize,
                                    kAlignSize,
                                    reinterpret_cast<const AX_S8 *>(kSessionName));
    } else {
        ret = AX_SYS_MemAlloc(reinterpret_cast<AX_U64 *>(&buf.phyAddr),
                              &buf.pVirAddr,
                              buf.nSize,
                              kAlignSize,
                              reinterpret_cast<const AX_S8 *>(kSessionName));
    }
    if (ret != 0) return ret;
    std::memset(buf.pVirAddr, 0, buf.nSize);
    return 0;
}

int prepare_io_with_alloc(AX_ENGINE_IO_INFO_T *info,
                          AX_ENGINE_IO_T *io,
                          std::pair<AllocStrategy, AllocStrategy> strategy,
                          const std::vector<std::string> &skip_names = {}) {
    int ret = prepare_io_struct_only(info, io);
    if (ret != 0) return ret;

    for (AX_U32 i = 0; i < info->nInputSize; ++i) {
        if (should_skip(skip_names, info->pInputs[i].pName)) continue;
        ret = alloc_buf(io->pInputs[i], strategy.first);
        if (ret != 0) return ret;
    }

    for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
        if (should_skip(skip_names, info->pOutputs[i].pName)) continue;
        ret = alloc_buf(io->pOutputs[i], strategy.second);
        if (ret != 0) return ret;
    }
    return 0;
}

AX_ENGINE_IO_BUFFER_T *find_input_buffer_by_name(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io, const std::string &name) {
    if (!info || !io) return nullptr;
    for (AX_U32 i = 0; i < info->nInputSize; ++i) {
        const char *tensor_name = info->pInputs[i].pName;
        if (tensor_name && name == tensor_name) return io->pInputs + i;
    }
    return nullptr;
}

AX_ENGINE_IO_BUFFER_T *find_output_buffer_by_name(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io, const std::string &name) {
    if (!info || !io) return nullptr;
    for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
        const char *tensor_name = info->pOutputs[i].pName;
        if (tensor_name && name == tensor_name) return io->pOutputs + i;
    }
    return nullptr;
}

int init_model_ctx(ModelCtx &ctx, const std::vector<char> &model_buf) {
    int ret = AX_ENGINE_CreateHandle(&ctx.handle, model_buf.data(), static_cast<AX_U32>(model_buf.size()));
    if (ret != 0) return ret;

    ret = AX_ENGINE_CreateContext(ctx.handle);
    if (ret != 0) return ret;

    ret = AX_ENGINE_CreateContextV2(ctx.handle, &ctx.context);
    if (ret != 0) return ret;

    AX_U32 io_count = 0;
    ret = AX_ENGINE_GetGroupIOInfoCount(ctx.handle, &io_count);
    if (ret != 0) return ret;

    ctx.io_info.resize(io_count);
    ctx.io_data.resize(io_count);

    const std::vector<std::string> skip_alloc_names = {"K_cache", "V_cache"};
    for (AX_U32 grpid = 0; grpid < io_count; ++grpid) {
        AX_ENGINE_IO_INFO_T *info = nullptr;
        ret = AX_ENGINE_GetGroupIOInfo(ctx.handle, grpid, &info);
        if (ret != 0) return ret;
        ctx.io_info[grpid] = info;

        if (grpid == 0) {
            ret = prepare_io_with_alloc(info, &ctx.io_data[grpid], {kDefault, kCached});
        } else if (grpid == io_count - 1) {
            ret = prepare_io_with_alloc(info, &ctx.io_data[grpid], {kDefault, kCached}, skip_alloc_names);
        } else {
            ret = prepare_io_struct_only(info, &ctx.io_data[grpid]);
        }
        if (ret != 0) return ret;
    }

    if (io_count > 0) {
        size_t max_k_bytes = 0;
        size_t max_v_bytes = 0;
        for (AX_U32 grpid = 0; grpid < io_count; ++grpid) {
            AX_ENGINE_IO_INFO_T *info = ctx.io_info[grpid];
            if (!info) continue;
            for (AX_U32 i = 0; i < info->nInputSize; ++i) {
                const char *name = info->pInputs[i].pName;
                if (!name) continue;
                if (std::strcmp(name, "K_cache") == 0) max_k_bytes = std::max(max_k_bytes, static_cast<size_t>(info->pInputs[i].nSize));
                if (std::strcmp(name, "V_cache") == 0) max_v_bytes = std::max(max_v_bytes, static_cast<size_t>(info->pInputs[i].nSize));
            }
        }

        auto realloc_kv_if_needed = [&](const char *name, size_t want_bytes) -> int {
            if (want_bytes == 0 || !ctx.io_info[0] || !ctx.io_data[0].pInputs) return 0;
            auto *first_info = ctx.io_info[0];
            auto &first_data = ctx.io_data[0];
            for (AX_U32 i = 0; i < first_info->nInputSize && i < first_data.nInputSize; ++i) {
                const char *n = first_info->pInputs[i].pName;
                if (!n || std::strcmp(n, name) != 0) continue;
                auto &buf = first_data.pInputs[i];
                if (buf.nSize >= want_bytes) return 0;
                if (buf.phyAddr != 0) AX_SYS_MemFree(buf.phyAddr, buf.pVirAddr);
                int rc = AX_SYS_MemAllocCached(reinterpret_cast<AX_U64 *>(&buf.phyAddr),
                                               &buf.pVirAddr,
                                               want_bytes,
                                               kAlignSize,
                                               reinterpret_cast<const AX_S8 *>(kSessionName));
                if (rc != 0) return rc;
                std::memset(buf.pVirAddr, 0, want_bytes);
                return 0;
            }
            return 0;
        };

        ret = realloc_kv_if_needed("K_cache", max_k_bytes);
        if (ret != 0) return ret;
        ret = realloc_kv_if_needed("V_cache", max_v_bytes);
        if (ret != 0) return ret;
    }

    if (io_count > 2) {
        auto &first_io_data = ctx.io_data[0];
        auto *first_io_info = ctx.io_info[0];
        auto &last_io_data = ctx.io_data[io_count - 1];
        auto *last_io_info = ctx.io_info[io_count - 1];

        for (AX_U32 i = 0; i < last_io_data.nInputSize; ++i) {
            const char *tensor_name = last_io_info->pInputs[i].pName;
            if (!tensor_name) continue;
            if (!should_skip(skip_alloc_names, tensor_name)) continue;
            auto *first_buf = find_input_buffer_by_name(first_io_info, &first_io_data, tensor_name);
            if (!first_buf) return -1;
            if (first_buf->nSize < last_io_data.pInputs[i].nSize) return -1;
            last_io_data.pInputs[i].phyAddr = first_buf->phyAddr;
            last_io_data.pInputs[i].pVirAddr = first_buf->pVirAddr;
        }

        for (AX_U32 grpid = 1; grpid + 1 < io_count; ++grpid) {
            auto *info = ctx.io_info[grpid];
            auto &io = ctx.io_data[grpid];

            for (AX_U32 i = 0; i < info->nInputSize; ++i) {
                const char *tensor_name = info->pInputs[i].pName;
                if (!tensor_name) continue;
                auto *shared = find_input_buffer_by_name(last_io_info, &last_io_data, tensor_name);
                if (!shared || shared->nSize < io.pInputs[i].nSize) return -1;
                io.pInputs[i].phyAddr = shared->phyAddr;
                io.pInputs[i].pVirAddr = shared->pVirAddr;
            }

            for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
                const char *tensor_name = info->pOutputs[i].pName;
                if (!tensor_name) continue;
                auto *shared = find_output_buffer_by_name(last_io_info, &last_io_data, tensor_name);
                if (!shared || shared->nSize < io.pOutputs[i].nSize) return -1;
                io.pOutputs[i].phyAddr = shared->phyAddr;
                io.pOutputs[i].pVirAddr = shared->pVirAddr;
            }
        }
    }

    return 0;
}

void deinit_model_ctx(ModelCtx &ctx) {
    std::unordered_set<unsigned long long> freed_phy_addrs;
    for (auto &io : ctx.io_data) {
        if (io.pInputs) {
            for (AX_U32 j = 0; j < io.nInputSize; ++j) {
                auto &buf = io.pInputs[j];
                if (buf.phyAddr != 0 && !freed_phy_addrs.count(buf.phyAddr)) {
                    AX_SYS_MemFree(buf.phyAddr, buf.pVirAddr);
                    freed_phy_addrs.insert(buf.phyAddr);
                }
            }
            delete[] io.pInputs;
            io.pInputs = nullptr;
        }
        if (io.pOutputs) {
            for (AX_U32 j = 0; j < io.nOutputSize; ++j) {
                auto &buf = io.pOutputs[j];
                if (buf.phyAddr != 0 && !freed_phy_addrs.count(buf.phyAddr)) {
                    AX_SYS_MemFree(buf.phyAddr, buf.pVirAddr);
                    freed_phy_addrs.insert(buf.phyAddr);
                }
            }
            delete[] io.pOutputs;
            io.pOutputs = nullptr;
        }
    }
    if (ctx.handle) AX_ENGINE_DestroyHandle(ctx.handle);
    ctx = {};
}

double to_mib(int64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: axmodel_cmm_probe <axmodel> [<axmodel> ...]\n";
        return 1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    std::memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;

    int ret = AX_SYS_Init();
    if (ret != 0) {
        std::cerr << "AX_SYS_Init failed: 0x" << std::hex << ret << std::dec << "\n";
        return ret;
    }

    ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) {
        std::cerr << "AX_ENGINE_Init failed: 0x" << std::hex << ret << std::dec << "\n";
        AX_SYS_Deinit();
        return ret;
    }

    std::cout << "path\tflash_bytes\tflash_mib\tengine_cmm_bytes\tengine_cmm_mib\tobserved_cmm_delta_bytes\tobserved_cmm_delta_mib\tused_before_kb\tused_after_kb\n";

    for (int i = 1; i < argc; ++i) {
        const std::string path = argv[i];
        std::vector<char> model_buf;
        if (!read_file(path, model_buf)) {
            std::cerr << "read failed: " << path << "\n";
            continue;
        }

        int64_t used_before_kb = -1;
        int64_t remain_before_kb = -1;
        read_cmm_used_kb(used_before_kb, remain_before_kb);

        ModelCtx ctx;
        ret = init_model_ctx(ctx, model_buf);
        if (ret != 0) {
            std::cerr << "init failed: " << path << " ret=0x" << std::hex << ret << std::dec << "\n";
            deinit_model_ctx(ctx);
            continue;
        }

        AX_ENGINE_CMM_INFO cmm_info{};
        ret = AX_ENGINE_GetCMMUsage(ctx.handle, &cmm_info);
        if (ret != 0) {
            std::cerr << "AX_ENGINE_GetCMMUsage failed: " << path << " ret=0x" << std::hex << ret << std::dec << "\n";
        }

        int64_t used_after_kb = -1;
        int64_t remain_after_kb = -1;
        read_cmm_used_kb(used_after_kb, remain_after_kb);

        const int64_t flash_bytes = static_cast<int64_t>(model_buf.size());
        const int64_t engine_cmm_bytes = static_cast<int64_t>(cmm_info.nCMMSize);
        const int64_t observed_delta_bytes =
            (used_before_kb >= 0 && used_after_kb >= 0) ? (used_after_kb - used_before_kb) * 1024LL : -1;

        std::cout << path << '\t'
                  << flash_bytes << '\t'
                  << to_mib(flash_bytes) << '\t'
                  << engine_cmm_bytes << '\t'
                  << to_mib(engine_cmm_bytes) << '\t'
                  << observed_delta_bytes << '\t'
                  << to_mib(observed_delta_bytes) << '\t'
                  << used_before_kb << '\t'
                  << used_after_kb << '\n';

        deinit_model_ctx(ctx);
    }

    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return 0;
}
