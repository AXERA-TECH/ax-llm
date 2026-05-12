#include "ax_model_runner_ax650.hpp"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <memory>
#include <unordered_set> // 用于去重释放物理内存
#include <ax_sys_api.h>
#include <ax_ivps_api.h>
#include <ax_engine_api.h>
#include <fcntl.h>
#include "memory_utils.hpp"
#include "sample_log.h"

#define AX_CMM_ALIGN_SIZE 128
const char *AX_CMM_SESSION_NAME = "npu";

typedef enum
{
    AX_ENGINE_ABST_DEFAULT = 0,
    AX_ENGINE_ABST_CACHED = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

// 封装一个结构体来管理内部句柄，方便管理
struct ax_runner_ax650_handle_t
{
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_CONTEXT_T context = 0;
    std::vector<AX_ENGINE_IO_INFO_T *> io_info;
    std::vector<AX_ENGINE_IO_T> io_data;
};

// 辅助：分配 IO 结构体数组（不分配物理内存）
static int prepare_io_struct_only(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data)
{
    memset(io_data, 0, sizeof(*io_data));
    io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    io_data->nInputSize = info->nInputSize;
    memset(io_data->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);

    io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    io_data->nOutputSize = info->nOutputSize;
    memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);

    // 仅设置 Size，方便后续逻辑
    for (uint i = 0; i < info->nInputSize; ++i)
        io_data->pInputs[i].nSize = info->pInputs[i].nSize;
    for (uint i = 0; i < info->nOutputSize; ++i)
        io_data->pOutputs[i].nSize = info->pOutputs[i].nSize;

    return 0;
}

// 辅助：分配 IO 结构体数组 + 物理内存
static int prepare_io_with_alloc(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data,
                                 std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T> strategy, std::vector<std::string> skip_alloc_names = {})
{
    int ret = prepare_io_struct_only(info, io_data);
    if (ret != 0)
        return ret;

    // Alloc Inputs
    for (uint i = 0; i < info->nInputSize; ++i)
    {
        auto &buffer = io_data->pInputs[i];
        if (std::find(skip_alloc_names.begin(), skip_alloc_names.end(), info->pInputs[i].pName) != skip_alloc_names.end())
        {
            continue;
        }
        if (strategy.first == AX_ENGINE_ABST_CACHED)
        {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer.phyAddr), &buffer.pVirAddr, buffer.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        else
        {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer.phyAddr), &buffer.pVirAddr, buffer.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        if (ret != 0)
        {
            ALOGE("Alloc input[%d] failed", i);
            return ret; // 注意：此处在实际工程中应跳转到 cleanup，为简化展示直接返回
        }
        memset(buffer.pVirAddr, 0, buffer.nSize);
    }

    // Alloc Outputs
    for (uint i = 0; i < info->nOutputSize; ++i)
    {
        auto &buffer = io_data->pOutputs[i];
        if (std::find(skip_alloc_names.begin(), skip_alloc_names.end(), info->pOutputs[i].pName) != skip_alloc_names.end())
        {
            continue;
        }
        if (strategy.second == AX_ENGINE_ABST_CACHED)
        {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer.phyAddr), &buffer.pVirAddr, buffer.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        else
        {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer.phyAddr), &buffer.pVirAddr, buffer.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        if (ret != 0)
        {
            ALOGE("Alloc output[%d] failed", i);
            return ret;
        }
        memset(buffer.pVirAddr, 0, buffer.nSize);
    }
    return 0;
}

static AX_ENGINE_IO_BUFFER_T *find_input_buffer_by_name(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data, const std::string &name)
{
    if (!info || !io_data)
        return nullptr;

    for (uint i = 0; i < info->nInputSize; ++i)
    {
        const char *tensor_name = info->pInputs[i].pName;
        if (tensor_name && name == tensor_name)
            return io_data->pInputs + i;
    }

    return nullptr;
}

static AX_ENGINE_IO_BUFFER_T *find_output_buffer_by_name(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data, const std::string &name)
{
    if (!info || !io_data)
        return nullptr;

    for (uint i = 0; i < info->nOutputSize; ++i)
    {
        const char *tensor_name = info->pOutputs[i].pName;
        if (tensor_name && name == tensor_name)
            return io_data->pOutputs + i;
    }

    return nullptr;
}

int ax_runner_ax650::sub_init()
{
    if (!m_handle)
        return -1;

    int ret = AX_ENGINE_CreateContext(m_handle->handle);
    if (ret != 0)
        return ret;

    ret = AX_ENGINE_CreateContextV2(m_handle->handle, &m_handle->context);
    if (ret != 0)
        return ret;

    AX_U32 io_count = 0;
    ret = AX_ENGINE_GetGroupIOInfoCount(m_handle->handle, &io_count);
    if (ret != 0)
        return ret;

    m_handle->io_info.resize(io_count);
    m_handle->io_data.resize(io_count);
    mgroup_input_tensors.resize(io_count);
    mgroup_output_tensors.resize(io_count);

    std::vector<std::string> skip_alloc_names = {"K_cache", "V_cache"};
    // 1. 分配 IO 资源
    for (size_t grpid = 0; grpid < io_count; grpid++)
    {
        AX_ENGINE_IO_INFO_T *io_info = nullptr;
        ret = AX_ENGINE_GetGroupIOInfo(m_handle->handle, grpid, &io_info);
        if (ret != 0)
            return ret;
        m_handle->io_info[grpid] = io_info;

        // 原有逻辑保持不变：Group 0 和 Last Group 分配物理内存，中间 Group 不分配
        if (grpid == 0)
        {
            ret = prepare_io_with_alloc(io_info, &m_handle->io_data[grpid], {AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED});
        }
        else if (grpid == io_count - 1)
        {
            ret = prepare_io_with_alloc(io_info, &m_handle->io_data[grpid], {AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED}, skip_alloc_names);
        }
        else
        {
            ret = prepare_io_struct_only(io_info, &m_handle->io_data[grpid]);
        }
        if (ret != 0)
            return ret;
    }

    // Ensure shared K_cache/V_cache buffers are large enough for the maximum group.
    // Some models have multiple decode groups (2k/4k/8k/16k) where group 0 is not the largest.
    // We share KV buffers across groups to save memory, so allocate KV based on max required size.
    {
        size_t max_k_bytes = 0;
        size_t max_v_bytes = 0;
        for (size_t grpid = 0; grpid < m_handle->io_info.size(); ++grpid)
        {
            auto *info = m_handle->io_info[grpid];
            if (!info) continue;
            for (size_t i = 0; i < info->nInputSize; ++i)
            {
                const char *name = info->pInputs[i].pName;
                if (!name) continue;
                if (0 == strcmp(name, "K_cache")) max_k_bytes = std::max(max_k_bytes, (size_t)info->pInputs[i].nSize);
                else if (0 == strcmp(name, "V_cache")) max_v_bytes = std::max(max_v_bytes, (size_t)info->pInputs[i].nSize);
            }
        }

        auto realloc_kv_if_needed = [&](const char *name, size_t want_bytes) -> int {
            if (want_bytes == 0) return 0;
            auto *first_info = m_handle->io_info[0];
            auto &first_data = m_handle->io_data[0];
            if (!first_info || !first_data.pInputs) return 0;

            for (size_t i = 0; i < first_info->nInputSize && i < first_data.nInputSize; ++i)
            {
                const char *n = first_info->pInputs[i].pName;
                if (!n || 0 != strcmp(n, name)) continue;
                auto &buf = first_data.pInputs[i];
                if ((size_t)buf.nSize >= want_bytes) return 0;
                const AX_U32 old_size = buf.nSize;

                if (buf.phyAddr != 0) AX_SYS_MemFree(buf.phyAddr, buf.pVirAddr);

                int ret = AX_SYS_MemAllocCached((AX_U64 *)(&buf.phyAddr),
                                               &buf.pVirAddr,
                                               want_bytes,
                                               AX_CMM_ALIGN_SIZE,
                                               (const AX_S8 *)(AX_CMM_SESSION_NAME));
                if (ret != 0)
                {
                    ALOGE("AX_SYS_MemAllocCached(%s, %zu) failed, ret=0x%x", name, want_bytes, ret);
                    buf.phyAddr = 0;
                    buf.pVirAddr = nullptr;
                    return ret;
                }
                buf.nSize = (AX_U32)want_bytes;
                memset(buf.pVirAddr, 0, want_bytes);
                ALOGD("realloc %s buffer: group0_size=%u -> max_group_size=%zu", name, old_size, want_bytes);
                return 0;
            }

            ALOGW("KV tensor %s not found in group 0", name);
            return 0;
        };

        int ret = realloc_kv_if_needed("K_cache", max_k_bytes);
        if (ret != 0) return ret;
        ret = realloc_kv_if_needed("V_cache", max_v_bytes);
        if (ret != 0) return ret;
    }

    // 2. 处理中间 Group 的内存共享逻辑 (原有逻辑的 Hack)
    if (io_count > 2)
    {
        auto &first_io_data = m_handle->io_data[0];
        auto &first_io_info = m_handle->io_info[0];
        auto &last_io_data = m_handle->io_data[io_count - 1];
        auto &last_io_info = m_handle->io_info[io_count - 1];
        for (uint i = 0; i < last_io_data.nInputSize; ++i)
        {
            const char *tensor_name = last_io_info->pInputs[i].pName;
            if (!tensor_name)
                continue;

            if (std::find(skip_alloc_names.begin(), skip_alloc_names.end(), tensor_name) != skip_alloc_names.end())
            {
                AX_ENGINE_IO_BUFFER_T *first_buf = find_input_buffer_by_name(first_io_info, &first_io_data, tensor_name);
                if (!first_buf)
                {
                    ALOGE("failed to find shared input buffer for %s in group0", tensor_name);
                    return -1;
                }

                if (first_buf->nSize < last_io_data.pInputs[i].nSize)
                {
                    ALOGE("shared input buffer too small for %s: src=%u dst=%u",
                          tensor_name,
                          first_buf->nSize,
                          last_io_data.pInputs[i].nSize);
                    return -1;
                }

                last_io_data.pInputs[i].phyAddr = first_buf->phyAddr;
                last_io_data.pInputs[i].pVirAddr = first_buf->pVirAddr;
                // Intermediate groups reuse buffers by referencing `last_io_data` as the shared pool.
                // Make sure the shared pool reflects the actual allocated bytes (which may exceed the
                // last group's symbolic tensor size for multi-shape models).
                last_io_data.pInputs[i].nSize = first_buf->nSize;
            }
        }

        for (size_t grpid = 1; grpid < io_count - 1; grpid++)
        {
            auto &io_info = m_handle->io_info[grpid];
            auto &io_data = m_handle->io_data[grpid];

            // Gemma4 prefill groups do not guarantee identical tensor ordering across groups.
            // Reuse buffers by tensor name instead of index to avoid wrong buffer aliasing.
            for (size_t i = 0; i < io_info->nInputSize; i++)
            {
                const char *tensor_name = io_info->pInputs[i].pName;
                if (!tensor_name)
                    continue;

                AX_ENGINE_IO_BUFFER_T *shared = find_input_buffer_by_name(last_io_info, &last_io_data, tensor_name);
                if (!shared)
                {
                    ALOGE("failed to find shared input buffer for group %zu tensor %s", grpid, tensor_name);
                    return -1;
                }

                if (shared->nSize < io_data.pInputs[i].nSize)
                {
                    ALOGE("shared input buffer too small for group %zu tensor %s: src=%u dst=%u",
                          grpid,
                          tensor_name,
                          shared->nSize,
                          io_data.pInputs[i].nSize);
                    return -1;
                }

                io_data.pInputs[i].phyAddr = shared->phyAddr;
                io_data.pInputs[i].pVirAddr = shared->pVirAddr;
            }

            for (size_t i = 0; i < io_info->nOutputSize; i++)
            {
                const char *tensor_name = io_info->pOutputs[i].pName;
                if (!tensor_name)
                    continue;

                AX_ENGINE_IO_BUFFER_T *shared = find_output_buffer_by_name(last_io_info, &last_io_data, tensor_name);
                if (!shared)
                {
                    ALOGE("failed to find shared output buffer for group %zu tensor %s", grpid, tensor_name);
                    return -1;
                }

                if (shared->nSize < io_data.pOutputs[i].nSize)
                {
                    ALOGE("shared output buffer too small for group %zu tensor %s: src=%u dst=%u",
                          grpid,
                          tensor_name,
                          shared->nSize,
                          io_data.pOutputs[i].nSize);
                    return -1;
                }

                io_data.pOutputs[i].phyAddr = shared->phyAddr;
                io_data.pOutputs[i].pVirAddr = shared->pVirAddr;
            }
        }
    }

    // 3. 构建 Tensor 对象
    for (size_t grpid = 0; grpid < io_count; grpid++)
    {
        auto &io_info = m_handle->io_info[grpid];
        auto &io_data = m_handle->io_data[grpid];

        // Process Outputs
        for (size_t i = 0; i < io_info->nOutputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = io_info->pOutputs[i].pName ? std::string(io_info->pOutputs[i].pName) : "";
            tensor.nSize = io_info->pOutputs[i].nSize;
            tensor.phyAddr = io_data.pOutputs[i].phyAddr;
            tensor.pVirAddr = io_data.pOutputs[i].pVirAddr;
            for (size_t j = 0; j < io_info->pOutputs[i].nShapeSize; j++)
            {
                tensor.vShape.push_back(io_info->pOutputs[i].pShape[j]);
            }
            mgroup_output_tensors[grpid].push_back(tensor);
        }

        // Process Inputs
        for (size_t i = 0; i < io_info->nInputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = io_info->pInputs[i].pName ? std::string(io_info->pInputs[i].pName) : "";
            tensor.nSize = io_info->pInputs[i].nSize;
            tensor.phyAddr = io_data.pInputs[i].phyAddr;
            tensor.pVirAddr = io_data.pInputs[i].pVirAddr;
            for (size_t j = 0; j < io_info->pInputs[i].nShapeSize; j++)
            {
                tensor.vShape.push_back(io_info->pInputs[i].pShape[j]);
            }
            mgroup_input_tensors[grpid].push_back(tensor);
        }
    }

    if (!mgroup_output_tensors.empty())
        moutput_tensors = mgroup_output_tensors[0];
    if (!mgroup_input_tensors.empty())
        minput_tensors = mgroup_input_tensors[0];

    // 4. 构建查找表
    build_tensor_maps();

    return 0;
}

int ax_runner_ax650::init(const char *model_file, int /*devid*/)
{
    MMap model_buffer;
    if (!model_buffer.open_file(model_file))
    {
        ALOGE("model file(%s) open failed", model_file);
        return -1;
    }
    auto ret = init((char *)model_buffer.data(), model_buffer.size(), -1);
    return ret;
}

int ax_runner_ax650::init(char *model_buffer, size_t model_size, int /*devid*/)
{
    if (m_handle)
        deinit(); // 防止多次 init 导致泄漏
    m_handle = new ax_runner_ax650_handle_t;
    int ret = AX_ENGINE_CreateHandle(&m_handle->handle, model_buffer, model_size);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle failed: 0x%x", ret);
        delete m_handle;
        m_handle = nullptr;
        return ret;
    }
    return sub_init();
}

void ax_runner_ax650::deinit()
{
    if (!m_handle)
        return;

    // 使用 Set 防止物理内存被重复释放 (Double Free)
    std::unordered_set<unsigned long long> freed_phy_addrs;

    // 遍历所有 Group
    for (size_t g = 0; g < m_handle->io_data.size(); ++g)
    {
        auto &io = m_handle->io_data[g];

        // 1. 清理 Inputs
        if (io.pInputs)
        {
            for (size_t j = 0; j < io.nInputSize; ++j)
            {
                AX_ENGINE_IO_BUFFER_T *pBuf = io.pInputs + j;
                if (pBuf->phyAddr != 0)
                {
                    // 如果这个物理地址还没被释放过，则释放
                    if (freed_phy_addrs.find(pBuf->phyAddr) == freed_phy_addrs.end())
                    {
                        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
                        freed_phy_addrs.insert(pBuf->phyAddr);
                    }
                }
            }
            // 释放数组本身 (C++ heap memory)
            delete[] io.pInputs;
            io.pInputs = nullptr;
        }

        // 2. 清理 Outputs
        if (io.pOutputs)
        {
            for (size_t j = 0; j < io.nOutputSize; ++j)
            {
                AX_ENGINE_IO_BUFFER_T *pBuf = io.pOutputs + j;
                if (pBuf->phyAddr != 0)
                {
                    if (freed_phy_addrs.find(pBuf->phyAddr) == freed_phy_addrs.end())
                    {
                        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
                        freed_phy_addrs.insert(pBuf->phyAddr);
                    }
                }
            }
            delete[] io.pOutputs;
            io.pOutputs = nullptr;
        }
    }

    if (m_handle->handle)
    {
        AX_ENGINE_DestroyHandle(m_handle->handle);
    }

    delete m_handle;
    m_handle = nullptr;

    // 清空容器
    moutput_tensors.clear();
    minput_tensors.clear();
    map_input_tensors.clear();
    map_output_tensors.clear();
    mgroup_output_tensors.clear();
    mgroup_input_tensors.clear();
    map_group_input_tensors.clear();
    map_group_output_tensors.clear();
}

int ax_runner_ax650::inference()
{
    if (!m_handle)
        return -1;
    // 刷 Cache 保证数据一致性
    for (size_t i = 0; i < get_num_inputs(); i++)
    {
        auto &tensor = get_input(i);
        AX_SYS_MflushCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }

    int ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[0]);

    for (size_t i = 0; i < get_num_outputs(); i++)
    {
        auto &tensor = get_output(i);
        AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }
    return ret;
}

int ax_runner_ax650::inference(int grpid)
{
    if (!m_handle)
        return -1;
    if (grpid < 0 || grpid >= (int)m_handle->io_data.size())
        return -1;

    // 刷 Cache (Input)
    for (size_t i = 0; i < mgroup_input_tensors[grpid].size(); i++)
    {
        auto &tensor = mgroup_input_tensors[grpid][i];
        AX_SYS_MflushCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }

    int ret = AX_ENGINE_RunGroupIOSync(m_handle->handle, m_handle->context, grpid, &m_handle->io_data[grpid]);

    // 刷 Cache (Output)
    for (size_t i = 0; i < mgroup_output_tensors[grpid].size(); i++)
    {
        auto &tensor = mgroup_output_tensors[grpid][i];
        AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }
    return ret;
}
