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

    // 2. 处理中间 Group 的内存共享逻辑 (原有逻辑的 Hack)
    if (io_count > 2)
    {
        auto &first_io_data = m_handle->io_data[0];
        auto &first_io_info = m_handle->io_info[0];
        auto &last_io_data = m_handle->io_data[io_count - 1];
        auto &last_io_info = m_handle->io_info[io_count - 1];
        for (uint i = 0; i < last_io_data.nInputSize; ++i)
        {
            if (std::find(skip_alloc_names.begin(), skip_alloc_names.end(), last_io_info->pInputs[i].pName) != skip_alloc_names.end())
            {
                for (uint j = 0; j < first_io_data.nInputSize; ++j)
                {
                    if (first_io_info->pInputs[j].pName == last_io_info->pInputs[i].pName)
                    {
                        last_io_data.pInputs[i].phyAddr = first_io_data.pInputs[j].phyAddr;
                        last_io_data.pInputs[i].pVirAddr = first_io_data.pInputs[j].pVirAddr;
                    }
                }
            }
        }

        for (size_t grpid = 1; grpid < io_count - 1; grpid++)
        {
            auto &io_info = m_handle->io_info[grpid];
            auto &io_data = m_handle->io_data[grpid];

            // 安全检查：确保维度匹配再拷贝
            size_t min_inputs = std::min(io_info->nInputSize, last_io_data.nInputSize);
            for (size_t i = 0; i < min_inputs; i++)
            {
                io_data.pInputs[i].phyAddr = last_io_data.pInputs[i].phyAddr;
                io_data.pInputs[i].pVirAddr = last_io_data.pInputs[i].pVirAddr;
            }

            size_t min_outputs = std::min(io_info->nOutputSize, last_io_data.nOutputSize);
            for (size_t i = 0; i < min_outputs; i++)
            {
                io_data.pOutputs[i].phyAddr = last_io_data.pOutputs[i].phyAddr;
                io_data.pOutputs[i].pVirAddr = last_io_data.pOutputs[i].pVirAddr;
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