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
    const bool force_uncached = std::getenv("AXLLM_AX650_UNCACHED_IO") != nullptr;
    if (force_uncached)
    {
        strategy.first = AX_ENGINE_ABST_DEFAULT;
        strategy.second = AX_ENGINE_ABST_DEFAULT;
    }
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

    int ret = 0;
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
        if (io_count == 1)
        {
            ret = AX_ENGINE_GetIOInfo(m_handle->handle, &io_info);
        }
        else
        {
            ret = AX_ENGINE_GetGroupIOInfo(m_handle->handle, grpid, &io_info);
        }
        if (ret != 0)
            return ret;
        m_handle->io_info[grpid] = io_info;

        if (io_count == 1)
        {
            ret = prepare_io_with_alloc(io_info,
                                        &m_handle->io_data[grpid],
                                        {AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED});
        }
        // Keep the original AX650 multi-group behavior for KV-cache sharing.
        else if (grpid == 0)
        {
            ret = prepare_io_with_alloc(io_info,
                                        &m_handle->io_data[grpid],
                                        {AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED});
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
            const bool force_uncached = std::getenv("AXLLM_AX650_UNCACHED_IO") != nullptr;

            for (size_t i = 0; i < first_info->nInputSize && i < first_data.nInputSize; ++i)
            {
                const char *n = first_info->pInputs[i].pName;
                if (!n || 0 != strcmp(n, name)) continue;
                auto &buf = first_data.pInputs[i];
                if ((size_t)buf.nSize >= want_bytes) return 0;
                const AX_U32 old_size = buf.nSize;

                if (buf.phyAddr != 0) AX_SYS_MemFree(buf.phyAddr, buf.pVirAddr);

                int ret = force_uncached
                              ? AX_SYS_MemAlloc((AX_U64 *)(&buf.phyAddr),
                                                &buf.pVirAddr,
                                                want_bytes,
                                                AX_CMM_ALIGN_SIZE,
                                                (const AX_S8 *)(AX_CMM_SESSION_NAME))
                              : AX_SYS_MemAllocCached((AX_U64 *)(&buf.phyAddr),
                                                      &buf.pVirAddr,
                                                      want_bytes,
                                                      AX_CMM_ALIGN_SIZE,
                                                      (const AX_S8 *)(AX_CMM_SESSION_NAME));
                if (ret != 0)
                {
                    ALOGE("AX_SYS_MemAlloc%s(%s, %zu) failed, ret=0x%x",
                          force_uncached ? "" : "Cached",
                          name,
                          want_bytes,
                          ret);
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

    // 2. Share skipped KV inputs across groups.
    // Two-group models (decode + one prefill group) still need the last
    // group's skipped K/V inputs to alias group0; otherwise the prefill
    // group's fake K/V tensors can keep null buffers and poison layer0.
    if (io_count > 1)
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
    }

    // 3. 处理中间 Group 的内存共享逻辑 (原有逻辑的 Hack)
    if (io_count > 2)
    {
        auto &last_io_data = m_handle->io_data[io_count - 1];
        auto &last_io_info = m_handle->io_info[io_count - 1];
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

    // 4. 构建 Tensor 对象
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
            tensor.eDataType = static_cast<int>(io_info->pOutputs[i].eDataType);
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
            tensor.eDataType = static_cast<int>(io_info->pInputs[i].eDataType);
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
    if (m_handle)
        deinit();

    MMap model_buffer;
    if (!model_buffer.open_file(model_file))
    {
        ALOGE("model file(%s) open failed", model_file);
        return -1;
    }
    // Feed the mmap'd data straight to CreateHandle (in the buffer overload below).
    // The engine loads it into CMM during init(); we never keep a host-side copy of
    // the model (that copy was ~= model size per axmodel and dominated host DDR).
    // MMap stays alive for the duration of this call.
    return init(reinterpret_cast<char *>(model_buffer.data()), model_buffer.size(), -1);
}

int ax_runner_ax650::init(char *model_buffer, size_t model_size, int /*devid*/)
{
    if (m_handle)
        deinit(); // 防止多次 init 导致泄漏

    // CreateHandle loads the model into CMM during this call; we deliberately do NOT
    // retain a host-side copy of `model_buffer` (the caller owns it and it can be
    // released right after). This keeps host DDR ~= working set, not model size.
    m_handle = new ax_runner_ax650_handle_t;
    AX_ENGINE_HANDLE_EXTRA_T extra{};
    extra.pName = (AX_S8 *)(AX_CMM_SESSION_NAME);
    int ret = AX_ENGINE_CreateHandleV2(&m_handle->handle, model_buffer, model_size, &extra);
    if (0 != ret)
    {
        ret = AX_ENGINE_CreateHandle(&m_handle->handle, model_buffer, model_size);
    }
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle failed: 0x%x", ret);
        delete m_handle;
        m_handle = nullptr;
        return ret;
    }
    ret = AX_ENGINE_CreateContextV2(m_handle->handle, &m_handle->context);
    if (ret != 0)
    {
        ALOGW("AX_ENGINE_CreateContextV2 failed: 0x%x, fallback to AX_ENGINE_CreateContext", ret);
        ret = AX_ENGINE_CreateContext(m_handle->handle);
        if (ret != 0)
        {
            ALOGE("AX_ENGINE_CreateContext failed: 0x%x", ret);
            deinit();
            return ret;
        }
        m_handle->context = 0;
    }
    return sub_init();
}

bool ax_runner_ax650::has_handle_loaded() const
{
    return m_handle && m_handle->handle != nullptr;
}

int ax_runner_ax650::load_handle_reuse_io(const char *model_file, int /*devid*/)
{
    if (has_handle_loaded())
        return 0;

    // No prepared IO buffers yet → fall back to full init (allocates IO buffers).
    if (!m_handle || m_handle->io_data.empty() || mgroup_input_tensors.empty() || mgroup_output_tensors.empty())
    {
        return init(model_file, -1);
    }

    if (!model_file || !model_file[0])
    {
        ALOGE("load_handle_reuse_io: model_file is empty");
        return -1;
    }

    // Re-mmap the file on each reload (dynamic layer loading). We keep no persistent
    // host copy of the model — trading a little reload I/O for much lower host DDR.
    // MMap stays alive through CreateHandle below, then unmaps on return.
    MMap model_buffer;
    if (!model_buffer.open_file(model_file))
    {
        ALOGE("model file(%s) open failed", model_file);
        return -1;
    }

    AX_ENGINE_HANDLE_EXTRA_T extra{};
    extra.pName = (AX_S8 *)(AX_CMM_SESSION_NAME);

    int ret = AX_ENGINE_CreateHandleV2(&m_handle->handle, reinterpret_cast<char *>(model_buffer.data()), model_buffer.size(), &extra);
    if (ret != 0)
    {
        ret = AX_ENGINE_CreateHandle(&m_handle->handle, reinterpret_cast<char *>(model_buffer.data()), model_buffer.size());
    }
    if (ret != 0)
    {
        ALOGE("AX_ENGINE_CreateHandle failed: 0x%x", ret);
        m_handle->handle = nullptr;
        return ret;
    }

    ret = AX_ENGINE_CreateContextV2(m_handle->handle, &m_handle->context);
    if (ret != 0)
    {
        ALOGW("AX_ENGINE_CreateContextV2 failed: 0x%x, fallback to AX_ENGINE_CreateContext", ret);
        ret = AX_ENGINE_CreateContext(m_handle->handle);
        if (ret != 0)
        {
            ALOGE("AX_ENGINE_CreateContext failed: 0x%x", ret);
            AX_ENGINE_DestroyHandle(m_handle->handle);
            m_handle->handle = nullptr;
            m_handle->context = 0;
            return ret;
        }
        m_handle->context = 0;
    }

    // Keep existing io_data + buffers. Clear io_info pointers to avoid dangling refs
    // from the previous handle instance (they are owned by the engine).
    for (auto &p : m_handle->io_info) p = nullptr;

    return 0;
}

void ax_runner_ax650::unload_handle_keep_io()
{
    if (!has_handle_loaded())
        return;

    AX_ENGINE_DestroyHandle(m_handle->handle);
    m_handle->handle = nullptr;
    m_handle->context = 0;
    for (auto &p : m_handle->io_info) p = nullptr;
}

void ax_runner_ax650::deinit()
{
    if (!m_handle)
        return;

    // Restore slot-0 binding and free extra slot buffers before the engine
    // buffers are freed below, so the dedup-free loop sees the genuine engine
    // K/V buffers (slot 0) rather than a leftover extra slot.
    kv_cache_slots_release();

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
    if (!has_handle_loaded())
        return -1;

    const bool force_runsync = std::getenv("AXLLM_AX650_FORCE_RUNSYNC") != nullptr;
    if (_auto_sync_before_inference)
    {
        for (size_t i = 0; i < get_num_inputs(); i++)
        {
            auto &tensor = get_input(i);
            int sync_ret = AX_SYS_MflushCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
            if (sync_ret != 0)
            {
                ALOGE("AX_SYS_MflushCache(input[%zu]) failed: 0x%x", i, sync_ret);
                return sync_ret;
            }
        }
    }

    int ret = 0;
    if (m_handle->io_data.size() == 1)
    {
        // Some single-group AX650 models, including the SD1.5 text encoder, are
        // stable with the legacy RunSync path but can fail with 0x8006008a when
        // driven through RunSyncV2. Keep V2 contexts for multi-group models and
        // always use RunSync for single-group execution.
        ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[0]);
    }
    else
    {
        if (!force_runsync && m_handle->context != 0)
        {
            ret = AX_ENGINE_RunGroupIOSync(m_handle->handle, m_handle->context, 0, &m_handle->io_data[0]);
            if (ret != 0)
            {
                ALOGW("AX_ENGINE_RunGroupIOSync(grpid=0) failed: 0x%x, fallback to AX_ENGINE_RunSync", ret);
                ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[0]);
            }
        }
        else
        {
            ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[0]);
        }
    }
    if (ret != 0)
    {
        ALOGE("AX_ENGINE_RunSync failed: 0x%x", ret);
        return ret;
    }

    if (_auto_sync_after_inference)
    {
        for (size_t i = 0; i < get_num_outputs(); i++)
        {
            auto &tensor = get_output(i);
            int sync_ret = AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
            if (sync_ret != 0)
            {
                ALOGE("AX_SYS_MinvalidateCache(output[%zu]) failed: 0x%x", i, sync_ret);
                return sync_ret;
            }
        }
    }
    return 0;
}

int ax_runner_ax650::inference(int grpid)
{
    if (!has_handle_loaded())
        return -1;
    if (grpid < 0 || grpid >= (int)m_handle->io_data.size())
        return -1;
    const bool force_runsync = std::getenv("AXLLM_AX650_FORCE_RUNSYNC") != nullptr;

    if (std::getenv("AXLLM_DEBUG_LAYER0_IO"))
    {
        auto *info = m_handle->io_info[(size_t)grpid];
        auto &io = m_handle->io_data[(size_t)grpid];
        ALOGI("AX650 inference grpid=%d io_inputs=%u io_outputs=%u", grpid, io.nInputSize, io.nOutputSize);
        if (info)
        {
            for (size_t i = 0; i < io.nInputSize && i < info->nInputSize; ++i)
            {
                const char *name = info->pInputs[i].pName ? info->pInputs[i].pName : "(null)";
                ALOGI("AX650 io_in[%zu] name=%s phy=%p vir=%p data_bytes=%u info_bytes=%u",
                      i,
                      name,
                      (void *)io.pInputs[i].phyAddr,
                      io.pInputs[i].pVirAddr,
                      io.pInputs[i].nSize,
                      info->pInputs[i].nSize);
            }
            for (size_t i = 0; i < io.nOutputSize && i < info->nOutputSize; ++i)
            {
                const char *name = info->pOutputs[i].pName ? info->pOutputs[i].pName : "(null)";
                ALOGI("AX650 io_out[%zu] name=%s phy=%p vir=%p data_bytes=%u info_bytes=%u",
                      i,
                      name,
                      (void *)io.pOutputs[i].phyAddr,
                      io.pOutputs[i].pVirAddr,
                      io.pOutputs[i].nSize,
                      info->pOutputs[i].nSize);
            }
        }
    }

    if (_auto_sync_before_inference)
    {
        for (size_t i = 0; i < mgroup_input_tensors[grpid].size(); i++)
        {
            auto &tensor = mgroup_input_tensors[grpid][i];
            int sync_ret = AX_SYS_MflushCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
            if (sync_ret != 0)
            {
                ALOGE("AX_SYS_MflushCache(group=%d input[%zu]) failed: 0x%x", grpid, i, sync_ret);
                return sync_ret;
            }
        }
    }

    int ret = 0;
    if (m_handle->io_data.size() == 1)
    {
        ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[0]);
    }
    else
    {
        if (!force_runsync && m_handle->context != 0)
        {
            ret = AX_ENGINE_RunGroupIOSync(m_handle->handle, m_handle->context, grpid, &m_handle->io_data[grpid]);
            if (ret != 0)
            {
                ALOGW("AX_ENGINE_RunGroupIOSync(grpid=%d) failed: 0x%x, fallback to AX_ENGINE_RunSync", grpid, ret);
                ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[grpid]);
            }
        }
        else
        {
            ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[grpid]);
        }
    }
    if (ret != 0)
    {
        ALOGE("AX_ENGINE_Run%s failed: 0x%x", m_handle->io_data.size() == 1 ? "Sync" : "GroupIOSync", ret);
        return ret;
    }
    else if (std::getenv("AXLLM_DEBUG_LAYER0_IO"))
    {
        ALOGI("AX_ENGINE_Run%s(grpid=%d) ok",
              m_handle->io_data.size() == 1 ? "Sync" : "GroupIOSync",
              grpid);
    }

    if (_auto_sync_after_inference)
    {
        for (size_t i = 0; i < mgroup_output_tensors[grpid].size(); i++)
        {
            auto &tensor = mgroup_output_tensors[grpid][i];
            int sync_ret = AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
            if (sync_ret != 0)
            {
                ALOGE("AX_SYS_MinvalidateCache(group=%d output[%zu]) failed: 0x%x", grpid, i, sync_ret);
                return sync_ret;
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Multi-slot KV cache (device-resident prefix cache)
// ---------------------------------------------------------------------------
size_t ax_runner_ax650::kv_cache_slots_prepare()
{
    kv_slot_tensors_.clear();
    kv_num_slots_ = 1;
    kv_active_slot_ = 0;
    if (!m_handle || m_handle->io_data.empty() || m_handle->io_info.empty())
        return 0;

    const int ngrp = (int)m_handle->io_data.size();
    const std::vector<std::string> kv_names = {"K_cache", "V_cache"};
    size_t per_slot_bytes = 0;
    for (const auto &name : kv_names)
    {
        KvSlotTensor st;
        st.name = name;
        st.idx_in_group.assign(ngrp, -1);
        unsigned long long slot0_phy = 0;
        void *slot0_vir = nullptr;
        size_t max_bytes = 0;
        for (int g = 0; g < ngrp; ++g)
        {
            auto *info = m_handle->io_info[g];
            auto &io = m_handle->io_data[g];
            if (!info || !io.pInputs) continue;
            for (uint i = 0; i < info->nInputSize; ++i)
            {
                const char *pn = info->pInputs[i].pName;
                if (!pn || name != pn) continue;
                st.idx_in_group[g] = (int)i;
                max_bytes = std::max(max_bytes, (size_t)io.pInputs[i].nSize);
                if (io.pInputs[i].phyAddr != 0 && slot0_phy == 0)
                {
                    slot0_phy = io.pInputs[i].phyAddr;
                    slot0_vir = io.pInputs[i].pVirAddr;
                }
                break;
            }
        }
        if (max_bytes == 0 || slot0_phy == 0)
        {
            ALOGW("kv slot: tensor %s not found / unallocated, skip", name.c_str());
            continue;
        }
        st.bytes = max_bytes;
        st.slot_phy.assign(1, slot0_phy); // slot 0 = borrowed engine buffer
        st.slot_vir.assign(1, slot0_vir);
        per_slot_bytes += max_bytes;
        kv_slot_tensors_.push_back(std::move(st));
    }
    return kv_slot_tensors_.empty() ? 0 : per_slot_bytes;
}

int ax_runner_ax650::kv_cache_slots_alloc(int num_slots)
{
    if (num_slots <= 1 || kv_slot_tensors_.empty())
    {
        kv_num_slots_ = 1;
        return 1;
    }
    const bool force_uncached = std::getenv("AXLLM_AX650_UNCACHED_IO") != nullptr;
    for (auto &st : kv_slot_tensors_) { st.slot_phy.resize(num_slots, 0); st.slot_vir.resize(num_slots, nullptr); }

    int achieved = 1;
    for (int s = 1; s < num_slots; ++s)
    {
        bool ok = true;
        for (auto &st : kv_slot_tensors_)
        {
            AX_U64 phy = 0;
            void *vir = nullptr;
            int ret = force_uncached
                          ? AX_SYS_MemAlloc(&phy, &vir, st.bytes, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME))
                          : AX_SYS_MemAllocCached(&phy, &vir, st.bytes, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
            if (ret != 0)
            {
                ALOGW("kv slot: alloc %s slot %d (%zu bytes) failed: 0x%x", st.name.c_str(), s, st.bytes, ret);
                ok = false;
                break;
            }
            memset(vir, 0, st.bytes);
            st.slot_phy[s] = phy;
            st.slot_vir[s] = vir;
        }
        if (!ok)
        {
            for (auto &st : kv_slot_tensors_)
                if (st.slot_phy[s]) { AX_SYS_MemFree(st.slot_phy[s], st.slot_vir[s]); st.slot_phy[s] = 0; st.slot_vir[s] = nullptr; }
            break;
        }
        achieved = s + 1;
    }
    for (auto &st : kv_slot_tensors_) { st.slot_phy.resize(achieved); st.slot_vir.resize(achieved); }
    kv_num_slots_ = achieved;
    kv_active_slot_ = 0;
    return achieved;
}

int ax_runner_ax650::kv_cache_slots_set_count(int n)
{
    if (n < 1) n = 1;
    if (kv_slot_tensors_.empty()) { kv_num_slots_ = 1; return 1; }
    if (kv_active_slot_ >= n) kv_cache_slots_activate(0);
    for (auto &st : kv_slot_tensors_)
    {
        for (int s = n; s < (int)st.slot_phy.size(); ++s)
            if (st.slot_phy[s]) { AX_SYS_MemFree(st.slot_phy[s], st.slot_vir[s]); st.slot_phy[s] = 0; st.slot_vir[s] = nullptr; }
        if ((int)st.slot_phy.size() > n) { st.slot_phy.resize(n); st.slot_vir.resize(n); }
    }
    kv_num_slots_ = n;
    return n;
}

int ax_runner_ax650::kv_cache_slots_activate(int slot)
{
    if (kv_num_slots_ <= 1) return slot == 0 ? 0 : -1;
    if (slot < 0 || slot >= kv_num_slots_) return -1;
    if (slot == kv_active_slot_) return 0;

    const int ngrp = (int)m_handle->io_data.size();
    for (auto &st : kv_slot_tensors_)
    {
        const unsigned long long phy = st.slot_phy[slot];
        void *vir = st.slot_vir[slot];
        for (int g = 0; g < ngrp; ++g)
        {
            const int idx = st.idx_in_group[g];
            if (idx < 0) continue;
            auto &io = m_handle->io_data[g];
            if (io.pInputs && idx < (int)io.nInputSize)
            {
                io.pInputs[idx].phyAddr = phy;
                io.pInputs[idx].pVirAddr = vir;
            }
            if (g < (int)mgroup_input_tensors.size() && idx < (int)mgroup_input_tensors[g].size())
            {
                mgroup_input_tensors[g][idx].phyAddr = phy;
                mgroup_input_tensors[g][idx].pVirAddr = vir;
            }
        }
    }
    if (!mgroup_input_tensors.empty())
        minput_tensors = mgroup_input_tensors[0];
    build_tensor_maps();
    kv_active_slot_ = slot;
    return 0;
}

void ax_runner_ax650::kv_cache_slots_release()
{
    if (kv_slot_tensors_.empty())
    {
        kv_num_slots_ = 1;
        kv_active_slot_ = 0;
        return;
    }
    // Rebind descriptors/io_data back to slot 0 (the engine's own buffers).
    if (kv_active_slot_ != 0)
        kv_cache_slots_activate(0);
    // Free owned (slot >= 1) buffers.
    for (auto &st : kv_slot_tensors_)
    {
        for (int s = 1; s < (int)st.slot_phy.size(); ++s)
        {
            if (st.slot_phy[s])
                AX_SYS_MemFree(st.slot_phy[s], st.slot_vir[s]);
            st.slot_phy[s] = 0;
            st.slot_vir[s] = nullptr;
        }
    }
    kv_slot_tensors_.clear();
    kv_num_slots_ = 1;
    kv_active_slot_ = 0;
}
