#include "ax_model_runner_axcl.hpp"
#include <string.h>
#include <fstream>
#include <memory>
#include <fcntl.h>
#include <algorithm>
#include "memory_utils.hpp"
#include "sample_log.h"
#include "axcl_manager.h"

#define AX_CMM_ALIGN_SIZE 128

static const char *AX_CMM_SESSION_NAME = "npu";

static void print_io_info(std::vector<ax_runner_tensor_t> &input, std::vector<ax_runner_tensor_t> &output)
{
    printf("\ninput size: %ld\n", input.size());
    for (uint32_t i = 0; i < input.size(); ++i)
    {
        // print shape info,like [batchsize x channel x height x width]
        auto &info = input[i];
        printf("    name: \e[1;32m%8s", info.sName.c_str());

        std::string dt = "unknown";

        printf(" \e[1;31m[%s] ", dt.c_str());

        std::string ct = "unknown";

        printf("\e[1;31m[%s]", ct.c_str());

        printf(" \n        \e[1;31m");

        for (int s = 0; s < info.vShape.size(); s++)
        {
            printf("%d", info.vShape[s]);
            if (s != info.vShape.size() - 1)
            {
                printf(" x ");
            }
        }
        printf("\e[0m\n\n");
    }

    printf("\noutput size: %ld\n", output.size());
    for (uint32_t i = 0; i < output.size(); ++i)
    {
        // print shape info,like [batchsize x channel x height x width]
        auto &info = output[i];
        printf("    name: \e[1;32m%8s \e[0m\n        \e[1;31m", info.sName.c_str());
        for (int s = 0; s < info.vShape.size(); s++)
        {
            printf("%d", info.vShape[s]);
            if (s != info.vShape.size() - 1)
            {
                printf(" x ");
            }
        }
        printf("\e[0m\n\n");
    }
}

static int prepare_io_struct_only(int grpid, axclrtEngineIOInfo io_info, axclrtEngineIO io, std::vector<ax_runner_tensor_t> &input, std::vector<ax_runner_tensor_t> &output, int _devid)
{
    auto inputNum = axcl_EngineGetNumInputs(io_info, _devid);
    auto outputNum = axcl_EngineGetNumOutputs(io_info, _devid);
    input.resize(inputNum);
    output.resize(outputNum);

    for (int32_t i = 0; i < inputNum; i++)
    {
        axclrtEngineIODims dims = {0};
        int ret = axcl_EngineGetInputDims(io_info, grpid, i, &dims, _devid);
        if (ret != 0)
        {
            printf("axcl_EngineGetInputDims failed, ret: %d\n", ret);
            return ret;
        }
        input[i].sName = axcl_EngineGetInputNameByIndex(io_info, i, _devid);
        input[i].vShape.resize(dims.dimCount);
        for (int32_t j = 0; j < dims.dimCount; j++)
        {
            input[i].vShape[j] = dims.dims[j];
        }
        input[i].nIdx = i;
        input[i].nSize = axcl_EngineGetInputSizeByIndex(io_info, grpid, i, _devid);

        input[i].phyAddr = 0;
        input[i].pVirAddr = 0;
    }

    for (int32_t i = 0; i < outputNum; i++)
    {
        axclrtEngineIODims dims = {0};
        int ret = axcl_EngineGetOutputDims(io_info, grpid, i, &dims, _devid);
        if (ret != 0)
        {
            printf("axcl_EngineGetOutputDims failed, ret: %d\n", ret);
            return ret;
        }
        output[i].sName = axcl_EngineGetOutputNameByIndex(io_info, i, _devid);
        output[i].vShape.resize(dims.dimCount);
        for (int32_t j = 0; j < dims.dimCount; j++)
        {
            output[i].vShape[j] = dims.dims[j];
        }
        output[i].nIdx = i;
        output[i].nSize = axcl_EngineGetOutputSizeByIndex(io_info, grpid, i, _devid);
        output[i].phyAddr = 0;
        output[i].pVirAddr = 0;
    }
    return 0;
}

static int alloc_host_buffer(void **host_ptr, size_t size, int devid);
static void free_host_buffer(void *host_ptr, int devid);

static int prepare_io_with_alloc(int grpid, axclrtEngineIOInfo io_info, axclrtEngineIO io,
                                 std::vector<ax_runner_tensor_t> &input, std::vector<ax_runner_tensor_t> &output,
                                 int _devid, std::vector<std::string> skip_alloc_input_names = {}, std::vector<std::string> skip_alloc_output_names = {})
{
    int ret = prepare_io_struct_only(grpid, io_info, io, input, output, _devid);
    if (ret != 0)
    {
        printf("prepare_io_struct_only failed, ret: %d\n", ret);
        return ret;
    }

    for (int32_t i = 0; i < input.size(); i++)
    {
        if (std::find(skip_alloc_input_names.begin(), skip_alloc_input_names.end(), input[i].sName) != skip_alloc_input_names.end())
        {
            continue;
        }
        void *devPtr = nullptr;
        ret = axcl_Malloc(&devPtr, input[i].nSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        if (ret != 0)
        {
            printf("axcl_Malloc failed, ret: %d\n", ret);
            return ret;
        }
        input[i].phyAddr = (unsigned long long)devPtr;
        ret = alloc_host_buffer(&input[i].pVirAddr, input[i].nSize, _devid);
        if (ret != 0)
        {
            axcl_Free(devPtr, _devid);
            input[i].phyAddr = 0;
            return ret;
        }
        axcl_Memset(devPtr, 0, input[i].nSize, _devid);
    }

    for (int32_t i = 0; i < output.size(); i++)
    {
        if (std::find(skip_alloc_output_names.begin(), skip_alloc_output_names.end(), output[i].sName) != skip_alloc_output_names.end())
        {
            continue;
        }
        void *devPtr = nullptr;
        ret = axcl_Malloc(&devPtr, output[i].nSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        if (ret != 0)
        {
            printf("axcl_Malloc failed, ret: %d\n", ret);
            return ret;
        }
        output[i].phyAddr = (unsigned long long)devPtr;
        ret = alloc_host_buffer(&output[i].pVirAddr, output[i].nSize, _devid);
        if (ret != 0)
        {
            axcl_Free(devPtr, _devid);
            output[i].phyAddr = 0;
            return ret;
        }
        axcl_Memset(devPtr, 0, output[i].nSize, _devid);
    }
    return 0;
}

static ax_runner_tensor_t *find_tensor_by_name(std::vector<ax_runner_tensor_t> &tensors, const std::string &name)
{
    for (auto &tensor : tensors)
    {
        if (tensor.sName == name)
        {
            return &tensor;
        }
    }
    return nullptr;
}

static bool debug_bindings_enabled()
{
    return std::getenv("AXLLM_AXCL_DEBUG_BINDINGS") != nullptr;
}

static bool should_log_binding_name(const std::string &name)
{
    return name == "indices" ||
           name == "input" ||
           name == "mask" ||
           name == "per_layer_input" ||
           name == "K_cache" ||
           name == "V_cache" ||
           name == "output" ||
           name == "K_cache_out" ||
           name == "V_cache_out";
}

static void dump_group_tensor_bindings(const char *kind, size_t grpid, const std::vector<ax_runner_tensor_t> &tensors)
{
    if (!debug_bindings_enabled())
    {
        return;
    }
    for (const auto &tensor : tensors)
    {
        if (!should_log_binding_name(tensor.sName))
        {
            continue;
        }
        ALOGI("AXCL_BIND group=%zu kind=%s name=%s idx=%d size=%zu phy=0x%llx host=%p",
              grpid,
              kind,
              tensor.sName.c_str(),
              tensor.nIdx,
              (size_t)tensor.nSize,
              (unsigned long long)tensor.phyAddr,
              tensor.pVirAddr);
    }
}

static int allocate_tensor_storage(ax_runner_tensor_t &tensor, int devid)
{
    void *devPtr = nullptr;
    int ret = axcl_Malloc(&devPtr, tensor.nSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, devid);
    if (ret != 0)
    {
        ALOGE("axcl_Malloc(%s, %zu) failed, ret=%d", tensor.sName.c_str(), (size_t)tensor.nSize, ret);
        return ret;
    }

    tensor.phyAddr = (unsigned long long)devPtr;
    ret = alloc_host_buffer(&tensor.pVirAddr, tensor.nSize, devid);
    if (ret != 0)
    {
        axcl_Free(devPtr, devid);
        tensor.phyAddr = 0;
        return ret;
    }
    axcl_Memset(devPtr, 0, tensor.nSize, devid);
    return 0;
}

static int alloc_host_buffer(void **host_ptr, size_t size, int devid)
{
    int ret = axcl_MallocHost(host_ptr, size, devid);
    if (ret != 0)
    {
        ALOGE("axcl_MallocHost(%zu) failed, ret=%d", size, ret);
        return ret;
    }
    memset(*host_ptr, 0, size);
    return 0;
}

static void free_host_buffer(void *host_ptr, int devid)
{
    if (host_ptr)
    {
        axcl_FreeHost(host_ptr, devid);
    }
}

struct ax_joint_runner_ax650_handle_t
{
    uint64_t handle = 0;
    uint64_t context = 0;
    axclrtEngineIOInfo io_info = 0;
    std::vector<axclrtEngineIO> ios;
};

int ax_runner_axcl::sub_init()
{
    // 4. create context
    int ret = axcl_EngineCreateContext(m_handle->handle, &m_handle->context, dev_id);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateContext");
        return ret;
    }
    // fprintf(stdout, "Engine creating context is done.\n");

    // 5. set io
    ret = axcl_EngineGetIOInfo(m_handle->handle, &m_handle->io_info, dev_id);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_GetIOInfo");
        return ret;
    }
    // fprintf(stdout, "Engine get io info is done. \n");

    ret = axcl_EngineGetShapeGroupsCount(m_handle->io_info, &group_count, dev_id);
    if (ret != 0)
    {
        axcl_EngineUnload(m_handle->handle, dev_id);
        return ret;
    }

    // 4. create io

    // fprintf(stdout, "Engine creating io is done. \n");

    // 6. alloc io

    m_handle->ios.resize(group_count);
    // m_handle->io_datas.resize(group_count);
    mgroup_input_tensors.resize(group_count);
    mgroup_output_tensors.resize(group_count);

    // memset(&m_handle->io_datas[0], 0, sizeof(AXCL_IO_DATA_T) * group_count);

    std::vector<std::string> skip_alloc_input_names = {"K_cache", "V_cache"};
    const bool disable_group_alias = std::getenv("AXLLM_AXCL_DISABLE_GROUP_ALIAS") != nullptr;
    const bool dedicate_prefill_indices = std::getenv("AXLLM_AXCL_DEDICATE_PREFILL_INDICES") != nullptr;
    const bool dedicate_prefill_kv = std::getenv("AXLLM_AXCL_DEDICATE_PREFILL_KV") != nullptr;
    const bool dedicate_prefill_outputs = std::getenv("AXLLM_AXCL_DEDICATE_PREFILL_OUTPUTS") != nullptr;
    // 1. 分配 IO 资源
    for (size_t grpid = 0; grpid < group_count; grpid++)
    {
        ret = axcl_EngineCreateIO(m_handle->io_info, &m_handle->ios[grpid], dev_id);

        if (disable_group_alias)
        {
            ret = prepare_io_with_alloc(grpid, m_handle->io_info, m_handle->ios[grpid], mgroup_input_tensors[grpid], mgroup_output_tensors[grpid], dev_id);
        }
        // 原有逻辑保持不变：Group 0 和 Last Group 分配物理内存，中间 Group 不分配
        else if (grpid == 0)
        {
            ret = prepare_io_with_alloc(grpid, m_handle->io_info, m_handle->ios[grpid], mgroup_input_tensors[grpid], mgroup_output_tensors[grpid], dev_id);
        }
        else if (grpid == group_count - 1)
        {
            ret = prepare_io_with_alloc(grpid, m_handle->io_info, m_handle->ios[grpid], mgroup_input_tensors[grpid], mgroup_output_tensors[grpid], dev_id, skip_alloc_input_names);
        }
        else
        {
            ret = prepare_io_struct_only(grpid, m_handle->io_info, m_handle->ios[grpid], mgroup_input_tensors[grpid], mgroup_output_tensors[grpid], dev_id);
        }
        if (ret != 0)
            return ret;
    }

    // Ensure K_cache/V_cache buffers are large enough for the maximum group.
    // We share KV buffers across shape groups to save memory, but some models
    // (e.g. multi-decode 2k/4k/8k/16k) have different KV sizes per group.
    // Group 0 might not be the largest, so allocate KV based on max required size.
    if (!disable_group_alias)
    {
        size_t max_k_bytes = 0;
        size_t max_v_bytes = 0;
        for (size_t grpid = 0; grpid < mgroup_input_tensors.size(); ++grpid)
        {
            for (const auto &t : mgroup_input_tensors[grpid])
            {
                if (t.sName == "K_cache") max_k_bytes = std::max(max_k_bytes, (size_t)t.nSize);
                else if (t.sName == "V_cache") max_v_bytes = std::max(max_v_bytes, (size_t)t.nSize);
            }
        }

        auto realloc_kv_if_needed = [&](const char *name, size_t want_bytes) -> int {
            if (want_bytes == 0) return 0;
            auto &first_input = mgroup_input_tensors[0];
            for (auto &t : first_input)
            {
                if (t.sName != name) continue;
                if ((size_t)t.nSize >= want_bytes) return 0;
                const size_t old_size = (size_t)t.nSize;

                // Free the original (smaller) buffers from group 0.
                if (t.phyAddr) axcl_Free((void *)t.phyAddr, dev_id);
                free_host_buffer(t.pVirAddr, dev_id);

                void *devPtr = nullptr;
                int ret = axcl_Malloc(&devPtr, want_bytes, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, dev_id);
                if (ret != 0)
                {
                    ALOGE("axcl_Malloc(%s, %zu) failed, ret=%d", name, want_bytes, ret);
                    t.phyAddr = 0;
                    t.pVirAddr = nullptr;
                    return ret;
                }
                t.phyAddr = (unsigned long long)devPtr;
                ret = alloc_host_buffer(&t.pVirAddr, want_bytes, dev_id);
                if (ret != 0)
                {
                    axcl_Free(devPtr, dev_id);
                    t.phyAddr = 0;
                    t.pVirAddr = nullptr;
                    return ret;
                }
                t.nSize = want_bytes;
                axcl_Memset(devPtr, 0, want_bytes, dev_id);
                ALOGD("realloc %s buffer: group0_size=%zu -> max_group_size=%zu", name, old_size, want_bytes);
                return 0;
            }
            // No KV tensor in group 0 (should not happen).
            ALOGW("KV tensor %s not found in group 0", name);
            return 0;
        };

        // Realloc in group 0 if needed (others will alias to group 0 later).
        int ret = realloc_kv_if_needed("K_cache", max_k_bytes);
        if (ret != 0) return ret;
        ret = realloc_kv_if_needed("V_cache", max_v_bytes);
        if (ret != 0) return ret;
    }

    if (!disable_group_alias && group_count > 2)
    {
        auto &first_input = mgroup_input_tensors[0];
        auto &last_input = mgroup_input_tensors[group_count - 1];
        auto &last_output = mgroup_output_tensors[group_count - 1];
        for (size_t i = 0; i < last_input.size(); ++i)
        {
            if (std::find(skip_alloc_input_names.begin(), skip_alloc_input_names.end(), last_input[i].sName) != skip_alloc_input_names.end())
            {
                ax_runner_tensor_t *shared = find_tensor_by_name(first_input, last_input[i].sName);
                if (shared == nullptr)
                {
                    ALOGE("failed to find shared input buffer for %s in group0", last_input[i].sName.c_str());
                    return -1;
                }
                if (shared->nSize < last_input[i].nSize)
                {
                    ALOGE("shared input buffer too small for %s: src=%zu dst=%zu",
                          last_input[i].sName.c_str(),
                          (size_t)shared->nSize,
                          (size_t)last_input[i].nSize);
                    return -1;
                }
                last_input[i].phyAddr = shared->phyAddr;
                last_input[i].pVirAddr = shared->pVirAddr;
            }
        }

        for (size_t grpid = 1; grpid < group_count - 1; grpid++)
        {
            auto &input = mgroup_input_tensors[grpid];

            // Gemma4 prefill groups may not keep identical tensor ordering across groups.
            for (size_t i = 0; i < input.size(); i++)
            {
                ax_runner_tensor_t *shared = find_tensor_by_name(last_input, input[i].sName);
                if (shared == nullptr)
                {
                    ALOGE("failed to find shared input buffer for group %zu tensor %s", grpid, input[i].sName.c_str());
                    return -1;
                }
                if (shared->nSize < input[i].nSize)
                {
                    ALOGE("shared input buffer too small for group %zu tensor %s: src=%zu dst=%zu",
                          grpid,
                          input[i].sName.c_str(),
                          (size_t)shared->nSize,
                          (size_t)input[i].nSize);
                    return -1;
                }
                input[i].phyAddr = shared->phyAddr;
                input[i].pVirAddr = shared->pVirAddr;
            }

            auto &output = mgroup_output_tensors[grpid];
            for (size_t i = 0; i < output.size(); i++)
            {
                ax_runner_tensor_t *shared = find_tensor_by_name(last_output, output[i].sName);
                if (shared == nullptr)
                {
                    ALOGE("failed to find shared output buffer for group %zu tensor %s", grpid, output[i].sName.c_str());
                    return -1;
                }
                if (shared->nSize < output[i].nSize)
                {
                    ALOGE("shared output buffer too small for group %zu tensor %s: src=%zu dst=%zu",
                          grpid,
                          output[i].sName.c_str(),
                          (size_t)shared->nSize,
                          (size_t)output[i].nSize);
                    return -1;
                }
                output[i].phyAddr = shared->phyAddr;
                output[i].pVirAddr = shared->pVirAddr;
            }
        }
    }

    if (!disable_group_alias && dedicate_prefill_indices)
    {
        for (size_t grpid = 1; grpid < group_count; ++grpid)
        {
            auto &inputs = mgroup_input_tensors[grpid];
            ax_runner_tensor_t *tensor = find_tensor_by_name(inputs, "indices");
            if (!tensor)
            {
                continue;
            }
            ret = allocate_tensor_storage(*tensor, dev_id);
            if (ret != 0)
            {
                return ret;
            }
        }
    }

    if (!disable_group_alias && dedicate_prefill_kv)
    {
        for (size_t grpid = 1; grpid < group_count; ++grpid)
        {
            auto &inputs = mgroup_input_tensors[grpid];
            for (const char *name : {"K_cache", "V_cache"})
            {
                ax_runner_tensor_t *tensor = find_tensor_by_name(inputs, name);
                if (!tensor)
                {
                    continue;
                }
                ret = allocate_tensor_storage(*tensor, dev_id);
                if (ret != 0)
                {
                    return ret;
                }
            }
        }
    }

    if (!disable_group_alias && dedicate_prefill_outputs)
    {
        for (size_t grpid = 1; grpid < group_count; ++grpid)
        {
            auto &outputs = mgroup_output_tensors[grpid];
            for (const char *name : {"K_cache_out", "V_cache_out", "output"})
            {
                ax_runner_tensor_t *tensor = find_tensor_by_name(outputs, name);
                if (!tensor)
                {
                    continue;
                }
                ret = allocate_tensor_storage(*tensor, dev_id);
                if (ret != 0)
                {
                    return ret;
                }
            }
        }
    }

    for (size_t grpid = 0; grpid < mgroup_input_tensors.size(); grpid++)
    {
        for (size_t i = 0; i < mgroup_input_tensors[grpid].size(); i++)
        {
            axcl_EngineSetInputBufferByIndex(m_handle->ios[grpid], i, (void *)mgroup_input_tensors[grpid][i].phyAddr, mgroup_input_tensors[grpid][i].nSize, dev_id);
        }
    }

    for (size_t grpid = 0; grpid < mgroup_output_tensors.size(); grpid++)
    {
        for (size_t i = 0; i < mgroup_output_tensors[grpid].size(); i++)
        {
            axcl_EngineSetOutputBufferByIndex(m_handle->ios[grpid], i, (void *)mgroup_output_tensors[grpid][i].phyAddr, mgroup_output_tensors[grpid][i].nSize, dev_id);
        }
    }

    if (!mgroup_output_tensors.empty())
        moutput_tensors = mgroup_output_tensors[0];
    if (!mgroup_input_tensors.empty())
        minput_tensors = mgroup_input_tensors[0];

    if (debug_bindings_enabled())
    {
        const size_t last_grpid = group_count > 0 ? group_count - 1 : 0;
        for (size_t grpid = 0; grpid < group_count; ++grpid)
        {
            if (grpid != 0 && grpid != 1 && grpid != last_grpid)
            {
                continue;
            }
            dump_group_tensor_bindings("input", grpid, mgroup_input_tensors[grpid]);
            dump_group_tensor_bindings("output", grpid, mgroup_output_tensors[grpid]);
        }
    }

    // print_io_info(minput_tensors, mtensors);

    build_tensor_maps();

    return ret;
}

int ax_runner_axcl::init(const char *model_file, int devid)
{
    if (m_handle)
    {
        deinit();
    }
    m_handle = new ax_joint_runner_ax650_handle_t;
    this->dev_id = devid;

    int ret = axcl_EngineLoadFromFile(model_file, &m_handle->handle, dev_id);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    return sub_init();
}

int ax_runner_axcl::init(char *model_buffer, size_t model_size, int devid)
{
    if (m_handle)
    {
        deinit();
    }
    m_handle = new ax_joint_runner_ax650_handle_t;
    this->dev_id = devid;

    void *devMem = nullptr;
    axcl_Malloc(&devMem, model_size, AXCL_MEM_MALLOC_NORMAL_ONLY, dev_id);

    axcl_Memcpy(devMem, model_buffer, model_size, AXCL_MEMCPY_HOST_TO_DEVICE, dev_id);

    int ret = axcl_EngineLoadFromMem(devMem, model_size, &m_handle->handle, dev_id);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    axcl_Free(devMem, dev_id);

    return sub_init();
}

void ax_runner_axcl::deinit()
{
    if (m_handle && m_handle->handle)
    {
        std::vector<unsigned long long> free_phy_addr;
        std::vector<void *> free_vir_addr;
        for (int grpid = 0; grpid < group_count; grpid++)
        {
            for (auto &tensor : mgroup_output_tensors[grpid])
            {
                if (free_phy_addr.end() == std::find(free_phy_addr.begin(), free_phy_addr.end(), tensor.phyAddr))
                {
                    axcl_Free((void *)tensor.phyAddr, dev_id);
                    free_phy_addr.push_back(tensor.phyAddr);
                }
                if (free_vir_addr.end() == std::find(free_vir_addr.begin(), free_vir_addr.end(), tensor.pVirAddr))
                {
                    free_host_buffer(tensor.pVirAddr, dev_id);
                    free_vir_addr.push_back(tensor.pVirAddr);
                }
            }
            for (auto &tensor : mgroup_input_tensors[grpid])
            {
                if (free_phy_addr.end() == std::find(free_phy_addr.begin(), free_phy_addr.end(), tensor.phyAddr))
                {
                    axcl_Free((void *)tensor.phyAddr, dev_id);
                    free_phy_addr.push_back(tensor.phyAddr);
                }
                if (free_vir_addr.end() == std::find(free_vir_addr.begin(), free_vir_addr.end(), tensor.pVirAddr))
                {
                    free_host_buffer(tensor.pVirAddr, dev_id);
                    free_vir_addr.push_back(tensor.pVirAddr);
                }
            }
            axcl_EngineDestroyIO(m_handle->ios[grpid], dev_id);
        }

        axcl_EngineUnload(m_handle->handle, dev_id);
        m_handle->handle = 0;
    }

    if (m_handle)
    {
        delete m_handle;
        m_handle = nullptr;
    }

    minput_tensors.clear();
    moutput_tensors.clear();

    map_input_tensors.clear();
    map_output_tensors.clear();

    mgroup_input_tensors.clear();
    mgroup_output_tensors.clear();

    map_group_input_tensors.clear();
    map_group_output_tensors.clear();
}

int ax_runner_axcl::get_algo_width() { return -1; }
int ax_runner_axcl::get_algo_height() { return -1; }

int ax_runner_axcl::set_input(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    if (size < get_input(grpid, idx).nSize)
    {
        ALOGE("set_input size %ld < %d", size, get_input(grpid, idx).nSize);
        return -1;
    }

    int ret = axcl_EngineSetInputBufferByIndex(m_handle->ios[grpid], idx, (void *)phy_addr, size, dev_id);
    if (0 != ret)
    {
        ALOGE("axcl_EngineSetInputBufferByIndex %d", ret);
        return ret;
    }
    auto &input = mgroup_input_tensors[grpid][idx];
    input.phyAddr = phy_addr;
    return ret;
}
int ax_runner_axcl::set_output(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    if (size < get_output(grpid, idx).nSize)
    {
        ALOGE("set_output size %ld < %d", size, get_output(grpid, idx).nSize);
        return -1;
    }

    return axcl_EngineSetOutputBufferByIndex(m_handle->ios[grpid], idx, (void *)phy_addr, size, dev_id);
}

int ax_runner_axcl::set_input(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    if (size < get_input(grpid, name).nSize)
    {
        ALOGE("set_input size %ld < %d", size, mgroup_input_tensors[grpid][get_input(grpid, name).nIdx].nSize);
        return -1;
    }

    return axcl_EngineSetInputBufferByIndex(m_handle->ios[grpid], get_input(grpid, name).nIdx, (void *)phy_addr, size, dev_id);
}

int ax_runner_axcl::set_output(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    if (size < get_output(grpid, name).nSize)
    {
        ALOGE("set_output size %ld < %d", size, get_output(grpid, name).nSize);
        return -1;
    }

    return axcl_EngineSetOutputBufferByIndex(m_handle->ios[grpid], get_output(grpid, name).nIdx, (void *)phy_addr, size, dev_id);
}

ax_color_space_e ax_runner_axcl::get_color_space()
{
    return axdl_color_space_unknown;
}

int ax_runner_axcl::inference()
{
    return inference(0);
}

int ax_runner_axcl::inference(int grpid)
{
    if (_auto_sync_before_inference)
        for (size_t i = 0; i < mgroup_input_tensors[grpid].size(); i++)
            axcl_Memcpy((void *)mgroup_input_tensors[grpid][i].phyAddr, mgroup_input_tensors[grpid][i].pVirAddr, mgroup_input_tensors[grpid][i].nSize, AXCL_MEMCPY_HOST_TO_DEVICE, dev_id);

    auto ret = axcl_EngineExecute(m_handle->handle, m_handle->context, grpid, m_handle->ios[grpid], dev_id);
    if (ret != 0)
    {
        ALOGE("AX_ENGINE_Execute");
        return ret;
    }

    if (_auto_sync_after_inference)
        for (size_t i = 0; i < mgroup_output_tensors[grpid].size(); i++)
            axcl_Memcpy(mgroup_output_tensors[grpid][i].pVirAddr, (void *)mgroup_output_tensors[grpid][i].phyAddr, mgroup_output_tensors[grpid][i].nSize, AXCL_MEMCPY_DEVICE_TO_HOST, dev_id);

    return 0;
}
