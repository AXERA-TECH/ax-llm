#include "ax_model_runner_ax650.hpp"
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
        input[i].pVirAddr = malloc(input[i].nSize);
        if (input[i].pVirAddr == nullptr)
        {
            printf("malloc failed, ret: %d\n", ret);
            return ret;
        }
        memset(input[i].pVirAddr, 0, input[i].nSize);
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
        output[i].pVirAddr = malloc(output[i].nSize);
        if (output[i].pVirAddr == nullptr)
        {
            printf("malloc failed, ret: %d\n", ret);
            return ret;
        }
        memset(output[i].pVirAddr, 0, output[i].nSize);
        axcl_Memset(devPtr, 0, output[i].nSize, _devid);
    }
    return 0;
}

struct ax_joint_runner_ax650_handle_t
{
    uint64_t handle = 0;
    uint64_t context = 0;
    axclrtEngineIOInfo io_info = 0;
    std::vector<axclrtEngineIO> ios;
};

int ax_runner_ax650::sub_init()
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
    // 1. 分配 IO 资源
    for (size_t grpid = 0; grpid < group_count; grpid++)
    {
        ret = axcl_EngineCreateIO(m_handle->io_info, &m_handle->ios[grpid], dev_id);

        // 原有逻辑保持不变：Group 0 和 Last Group 分配物理内存，中间 Group 不分配
        if (grpid == 0)
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

    if (group_count > 2)
    {
        auto &first_input = mgroup_input_tensors[0];
        auto &last_input = mgroup_input_tensors[group_count - 1];
        auto &last_output = mgroup_output_tensors[group_count - 1];
        for (uint i = 0; i < last_input.size(); ++i)
        {
            if (std::find(skip_alloc_input_names.begin(), skip_alloc_input_names.end(), last_input[i].sName) != skip_alloc_input_names.end())
            {
                for (uint j = 0; j < first_input.size(); ++j)
                {
                    if (first_input[j].sName == last_input[i].sName)
                    {
                        last_input[i].phyAddr = first_input[j].phyAddr;
                        last_input[i].pVirAddr = first_input[j].pVirAddr;
                    }
                }
            }
        }

        for (size_t grpid = 1; grpid < group_count - 1; grpid++)
        {
            auto &input = mgroup_input_tensors[grpid];

            // 安全检查：确保维度匹配再拷贝
            size_t min_inputs = std::min(input.size(), last_input.size());
            for (size_t i = 0; i < min_inputs; i++)
            {
                input[i].phyAddr = last_input[i].phyAddr;
                input[i].pVirAddr = last_input[i].pVirAddr;
            }

            auto &output = mgroup_output_tensors[grpid];
            size_t min_outputs = std::min(output.size(), last_output.size());
            for (size_t i = 0; i < min_outputs; i++)
            {
                output[i].phyAddr = last_output[i].phyAddr;
                output[i].pVirAddr = last_output[i].pVirAddr;
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

    // print_io_info(minput_tensors, mtensors);

    build_tensor_maps();

    return ret;
}

int ax_runner_ax650::init(const char *model_file, int devid)
{
    if (!m_handle)
    {
        m_handle = new ax_joint_runner_ax650_handle_t;
    }
    memset(m_handle, 0, sizeof(ax_joint_runner_ax650_handle_t));
    this->dev_id = devid;

    this->dev_id = devid;
    int ret = axcl_EngineLoadFromFile(model_file, &m_handle->handle, dev_id);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    return sub_init();
}

int ax_runner_ax650::init(char *model_buffer, size_t model_size, int devid)
{
    if (!m_handle)
    {
        m_handle = new ax_joint_runner_ax650_handle_t;
    }
    memset(m_handle, 0, sizeof(ax_joint_runner_ax650_handle_t));
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

void ax_runner_ax650::deinit()
{
    if (m_handle && m_handle->handle)
    {
        std::vector<unsigned long long> free_phy_addr;
        for (int grpid = 0; grpid < group_count; grpid++)
        {
            for (auto &tensor : mgroup_output_tensors[grpid])
            {
                if (free_phy_addr.end() == std::find(free_phy_addr.begin(), free_phy_addr.end(), tensor.phyAddr))
                {
                    axcl_Free((void *)tensor.phyAddr, dev_id);
                    free_phy_addr.push_back(tensor.phyAddr);
                }
            }
            for (auto &tensor : mgroup_input_tensors[grpid])
            {
                if (free_phy_addr.end() == std::find(free_phy_addr.begin(), free_phy_addr.end(), tensor.phyAddr))
                {
                    axcl_Free((void *)tensor.phyAddr, dev_id);
                    free_phy_addr.push_back(tensor.phyAddr);
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

int ax_runner_ax650::get_algo_width() { return -1; }
int ax_runner_ax650::get_algo_height() { return -1; }

int ax_runner_ax650::set_input(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
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
int ax_runner_ax650::set_output(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    if (size < get_output(grpid, idx).nSize)
    {
        ALOGE("set_output size %ld < %d", size, get_output(grpid, idx).nSize);
        return -1;
    }

    return axcl_EngineSetOutputBufferByIndex(m_handle->ios[grpid], idx, (void *)phy_addr, size, dev_id);
}

int ax_runner_ax650::set_input(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    if (size < get_input(grpid, name).nSize)
    {
        ALOGE("set_input size %ld < %d", size, mgroup_input_tensors[grpid][get_input(grpid, name).nIdx].nSize);
        return -1;
    }

    return axcl_EngineSetInputBufferByIndex(m_handle->ios[grpid], get_input(grpid, name).nIdx, (void *)phy_addr, size, dev_id);
}

int ax_runner_ax650::set_output(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    if (size < get_output(grpid, name).nSize)
    {
        ALOGE("set_output size %ld < %d", size, get_output(grpid, name).nSize);
        return -1;
    }

    return axcl_EngineSetOutputBufferByIndex(m_handle->ios[grpid], get_output(grpid, name).nIdx, (void *)phy_addr, size, dev_id);
}

ax_color_space_e ax_runner_ax650::get_color_space()
{
    return axdl_color_space_unknown;
}

int ax_runner_ax650::inference()
{
    return inference(0);
}

int ax_runner_ax650::inference(int grpid)
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