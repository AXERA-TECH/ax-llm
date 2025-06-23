#include "ax_model_runner_ax650.hpp"
#include "string.h"
#include "fstream"
#include "memory"
// #include "utilities/file.hpp"
// #include <ax_sys_api.h>
// #include <ax_ivps_api.h>
// #include <ax_engine_api.h>
#include <fcntl.h>
#include "memory_utils.hpp"
#include "sample_log.h"

// #include <axcl/native/ax_sys_api.h>
#include "axcl_manager.h"

#define AX_CMM_ALIGN_SIZE 128

static const char *AX_CMM_SESSION_NAME = "npu";

typedef enum
{
    AX_ENGINE_ABST_DEFAULT = 0,
    AX_ENGINE_ABST_CACHED = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

typedef std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T> INPUT_OUTPUT_ALLOC_STRATEGY;

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
        printf("\e[0m size:%d\n\n", info.nSize);
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
        printf("\e[0m size:%d\n\n", info.nSize);
    }
}

typedef struct
{
    int nIndex;
    int nSize;
    void *pBuf;
    void *pVirAddr;

    std::string Name;

    axclrtEngineIODims dims;
} AXCL_IO_BUF_T;

typedef struct
{
    uint32_t nInputSize;
    uint32_t nOutputSize;
    AXCL_IO_BUF_T *pInputs;
    AXCL_IO_BUF_T *pOutputs;
} AXCL_IO_DATA_T;

static void free_io_index(AXCL_IO_BUF_T *pBuf, size_t index, int _devid)
{
    for (size_t i = 0; i < index; ++i)
    {
        axcl_Free(pBuf[i].pBuf, _devid);
    }
}

static void free_io(AXCL_IO_DATA_T *io_data, int _devid)
{
    for (size_t j = 0; j < io_data->nInputSize; ++j)
    {
        axcl_Free(io_data->pInputs[j].pBuf, _devid);
        free(io_data->pInputs[j].pVirAddr);
    }
    for (size_t j = 0; j < io_data->nOutputSize; ++j)
    {
        axcl_Free(io_data->pOutputs[j].pBuf, _devid);
        free(io_data->pOutputs[j].pVirAddr);
    }
    delete[] io_data->pInputs;
    delete[] io_data->pOutputs;
}

static inline int prepare_io(int grpid, axclrtEngineIOInfo io_info, axclrtEngineIO io, AXCL_IO_DATA_T *io_data, INPUT_OUTPUT_ALLOC_STRATEGY strategy, int _devid)
{
    memset(io_data, 0, sizeof(AXCL_IO_DATA_T));

    auto inputNum = axcl_EngineGetNumInputs(io_info, _devid);
    auto outputNum = axcl_EngineGetNumOutputs(io_info, _devid);
    io_data->nInputSize = inputNum;
    io_data->nOutputSize = outputNum;
    io_data->pInputs = new AXCL_IO_BUF_T[inputNum];
    io_data->pOutputs = new AXCL_IO_BUF_T[outputNum];

    // 1. alloc inputs
    for (int32_t i = 0; i < inputNum; i++)
    {
        auto bufSize = axcl_EngineGetInputSizeByIndex(io_info, grpid, i, _devid);
        void *devPtr = nullptr;
        axclError ret = 0;
        if (AX_ENGINE_ABST_DEFAULT == strategy.first)
        {
            ret = axcl_Malloc(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        }
        else
        {
            ret = axcl_MallocCached(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        }

        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i, _devid);
            fprintf(stderr, "Malloc input(index: %d, size: %ld) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
        std::vector<char> tmp(bufSize, 0);
        axcl_Memcpy(devPtr, tmp.data(), bufSize, axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _devid);
        // axclrtMemset(devPtr, 0, bufSize);

        axclrtEngineIODims dims;
        ret = axcl_EngineGetInputDims(io_info, grpid, i, &dims, _devid);
        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i, _devid);
            fprintf(stderr, "Get input dims(index: %d) failed! ret=0x%x\n", i, ret);
            return -1;
        }

        io_data->pInputs[i].nIndex = i;
        io_data->pInputs[i].nSize = bufSize;
        io_data->pInputs[i].pBuf = devPtr;
        io_data->pInputs[i].dims = dims;
        io_data->pInputs[i].Name = axcl_EngineGetInputNameByIndex(io_info, i, _devid);
        io_data->pInputs[i].pVirAddr = malloc(bufSize);
        memset(io_data->pInputs[i].pVirAddr, 0, bufSize);
        ret = axcl_EngineSetInputBufferByIndex(io, i, devPtr, bufSize, _devid);
        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i, _devid);
            fprintf(stderr, "Set input buffer(index: %d, size: %lu) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
    }

    // 2. alloc outputs
    for (int32_t i = 0; i < outputNum; i++)
    {
        auto bufSize = axcl_EngineGetOutputSizeByIndex(io_info, grpid, i, _devid);
        void *devPtr = NULL;
        axclError ret = 0;
        if (AX_ENGINE_ABST_DEFAULT == strategy.first)
        {
            ret = axcl_Malloc(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        }
        else
        {
            ret = axcl_MallocCached(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        }

        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i, _devid);
            fprintf(stderr, "Malloc output(index: %d, size: %ld) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
        std::vector<char> tmp(bufSize, 0);
        axcl_Memcpy(devPtr, tmp.data(), bufSize, axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _devid);
        // axclrtMemset(devPtr, 0, bufSize);
        axclrtEngineIODims dims;
        ret = axcl_EngineGetOutputDims(io_info, grpid, i, &dims, _devid);
        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i, _devid);
            fprintf(stderr, "Get output dims(index: %d) failed! ret=0x%x\n", i, ret);
            return -1;
        }

        io_data->pOutputs[i].nIndex = i;
        io_data->pOutputs[i].nSize = bufSize;
        io_data->pOutputs[i].pBuf = devPtr;
        io_data->pOutputs[i].dims = dims;
        io_data->pOutputs[i].Name = axcl_EngineGetOutputNameByIndex(io_info, i, _devid);
        io_data->pOutputs[i].pVirAddr = malloc(bufSize);
        memset(io_data->pOutputs[i].pVirAddr, 0, bufSize);
        ret = axcl_EngineSetOutputBufferByIndex(io, i, devPtr, bufSize, _devid);
        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i, _devid);
            fprintf(stderr, "Set output buffer(index: %d, size: %lu) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
    }

    return 0;
}

struct ax_runner_ax650_handle_t
{
    uint64_t handle = 0;
    uint64_t context = 0;
    axclrtEngineIOInfo io_info = 0;
    std::vector<axclrtEngineIO> ios;
    std::vector<AXCL_IO_DATA_T> io_datas;

    // int algo_width, algo_height;
    // int algo_colorformat;
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
    if (!_parepare_io)
    {
        m_handle->ios.resize(group_count);
        m_handle->io_datas.resize(group_count);
        mgroup_input_tensors.resize(group_count);
        mgroup_output_tensors.resize(group_count);

        memset(&m_handle->io_datas[0], 0, sizeof(AXCL_IO_DATA_T) * group_count);

        auto malloc_strategy = std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_DEFAULT);

        for (int grpid = 0; grpid < group_count; grpid++)
        {
            ret = axcl_EngineCreateIO(m_handle->io_info, &m_handle->ios[grpid], dev_id);
            if (ret != 0)
            {
                axcl_EngineUnload(m_handle->handle, dev_id);
                fprintf(stderr, "Create io failed. ret=0x%x\n", ret);
                return -1;
            }

            ret = prepare_io(grpid, m_handle->io_info, m_handle->ios[grpid], &m_handle->io_datas[grpid], malloc_strategy, dev_id);
            if (ret != 0)
            {
                free_io(&m_handle->io_datas[grpid], dev_id);
                axcl_EngineDestroyIO(m_handle->ios[grpid], dev_id);
                axcl_EngineUnload(m_handle->handle, dev_id);

                fprintf(stderr, "prepare_io failed.\n");
                return ret;
            }
        }

        for (size_t grpid = 0; grpid < group_count; grpid++)
        {
            // auto &io_info = m_handle->io_info[grpid];
            auto &io_data = m_handle->io_datas[grpid];
            for (size_t i = 0; i < io_data.nOutputSize; i++)
            {
                ax_runner_tensor_t tensor;
                tensor.nIdx = i;
                tensor.sName = std::string(io_data.pOutputs[i].Name);
                tensor.nSize = io_data.pOutputs[i].nSize;
                for (size_t j = 0; j < io_data.pOutputs[i].dims.dimCount; j++)
                {
                    tensor.vShape.push_back(io_data.pOutputs[i].dims.dims[j]);
                }
                tensor.phyAddr = (unsigned long long)io_data.pOutputs[i].pBuf;
                tensor.pVirAddr = io_data.pOutputs[i].pVirAddr;
                mgroup_output_tensors[grpid].push_back(tensor);
            }

            for (size_t i = 0; i < io_data.nInputSize; i++)
            {
                ax_runner_tensor_t tensor;
                tensor.nIdx = i;
                tensor.sName = std::string(io_data.pInputs[i].Name);
                tensor.nSize = io_data.pInputs[i].nSize;
                for (size_t j = 0; j < io_data.pInputs[i].dims.dimCount; j++)
                {
                    tensor.vShape.push_back(io_data.pInputs[i].dims.dims[j]);
                }
                tensor.phyAddr = (unsigned long long)io_data.pInputs[i].pBuf;
                tensor.pVirAddr = io_data.pInputs[i].pVirAddr;
                mgroup_input_tensors[grpid].push_back(tensor);
            }
        }

        moutput_tensors = mgroup_output_tensors[0];
        minput_tensors = mgroup_input_tensors[0];
        _parepare_io = true;
    }
    else
    {
    }
    // print_io_info(minput_tensors, mtensors);

    return ret;
}

int ax_runner_ax650::init(const char *model_file, int devid)
{
    this->dev_id = devid;
    char *model_buffer;
    size_t len;
    if (!read_file(model_file, &model_buffer, &len))
    {
        ALOGE("read_file");
        return -1;
    }
    auto ret = init(model_buffer, len);
    delete[] model_buffer;
    return ret;
}

int ax_runner_ax650::init(char *model_buffer, size_t model_size)
{
    if (!m_handle)
    {
        m_handle = new ax_runner_ax650_handle_t;
    }
    memset(m_handle, 0, sizeof(ax_runner_ax650_handle_t));

    // static bool b_init = false;
    // if (!b_init)
    // {
    //     // 1. init engine
    //     // AX_ENGINE_NPU_ATTR_T npu_attr;
    //     // memset(&npu_attr, 0, sizeof(npu_attr));
    //     // npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    //     // AX_SYS_Init();

    //     axclrtDeviceList lst;
    //     if (const auto ret = axclrtGetDeviceList(&lst); 0 != ret || 0 == lst.num)
    //     {
    //         ALOGE("Get AXCL device failed{0x%8x}, find total %d device.\n", ret, lst.num);
    //         return -1;
    //     }
    //     if (const auto ret = axclrtSetDevice(lst.devices[0]); 0 != ret)
    //     {
    //         ALOGE("Set AXCL device failed{0x%8x}.\n", ret);
    //         return -1;
    //     }

    //     int ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
    //     if (0 != ret)
    //     {
    //         ALOGE("axclrtEngineInit %d\n", ret);
    //         return ret;
    //     }
    //     b_init = true;
    // }

    // 3. create handle
    void *devMem = nullptr;
    axcl_Malloc(&devMem, model_size, AXCL_MEM_MALLOC_NORMAL_ONLY, dev_id);

    // 4. copy model to device
    axcl_Memcpy(devMem, model_buffer, model_size, AXCL_MEMCPY_HOST_TO_DEVICE, dev_id);

    int ret = axcl_EngineLoadFromMem(devMem, model_size, &m_handle->handle, dev_id);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    axcl_Free(devMem, dev_id);
    // fprintf(stdout, "Engine creating handle is done.\n");

    return sub_init();
}

void ax_runner_ax650::release()
{
    if (m_handle && m_handle->handle)
    {
        for (int grpid = 0; grpid < group_count; grpid++)
        {
            free_io(&m_handle->io_datas[grpid], dev_id);
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

    // AX_ENGINE_Deinit();
}

void ax_runner_ax650::deinit()
{
    if (m_handle && m_handle->handle)
    {
        // free_io(&m_handle->io_data);
        // mtensors.clear();
        // minput_tensors.clear();
        // map_input_tensors.clear();
        // map_tensors.clear();
        // AX_ENGINE_DestroyHandle(m_handle->handle);
        // axclrtEngineDestroyIO(m_handle->io);
        axcl_EngineUnload(m_handle->handle, dev_id);
        m_handle->handle = 0;
        // delete m_handle;
        // m_handle = nullptr;
    }

    // AX_ENGINE_Deinit();
}

void ax_runner_ax650::print_info(int grpid)
{
    print_io_info(mgroup_input_tensors[grpid], mgroup_output_tensors[grpid]);
}

int ax_runner_ax650::get_algo_width() { return -1; }
int ax_runner_ax650::get_algo_height() { return -1; }

int ax_runner_ax650::set_input(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    return axcl_EngineSetInputBufferByIndex(m_handle->ios[grpid], idx, (void *)phy_addr, size, dev_id);
}
int ax_runner_ax650::set_output(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    return axcl_EngineSetOutputBufferByIndex(m_handle->ios[grpid], idx, (void *)phy_addr, size, dev_id);
}

int ax_runner_ax650::set_input(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    return axcl_EngineSetInputBufferByIndex(m_handle->ios[grpid], get_input(grpid, name).nIdx, (void *)phy_addr, size, dev_id);
}

int ax_runner_ax650::set_output(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
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