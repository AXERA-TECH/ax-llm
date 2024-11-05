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
#include <axcl.h>

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
    printf("\ninput size: %d\n", input.size());
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

    printf("\noutput size: %d\n", output.size());
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

typedef struct
{
    int nIndex;
    int nSize;
    void *pBuf;

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

static void free_io_index(AXCL_IO_BUF_T *pBuf, size_t index)
{
    for (size_t i = 0; i < index; ++i)
    {
        axclrtFree(pBuf[i].pBuf);
    }
}

static void free_io(AXCL_IO_DATA_T *io_data)
{
    for (size_t j = 0; j < io_data->nInputSize; ++j)
    {
        axclrtFree(io_data->pInputs[j].pBuf);
    }
    for (size_t j = 0; j < io_data->nOutputSize; ++j)
    {
        axclrtFree(io_data->pOutputs[j].pBuf);
    }
    delete[] io_data->pInputs;
    delete[] io_data->pOutputs;
}

static inline int prepare_io(uint64_t handle, uint64_t context, axclrtEngineIOInfo io_info, axclrtEngineIO io, AXCL_IO_DATA_T *io_data, INPUT_OUTPUT_ALLOC_STRATEGY strategy)
{
    memset(io_data, 0, sizeof(AXCL_IO_DATA_T));

    auto inputNum = axclrtEngineGetNumInputs(io_info);
    auto outputNum = axclrtEngineGetNumOutputs(io_info);
    io_data->nInputSize = inputNum;
    io_data->nOutputSize = outputNum;
    io_data->pInputs = new AXCL_IO_BUF_T[inputNum];
    io_data->pOutputs = new AXCL_IO_BUF_T[outputNum];

    // 1. alloc inputs
    for (int32_t i = 0; i < inputNum; i++)
    {
        auto bufSize = axclrtEngineGetInputSizeByIndex(io_info, 0, i);
        void *devPtr = nullptr;
        axclError ret = 0;
        if (AX_ENGINE_ABST_DEFAULT == strategy.first)
        {
            ret = axclrtMalloc(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
        }
        else
        {
            ret = axclrtMallocCached(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
        }

        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i);
            fprintf(stderr, "Malloc input(index: %d, size: %d) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
        std::vector<char> tmp(bufSize, 0);
        axclrtMemcpy(devPtr, tmp.data(), bufSize, axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE);
        // axclrtMemset(devPtr, 0, bufSize);

        axclrtEngineIODims dims;
        ret = axclrtEngineGetInputDims(io_info, 0, i, &dims);
        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i);
            fprintf(stderr, "Get input dims(index: %d) failed! ret=0x%x\n", i, ret);
            return -1;
        }

        io_data->pInputs[i].nIndex = i;
        io_data->pInputs[i].nSize = bufSize;
        io_data->pInputs[i].pBuf = devPtr;
        io_data->pInputs[i].dims = dims;
        io_data->pInputs[i].Name = axclrtEngineGetInputNameByIndex(io_info, i);
        ret = axclrtEngineSetInputBufferByIndex(io, i, devPtr, bufSize);
        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i);
            fprintf(stderr, "Set input buffer(index: %d, size: %lu) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
    }

    // 2. alloc outputs
    for (int32_t i = 0; i < outputNum; i++)
    {
        auto bufSize = axclrtEngineGetOutputSizeByIndex(io_info, 0, i);
        void *devPtr = NULL;
        axclError ret = 0;
        if (AX_ENGINE_ABST_DEFAULT == strategy.first)
        {
            ret = axclrtMalloc(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
        }
        else
        {
            ret = axclrtMallocCached(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
        }

        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i);
            fprintf(stderr, "Malloc output(index: %d, size: %d) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
        std::vector<char> tmp(bufSize, 0);
        axclrtMemcpy(devPtr, tmp.data(), bufSize, axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE);
        // axclrtMemset(devPtr, 0, bufSize);
        axclrtEngineIODims dims;
        ret = axclrtEngineGetOutputDims(io_info, 0, i, &dims);
        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i);
            fprintf(stderr, "Get output dims(index: %d) failed! ret=0x%x\n", i, ret);
            return -1;
        }

        io_data->pOutputs[i].nIndex = i;
        io_data->pOutputs[i].nSize = bufSize;
        io_data->pOutputs[i].pBuf = devPtr;
        io_data->pOutputs[i].dims = dims;
        io_data->pOutputs[i].Name = axclrtEngineGetOutputNameByIndex(io_info, i);
        ret = axclrtEngineSetOutputBufferByIndex(io, i, devPtr, bufSize);
        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i);
            fprintf(stderr, "Set output buffer(index: %d, size: %lu) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
    }

    return 0;
}

struct ax_joint_runner_ax650_handle_t
{
    uint64_t handle = 0;
    uint64_t context = 0;
    axclrtEngineIOInfo io_info = 0;
    axclrtEngineIO io = 0;
    AXCL_IO_DATA_T io_data = {0};

    // int algo_width, algo_height;
    // int algo_colorformat;
};

int ax_runner_ax650::sub_init()
{
    // 4. create context
    int ret = axclrtEngineCreateContext(m_handle->handle, &m_handle->context);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateContext");
        return ret;
    }
    // fprintf(stdout, "Engine creating context is done.\n");

    // 5. set io

    ret = axclrtEngineGetIOInfo(m_handle->handle, &m_handle->io_info);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_GetIOInfo");
        return ret;
    }
    // fprintf(stdout, "Engine get io info is done. \n");

    // 4. create io
    ret = axclrtEngineCreateIO(m_handle->io_info, &m_handle->io);
    if (ret != 0)
    {
        axclrtEngineUnload(m_handle->handle);
        fprintf(stderr, "Create io failed. ret=0x%x\n", ret);
        return -1;
    }

    // fprintf(stdout, "Engine creating io is done. \n");

    // 6. alloc io
    if (!_parepare_io)
    {
        auto malloc_strategy = std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_DEFAULT);
        ret = prepare_io(m_handle->handle, m_handle->context, m_handle->io_info, m_handle->io, &m_handle->io_data, malloc_strategy);
        if (ret != 0)
        {
            free_io(&m_handle->io_data);
            axclrtEngineDestroyIO(m_handle->io);
            axclrtEngineUnload(m_handle->handle);

            fprintf(stderr, "prepare_io failed.\n");
            return ret;
        }

        for (size_t i = 0; i < m_handle->io_data.nOutputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = m_handle->io_data.pOutputs[i].Name;
            tensor.nSize = m_handle->io_data.pOutputs[i].nSize;
            for (size_t j = 0; j < m_handle->io_data.pOutputs[i].dims.dimCount; j++)
            {
                tensor.vShape.push_back(m_handle->io_data.pOutputs[i].dims.dims[j]);
            }
            tensor.phyAddr = (unsigned long long)m_handle->io_data.pOutputs[i].pBuf;
            tensor.pVirAddr = malloc(tensor.nSize);
            memset(tensor.pVirAddr, 0, tensor.nSize);
            moutput_tensors.push_back(tensor);
        }

        for (size_t i = 0; i < m_handle->io_data.nInputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = m_handle->io_data.pInputs[i].Name;
            tensor.nSize = m_handle->io_data.pInputs[i].nSize;
            for (size_t j = 0; j < m_handle->io_data.pInputs[i].dims.dimCount; j++)
            {
                tensor.vShape.push_back(m_handle->io_data.pInputs[i].dims.dims[j]);
            }
            tensor.phyAddr = (unsigned long long)m_handle->io_data.pInputs[i].pBuf;
            // tensor.pVirAddr = m_handle->io_data.pInputs[i].pVirAddr;
            tensor.pVirAddr = malloc(tensor.nSize);
            memset(tensor.pVirAddr, 0, tensor.nSize);
            minput_tensors.push_back(tensor);
        }
        _parepare_io = true;
    }
    else
    {
    }
    // print_io_info(minput_tensors, mtensors);

    return ret;
}

int ax_runner_ax650::init(const char *model_file, bool use_mmap)
{
    if (use_mmap)
    {
        MMap model_buffer(model_file);
        if (!model_buffer.data())
        {
            ALOGE("mmap");
            return -1;
        }
        auto ret = init((char *)model_buffer.data(), model_buffer.size());
        model_buffer.close_file();
        return ret;
    }
    else
    {
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
}

int ax_runner_ax650::init(char *model_buffer, size_t model_size)
{
    if (!m_handle)
    {
        m_handle = new ax_joint_runner_ax650_handle_t;
    }
    memset(m_handle, 0, sizeof(ax_joint_runner_ax650_handle_t));

    static bool b_init = false;
    if (!b_init)
    {
        // 1. init engine
        // AX_ENGINE_NPU_ATTR_T npu_attr;
        // memset(&npu_attr, 0, sizeof(npu_attr));
        // npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        // AX_SYS_Init();
        auto ret = axclInit(nullptr);
        if (0 != ret)
        {
            return ret;
        }

        axclrtDeviceList lst;
        if (const auto ret = axclrtGetDeviceList(&lst); 0 != ret || 0 == lst.num)
        {
            ALOGE("Get AXCL device failed{0x%8x}, find total %d device.\n", ret, lst.num);
            return -1;
        }
        if (const auto ret = axclrtSetDevice(lst.devices[0]); 0 != ret)
        {
            ALOGE("Set AXCL device failed{0x%8x}.\n", ret);
            return -1;
        }

        ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
        if (0 != ret)
        {
            ALOGE("axclrtEngineInit %d\n", ret);
            return ret;
        }
        b_init = true;
    }

    // 3. create handle
    void *devMem = nullptr;
    axclrtMalloc(&devMem, model_size, AXCL_MEM_MALLOC_NORMAL_ONLY);

    // 4. copy model to device
    axclrtMemcpy(devMem, model_buffer, model_size, AXCL_MEMCPY_HOST_TO_DEVICE);

    int ret = axclrtEngineLoadFromMem(devMem, model_size, &m_handle->handle);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    axclrtFree(devMem);
    // fprintf(stdout, "Engine creating handle is done.\n");

    return sub_init();
}

void ax_runner_ax650::release()
{
    if (m_handle && m_handle->handle)
    {
        free_io(&m_handle->io_data);
        axclrtEngineDestroyIO(m_handle->io);
        axclrtEngineUnload(m_handle->handle);
        m_handle->handle = 0;
    }

    if (m_handle)
    {
        delete m_handle;
        m_handle = nullptr;
    }

    moutput_tensors.clear();
    minput_tensors.clear();
    map_input_tensors.clear();
    map_output_tensors.clear();

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
        axclrtEngineDestroyIO(m_handle->io);
        axclrtEngineUnload(m_handle->handle);
        m_handle->handle = 0;
        // delete m_handle;
        // m_handle = nullptr;
    }

    // AX_ENGINE_Deinit();
}

int ax_runner_ax650::get_algo_width() { return -1; }
int ax_runner_ax650::get_algo_height() { return -1; }

int ax_runner_ax650::set_input(int idx, unsigned long long int phy_addr, unsigned long size)
{
    return axclrtEngineSetInputBufferByIndex(m_handle->io, idx, (void *)phy_addr, size);
}
int ax_runner_ax650::set_output(int idx, unsigned long long int phy_addr, unsigned long size)
{
    return axclrtEngineSetOutputBufferByIndex(m_handle->io, idx, (void *)phy_addr, size);
}

ax_color_space_e ax_runner_ax650::get_color_space()
{
    // switch (m_handle->algo_colorformat)
    // {
    // case AX_FORMAT_RGB888:
    //     return ax_color_space_e::axdl_color_space_rgb;
    // case AX_FORMAT_BGR888:
    //     return ax_color_space_e::axdl_color_space_bgr;
    // case AX_FORMAT_YUV420_SEMIPLANAR:
    //     return ax_color_space_e::axdl_color_space_nv12;
    // default:
    //     return axdl_color_space_unknown;
    // }
    return axdl_color_space_unknown;
}

int ax_runner_ax650::inference(ax_image_t *pstFrame)
{
    // unsigned char *dst = (unsigned char *)minput_tensors[0].pVirAddr;
    // unsigned char *src = (unsigned char *)pstFrame->pVir;

    // switch (m_handle->algo_colorformat)
    // {
    // case AX_FORMAT_RGB888:
    // case AX_FORMAT_BGR888:
    //     for (size_t i = 0; i < pstFrame->nHeight; i++)
    //     {
    //         memcpy(dst + i * pstFrame->nWidth * 3, src + i * pstFrame->tStride_W * 3, pstFrame->nWidth * 3);
    //     }
    //     break;
    // case AX_FORMAT_YUV420_SEMIPLANAR:
    // case AX_FORMAT_YUV420_SEMIPLANAR_VU:
    //     for (size_t i = 0; i < pstFrame->nHeight * 1.5; i++)
    //     {
    //         memcpy(dst + i * pstFrame->nWidth, src + i * pstFrame->tStride_W, pstFrame->nWidth);
    //     }
    //     break;
    // default:
    //     break;
    // }

    // memcpy(minput_tensors[0].pVirAddr, pstFrame->pVir, minput_tensors[0].nSize);
    return inference();
}
int ax_runner_ax650::inference()
{
    // for (size_t i = 0; i < minput_tensors.size(); i++)
    // {
    //     axclrtMemcpy(
    //         (void *)minput_tensors[i].phyAddr, minput_tensors[i].nSize,
    //         minput_tensors[i].pVirAddr, minput_tensors[i].nSize, AXCL_MEMCPY_HOST_TO_DEVICE);
    // }
    auto ret = axclrtEngineExecute(m_handle->handle, m_handle->context, 0, m_handle->io);
    if (ret != 0)
    {
        ALOGE("AX_ENGINE_Execute");
        return ret;
    }

    // for (size_t i = 0; i < mtensors.size(); i++)
    // {
    //     axclrtMemcpy(
    //         mtensors[i].pVirAddr, mtensors[i].nSize,
    //         (void *)mtensors[i].phyAddr, mtensors[i].nSize, AXCL_MEMCPY_DEVICE_TO_HOST);
    // }
    return 0;
}

// int ax_cmmcpy(unsigned long long int dst, unsigned long long int src, int size)
// {
//     return AX_IVPS_CmmCopyTdp(dst, src, size);
// }