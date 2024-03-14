#include "ax_model_runner_ax650.hpp"
#include "string.h"
#include "fstream"
#include "memory"
// #include "utilities/file.hpp"
#include <ax_sys_api.h>
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

typedef std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T> INPUT_OUTPUT_ALLOC_STRATEGY;

static void print_io_info(AX_ENGINE_IO_INFO_T *io_info)
{
    static std::map<AX_ENGINE_DATA_TYPE_T, const char *> data_type = {
        {AX_ENGINE_DT_UNKNOWN, "UNKNOWN"},
        {AX_ENGINE_DT_UINT8, "UINT8"},
        {AX_ENGINE_DT_UINT16, "UINT16"},
        {AX_ENGINE_DT_FLOAT32, "FLOAT32"},
        {AX_ENGINE_DT_SINT16, "SINT16"},
        {AX_ENGINE_DT_SINT8, "SINT8"},
        {AX_ENGINE_DT_SINT32, "SINT32"},
        {AX_ENGINE_DT_UINT32, "UINT32"},
        {AX_ENGINE_DT_FLOAT64, "FLOAT64"},
        {AX_ENGINE_DT_UINT10_PACKED, "UINT10_PACKED"},
        {AX_ENGINE_DT_UINT12_PACKED, "UINT12_PACKED"},
        {AX_ENGINE_DT_UINT14_PACKED, "UINT14_PACKED"},
        {AX_ENGINE_DT_UINT16_PACKED, "UINT16_PACKED"},
    };

    static std::map<AX_ENGINE_COLOR_SPACE_T, const char *> color_type = {
        {AX_ENGINE_CS_FEATUREMAP, "FEATUREMAP"},
        {AX_ENGINE_CS_RAW8, "RAW8"},
        {AX_ENGINE_CS_RAW10, "RAW10"},
        {AX_ENGINE_CS_RAW12, "RAW12"},
        {AX_ENGINE_CS_RAW14, "RAW14"},
        {AX_ENGINE_CS_RAW16, "RAW16"},
        {AX_ENGINE_CS_NV12, "NV12"},
        {AX_ENGINE_CS_NV21, "NV21"},
        {AX_ENGINE_CS_RGB, "RGB"},
        {AX_ENGINE_CS_BGR, "BGR"},
        {AX_ENGINE_CS_RGBA, "RGBA"},
        {AX_ENGINE_CS_GRAY, "GRAY"},
        {AX_ENGINE_CS_YUV444, "YUV444"},
    };
    printf("\ninput size: %d\n", io_info->nInputSize);
    for (uint32_t i = 0; i < io_info->nInputSize; ++i)
    {
        // print shape info,like [batchsize x channel x height x width]
        auto &info = io_info->pInputs[i];
        printf("    name: \e[1;32m%8s", info.pName);

        std::string dt = "unknown";
        if (data_type.find(info.eDataType) != data_type.end())
        {
            dt = data_type[info.eDataType];
            printf(" \e[1;34m[%s] ", dt.c_str());
        }
        else
        {
            printf(" \e[1;31m[%s] ", dt.c_str());
        }

        std::string ct = "unknown";
        if (info.pExtraMeta && color_type.find(info.pExtraMeta->eColorSpace) != color_type.end())
        {
            ct = color_type[info.pExtraMeta->eColorSpace];
            printf("\e[1;34m[%s]", ct.c_str());
        }
        else
        {
            printf("\e[1;31m[%s]", ct.c_str());
        }
        printf(" \n        \e[1;31m");

        for (AX_U8 s = 0; s < info.nShapeSize; s++)
        {
            printf("%d", info.pShape[s]);
            if (s != info.nShapeSize - 1)
            {
                printf(" x ");
            }
        }
        printf("\e[0m\n\n");
    }

    printf("\noutput size: %d\n", io_info->nOutputSize);
    for (uint32_t i = 0; i < io_info->nOutputSize; ++i)
    {
        // print shape info,like [batchsize x channel x height x width]
        auto &info = io_info->pOutputs[i];
        printf("    name: \e[1;32m%8s \e[1;34m[%s]\e[0m\n        \e[1;31m", info.pName, data_type[info.eDataType]);
        for (AX_U8 s = 0; s < info.nShapeSize; s++)
        {
            printf("%d", info.pShape[s]);
            if (s != info.nShapeSize - 1)
            {
                printf(" x ");
            }
        }
        printf("\e[0m\n\n");
    }
}

void free_io_index(AX_ENGINE_IO_BUFFER_T *io_buf, int index)
{
    for (int i = 0; i < index; ++i)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io_buf + i;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
}

void free_io(AX_ENGINE_IO_T *io)
{
    for (size_t j = 0; j < io->nInputSize; ++j)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io->pInputs + j;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    for (size_t j = 0; j < io->nOutputSize; ++j)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io->pOutputs + j;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    delete[] io->pInputs;
    delete[] io->pOutputs;
}

static inline int prepare_io(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data, INPUT_OUTPUT_ALLOC_STRATEGY strategy)
{
    memset(io_data, 0, sizeof(*io_data));
    io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    io_data->nInputSize = info->nInputSize;

    auto ret = 0;
    for (uint i = 0; i < info->nInputSize; ++i)
    {
        auto meta = info->pInputs[i];
        auto buffer = &io_data->pInputs[i];
        if (strategy.first == AX_ENGINE_ABST_CACHED)
        {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        else
        {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }

        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i);
            fprintf(stderr, "Allocate input{%d} { phy: %p, vir: %p, size: %lu Bytes }. fail \n", i, (void *)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
            return ret;
        }
        memset(buffer->pVirAddr, 0, meta.nSize);
        // fprintf(stderr, "Allocate input{%d} { phy: %p, vir: %p, size: %lu Bytes }. \n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
    }

    io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    io_data->nOutputSize = info->nOutputSize;
    for (uint i = 0; i < info->nOutputSize; ++i)
    {
        auto meta = info->pOutputs[i];
        auto buffer = &io_data->pOutputs[i];
        buffer->nSize = meta.nSize;
        if (strategy.second == AX_ENGINE_ABST_CACHED)
        {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        else
        {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        if (ret != 0)
        {
            fprintf(stderr, "Allocate output{%d} { phy: %p, vir: %p, size: %lu Bytes }. fail \n", i, (void *)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
            free_io_index(io_data->pInputs, io_data->nInputSize);
            free_io_index(io_data->pOutputs, i);
            return ret;
        }
        memset(buffer->pVirAddr, 0, meta.nSize);
        // fprintf(stderr, "Allocate output{%d} { phy: %p, vir: %p, size: %lu Bytes }.\n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
    }

    return 0;
}

struct ax_joint_runner_ax650_handle_t
{
    AX_ENGINE_HANDLE handle;
    AX_ENGINE_IO_INFO_T *io_info;
    AX_ENGINE_IO_T io_data;

    // int algo_width, algo_height;
    // int algo_colorformat;
};

int ax_runner_ax650::sub_init()
{
    // 4. create context
    int ret = AX_ENGINE_CreateContext(m_handle->handle);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateContext");
        return ret;
    }
    // fprintf(stdout, "Engine creating context is done.\n");

    // 5. set io

    ret = AX_ENGINE_GetIOInfo(m_handle->handle, &m_handle->io_info);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_GetIOInfo");
        return ret;
    }
    // fprintf(stdout, "Engine get io info is done. \n");

    // 6. alloc io
    if (!_parepare_io)
    {
        ret = prepare_io(m_handle->io_info, &m_handle->io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_DEFAULT));
        if (0 != ret)
        {
            ALOGE("prepare_io");
            return ret;
        }

        for (size_t i = 0; i < m_handle->io_info->nOutputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = std::string(m_handle->io_info->pOutputs[i].pName);
            tensor.nSize = m_handle->io_info->pOutputs[i].nSize;
            for (size_t j = 0; j < m_handle->io_info->pOutputs[i].nShapeSize; j++)
            {
                tensor.vShape.push_back(m_handle->io_info->pOutputs[i].pShape[j]);
            }
            tensor.phyAddr = m_handle->io_data.pOutputs[i].phyAddr;
            tensor.pVirAddr = m_handle->io_data.pOutputs[i].pVirAddr;
            mtensors.push_back(tensor);
        }

        for (size_t i = 0; i < m_handle->io_info->nInputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = std::string(m_handle->io_info->pInputs[i].pName);
            tensor.nSize = m_handle->io_info->pInputs[i].nSize;
            for (size_t j = 0; j < m_handle->io_info->pInputs[i].nShapeSize; j++)
            {
                tensor.vShape.push_back(m_handle->io_info->pInputs[i].pShape[j]);
            }
            tensor.phyAddr = m_handle->io_data.pInputs[i].phyAddr;
            tensor.pVirAddr = m_handle->io_data.pInputs[i].pVirAddr;
            minput_tensors.push_back(tensor);
        }
        _parepare_io = true;
    }
    else
    {
    }

    return ret;
}

int ax_runner_ax650::init(const char *model_file)
{
    // 2. load model
    std::shared_ptr<MMap> model_buffer(new MMap(model_file));
    if (!model_buffer->data())
    {
        ALOGE("mmap");
        return -1;
    }
    return init(*model_buffer.get());
    // std::shared_ptr<std::vector<char>> model_buffer((new std::vector<char>()));
    // if (!read_file(model_file, *model_buffer.get()))
    // {
    //     ALOGE("read_file");
    //     return -1;
    // }

    // 3. create handle
}

int ax_runner_ax650::init(MMap &model_buffer)
{
    if (!m_handle)
    {
        m_handle = new ax_joint_runner_ax650_handle_t;
    }

    static bool b_init = false;
    if (!b_init)
    {
        // 1. init engine
        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        AX_SYS_Init();
        auto ret = AX_ENGINE_Init(&npu_attr);
        if (0 != ret)
        {
            return ret;
        }
        b_init = true;
    }

    // 3. create handle

    int ret = AX_ENGINE_CreateHandle(&m_handle->handle, model_buffer.data(), model_buffer.size());
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    // fprintf(stdout, "Engine creating handle is done.\n");

    return sub_init();
}

int ax_runner_ax650::init(std::vector<char> &model_buffer)
{
    if (!m_handle)
    {
        m_handle = new ax_joint_runner_ax650_handle_t;
    }

    static bool b_init = false;
    if (!b_init)
    {
        // 1. init engine
        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        AX_SYS_Init();
        auto ret = AX_ENGINE_Init(&npu_attr);
        if (0 != ret)
        {
            return ret;
        }
        b_init = true;
    }

    // 3. create handle

    int ret = AX_ENGINE_CreateHandle(&m_handle->handle, model_buffer.data(), model_buffer.size());
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    // fprintf(stdout, "Engine creating handle is done.\n");

    return sub_init();
}

void ax_runner_ax650::release()
{
    if (m_handle && m_handle->handle)
    {
        free_io(&m_handle->io_data);
        AX_ENGINE_DestroyHandle(m_handle->handle);
        m_handle->handle = nullptr;
    }

    if (m_handle)
    {
        delete m_handle;
        m_handle = nullptr;
    }

    mtensors.clear();
    minput_tensors.clear();
    map_input_tensors.clear();
    map_tensors.clear();

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
        AX_ENGINE_DestroyHandle(m_handle->handle);
        m_handle->handle = nullptr;
        // delete m_handle;
        // m_handle = nullptr;
    }

    // AX_ENGINE_Deinit();
}

int ax_runner_ax650::get_algo_width() { return -1; }
int ax_runner_ax650::get_algo_height() { return -1; }
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
    return AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data);
}