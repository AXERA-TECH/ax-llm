// AXCL metadata and IO layout probe for single-axmodel debugging.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <axcl.h>

static const char *dtype_name(axclrtEngineDataType dtype) {
    switch (dtype) {
    case AXCL_DATA_TYPE_NONE: return "none";
    case AXCL_DATA_TYPE_INT4: return "int4";
    case AXCL_DATA_TYPE_UINT4: return "uint4";
    case AXCL_DATA_TYPE_INT8: return "int8";
    case AXCL_DATA_TYPE_UINT8: return "uint8";
    case AXCL_DATA_TYPE_INT16: return "int16";
    case AXCL_DATA_TYPE_UINT16: return "uint16";
    case AXCL_DATA_TYPE_INT32: return "int32";
    case AXCL_DATA_TYPE_UINT32: return "uint32";
    case AXCL_DATA_TYPE_INT64: return "int64";
    case AXCL_DATA_TYPE_UINT64: return "uint64";
    case AXCL_DATA_TYPE_FP4: return "fp4";
    case AXCL_DATA_TYPE_FP8: return "fp8";
    case AXCL_DATA_TYPE_FP16: return "fp16";
    case AXCL_DATA_TYPE_BF16: return "bf16";
    case AXCL_DATA_TYPE_FP32: return "fp32";
    case AXCL_DATA_TYPE_FP64: return "fp64";
    default: return "unknown";
    }
}

static const char *layout_name(axclrtEngineDataLayout layout) {
    if (layout == AXCL_DATA_LAYOUT_NCHW) return "nchw";
    if (layout == AXCL_DATA_LAYOUT_NHWC) return "nhwc/none";
    return "unknown";
}

static void print_dims(const axclrtEngineIODims &dims) {
    std::cout << "[";
    for (int i = 0; i < dims.dimCount; ++i) {
        if (i) std::cout << " x ";
        std::cout << dims.dims[i];
    }
    std::cout << "]";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: axcl_io_probe <axmodel_path> [device_id]\n";
        return 1;
    }

    const char *model_path = argv[1];
    int device_id = argc >= 3 ? std::atoi(argv[2]) : 0;

    axclError ret = axclInit(nullptr);
    if (ret != 0) {
        std::cerr << "axclInit failed: " << ret << "\n";
        return 2;
    }

    int32_t rt_major = 0;
    int32_t rt_minor = 0;
    int32_t rt_patch = 0;
    if (axclrtGetVersion(&rt_major, &rt_minor, &rt_patch) == 0) {
        std::cout << "axclrt_version=" << rt_major << "." << rt_minor << "." << rt_patch << "\n";
    }
    const char *soc_name = axclrtGetSocName();
    if (soc_name) {
        std::cout << "axclrt_soc=" << soc_name << "\n";
    }

    axclrtDeviceList lst;
    ret = axclrtGetDeviceList(&lst);
    if (ret != 0 || lst.num == 0 || device_id < 0 || device_id >= (int)lst.num) {
        std::cerr << "axclrtGetDeviceList failed or invalid device id, ret=" << ret << " num=" << lst.num << "\n";
        axclFinalize();
        return 3;
    }

    ret = axclrtSetDevice(lst.devices[device_id]);
    if (ret != 0) {
        std::cerr << "axclrtSetDevice failed: " << ret << "\n";
        axclFinalize();
        return 4;
    }

    ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
    if (ret != 0) {
        std::cerr << "axclrtEngineInit failed: " << ret << "\n";
        axclFinalize();
        return 5;
    }

    uint64_t model_id = 0;
    ret = axclrtEngineLoadFromFile(model_path, &model_id);
    if (ret != 0) {
        std::cerr << "axclrtEngineLoadFromFile failed: " << ret << "\n";
        axclrtEngineFinalize();
        axclFinalize();
        return 6;
    }

    axclrtEngineIOInfo io_info = nullptr;
    ret = axclrtEngineGetIOInfo(model_id, &io_info);
    if (ret != 0) {
        std::cerr << "axclrtEngineGetIOInfo failed: " << ret << "\n";
        axclrtEngineUnload(model_id);
        axclrtEngineFinalize();
        axclFinalize();
        return 7;
    }

    int32_t groups = 0;
    ret = axclrtEngineGetShapeGroupsCount(io_info, &groups);
    if (ret != 0) {
        std::cerr << "axclrtEngineGetShapeGroupsCount failed: " << ret << "\n";
        axclrtEngineDestroyIOInfo(io_info);
        axclrtEngineUnload(model_id);
        axclrtEngineFinalize();
        axclFinalize();
        return 8;
    }

    const uint32_t input_num = axclrtEngineGetNumInputs(io_info);
    const uint32_t output_num = axclrtEngineGetNumOutputs(io_info);
    std::cout << "model=" << model_path << "\n";
    const char *compiler_version = axclrtEngineGetModelCompilerVersion(model_id);
    if (compiler_version) {
        std::cout << "model_compiler_version=" << compiler_version << "\n";
    }
    std::cout << "groups=" << groups << " inputs=" << input_num << " outputs=" << output_num << "\n";

    for (uint32_t i = 0; i < input_num; ++i) {
        axclrtEngineDataType dtype = AXCL_DATA_TYPE_NONE;
        axclrtEngineDataLayout layout = AXCL_DATA_LAYOUT_NONE;
        axclrtEngineIODims dims = {0};
        axclrtEngineGetInputDataType(io_info, i, &dtype);
        axclrtEngineGetInputDataLayout(io_info, i, &layout);
        std::cout << "input[" << i << "] name=" << axclrtEngineGetInputNameByIndex(io_info, i)
                  << " dtype=" << dtype_name(dtype)
                  << " layout=" << layout_name(layout);
        for (int32_t g = 0; g < groups; ++g) {
            std::memset(&dims, 0, sizeof(dims));
            ret = axclrtEngineGetInputDims(io_info, (uint32_t)g, i, &dims);
            std::cout << " g" << g << "=";
            if (ret == 0) {
                print_dims(dims);
                std::cout << ":" << axclrtEngineGetInputSizeByIndex(io_info, (uint32_t)g, i);
            } else {
                std::cout << "<err:" << ret << ">";
            }
        }
        std::cout << "\n";
    }

    for (uint32_t i = 0; i < output_num; ++i) {
        axclrtEngineDataType dtype = AXCL_DATA_TYPE_NONE;
        axclrtEngineDataLayout layout = AXCL_DATA_LAYOUT_NONE;
        axclrtEngineIODims dims = {0};
        axclrtEngineGetOutputDataType(io_info, i, &dtype);
        axclrtEngineGetOutputDataLayout(io_info, i, &layout);
        std::cout << "output[" << i << "] name=" << axclrtEngineGetOutputNameByIndex(io_info, i)
                  << " dtype=" << dtype_name(dtype)
                  << " layout=" << layout_name(layout);
        for (int32_t g = 0; g < groups; ++g) {
            std::memset(&dims, 0, sizeof(dims));
            ret = axclrtEngineGetOutputDims(io_info, (uint32_t)g, i, &dims);
            std::cout << " g" << g << "=";
            if (ret == 0) {
                print_dims(dims);
                std::cout << ":" << axclrtEngineGetOutputSizeByIndex(io_info, (uint32_t)g, i);
            } else {
                std::cout << "<err:" << ret << ">";
            }
        }
        std::cout << "\n";
    }

    axclrtEngineDestroyIOInfo(io_info);
    axclrtEngineUnload(model_id);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
