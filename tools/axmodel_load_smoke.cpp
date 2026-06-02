#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

static void trace(const std::string& msg)
{
    const std::string line = "[axmodel_load_smoke] " + msg + "\n";
    ::write(2, line.data(), line.size());
}

static bool read_file(const char* path, std::vector<char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg(0, std::ios::end);
    const auto n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0) return false;
    out.resize((size_t)n);
    f.read(out.data(), n);
    return (bool)f;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "Usage: axmodel_load_smoke <model.axmodel> [model2.axmodel ...]\n");
        return 1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    std::memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;

    trace("AX_SYS_Init begin");
    AX_SYS_Init();
    trace("AX_SYS_Init done");

    trace("AX_ENGINE_Init begin");
    int ret = AX_ENGINE_Init(&npu_attr);
    trace("AX_ENGINE_Init ret=" + std::to_string(ret));
    if (ret != 0) return ret;

    std::vector<AX_ENGINE_HANDLE> handles;
    for (int argi = 1; argi < argc; ++argi) {
        const char* model_path = argv[argi];
        trace("load_index=" + std::to_string(argi - 1) + " model=" + model_path);

        std::vector<char> model;
        if (!read_file(model_path, model)) {
            std::fprintf(stderr, "read model failed: %s\n", model_path);
            ret = 2;
            break;
        }
        trace("model_size=" + std::to_string(model.size()));

        AX_ENGINE_HANDLE handle = nullptr;
        const char* use_v2 = std::getenv("AX_RUNNER_USE_CREATE_HANDLE_V2");
        if (use_v2 && std::string(use_v2) == "1") {
            AX_ENGINE_HANDLE_EXTRA_T extra;
            std::memset(&extra, 0, sizeof(extra));
            extra.pName = (AX_S8*)"axmodel_load_smoke";
            trace("AX_ENGINE_CreateHandleV2 begin index=" + std::to_string(argi - 1));
            ret = AX_ENGINE_CreateHandleV2(&handle, model.data(), (AX_U32)model.size(), &extra);
            trace("AX_ENGINE_CreateHandleV2 ret=" + std::to_string(ret));
        } else {
            trace("AX_ENGINE_CreateHandle begin index=" + std::to_string(argi - 1));
            ret = AX_ENGINE_CreateHandle(&handle, model.data(), (AX_U32)model.size());
            trace("AX_ENGINE_CreateHandle ret=" + std::to_string(ret));
        }
        if (ret != 0) break;
        handles.push_back(handle);

        AX_ENGINE_CONTEXT_T context = 0;
        trace("AX_ENGINE_CreateContextV2 begin index=" + std::to_string(argi - 1));
        ret = AX_ENGINE_CreateContextV2(handle, &context);
        trace("AX_ENGINE_CreateContextV2 ret=" + std::to_string(ret));
        if (ret != 0) {
            trace("AX_ENGINE_CreateContext fallback begin index=" + std::to_string(argi - 1));
            ret = AX_ENGINE_CreateContext(handle);
            trace("AX_ENGINE_CreateContext fallback ret=" + std::to_string(ret));
        }

        AX_U32 group_count = 0;
        trace("AX_ENGINE_GetGroupIOInfoCount begin index=" + std::to_string(argi - 1));
        int io_ret = AX_ENGINE_GetGroupIOInfoCount(handle, &group_count);
        trace("AX_ENGINE_GetGroupIOInfoCount ret=" + std::to_string(io_ret) +
              " count=" + std::to_string(group_count));
        if (io_ret == 0) {
            for (AX_U32 gid = 0; gid < group_count; ++gid) {
                AX_ENGINE_IO_INFO_T* info = nullptr;
                int gr = group_count == 1 ? AX_ENGINE_GetIOInfo(handle, &info)
                                          : AX_ENGINE_GetGroupIOInfo(handle, gid, &info);
                trace("group " + std::to_string(gid) + " io_ret=" + std::to_string(gr) +
                      (info ? (" inputs=" + std::to_string(info->nInputSize) +
                               " outputs=" + std::to_string(info->nOutputSize)) : ""));
            }
        }
        trace("load_index=" + std::to_string(argi - 1) + " done");
    }

    for (auto it = handles.rbegin(); it != handles.rend(); ++it) {
        trace("AX_ENGINE_DestroyHandle begin");
        AX_ENGINE_DestroyHandle(*it);
        trace("AX_ENGINE_DestroyHandle done");
    }
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    trace("done");
    return ret == 0 ? 0 : ret;
}
