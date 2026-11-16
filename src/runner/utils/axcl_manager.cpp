#include "axcl_manager.h"
#include "axcl_manager.hpp"
#include "ax_cmm_utils.hpp"

#include <map>

#include "axcl_sys_type.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static std::map<int, std::shared_ptr<AXCLWorker>> g_devices;

static bool axcl_contains(int devid)
{
    return g_devices.find(devid) != g_devices.end();
}

axclError axcl_Init(int devid)
{
    if (axcl_contains(devid))
    {
        ALOGI("AXCL device %d already inited\n", devid);
        return 0;
    }
    g_devices[devid] = std::make_shared<AXCLWorker>();
    if (!g_devices[devid]->Run(devid))
    {
        return -1;
    }
    return 0;
}

axclError axcl_Exit(int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
        return -1;
    }
    g_devices[devid]->Stop();
    g_devices.erase(devid);
    return 0;
}

namespace
{
// AXCL SYS entry points, looked up at run time so the build does not need to link libaxcl_sys.
struct AxclSysApi
{
    using VoidFn = AX_S32 (*)();
    using QueryFn = AX_S32 (*)(AX_CMM_STATUS_T *);
    VoidFn init = nullptr;
    VoidFn deinit = nullptr;
    QueryFn query = nullptr;

    AxclSysApi()
    {
#ifdef _WIN32
        HMODULE lib = LoadLibraryA("libaxcl_sys.dll");
        auto sym = [lib](const char *name) { return lib ? (void *)GetProcAddress(lib, name) : nullptr; };
#else
        void *lib = dlopen("libaxcl_sys.so", RTLD_NOW | RTLD_LOCAL);
        auto sym = [lib](const char *name) { return lib ? dlsym(lib, name) : nullptr; };
#endif
        init = reinterpret_cast<VoidFn>(sym("AXCL_SYS_Init"));
        deinit = reinterpret_cast<VoidFn>(sym("AXCL_SYS_Deinit"));
        query = reinterpret_cast<QueryFn>(sym("AXCL_SYS_MemQueryStatus"));
    }
};

const AxclSysApi &axcl_sys_api()
{
    static const AxclSysApi api;
    return api;
}
} // namespace

int axcl_sys_cmm_remain_mb(bool *sys_inited)
{
    const AxclSysApi &api = axcl_sys_api();
    if (!api.query) return -1;
    AX_CMM_STATUS_T status{};
    AX_S32 ret = api.query(&status);
    if (ret != 0 && api.init && !*sys_inited && api.init() == 0)
    {
        *sys_inited = true;
        status = AX_CMM_STATUS_T{};
        ret = api.query(&status);
    }
    if (ret != 0) return -1;
    return (int)(status.RemainSize / 1024); // KB
}

void axcl_sys_deinit()
{
    if (axcl_sys_api().deinit) axcl_sys_api().deinit();
}

int axcl_GetCMMRemain(int devid)
{
    // AXCL_SYS_MemQueryStatus on the device's worker thread. The fallback runs
    // `axcl-smi sh cat /proc/ax_proc/mem_cmm_info`, ~0.5 s per call (the loader asks twice per
    // layer), and that call has been seen to hang on the first use after a killed process.
    if (axcl_contains(devid))
    {
        const int mb = g_devices[devid]->cmmRemainMB();
        if (mb >= 0) return mb;
    }
    return get_pcie_remaining_cmm_size(devid);
}

axclError axcl_Malloc(void **devPtr, size_t size, axclrtMemMallocPolicy policy, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclMalloc(devPtr, size, policy);
}

axclError axcl_MallocCached(void **devPtr, size_t size, axclrtMemMallocPolicy policy, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclMallocCached(devPtr, size, policy);
}
axclError axcl_Free(void *devPtr, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclFree(devPtr);
}
axclError axcl_MemFlush(void *devPtr, size_t size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclMemFlush(devPtr, size);
}
axclError axcl_MemInvalidate(void *devPtr, size_t size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclMemInvalidate(devPtr, size);
}
axclError axcl_MallocHost(void **hostPtr, size_t size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclMallocHost(hostPtr, size);
}
axclError axcl_FreeHost(void *hostPtr, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclFreeHost(hostPtr);
}
axclError axcl_Memset(void *devPtr, uint8_t value, size_t count, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclMemset(devPtr, value, count);
}
axclError axcl_Memcpy(void *dstPtr, const void *srcPtr, size_t count, axclrtMemcpyKind kind, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclMemcpy(dstPtr, srcPtr, count, kind);
}
axclError axcl_Memcmp(const void *devPtr1, const void *devPtr2, size_t count, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclMemcmp(devPtr1, devPtr2, count);
}

axclError axcl_EngineLoadFromFile(const char *modelPath, uint64_t *modelId, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineLoadFromFile(modelPath, modelId);
}
axclError axcl_EngineLoadFromMem(const void *model, uint64_t modelSize, uint64_t *modelId, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineLoadFromMem(model, modelSize, modelId);
}
axclError axcl_EngineUnload(uint64_t modelId, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineUnload(modelId);
}
const char *axcl_EngineGetModelCompilerVersion(uint64_t modelId, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetModelCompilerVersion(modelId);
}
axclError axcl_EngineSetAffinity(uint64_t modelId, axclrtEngineSet set, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineSetAffinity(modelId, set);
}
axclError axcl_EngineGetAffinity(uint64_t modelId, axclrtEngineSet *set, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetAffinity(modelId, set);
}
axclError axcl_EngineGetUsage(const char *modelPath, int64_t *sysSize, int64_t *cmmSize, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetUsage(modelPath, sysSize, cmmSize);
}
axclError axcl_EngineGetUsageFromMem(const void *model, uint64_t modelSize, int64_t *sysSize, int64_t *cmmSize, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetUsageFromMem(model, modelSize, sysSize, cmmSize);
}
axclError axcl_EngineGetUsageFromModelId(uint64_t modelId, int64_t *sysSize, int64_t *cmmSize, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetUsageFromModelId(modelId, sysSize, cmmSize);
}
axclError axcl_EngineGetModelType(const char *modelPath, axclrtEngineModelKind *modelType, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetModelType(modelPath, modelType);
}
axclError axcl_EngineGetModelTypeFromMem(const void *model, uint64_t modelSize, axclrtEngineModelKind *modelType, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetModelTypeFromMem(model, modelSize, modelType);
}
axclError axcl_EngineGetModelTypeFromModelId(uint64_t modelId, axclrtEngineModelKind *modelType, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetModelTypeFromModelId(modelId, modelType);
}
axclError axcl_EngineGetIOInfo(uint64_t modelId, axclrtEngineIOInfo *ioInfo, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetIOInfo(modelId, ioInfo);
}
axclError axcl_EngineDestroyIOInfo(axclrtEngineIOInfo ioInfo, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineDestroyIOInfo(ioInfo);
}
axclError axcl_EngineGetShapeGroupsCount(axclrtEngineIOInfo ioInfo, int32_t *count, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetShapeGroupsCount(ioInfo, count);
}
uint32_t axcl_EngineGetNumInputs(axclrtEngineIOInfo ioInfo, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetNumInputs(ioInfo);
}
uint32_t axcl_EngineGetNumOutputs(axclrtEngineIOInfo ioInfo, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetNumOutputs(ioInfo);
}
uint64_t axcl_EngineGetInputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetInputSizeByIndex(ioInfo, group, index);
}
uint64_t axcl_EngineGetOutputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetOutputSizeByIndex(ioInfo, group, index);
}
const char *axcl_EngineGetInputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetInputNameByIndex(ioInfo, index);
}
const char *axcl_EngineGetOutputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetOutputNameByIndex(ioInfo, index);
}
int32_t axcl_EngineGetInputIndexByName(axclrtEngineIOInfo ioInfo, const char *name, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetInputIndexByName(ioInfo, name);
}
int32_t axcl_EngineGetOutputIndexByName(axclrtEngineIOInfo ioInfo, const char *name, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetOutputIndexByName(ioInfo, name);
}
axclError axcl_EngineGetInputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetInputDims(ioInfo, group, index, dims);
}
axclError axcl_EngineGetOutputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetOutputDims(ioInfo, group, index, dims);
}
axclError axcl_EngineCreateIO(axclrtEngineIOInfo ioInfo, axclrtEngineIO *io, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineCreateIO(ioInfo, io);
}
axclError axcl_EngineDestroyIO(axclrtEngineIO io, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineDestroyIO(io);
}
axclError axcl_EngineSetInputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineSetInputBufferByIndex(io, index, dataBuffer, size);
}
axclError axcl_EngineSetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineSetOutputBufferByIndex(io, index, dataBuffer, size);
}
axclError axcl_EngineSetInputBufferByName(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineSetInputBufferByName(io, name, dataBuffer, size);
}
axclError axcl_EngineSetOutputBufferByName(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineSetOutputBufferByName(io, name, dataBuffer, size);
}
axclError axcl_EngineGetInputBufferByIndex(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetInputBufferByIndex(io, index, dataBuffer, size);
}
axclError axcl_EngineGetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetOutputBufferByIndex(io, index, dataBuffer, size);
}
axclError axcl_EngineGetInputBufferByName(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetInputBufferByName(io, name, dataBuffer, size);
}
axclError axcl_EngineGetOutputBufferByName(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineGetOutputBufferByName(io, name, dataBuffer, size);
}
axclError axcl_EngineSetDynamicBatchSize(axclrtEngineIO io, uint32_t batchSize, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineSetDynamicBatchSize(io, batchSize);
}
axclError axcl_EngineCreateContext(uint64_t modelId, uint64_t *contextId, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineCreateContext(modelId, contextId);
}
axclError axcl_EngineExecute(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineExecute(modelId, contextId, group, io);
}
axclError axcl_EngineExecuteAsync(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io, axclrtStream stream, int devid)
{
    if (!axcl_contains(devid))
    {
        ALOGE("AXCL device %d not inited\n", devid);
    }
    return g_devices[devid]->axclEngineExecuteAsync(modelId, contextId, group, io, stream);
}