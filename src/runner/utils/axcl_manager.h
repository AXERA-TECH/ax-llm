#ifndef _AXCL_MANAGER_H_
#define _AXCL_MANAGER_H_
#include <axcl.h>

#ifdef __cplusplus
extern "C"
{
#endif
    axclError axcl_Init(int devid);
    axclError axcl_Exit(int devid);

    int axcl_GetCMMRemain(int devid);

    axclError axcl_Malloc(void **devPtr, size_t size, axclrtMemMallocPolicy policy, int devid);
    axclError axcl_MallocCached(void **devPtr, size_t size, axclrtMemMallocPolicy policy, int devid);
    axclError axcl_Free(void *devPtr, int devid);
    axclError axcl_MemFlush(void *devPtr, size_t size, int devid);
    axclError axcl_MemInvalidate(void *devPtr, size_t size, int devid);
    axclError axcl_MallocHost(void **hostPtr, size_t size, int devid);
    axclError axcl_FreeHost(void *hostPtr, int devid);
    axclError axcl_Memset(void *devPtr, uint8_t value, size_t count, int devid);
    axclError axcl_Memcpy(void *dstPtr, const void *srcPtr, size_t count, axclrtMemcpyKind kind, int devid);
    axclError axcl_Memcmp(const void *devPtr1, const void *devPtr2, size_t count, int devid);

    axclError axcl_EngineLoadFromFile(const char *modelPath, uint64_t *modelId, int devid);
    axclError axcl_EngineLoadFromMem(const void *model, uint64_t modelSize, uint64_t *modelId, int devid);
    axclError axcl_EngineUnload(uint64_t modelId, int devid);
    const char* axcl_EngineGetModelCompilerVersion(uint64_t modelId, int devid);
    axclError axcl_EngineSetAffinity(uint64_t modelId, axclrtEngineSet set, int devid);
    axclError axcl_EngineGetAffinity(uint64_t modelId, axclrtEngineSet *set, int devid);
    axclError axcl_EngineGetUsage(const char *modelPath, int64_t *sysSize, int64_t *cmmSize, int devid);
    axclError axcl_EngineGetUsageFromMem(const void *model, uint64_t modelSize, int64_t *sysSize, int64_t *cmmSize, int devid);
    axclError axcl_EngineGetUsageFromModelId(uint64_t modelId, int64_t *sysSize, int64_t *cmmSize, int devid);
    axclError axcl_EngineGetModelType(const char *modelPath, axclrtEngineModelKind *modelType, int devid);
    axclError axcl_EngineGetModelTypeFromMem(const void *model, uint64_t modelSize, axclrtEngineModelKind *modelType, int devid);
    axclError axcl_EngineGetModelTypeFromModelId(uint64_t modelId, axclrtEngineModelKind *modelType, int devid);
    axclError axcl_EngineGetIOInfo(uint64_t modelId, axclrtEngineIOInfo *ioInfo, int devid);
    axclError axcl_EngineDestroyIOInfo(axclrtEngineIOInfo ioInfo, int devid);
    axclError axcl_EngineGetShapeGroupsCount(axclrtEngineIOInfo ioInfo, int32_t *count, int devid);
    uint32_t axcl_EngineGetNumInputs(axclrtEngineIOInfo ioInfo, int devid);
    uint32_t axcl_EngineGetNumOutputs(axclrtEngineIOInfo ioInfo, int devid);
    uint64_t axcl_EngineGetInputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, int devid);
    uint64_t axcl_EngineGetOutputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, int devid);
    const char* axcl_EngineGetInputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index, int devid);
    const char* axcl_EngineGetOutputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index, int devid);
    int32_t axcl_EngineGetInputIndexByName(axclrtEngineIOInfo ioInfo, const char *name, int devid);
    int32_t axcl_EngineGetOutputIndexByName(axclrtEngineIOInfo ioInfo, const char *name, int devid);
    axclError axcl_EngineGetInputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims, int devid);
    axclError axcl_EngineGetOutputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims, int devid);
    axclError axcl_EngineCreateIO(axclrtEngineIOInfo ioInfo, axclrtEngineIO *io, int devid);
    axclError axcl_EngineDestroyIO(axclrtEngineIO io, int devid);
    axclError axcl_EngineSetInputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size, int devid);
    axclError axcl_EngineSetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size, int devid);
    axclError axcl_EngineSetInputBufferByName(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size, int devid);
    axclError axcl_EngineSetOutputBufferByName(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size, int devid);
    axclError axcl_EngineGetInputBufferByIndex(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size, int devid);
    axclError axcl_EngineGetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size, int devid);
    axclError axcl_EngineGetInputBufferByName(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size, int devid);
    axclError axcl_EngineGetOutputBufferByName(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size, int devid);
    axclError axcl_EngineSetDynamicBatchSize(axclrtEngineIO io, uint32_t batchSize, int devid);
    axclError axcl_EngineCreateContext(uint64_t modelId, uint64_t *contextId, int devid);
    axclError axcl_EngineExecute(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io, int devid);
    axclError axcl_EngineExecuteAsync(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io, axclrtStream stream, int devid);
#ifdef __cplusplus
}
#endif

#endif