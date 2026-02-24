#pragma once
#include <iostream>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <chrono>

#include <axcl.h>
#include "ax_cmm_utils.hpp"
#include "sample_log.h"

class AXCLWorker
{
private:
    using Task = std::function<void()>;
    std::thread worker_thread;
    std::queue<Task> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop_flag;

    std::promise<bool> initPromise;

    void run(int devid)
    {
        ALOGI("AXCLWorker start with devid %d", devid);

        axclrtDeviceList lst;
        if (const auto ret = axclrtGetDeviceList(&lst); 0 != ret || 0 == lst.num)
        {
            ALOGE("Get AXCL device failed{0x%8x}, find total %d device.", ret, lst.num);
            initPromise.set_value(false); // 初始化失败
            return;
        }
        if (devid >= lst.num)
        {
            ALOGE("Invalid AXCL device id %d, find total %d device.", devid, lst.num);
            initPromise.set_value(false); // 初始化失败
            return;
        }

        // ALOGI("AXCLWorker start with devidx-%d, bus-id-%d", devidx, lst.devices[devidx]);

        if (const auto ret = axclrtSetDevice(lst.devices[devid]); 0 != ret)
        {
            ALOGE("Set AXCL device failed{0x%8x}.", ret);
            initPromise.set_value(false); // 初始化失败
            return;
        }
        if (const auto ret = axclrtEngineInit(AXCL_VNPU_DISABLE); 0 != ret)
        {
            ALOGE("axclrtEngineInit %d", ret);
            initPromise.set_value(false); // 初始化失败
            return;
        }

        // 初始化成功，通知 Run 可以返回 true
        initPromise.set_value(true);

        while (true)
        {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this]
                               { return stop_flag || !tasks.empty(); });
                if (stop_flag && tasks.empty())
                {
                    break;
                }
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
        }
        ALOGI("AXCLWorker exit with devid %d", devid);
    }

    // 添加任务的接口（无返回值版本）
    void addTask(Task task)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.push(task);
        }
        condition.notify_one();
    }

    // 模板接口：添加任务并返回 std::future 用于获取返回值
    template <typename F, typename... Args>
    auto addTaskWithResult(F &&f, Args &&...args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using result_type = typename std::result_of<F(Args...)>::type;
        // 将函数及其参数绑定成一个无参函数
        auto task = std::make_shared<std::packaged_task<result_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<result_type> res = task->get_future();
        // 将任务封装为 lambda，确保在工作线程中执行
        addTask([task]()
                { (*task)(); });
        return res;
    }

    axclError axclrtMalloc_func(void **devPtr, size_t size, axclrtMemMallocPolicy policy)
    {
        return axclrtMalloc(devPtr, size, policy);
    }
    axclError axclrtMallocCached_func(void **devPtr, size_t size, axclrtMemMallocPolicy policy)
    {
        return axclrtMallocCached(devPtr, size, policy);
    }
    axclError axclrtFree_func(void *devPtr)
    {
        return axclrtFree(devPtr);
    }
    axclError axclrtMemFlush_func(void *devPtr, size_t size)
    {
        return axclrtMemFlush(devPtr, size);
    }
    axclError axclrtMemInvalidate_func(void *devPtr, size_t size)
    {
        return axclrtMemInvalidate(devPtr, size);
    }
    axclError axclrtMallocHost_func(void **hostPtr, size_t size)
    {
        return axclrtMallocHost(hostPtr, size);
    }
    axclError axclrtFreeHost_func(void *hostPtr)
    {
        return axclrtFreeHost(hostPtr);
    }
    axclError axclrtMemset_func(void *devPtr, uint8_t value, size_t count)
    {
        return axclrtMemset(devPtr, value, count);
    }
    axclError axclrtMemcpy_func(void *dstPtr, const void *srcPtr, size_t count, axclrtMemcpyKind kind)
    {
        return axclrtMemcpy(dstPtr, srcPtr, count, kind);
    }
    axclError axclrtMemcmp_func(const void *devPtr1, const void *devPtr2, size_t count)
    {
        return axclrtMemcmp(devPtr1, devPtr2, count);
    }

    // ────────── 以下为各 API 的内部实现（私有部分，后缀 _func） ──────────
    // 1. axclrtEngineLoadFromFile
    axclError axclrtEngineLoadFromFile_func(const char *modelPath, uint64_t *modelId)
    {
        return axclrtEngineLoadFromFile(modelPath, modelId);
    }
    // 2. axclrtEngineLoadFromMem
    axclError axclrtEngineLoadFromMem_func(const void *model, uint64_t modelSize, uint64_t *modelId)
    {
        return axclrtEngineLoadFromMem(model, modelSize, modelId);
    }
    // 3. axclrtEngineUnload
    axclError axclrtEngineUnload_func(uint64_t modelId)
    {
        return axclrtEngineUnload(modelId);
    }
    // 4. axclrtEngineGetModelCompilerVersion
    const char *axclrtEngineGetModelCompilerVersion_func(uint64_t modelId)
    {
        return axclrtEngineGetModelCompilerVersion(modelId);
    }
    // 5. axclrtEngineSetAffinity
    axclError axclrtEngineSetAffinity_func(uint64_t modelId, axclrtEngineSet set)
    {
        return axclrtEngineSetAffinity(modelId, set);
    }
    // 6. axclrtEngineGetAffinity
    axclError axclrtEngineGetAffinity_func(uint64_t modelId, axclrtEngineSet *set)
    {
        return axclrtEngineGetAffinity(modelId, set);
    }
    // 7. axclrtEngineGetUsage
    axclError axclrtEngineGetUsage_func(const char *modelPath, int64_t *sysSize, int64_t *cmmSize)
    {
        return axclrtEngineGetUsage(modelPath, sysSize, cmmSize);
    }
    // 8. axclrtEngineGetUsageFromMem
    axclError axclrtEngineGetUsageFromMem_func(const void *model, uint64_t modelSize, int64_t *sysSize, int64_t *cmmSize)
    {
        return axclrtEngineGetUsageFromMem(model, modelSize, sysSize, cmmSize);
    }
    // 9. axclrtEngineGetUsageFromModelId
    axclError axclrtEngineGetUsageFromModelId_func(uint64_t modelId, int64_t *sysSize, int64_t *cmmSize)
    {
        return axclrtEngineGetUsageFromModelId(modelId, sysSize, cmmSize);
    }
    // 10. axclrtEngineGetModelType
    axclError axclrtEngineGetModelType_func(const char *modelPath, axclrtEngineModelKind *modelType)
    {
        return axclrtEngineGetModelType(modelPath, modelType);
    }
    // 11. axclrtEngineGetModelTypeFromMem
    axclError axclrtEngineGetModelTypeFromMem_func(const void *model, uint64_t modelSize, axclrtEngineModelKind *modelType)
    {
        return axclrtEngineGetModelTypeFromMem(model, modelSize, modelType);
    }
    // 12. axclrtEngineGetModelTypeFromModelId
    axclError axclrtEngineGetModelTypeFromModelId_func(uint64_t modelId, axclrtEngineModelKind *modelType)
    {
        return axclrtEngineGetModelTypeFromModelId(modelId, modelType);
    }
    // 13. axclrtEngineGetIOInfo
    axclError axclrtEngineGetIOInfo_func(uint64_t modelId, axclrtEngineIOInfo *ioInfo)
    {
        return axclrtEngineGetIOInfo(modelId, ioInfo);
    }
    // 14. axclrtEngineDestroyIOInfo
    axclError axclrtEngineDestroyIOInfo_func(axclrtEngineIOInfo ioInfo)
    {
        return axclrtEngineDestroyIOInfo(ioInfo);
    }
    // 15. axclrtEngineGetShapeGroupsCount
    axclError axclrtEngineGetShapeGroupsCount_func(axclrtEngineIOInfo ioInfo, int32_t *count)
    {
        return axclrtEngineGetShapeGroupsCount(ioInfo, count);
    }
    // 16. axclrtEngineGetNumInputs
    uint32_t axclrtEngineGetNumInputs_func(axclrtEngineIOInfo ioInfo)
    {
        return axclrtEngineGetNumInputs(ioInfo);
    }
    // 17. axclrtEngineGetNumOutputs
    uint32_t axclrtEngineGetNumOutputs_func(axclrtEngineIOInfo ioInfo)
    {
        return axclrtEngineGetNumOutputs(ioInfo);
    }
    // 18. axclrtEngineGetInputSizeByIndex
    uint64_t axclrtEngineGetInputSizeByIndex_func(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index)
    {
        return axclrtEngineGetInputSizeByIndex(ioInfo, group, index);
    }
    // 19. axclrtEngineGetOutputSizeByIndex
    uint64_t axclrtEngineGetOutputSizeByIndex_func(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index)
    {
        return axclrtEngineGetOutputSizeByIndex(ioInfo, group, index);
    }
    // 20. axclrtEngineGetInputNameByIndex
    const char *axclrtEngineGetInputNameByIndex_func(axclrtEngineIOInfo ioInfo, uint32_t index)
    {
        return axclrtEngineGetInputNameByIndex(ioInfo, index);
    }
    // 21. axclrtEngineGetOutputNameByIndex
    const char *axclrtEngineGetOutputNameByIndex_func(axclrtEngineIOInfo ioInfo, uint32_t index)
    {
        return axclrtEngineGetOutputNameByIndex(ioInfo, index);
    }
    // 22. axclrtEngineGetInputIndexByName
    int32_t axclrtEngineGetInputIndexByName_func(axclrtEngineIOInfo ioInfo, const char *name)
    {
        return axclrtEngineGetInputIndexByName(ioInfo, name);
    }
    // 23. axclrtEngineGetOutputIndexByName
    int32_t axclrtEngineGetOutputIndexByName_func(axclrtEngineIOInfo ioInfo, const char *name)
    {
        return axclrtEngineGetOutputIndexByName(ioInfo, name);
    }
    // 24. axclrtEngineGetInputDims
    axclError axclrtEngineGetInputDims_func(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims)
    {
        return axclrtEngineGetInputDims(ioInfo, group, index, dims);
    }
    // 25. axclrtEngineGetOutputDims
    axclError axclrtEngineGetOutputDims_func(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims)
    {
        return axclrtEngineGetOutputDims(ioInfo, group, index, dims);
    }
    // 26. axclrtEngineCreateIO
    axclError axclrtEngineCreateIO_func(axclrtEngineIOInfo ioInfo, axclrtEngineIO *io)
    {
        return axclrtEngineCreateIO(ioInfo, io);
    }
    // 27. axclrtEngineDestroyIO
    axclError axclrtEngineDestroyIO_func(axclrtEngineIO io)
    {
        return axclrtEngineDestroyIO(io);
    }
    // 28. axclrtEngineSetInputBufferByIndex
    axclError axclrtEngineSetInputBufferByIndex_func(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size)
    {
        return axclrtEngineSetInputBufferByIndex(io, index, dataBuffer, size);
    }
    // 29. axclrtEngineSetOutputBufferByIndex
    axclError axclrtEngineSetOutputBufferByIndex_func(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size)
    {
        return axclrtEngineSetOutputBufferByIndex(io, index, dataBuffer, size);
    }
    // 30. axclrtEngineSetInputBufferByName
    axclError axclrtEngineSetInputBufferByName_func(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size)
    {
        return axclrtEngineSetInputBufferByName(io, name, dataBuffer, size);
    }
    // 31. axclrtEngineSetOutputBufferByName
    axclError axclrtEngineSetOutputBufferByName_func(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size)
    {
        return axclrtEngineSetOutputBufferByName(io, name, dataBuffer, size);
    }
    // 32. axclrtEngineGetInputBufferByIndex
    axclError axclrtEngineGetInputBufferByIndex_func(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size)
    {
        return axclrtEngineGetInputBufferByIndex(io, index, dataBuffer, size);
    }
    // 33. axclrtEngineGetOutputBufferByIndex
    axclError axclrtEngineGetOutputBufferByIndex_func(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size)
    {
        return axclrtEngineGetOutputBufferByIndex(io, index, dataBuffer, size);
    }
    // 34. axclrtEngineGetInputBufferByName
    axclError axclrtEngineGetInputBufferByName_func(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size)
    {
        return axclrtEngineGetInputBufferByName(io, name, dataBuffer, size);
    }
    // 35. axclrtEngineGetOutputBufferByName
    axclError axclrtEngineGetOutputBufferByName_func(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size)
    {
        return axclrtEngineGetOutputBufferByName(io, name, dataBuffer, size);
    }
    // 36. axclrtEngineSetDynamicBatchSize
    axclError axclrtEngineSetDynamicBatchSize_func(axclrtEngineIO io, uint32_t batchSize)
    {
        return axclrtEngineSetDynamicBatchSize(io, batchSize);
    }
    // 37. axclrtEngineCreateContext
    axclError axclrtEngineCreateContext_func(uint64_t modelId, uint64_t *contextId)
    {
        return axclrtEngineCreateContext(modelId, contextId);
    }
    // 38. axclrtEngineExecute
    axclError axclrtEngineExecute_func(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io)
    {
        return axclrtEngineExecute(modelId, contextId, group, io);
    }
    // 39. axclrtEngineExecuteAsync
    axclError axclrtEngineExecuteAsync_func(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io, axclrtStream stream)
    {
        return axclrtEngineExecuteAsync(modelId, contextId, group, io, stream);
    }

public:
    AXCLWorker() : stop_flag(false)
    {
    }
    ~AXCLWorker()
    {
        Stop();
    }

    bool Run(int devid)
    {
        // 启动线程前先重置 promise，确保没有旧状态影响
        initPromise = std::promise<bool>();
        std::future<bool> initFuture = initPromise.get_future();

        worker_thread = std::thread(&AXCLWorker::run, this, devid);

        // 等待一定时间，防止无限等待（可根据需要设置超时时间）
        if (initFuture.wait_for(std::chrono::seconds(5)) == std::future_status::ready)
        {
            bool initSuccess = initFuture.get();
            return initSuccess;
        }
        else
        {
            Stop();
            ALOGE("AXCLWorker initialization timeout");
            return false;
        }
    }
    // 停止工作线程
    void Stop()
    {
        if (!stop_flag)
        {
            stop_flag = true;
            condition.notify_one();
            if (worker_thread.joinable())
            {
                worker_thread.join();
            }
        }
    }

    axclError axclMalloc(void **devPtr, size_t size, axclrtMemMallocPolicy policy)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtMalloc_func, this, devPtr, size, policy);
        return future_result.get();
    }
    axclError axclMallocCached(void **devPtr, size_t size, axclrtMemMallocPolicy policy)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtMallocCached_func, this, devPtr, size, policy);
        return future_result.get();
    }
    axclError axclFree(void *devPtr)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtFree_func, this, devPtr);
        return future_result.get();
    }
    axclError axclMemFlush(void *devPtr, size_t size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtMemFlush_func, this, devPtr, size);
        return future_result.get();
    }
    axclError axclMemInvalidate(void *devPtr, size_t size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtMemInvalidate_func, this, devPtr, size);
        return future_result.get();
    }
    axclError axclMallocHost(void **hostPtr, size_t size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtMallocHost_func, this, hostPtr, size);
        return future_result.get();
    }
    axclError axclFreeHost(void *hostPtr)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtFreeHost_func, this, hostPtr);
        return future_result.get();
    }
    axclError axclMemset(void *devPtr, uint8_t value, size_t count)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtMemset_func, this, devPtr, value, count);
        return future_result.get();
    }
    axclError axclMemcpy(void *dstPtr, const void *srcPtr, size_t count, axclrtMemcpyKind kind)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtMemcpy_func, this, dstPtr, srcPtr, count, kind);
        return future_result.get();
    }
    axclError axclMemcmp(const void *devPtr1, const void *devPtr2, size_t count)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtMemcmp_func, this, devPtr1, devPtr2, count);
        return future_result.get();
    }

    // ────────── 以下为对外的 API 封装，顺序按照需求排列 ──────────
    axclError axclEngineLoadFromFile(const char *modelPath, uint64_t *modelId)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineLoadFromFile_func, this, modelPath, modelId);
        return future_result.get();
    }
    axclError axclEngineLoadFromMem(const void *model, uint64_t modelSize, uint64_t *modelId)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineLoadFromMem_func, this, model, modelSize, modelId);
        return future_result.get();
    }
    axclError axclEngineUnload(uint64_t modelId)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineUnload_func, this, modelId);
        return future_result.get();
    }
    const char *axclEngineGetModelCompilerVersion(uint64_t modelId)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetModelCompilerVersion_func, this, modelId);
        return future_result.get();
    }
    axclError axclEngineSetAffinity(uint64_t modelId, axclrtEngineSet set)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineSetAffinity_func, this, modelId, set);
        return future_result.get();
    }
    axclError axclEngineGetAffinity(uint64_t modelId, axclrtEngineSet *set)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetAffinity_func, this, modelId, set);
        return future_result.get();
    }
    axclError axclEngineGetUsage(const char *modelPath, int64_t *sysSize, int64_t *cmmSize)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetUsage_func, this, modelPath, sysSize, cmmSize);
        return future_result.get();
    }
    axclError axclEngineGetUsageFromMem(const void *model, uint64_t modelSize, int64_t *sysSize, int64_t *cmmSize)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetUsageFromMem_func, this, model, modelSize, sysSize, cmmSize);
        return future_result.get();
    }
    axclError axclEngineGetUsageFromModelId(uint64_t modelId, int64_t *sysSize, int64_t *cmmSize)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetUsageFromModelId_func, this, modelId, sysSize, cmmSize);
        return future_result.get();
    }
    axclError axclEngineGetModelType(const char *modelPath, axclrtEngineModelKind *modelType)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetModelType_func, this, modelPath, modelType);
        return future_result.get();
    }
    axclError axclEngineGetModelTypeFromMem(const void *model, uint64_t modelSize, axclrtEngineModelKind *modelType)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetModelTypeFromMem_func, this, model, modelSize, modelType);
        return future_result.get();
    }
    axclError axclEngineGetModelTypeFromModelId(uint64_t modelId, axclrtEngineModelKind *modelType)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetModelTypeFromModelId_func, this, modelId, modelType);
        return future_result.get();
    }
    axclError axclEngineGetIOInfo(uint64_t modelId, axclrtEngineIOInfo *ioInfo)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetIOInfo_func, this, modelId, ioInfo);
        return future_result.get();
    }
    axclError axclEngineDestroyIOInfo(axclrtEngineIOInfo ioInfo)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineDestroyIOInfo_func, this, ioInfo);
        return future_result.get();
    }
    axclError axclEngineGetShapeGroupsCount(axclrtEngineIOInfo ioInfo, int32_t *count)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetShapeGroupsCount_func, this, ioInfo, count);
        return future_result.get();
    }
    uint32_t axclEngineGetNumInputs(axclrtEngineIOInfo ioInfo)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetNumInputs_func, this, ioInfo);
        return future_result.get();
    }
    uint32_t axclEngineGetNumOutputs(axclrtEngineIOInfo ioInfo)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetNumOutputs_func, this, ioInfo);
        return future_result.get();
    }
    uint64_t axclEngineGetInputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetInputSizeByIndex_func, this, ioInfo, group, index);
        return future_result.get();
    }
    uint64_t axclEngineGetOutputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetOutputSizeByIndex_func, this, ioInfo, group, index);
        return future_result.get();
    }
    const char *axclEngineGetInputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetInputNameByIndex_func, this, ioInfo, index);
        return future_result.get();
    }
    const char *axclEngineGetOutputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetOutputNameByIndex_func, this, ioInfo, index);
        return future_result.get();
    }
    int32_t axclEngineGetInputIndexByName(axclrtEngineIOInfo ioInfo, const char *name)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetInputIndexByName_func, this, ioInfo, name);
        return future_result.get();
    }
    int32_t axclEngineGetOutputIndexByName(axclrtEngineIOInfo ioInfo, const char *name)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetOutputIndexByName_func, this, ioInfo, name);
        return future_result.get();
    }
    axclError axclEngineGetInputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetInputDims_func, this, ioInfo, group, index, dims);
        return future_result.get();
    }
    axclError axclEngineGetOutputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetOutputDims_func, this, ioInfo, group, index, dims);
        return future_result.get();
    }
    axclError axclEngineCreateIO(axclrtEngineIOInfo ioInfo, axclrtEngineIO *io)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineCreateIO_func, this, ioInfo, io);
        return future_result.get();
    }
    axclError axclEngineDestroyIO(axclrtEngineIO io)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineDestroyIO_func, this, io);
        return future_result.get();
    }
    axclError axclEngineSetInputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineSetInputBufferByIndex_func, this, io, index, dataBuffer, size);
        return future_result.get();
    }
    axclError axclEngineSetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineSetOutputBufferByIndex_func, this, io, index, dataBuffer, size);
        return future_result.get();
    }
    axclError axclEngineSetInputBufferByName(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineSetInputBufferByName_func, this, io, name, dataBuffer, size);
        return future_result.get();
    }
    axclError axclEngineSetOutputBufferByName(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineSetOutputBufferByName_func, this, io, name, dataBuffer, size);
        return future_result.get();
    }
    axclError axclEngineGetInputBufferByIndex(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetInputBufferByIndex_func, this, io, index, dataBuffer, size);
        return future_result.get();
    }
    axclError axclEngineGetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetOutputBufferByIndex_func, this, io, index, dataBuffer, size);
        return future_result.get();
    }
    axclError axclEngineGetInputBufferByName(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetInputBufferByName_func, this, io, name, dataBuffer, size);
        return future_result.get();
    }
    axclError axclEngineGetOutputBufferByName(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineGetOutputBufferByName_func, this, io, name, dataBuffer, size);
        return future_result.get();
    }
    axclError axclEngineSetDynamicBatchSize(axclrtEngineIO io, uint32_t batchSize)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineSetDynamicBatchSize_func, this, io, batchSize);
        return future_result.get();
    }
    axclError axclEngineCreateContext(uint64_t modelId, uint64_t *contextId)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineCreateContext_func, this, modelId, contextId);
        return future_result.get();
    }
    axclError axclEngineExecute(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineExecute_func, this, modelId, contextId, group, io);
        return future_result.get();
    }
    axclError axclEngineExecuteAsync(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io, axclrtStream stream)
    {
        auto future_result = addTaskWithResult(&AXCLWorker::axclrtEngineExecuteAsync_func, this, modelId, contextId, group, io, stream);
        return future_result.get();
    }
};
