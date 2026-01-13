#include "runner/ax_model_runner/ax_model_runner_ax650.hpp"
#include "runner/ax_model_runner/ax_parallel_runner.hpp"
#include <axcl_rt_memory.h>
#include <axcl.h>
#include "cmdline.hpp"
#include "memory_utils.hpp"
#include "bfloat16.hpp"
#include <axcl_rt_p2p.h>

#include <cmath>
#include <timer.hpp>

float consine_similarity(const float *a, const float *b, int size)
{
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        dot_product += a[i] * b[i];
    }
    for (int i = 0; i < size; ++i)
    {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    return dot_product / (norm_a * norm_b);
}

int main(int argc, char **argv)
{
    std::vector<int> devices = {0, 1, 2, 3, 4, 5, 6, 7};
    AXCL_P2P_UNIT_HANDLE p2p_handle = nullptr;
    auto ret = axclInit(nullptr);
    if (0 != ret)
    {
        return ret;
    }

    for (auto &devid : devices)
    {
        if (axcl_Init(devid) != 0)
        {
            ALOGE("axcl_Init(%d) failed", devid);
            return false;
        }
    }

    {

        axclrtDeviceList device_list;
        if (const axclError ret = axclrtGetDeviceList(&device_list); AXCL_SUCC != ret || 0 == device_list.num)
        {
            printf("[ERROR] no device is connected.\n");
        }
        printf("[INFO] device num: %d.\n", device_list.num);
        if (2 > device_list.num)
        {
            printf("[ERROR] device num is less than 2.\n");
            return 2;
        }

        size_t p2p_cmm_size = 8 * 1024 * 1024;
        axclrtP2PUnitInfo p2p_unit;
        p2p_unit.u32DeviceNum = devices.size();
        for (uint32_t i = 0; i < p2p_unit.u32DeviceNum; ++i)
        {
            p2p_unit.n32DeviceId[i] = device_list.devices[devices[i]];
            p2p_unit.u32DeviceMemSize[i] = p2p_cmm_size;
        }

        if (const auto ret = axclrtCreateP2PUnit(&p2p_unit, &p2p_handle); AXCL_SUCC != ret)
        {
            printf("[ERROR] axcl init p2p unit fail, ret = 0x%x\n", ret);
            return -1;
        }
        else
        {
            std::cout << "[INFO] p2p unit created." << std::endl;
        }
    }

    ax_parallel_runner runner;

    runner.init("/home/axera/Qwen3-VL-8B-Instruct-GPTQ-Int4-AX650-C512-P1024-CTX2048-TP8/model_tar/qwen3_vl_text_p256_l0_together.tar", devices);

    timer t;
    for (int i = 0; i < 10; i++)
    {
        t.start();
        runner.inference(1);
        t.stop();
        printf("tp inference time: %0.2f ms\n", t.cost());
    }

    runner.deinit();

    axclrtDestoryP2PUnit(p2p_handle);

    for (auto &devid : devices)
    {
        axcl_Exit(devid);
    }

    axclFinalize();

    return 0;
}