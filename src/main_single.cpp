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
    std::vector<int> devices = {0, 1, 2, 3};
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

    ax_runner_ax650 runner_ax650;

    runner.init("/home/axera/Qwen2.5-1.5B-Instruct-Parallel4-ax650/qwen2_p128_l0_together.tar", devices);

    runner_ax650.init("axmodel_qwen2.5_1.5b_chunked_nogptq_bf16/qwen2_p128_l0_together.axmodel", 0);

    for (size_t i = 0; i < runner_ax650.get_num_inputs(); i++)
    {
        auto input = runner_ax650.get_input(i);
        std::vector<char> in_data;
        if (!read_file("dump/input/" + input.sName + ".bin", in_data))
        {
            std::cout << "read file failed" << std::endl;
            return -1;
        }
        printf("input name: %s, size: %d, data: %d\n", input.sName.c_str(), input.nSize, in_data.size());
        axcl_Memcpy((void *)input.phyAddr, in_data.data(), input.nSize, AXCL_MEMCPY_HOST_TO_DEVICE, 0);
    }

    for (size_t i = 0; i < runner.get_num_inputs(); i++)
    {
        for (int rankid = 0; rankid < devices.size(); rankid++)
        {
            auto input = runner.get_rank_input(rankid, i);

            if (input.sName.find("K_cache") != std::string::npos)
            {
                axcl_Memset((void *)input.phyAddr, 0, input.nSize, 0);
                printf("skip input name: %s, size: %d, data: %d\n", input.sName.c_str(), input.nSize, 0);
                continue;
            }

            if (input.sName.find("V_cache") != std::string::npos)
            {
                axcl_Memset((void *)input.phyAddr, 0, input.nSize, 0);
                printf("skip input name: %s, size: %d, data: %d\n", input.sName.c_str(), input.nSize, 0);
                continue;
            }

            std::vector<char> in_data;
            std::string path = "dump_tp/mingli/input/" + input.sName + ".bin";
            if (!read_file(path, in_data))
            {
                std::cout << "read file failed" << std::endl;
                return -1;
            }
            printf("input name: %s - %s, size: %d, data: %d\n", input.sName.c_str(), path.c_str(), input.nSize, in_data.size());
            axcl_Memcpy((void *)input.phyAddr, in_data.data(), input.nSize, AXCL_MEMCPY_HOST_TO_DEVICE, runner.get_devid(rankid));
        }
        // auto input = runner.get_input(i);
        // std::vector<char> in_data;
        // if (!read_file("dump_tp/input/" + input.sName + ".bin", in_data))
        // {
        //     std::cout << "read file failed" << std::endl;
        //     return -1;
        // }
        // printf("input name: %s, size: %d, data: %d\n", input.sName.c_str(), input.nSize, in_data.size());
        // runner.set_input(i, in_data.data(), input.nSize);
    }

    axcl_Memcpy((void *)runner_ax650.get_input("input").phyAddr,
                (void *)runner.get_input("input").phyAddr,
                runner_ax650.get_input("input").nSize, AXCL_MEMCPY_DEVICE_TO_DEVICE, runner.get_devid());

    printf("set input done\n");
    timer t;
    for (int i = 0; i < 10; i++)
    {
        t.start();
        runner.inference();
        t.stop();
        printf("tp inference time: %0.2f ms\n", t.cost());

        t.start();
        runner_ax650.inference();
        t.stop();
        printf("inference time: %0.2f ms\n", t.cost());
    }

    for (size_t i = 0; i < runner.get_num_outputs(); i++)
    {
        auto output = runner.get_output(i);
        FILE *fp = fopen(("dump_tp/output/" + output.sName + ".bin").c_str(), "wb");
        if (fp)
        {
            axcl_Memcpy(output.pVirAddr, (void *)output.phyAddr, output.nSize, AXCL_MEMCPY_DEVICE_TO_HOST, runner.get_devid());
            fwrite(output.pVirAddr, output.nSize, 1, fp);
            fclose(fp);
        }
    }

    axcl_Memcpy(runner_ax650.get_output("output").pVirAddr,
                (void *)runner_ax650.get_output("output").phyAddr,
                runner_ax650.get_output("output").nSize, AXCL_MEMCPY_DEVICE_TO_HOST, runner.get_devid());

    std::vector<float> output_data_650(runner_ax650.get_output("output").nSize / sizeof(unsigned short));

    unsigned short *output_data = (unsigned short *)runner_ax650.get_output("output").pVirAddr;
    printf("output_data size: %d\n", runner_ax650.get_output("output").nSize);
    printf("(");
    for (size_t i = 0; i < output_data_650.size(); i++)
    {
        output_data_650[i] = bfloat16(output_data[i]).fp32();
        printf("%f, ", output_data_650[i]);
    }
    printf(")\n");

    for (size_t rankid = 0; rankid < devices.size(); rankid++)
    {
        axcl_Memcpy(runner.get_rank_output(rankid, "output").pVirAddr,
                    (void *)runner.get_rank_output(rankid, "output").phyAddr,
                    runner.get_rank_output(rankid, "output").nSize,
                    AXCL_MEMCPY_DEVICE_TO_HOST, runner.get_devid(rankid));

        std::vector<float> buffer_out_fp32(runner.get_rank_output(rankid, "output").nSize / sizeof(unsigned short));

        unsigned short *buffer_out_fp16 = (unsigned short *)runner.get_rank_output(rankid, "output").pVirAddr;
        printf("rank-%d size: %d\n", rankid, runner.get_rank_output(rankid, "output").nSize);
        printf("(");
        for (size_t i = 0; i < buffer_out_fp32.size(); i++)
        {
            buffer_out_fp32[i] = bfloat16(buffer_out_fp16[i]).fp32();
            printf("%f, ", buffer_out_fp32[i]);
        }
        printf(")\n");

        FILE *fp = fopen(("rank" + std::to_string(rankid) + ".bin").c_str(), "wb");
        if (fp)
        {
            fwrite(runner.get_rank_output(rankid, "output").pVirAddr, runner.get_rank_output(rankid, "output").nSize, 1, fp);
            fclose(fp);
        }

        float sim = consine_similarity(output_data_650.data(), buffer_out_fp32.data(), output_data_650.size());
        printf("rank %d similarity: %f\n", rankid, sim);
    }

    axclrtDestoryP2PUnit(p2p_handle);

    for (auto &devid : devices)
    {
        axcl_Exit(devid);
    }

    axclFinalize();

    return 0;
}