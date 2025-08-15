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

float cosine_similarity(const float *a, const float *b, int size)
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

std::vector<float> tofp32(unsigned short *bf16, int size)
{
    std::vector<float> fp32(size);
    for (int i = 0; i < size; ++i)
    {
        fp32[i] = bfloat16(bf16[i]).fp32();
    }
    return fp32;
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

    ax_parallel_runner runner, runner_l1, runner_l2;

    runner.init("/home/axera/Qwen2.5-1.5B-Instruct-Parallel4-ax650/qwen2_p128_l0_together.tar", devices);
    runner_l1.init("/home/axera/Qwen2.5-1.5B-Instruct-Parallel4-ax650/qwen2_p128_l1_together.tar", devices);
    runner_l2.init("/home/axera/Qwen2.5-1.5B-Instruct-Parallel4-ax650/qwen2_p128_l2_together.tar", devices);

    for (size_t i = 0; i < runner_l1.get_num_groups(); i++)
    {
        runner.inference(0);
        runner_l1.inference(i);
        runner_l2.inference(i);
    }
    

    std::map<int, std::string> m_grp_gt = {
        {0, "/home/axera/ax-llm/build/qwen2.5_tp/sim_output_with_rank/"},
        {1, "/home/axera/ax-llm/build/qwen2.5_tp/sim_output_with_rank_group1/"},
    };

    for (auto &[grpid, gt_path] : m_grp_gt)
    {
        printf("group id: %d\n", grpid);
        for (size_t i = 0; i < runner.get_num_inputs(); i++)
        {
            for (int rankid = 0; rankid < devices.size(); rankid++)
            {
                auto input = runner.get_rank_input(rankid, grpid, i);

                std::vector<char> in_data;
                std::vector<std::string> paths = {
                    gt_path + input.sName + ".bin",
                    gt_path + input.sName + "_" + std::to_string(grpid) + ".bin",
                    gt_path + input.sName + "_rank" + std::to_string(rankid) + ".bin",
                    gt_path + input.sName + "_" + std::to_string(grpid) + "_rank" + std::to_string(rankid) + ".bin",
                };
                bool is_success = false;
                for (auto &path : paths)
                {
                    if (read_file(path, in_data))
                    {
                        is_success = true;
                        break;
                    }
                }
                if (!is_success)
                {
                    printf("read file failed for input %s\n", input.sName.c_str());
                    return -1;
                }
                printf("input name: %s , size: %d, data: %ld\n", input.sName.c_str(), input.nSize, in_data.size());
                axcl_Memcpy((void *)input.phyAddr, in_data.data(), input.nSize, AXCL_MEMCPY_HOST_TO_DEVICE, runner.get_devid(rankid));
            }
        }

        printf("set input done\n");
        timer t;
        for (int i = 0; i < 10; i++)
        {
            t.start();
            runner.inference(grpid);
            t.stop();
            printf("tp inference time: %0.2f ms\n", t.cost());
        }
        for (size_t i = 0; i < runner.get_num_outputs(); i++)
        {
            for (size_t rankid = 0; rankid < devices.size(); rankid++)
            {
                auto output = runner.get_rank_output(rankid, grpid, i);
                axcl_Memcpy(output.pVirAddr, (void *)output.phyAddr, output.nSize,
                            AXCL_MEMCPY_DEVICE_TO_HOST, runner.get_devid(rankid));

                std::vector<float> buffer_out_fp32 = tofp32((unsigned short *)output.pVirAddr, output.nSize / sizeof(unsigned short));

                std::vector<char> out_data;
                std::vector<std::string> paths = {
                    gt_path + output.sName + ".bin",
                    gt_path + output.sName + "_" + std::to_string(grpid) + ".bin",
                    gt_path + output.sName + "_rank" + std::to_string(rankid) + ".bin",
                    gt_path + output.sName + "_" + std::to_string(grpid) + "_rank" + std::to_string(rankid) + ".bin",
                };
                bool is_success = false;
                for (auto &path : paths)
                {
                    if (read_file(path, out_data))
                    {
                        is_success = true;
                        break;
                    }
                }
                if (!is_success)
                {
                    printf("read file failed for input %s\n", output.sName.c_str());
                    return -1;
                }
                std::vector<float> buffer_out_ref = tofp32((unsigned short *)out_data.data(), out_data.size() / sizeof(unsigned short));

                float sim = cosine_similarity(buffer_out_fp32.data(), buffer_out_ref.data(), buffer_out_fp32.size());
                printf("output name: %s rank-%ld sim: %f\n", output.sName.c_str(), rankid, sim);
            }
        }
        printf("group %d done\n", grpid);
        printf("==========================================================================================\n");
    }

    axclrtDestoryP2PUnit(p2p_handle);

    for (auto &devid : devices)
    {
        axcl_Exit(devid);
    }

    axclFinalize();

    return 0;
}