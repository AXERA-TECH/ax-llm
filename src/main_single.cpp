#include "runner/ax_model_runner/ax_model_runner_ax650.hpp"
#include "runner/ax_model_runner/ax_model_runner_ax650_host.hpp"
#include <axcl_rt_memory.h>
#include <axcl.h>
#include "cmdline.hpp"
#include "memory_utils.hpp"
#include "bfloat16.hpp"

int main(int argc, char **argv)
{
    cmdline::parser parser;
    parser.add<std::string>("model", 'm', "model file", true);
    parser.add<std::string>("input", 'i', "input folder", true);
    parser.parse_check(argc, argv);

    std::string model_file = parser.get<std::string>("model");
    std::string input_folder = parser.get<std::string>("input");

    ax_runner_ax650 runner;
    if (runner.init(model_file.c_str()) != 0)
    {
        std::cout << "init model error" << std::endl;
        return -1;
    }

    ax_runner_ax650_host runner_host;
    if (runner_host.init(model_file.c_str()) != 0)
    {
        std::cout << "init model error" << std::endl;
        return -1;
    }

    for (int i = 0; i < runner.get_num_inputs(); i++)
    {
        std::string file = input_folder + "/" + runner.get_input(i).sName + ".bin";
        std::vector<char> buffer;
        read_file(file, buffer);
        printf("name: %s size: %d\n", runner.get_input(i).sName.c_str(), buffer.size());
        axclrtMemcpy((void *)runner.get_input(i).phyAddr, buffer.data(), buffer.size(), AXCL_MEMCPY_HOST_TO_DEVICE);
        memcpy((void *)runner_host.get_input(i).pVirAddr, buffer.data(), buffer.size());
    }

    runner.inference();
    runner_host.inference();

    for (size_t i = 0; i < runner.get_num_outputs(); i++)
    {
        axclrtMemcpy(
            runner.get_output(i).pVirAddr,
            (void *)runner.get_output(i).phyAddr, runner.get_output(i).nSize, AXCL_MEMCPY_DEVICE_TO_HOST);
        
        unsigned short *embed = (unsigned short *)runner.get_output(i).pVirAddr;
        unsigned short *embed_host = (unsigned short *)runner_host.get_output(i).pVirAddr;

        printf("slave:%f %f %f %f %f\n", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
        printf(" host:%f %f %f %f %f\n", bfloat16(embed_host[0]).fp32(), bfloat16(embed_host[1]).fp32(), bfloat16(embed_host[2]).fp32(), bfloat16(embed_host[3]).fp32(), bfloat16(embed_host[4]).fp32());

        auto ret = memcmp(runner.get_output(i).pVirAddr, runner_host.get_output(i).pVirAddr, runner.get_output(i).nSize);
        if (ret != 0)
        {
            std::cout << "output " << i << " error" << std::endl;
            return -1;
        }
    }
    axclFinalize();

    return 0;
}