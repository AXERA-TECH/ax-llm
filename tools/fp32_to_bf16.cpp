#include <fstream>
#include <iostream>
#include "vector"
#include "../src/runner/utils/bfloat16.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <fp32_input_file> <bf16_output_file>" << std::endl;
        return 1;
    }
    std::ifstream fin(argv[1]);
    std::vector<char> data;

    fin.seekg(0, std::ios::end);
    data.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(data.data(), data.size());
    fin.close();

    std::ofstream fout(argv[2]);

    float *ptr = reinterpret_cast<float *>(data.data());
    for (int i = 0; i < data.size() / 4; i++)
    {
        bfloat16 bf16 = *ptr;
        fout.write(reinterpret_cast<const char *>(&bf16.data), 2);
        ptr++;
    }

    fout.close();

    return 0;
}