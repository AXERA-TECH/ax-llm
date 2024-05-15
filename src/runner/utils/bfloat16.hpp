#pragma once
#include <vector>

struct bfloat16
{
    unsigned short int data;

public:
    bfloat16()
    {
        data = 0;
    }
    bfloat16(float float_val)
    {
        *this = float_val;
    }

    bfloat16(unsigned short int _data)
    {
        data = _data;
    }

    float fp32()
    {
        return *this;
    }

    // cast to float
    operator float()
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
        unsigned int proc = data << 16;
        return *reinterpret_cast<float *>(&proc);
#pragma GCC diagnostic pop
    }
    // cast to bfloat16
    bfloat16 &operator=(float float_val)
    {
        data = (*reinterpret_cast<unsigned int *>(&float_val)) >> 16;
        return *this;
    }
};

static std::vector<std::pair<int, float>> topk_bfloat16(unsigned short *arr, int size, int k)
{
    std::vector<std::pair<int, float>> result;

    // Create a vector of pairs with index and value
    std::vector<std::pair<int, float>> indexedValues;
    indexedValues.reserve(size);
    for (int i = 0; i < size; ++i)
    {
        indexedValues.push_back(std::make_pair(i, bfloat16(arr[i])));
    }

    // Sort the vector based on the values in descending order
    std::sort(indexedValues.begin(), indexedValues.end(),
              [](const std::pair<int, float> &a, const std::pair<int, float> &b)
              {
                  return a.second > b.second;
              });

    // Take the top k elements
    for (int i = 0; i < k; ++i)
    {
        result.push_back(indexedValues[i]);
    }

    return result;
}