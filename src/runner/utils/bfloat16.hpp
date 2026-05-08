#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
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

    float fp32() const
    {
        return static_cast<float>(*this);
    }

    // cast to float
    operator float() const
    {
        uint32_t proc = static_cast<uint32_t>(data) << 16;
        float out = 0.0f;
        std::memcpy(&out, &proc, sizeof(out));
        return out;
    }
    // cast to bfloat16
    bfloat16 &operator=(float float_val)
    {
        uint32_t proc = 0;
        std::memcpy(&proc, &float_val, sizeof(proc));
        data = static_cast<unsigned short>(proc >> 16);
        return *this;
    }
};

static inline unsigned short fp32_to_bfloat16_rne(float float_val)
{
    std::uint32_t bits;
    std::memcpy(&bits, &float_val, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<unsigned short>(bits >> 16);
}

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
