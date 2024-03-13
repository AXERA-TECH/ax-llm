#pragma once

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
        unsigned int proc = data << 16;
        return *reinterpret_cast<float *>(&proc);
    }
    // cast to bfloat16
    bfloat16 &operator=(float float_val)
    {
        data = (*reinterpret_cast<unsigned int *>(&float_val)) >> 16;
        return *this;
    }
};