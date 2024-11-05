#pragma once
#include <vector>
#include <string>
#include <map>

typedef enum _color_space_e
{
    axdl_color_space_unknown,
    axdl_color_space_nv12,
    axdl_color_space_nv21,
    axdl_color_space_bgr,
    axdl_color_space_rgb,
} ax_color_space_e;

typedef struct _image_t
{
    unsigned long long int pPhy;
    void *pVir;
    unsigned int nSize;
    unsigned int nWidth;
    unsigned int nHeight;
    ax_color_space_e eDtype;
    union
    {
        int tStride_H, tStride_W, tStride_C;
    };
} ax_image_t;

typedef struct
{
    std::string sName;
    unsigned int nIdx;
    std::vector<unsigned int> vShape;
    int nSize;
    unsigned long long phyAddr;
    void *pVirAddr;
} ax_runner_tensor_t;

class ax_runner_base
{
protected:
    std::vector<ax_runner_tensor_t> moutput_tensors;
    std::vector<ax_runner_tensor_t> minput_tensors;

    std::map<std::string, ax_runner_tensor_t> map_output_tensors;
    std::map<std::string, ax_runner_tensor_t> map_input_tensors;

public:
    virtual int init(const char *model_file, bool use_mmap = false) = 0;
    virtual int init(char *model_buffer, size_t model_size) = 0;

    virtual void deinit() = 0;

    int get_num_inputs() { return minput_tensors.size(); }
    int get_num_outputs() { return moutput_tensors.size(); };

    const ax_runner_tensor_t &get_input(int idx) { return minput_tensors[idx]; }
    const ax_runner_tensor_t *get_inputs_ptr() { return minput_tensors.data(); }
    const ax_runner_tensor_t &get_input(std::string name)
    {
        if (map_input_tensors.size() == 0)
        {
            for (size_t i = 0; i < minput_tensors.size(); i++)
            {
                map_input_tensors[minput_tensors[i].sName] = minput_tensors[i];
            }
        }
        if (map_input_tensors.find(name) == map_input_tensors.end())
        {
            throw std::runtime_error("input tensor not found: " + name);
        }

        return map_input_tensors[name];
    }

    const ax_runner_tensor_t &get_output(int idx) { return moutput_tensors[idx]; }
    const ax_runner_tensor_t *get_outputs_ptr() { return moutput_tensors.data(); }
    const ax_runner_tensor_t &get_output(std::string name)
    {
        if (map_output_tensors.size() == 0)
        {
            for (size_t i = 0; i < moutput_tensors.size(); i++)
            {
                map_output_tensors[moutput_tensors[i].sName] = moutput_tensors[i];
            }
        }
        if (map_output_tensors.find(name) == map_output_tensors.end())
        {
            throw std::runtime_error("output tensor not found: " + name);
        }

        return map_output_tensors[name];
    }

    virtual int get_algo_width() = 0;
    virtual int get_algo_height() = 0;
    virtual ax_color_space_e get_color_space() = 0;

    virtual int inference(ax_image_t *pstFrame) = 0;
    virtual int inference() = 0;

    int operator()(ax_image_t *pstFrame)
    {
        return inference(pstFrame);
    }
};

// int ax_cmmcpy(unsigned long long int dst, unsigned long long int src, int size);