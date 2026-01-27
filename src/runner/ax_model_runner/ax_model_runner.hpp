#pragma once
#include <vector>
#include <string>
#include <map>
#include <utility>
#include <stdexcept>

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
public:
    std::vector<ax_runner_tensor_t> moutput_tensors;
    std::vector<ax_runner_tensor_t> minput_tensors;

    std::vector<std::vector<ax_runner_tensor_t>> mgroup_output_tensors;
    std::vector<std::vector<ax_runner_tensor_t>> mgroup_input_tensors;

    std::map<std::string, ax_runner_tensor_t> map_output_tensors;
    std::map<std::string, ax_runner_tensor_t> map_input_tensors;

    std::map<std::string, std::vector<ax_runner_tensor_t>> map_group_output_tensors;
    std::map<std::string, std::vector<ax_runner_tensor_t>> map_group_input_tensors;

    bool _auto_sync_before_inference = false;
    bool _auto_sync_after_inference = false;

    int dev_id = 0;

    // 辅助函数：初始化完成后构建映射表，提高后续查找速度
    void build_tensor_maps()
    {
        map_input_tensors.clear();
        for (const auto &t : minput_tensors)
            map_input_tensors[t.sName] = t;

        map_output_tensors.clear();
        for (const auto &t : moutput_tensors)
            map_output_tensors[t.sName] = t;

        map_group_input_tensors.clear();
        for (const auto &grp : mgroup_input_tensors)
        {
            for (const auto &t : grp)
                map_group_input_tensors[t.sName].push_back(t);
        }

        map_group_output_tensors.clear();
        for (const auto &grp : mgroup_output_tensors)
        {
            for (const auto &t : grp)
                map_group_output_tensors[t.sName].push_back(t);
        }
    }

public:
    virtual int init(const char *model_file, int devid) = 0;
    virtual int init(char *model_buffer, size_t model_size, int devid) = 0;

    virtual void deinit() = 0;

    int get_devid() { return dev_id; }

    int get_num_inputs() { return minput_tensors.size(); };
    int get_num_outputs() { return moutput_tensors.size(); };

    int get_num_input_groups() { return mgroup_input_tensors.size(); };
    int get_num_output_groups() { return mgroup_output_tensors.size(); };

    const ax_runner_tensor_t &get_input(int idx) { return minput_tensors[idx]; }
    const ax_runner_tensor_t *get_inputs_ptr() { return minput_tensors.data(); }

    const ax_runner_tensor_t &get_input(const std::string &name)
    {
        auto it = map_input_tensors.find(name);
        if (it == map_input_tensors.end())
            throw std::runtime_error("input tensor not found: " + name);
        return it->second;
    }

    const ax_runner_tensor_t &get_input(int grpid, int idx) { return mgroup_input_tensors[grpid][idx]; }
    const ax_runner_tensor_t *get_inputs_ptr(int grpid) { return mgroup_input_tensors[grpid].data(); }

    const ax_runner_tensor_t &get_input(int grpid, const std::string &name)
    {
        auto it = map_group_input_tensors.find(name);
        if (it == map_group_input_tensors.end())
            throw std::runtime_error("input tensor not found: " + name);
        // 简单的越界检查
        if (grpid < 0 || grpid >= (int)it->second.size())
            throw std::runtime_error("group id out of range for: " + name);
        return it->second[grpid];
    }

    const ax_runner_tensor_t &get_output(int idx) { return moutput_tensors[idx]; }
    const ax_runner_tensor_t *get_outputs_ptr() { return moutput_tensors.data(); }

    const ax_runner_tensor_t &get_output(const std::string &name)
    {
        auto it = map_output_tensors.find(name);
        if (it == map_output_tensors.end())
            throw std::runtime_error("output tensor not found: " + name);
        return it->second;
    }

    const ax_runner_tensor_t &get_output(int grpid, int idx) { return mgroup_output_tensors[grpid][idx]; }
    const ax_runner_tensor_t *get_outputs_ptr(int grpid) { return mgroup_output_tensors[grpid].data(); }

    const ax_runner_tensor_t &get_output(int grpid, const std::string &name)
    {
        auto it = map_group_output_tensors.find(name);
        if (it == map_group_output_tensors.end())
            throw std::runtime_error("output tensor not found: " + name);
        if (grpid < 0 || grpid >= (int)it->second.size())
            throw std::runtime_error("group id out of range for: " + name);
        return it->second[grpid];
    }

    virtual int get_algo_width() = 0;
    virtual int get_algo_height() = 0;
    virtual ax_color_space_e get_color_space() = 0;

    void set_auto_sync_before_inference(bool sync) { _auto_sync_before_inference = sync; }
    void set_auto_sync_after_inference(bool sync) { _auto_sync_after_inference = sync; }

    virtual int inference() = 0;
    virtual int inference(int grpid) = 0;
};