#pragma once
#include <vector>
#include <string>
#include <map>
#include <stdexcept>

// ---------------------------------------------------------------------------
// 通用类型：两个后端共用
// ---------------------------------------------------------------------------
typedef enum _color_space_e
{
    axdl_color_space_unknown,
    axdl_color_space_nv12,
    axdl_color_space_nv21,
    axdl_color_space_bgr,
    axdl_color_space_rgb,
} ax_color_space_e;

typedef struct
{
    std::string sName;
    unsigned int nIdx;
    std::vector<unsigned int> vShape;
    int nSize;
    unsigned long long phyAddr; // 统一 64-bit；AX650 aarch64 上 unsigned long 也是 64-bit
    void *pVirAddr;
} ax_runner_tensor_t;

// ---------------------------------------------------------------------------
// ax_runner_base — 统一基类
//
//   dev_id 约定：
//     -1   → AX650 片上后端（只有一个片上 NPU，无需设备号）
//     >= 0 → AXCL PCIe 后端的设备 ID（从 0 开始）
// ---------------------------------------------------------------------------
class ax_runner_base
{
protected:
    std::vector<ax_runner_tensor_t> moutput_tensors;
    std::vector<ax_runner_tensor_t> minput_tensors;

    std::vector<std::vector<ax_runner_tensor_t>> mgroup_output_tensors;
    std::vector<std::vector<ax_runner_tensor_t>> mgroup_input_tensors;

    std::map<std::string, ax_runner_tensor_t> map_output_tensors;
    std::map<std::string, ax_runner_tensor_t> map_input_tensors;

    std::map<std::string, std::vector<ax_runner_tensor_t>> map_group_output_tensors;
    std::map<std::string, std::vector<ax_runner_tensor_t>> map_group_input_tensors;

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
            for (const auto &t : grp)
                map_group_input_tensors[t.sName].push_back(t);

        map_group_output_tensors.clear();
        for (const auto &grp : mgroup_output_tensors)
            for (const auto &t : grp)
                map_group_output_tensors[t.sName].push_back(t);
    }

public:
    int dev_id = -1; // -1 = AX650 片上，>= 0 = AXCL 设备号

    bool _auto_sync_before_inference = false;
    bool _auto_sync_after_inference = false;

    virtual ~ax_runner_base() {}

    // devid = -1 → AX650（忽略），devid >= 0 → AXCL 设备
    virtual int init(const char *model_file, int devid = -1) = 0;
    virtual int init(char *model_buffer, size_t model_size, int devid = -1) = 0;
    virtual void deinit() = 0;

    int get_devid() const { return dev_id; }

    void set_auto_sync_before_inference(bool v) { _auto_sync_before_inference = v; }
    void set_auto_sync_after_inference(bool v) { _auto_sync_after_inference = v; }

    // 图像推理接口（LLM 不使用，提供默认实现避免子类必须重写）
    virtual int get_algo_width() { return 0; }
    virtual int get_algo_height() { return 0; }
    virtual ax_color_space_e get_color_space() { return axdl_color_space_unknown; }

    // ---- tensor 数量 ----
    int get_num_inputs() { return (int)minput_tensors.size(); }
    int get_num_outputs() { return (int)moutput_tensors.size(); }
    int get_num_input_groups() { return (int)mgroup_input_tensors.size(); }
    int get_num_output_groups() { return (int)mgroup_output_tensors.size(); }

    // ---- 无 group 的 tensor 访问 ----
    const ax_runner_tensor_t &get_input(int idx) { return minput_tensors[idx]; }
    const ax_runner_tensor_t *get_inputs_ptr() { return minput_tensors.data(); }

    const ax_runner_tensor_t &get_input(const std::string &name)
    {
        auto it = map_input_tensors.find(name);
        if (it == map_input_tensors.end())
            throw std::runtime_error("input tensor not found: " + name);
        return it->second;
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

    // ---- 带 group 的 tensor 访问 ----
    const ax_runner_tensor_t &get_input(int grpid, int idx) { return mgroup_input_tensors[grpid][idx]; }
    const ax_runner_tensor_t *get_inputs_ptr(int grpid) { return mgroup_input_tensors[grpid].data(); }

    const ax_runner_tensor_t &get_input(int grpid, const std::string &name)
    {
        auto it = map_group_input_tensors.find(name);
        if (it == map_group_input_tensors.end())
            throw std::runtime_error("input tensor not found: " + name);
        if (grpid < 0 || grpid >= (int)it->second.size())
            throw std::runtime_error("group id out of range for: " + name);
        return it->second[grpid];
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

    virtual int inference() = 0;
    virtual int inference(int grpid) = 0;
};
