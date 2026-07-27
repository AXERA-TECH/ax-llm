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
    int eDataType = 0; // Backend-native dtype value; 0 means unavailable/unknown.
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

    // ---- Multi-slot KV cache (device-resident prefix cache) ----
    // One entry per slottable input tensor (e.g. "K_cache" / "V_cache"). Each
    // holds, per slot, the device buffer backing that tensor. Slot 0 is the
    // engine's own buffer (borrowed, not owned by the slot pool); slots 1.. are
    // extra buffers allocated by the backend. Activating a slot rebinds every
    // group's tensor to that slot's buffer with zero data copy.
    struct KvSlotTensor
    {
        std::string name;
        size_t bytes = 0;                          // buffer size (max over groups)
        std::vector<int> idx_in_group;             // tensor index per group, -1 if absent
        std::vector<unsigned long long> slot_phy;  // [num_slots] device/phys addr
        std::vector<void *> slot_vir;              // [num_slots] cpu-mapped/scratch addr (may be null)
    };
    std::vector<KvSlotTensor> kv_slot_tensors_;
    int kv_num_slots_ = 1;
    int kv_active_slot_ = 0;

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
    int get_num_inputs(int grpid) { return (int)mgroup_input_tensors[grpid].size(); }
    int get_num_outputs(int grpid) { return (int)mgroup_output_tensors[grpid].size(); }

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
        if (grpid < 0 || grpid >= (int)mgroup_input_tensors.size())
            throw std::runtime_error("group id out of range for input: " + name);

        for (const auto &tensor : mgroup_input_tensors[(size_t)grpid])
        {
            if (tensor.sName == name)
                return tensor;
        }

        throw std::runtime_error("input tensor not found in group: " + name);
    }

    const ax_runner_tensor_t &get_output(int grpid, int idx) { return mgroup_output_tensors[grpid][idx]; }
    const ax_runner_tensor_t *get_outputs_ptr(int grpid) { return mgroup_output_tensors[grpid].data(); }

    const ax_runner_tensor_t &get_output(int grpid, const std::string &name)
    {
        if (grpid < 0 || grpid >= (int)mgroup_output_tensors.size())
            throw std::runtime_error("group id out of range for output: " + name);

        for (const auto &tensor : mgroup_output_tensors[(size_t)grpid])
        {
            if (tensor.sName == name)
                return tensor;
        }

        throw std::runtime_error("output tensor not found in group: " + name);
    }

    virtual int inference() = 0;
    virtual int inference(int grpid) = 0;

    // ---- Multi-slot KV cache API (device mode) ----
    // Phase 1: locate the slottable KV tensors (K_cache/V_cache), record slot 0
    // (the engine's own buffer), and return the per-slot device bytes that ONE
    // extra slot would consume (sum of K+V buffer sizes). 0 = unsupported / no KV.
    // Allocates nothing beyond bookkeeping, so the engine can budget first.
    virtual size_t kv_cache_slots_prepare() { return 0; }

    // Phase 2: allocate extra device buffers up to `num_slots` total (slot 0 is
    // the engine buffer). Allocates incrementally and stops at the first failure
    // instead of erroring out. Returns the number of slots actually available
    // (>=1). Default: only slot 0.
    virtual int kv_cache_slots_alloc(int num_slots) { return num_slots <= 1 ? 1 : 1; }

    // Free extra slot buffers so exactly `n` slots remain (>=1). Used to trim all
    // layers down to a common count. Returns the resulting count.
    virtual int kv_cache_slots_set_count(int n) { return n < 1 ? 1 : n; }

    // Rebind every group's KV tensors to the given slot's buffers (zero copy).
    // Returns 0 on success, <0 on error. Default: only slot 0 is valid.
    virtual int kv_cache_slots_activate(int slot)
    {
        return slot == 0 ? 0 : -1;
    }

    int kv_cache_slots_count() const { return kv_num_slots_; }
    int kv_cache_active_slot() const { return kv_active_slot_; }
};
