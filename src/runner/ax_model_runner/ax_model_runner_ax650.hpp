#pragma once
#include "ax_model_runner.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

struct ax_runner_ax650_handle_t;

class ax_runner_ax650 : public ax_runner_base
{
protected:
    struct ax_runner_ax650_handle_t *m_handle = nullptr;
    int sub_init();

public:
    ax_runner_ax650() = default;
    virtual ~ax_runner_ax650() { deinit(); }

    // devid 对 AX650 片上后端无意义，忽略（保留参数以满足统一接口）
    int init(const char *model_file, int devid = -1) override;
    int init(char *model_buffer, size_t model_size, int devid = -1) override;

    void deinit() override;

    bool has_handle_loaded() const;
    int load_handle_reuse_io(const char *model_file, int devid = -1);
    void unload_handle_keep_io();

    int inference() override;
    int inference(int grpid) override;

    // Temporarily rebind one group input to caller-owned memory.  This is used
    // by chunked prefill to share invariant mask/indices tensors between layer
    // AXModels with identical graph ABIs.  The original binding is recorded and
    // can be restored by restore_group_input_aliases().
    int alias_group_input(int grpid, const char *name, void *vir, uint64_t phy);
    int restore_group_input_aliases(int grpid);

    size_t kv_cache_slots_prepare() override;
    int kv_cache_slots_alloc(int num_slots) override;
    int kv_cache_slots_set_count(int n) override;
    int kv_cache_slots_activate(int slot) override;

protected:
    int flush_cached_inputs(const std::vector<ax_runner_tensor_t> &tensors);
    int invalidate_cached_outputs(const std::vector<ax_runner_tensor_t> &tensors);

    // Addresses in this set are mapped cacheable.  K/V inputs absent from the
    // set are non-cached maximum-capacity tensors and skip the full-range flush;
    // cached outputs are invalidated selectively.
    std::unordered_set<unsigned long long> mcached_phy_addrs_;

    struct InputAliasEntry
    {
        std::string name;
        uint64_t phy;
        void *vir;
        uint32_t nsize;
    };
    std::map<int, std::vector<InputAliasEntry>> input_alias_orig_;

    // Restore slot 0 binding and free slots 1.. (owned) device buffers.
    void kv_cache_slots_release();
};
