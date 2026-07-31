#pragma once
#include "ax_model_runner.hpp"
#include <vector>

struct ax_runner_ax650_handle_t;

struct ax_runner_shared_patch_t
{
    unsigned int offset = 0;
    unsigned int size = 0;
    unsigned int patch_offset = 0;
};

class ax_runner_ax650 : public ax_runner_base
{
protected:
    struct ax_runner_ax650_handle_t *m_handle = nullptr;
    std::vector<char> m_model_buffer;
    std::vector<char> m_shared_base_model_buffer;
    std::vector<char> m_shared_base_weight_buffer;
    std::vector<char> m_shared_adapter_weight_buffer;
    std::vector<ax_runner_shared_patch_t> m_shared_adapter_patches;
    int sub_init();

public:
    ax_runner_ax650() = default;
    virtual ~ax_runner_ax650() { deinit(); }

    // devid 对 AX650 片上后端无意义，忽略（保留参数以满足统一接口）
    int init(const char *model_file, int devid = -1) override;
    int init(char *model_buffer, size_t model_size, int devid = -1) override;
    int init_with_shared_weight_bundle(const char *model_file,
                                       const char *shared_base_model_file,
                                       const char *shared_base_weight_file,
                                       const char *shared_adapter_weight_file,
                                       const char *shared_adapter_patch_manifest_file,
                                       const char *task_id,
                                       int devid = -1) override;
    int switch_shared_weight_task(const char *shared_adapter_weight_file,
                                  const char *shared_adapter_patch_manifest_file,
                                  const char *task_id) override;

    int load_lora_adapter(const char *lora_bin_file);

    void deinit() override;

    bool has_handle_loaded() const;
    int load_handle_reuse_io(const char *model_file, int devid = -1);
    void unload_handle_keep_io();

    int inference() override;
    int inference(int grpid) override;

    size_t kv_cache_slots_prepare() override;
    int kv_cache_slots_alloc(int num_slots) override;
    int kv_cache_slots_set_count(int n) override;
    int kv_cache_slots_activate(int slot) override;

protected:
    // Restore slot 0 binding and free slots 1.. (owned) device buffers.
    void kv_cache_slots_release();
};
