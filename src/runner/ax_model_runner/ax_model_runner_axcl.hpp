#pragma once
#include "ax_model_runner.hpp"

class ax_runner_axcl : public ax_runner_base
{
protected:
    struct ax_joint_runner_ax650_handle_t *m_handle = nullptr;
    int group_count = 0;
    bool _parepare_io = false;

    int sub_init();
    int sub_init_reuse_io();

public:
    int init(const char *model_file, int devid) override;
    int init(char *model_buffer, size_t model_size, int devid) override;

    void deinit() override;

    bool has_handle_loaded() const;
    int load_handle_reuse_io(const char *model_file, int devid);
    void unload_handle_keep_io();

    int get_algo_width() override;
    int get_algo_height() override;
    ax_color_space_e get_color_space() override;

    int set_input(int grpid, int idx, unsigned long long int phy_addr, unsigned long size);
    int set_output(int grpid, int idx, unsigned long long int phy_addr, unsigned long size);

    int set_input(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size);
    int set_output(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size);

    // int inference(ax_image_t *pstFrame) override;
    int inference() override;
    int inference(int grpid) override;

    int kv_cache_slots_init(int num_slots) override;
    int kv_cache_slots_activate(int slot) override;

protected:
    // Restore slot 0 binding and free slots 1.. (owned) device buffers.
    void kv_cache_slots_release();
};
