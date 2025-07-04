#pragma once
#include "ax_model_runner.hpp"

class ax_runner_ax650_host : public ax_runner_base
{
protected:
    struct ax_joint_runner_ax650_host_handle_t *m_handle = nullptr;

    bool _parepare_io = false;

    int sub_init();

public:
    int init(const char *model_file, bool use_mmap = false) override;
    int init(char *model_buffer, size_t model_size) override;

    void release();
    void deinit() override;

    int get_algo_width() override;
    int get_algo_height() override;
    ax_color_space_e get_color_space() override;

    int inference(ax_image_t *pstFrame) override;
    int inference() override;
};