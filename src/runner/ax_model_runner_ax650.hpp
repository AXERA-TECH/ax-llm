#pragma once
#include "ax_model_runner.hpp"

class ax_runner_ax650 : public ax_runner_base
{
protected:
    struct ax_joint_runner_ax650_handle_t *m_handle = nullptr;

public:
    int init(const char *model_file) override;

    void deinit() override;

    int get_algo_width() override;
    int get_algo_height() override;
    ax_color_space_e get_color_space() override;

    int inference(ax_image_t *pstFrame) override;
    int inference() override;
};