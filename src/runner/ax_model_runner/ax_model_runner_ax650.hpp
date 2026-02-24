#pragma once
#include "ax_model_runner.hpp"

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

    int inference() override;
    int inference(int grpid) override;
};
