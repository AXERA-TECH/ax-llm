#pragma once
#include "ax_model_runner.hpp"

// 前置声明，隐藏实现细节
struct ax_runner_ax650_handle_t;

class ax_runner_ax650 : public ax_runner_base
{
protected:
    struct ax_runner_ax650_handle_t *m_handle = nullptr;
    int sub_init();

public:
    ax_runner_ax650() = default;
    virtual ~ax_runner_ax650() { deinit(); } // 析构时自动释放

    int init(const char *model_file, bool use_mmap = false) override;
    int init(char *model_buffer, size_t model_size) override;

    void deinit() override;

    int inference() override;
    int inference(int grpid) override;
};