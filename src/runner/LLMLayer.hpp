#pragma once

#include <string>
#include <vector>

#include "utils/memory_utils.hpp"  // MMap

// Backend runner type (compile-time selected). Hoisted out of LLM.cpp so LLMLayer
// (and the KV-slot manager that holds a std::vector<LLMLayer>&) can live in headers.
#ifdef USE_AXCL
#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "utils/axcl_manager.h"
using ax_runner_t = ax_runner_axcl;
#else
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include <ax_sys_api.h>
using ax_runner_t = ax_runner_ax650;
#endif

// One transformer layer: its engine handle plus the (optionally mmap'd) weight
// buffer backing it.
struct LLMLayer {
    ax_runner_t layer;
    std::string filename;
    MMap layer_buffer;
    std::vector<char> layer_buffer_vec;
};
