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

#include <cstring>
#ifdef USE_AXCL
#include <axcl.h>
#endif

// Backend memory ops + tensor addr/devid macros (moved out of LLM.cpp so headers
// like KvSlotManager.hpp that touch device KV can use them).
#ifdef USE_AXCL
static inline void llm_memset(void *phy, int val, size_t n, int devid) { axcl_Memset(phy, (uint8_t)val, n, devid); }
static inline void llm_h2d(void *phy_dst, const void *src, size_t n, int devid) { axcl_Memcpy(phy_dst, src, n, AXCL_MEMCPY_HOST_TO_DEVICE, devid); }
static inline void llm_d2h(void *dst, const void *phy_src, size_t n, int devid) { axcl_Memcpy(dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_HOST, devid); }
static inline void llm_d2d(void *phy_dst, const void *phy_src, size_t n, int devid) { axcl_Memcpy(phy_dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_DEVICE, devid); }
#define LLM_WADDR(t)      ((void *)(t).phyAddr)
#define LLM_RADDR(t)      ((const void *)(t).phyAddr)
#define LLM_DEVID(layer_obj) ((layer_obj).layer.get_devid())
#else
static inline void llm_memset(void *vir, int val, size_t n, int /*devid*/) { memset(vir, val, n); }
static inline void llm_h2d(void *vir_dst, const void *src, size_t n, int /*devid*/) { memcpy(vir_dst, src, n); }
static inline void llm_d2h(void *dst, const void *vir_src, size_t n, int /*devid*/) { memcpy(dst, vir_src, n); }
static inline void llm_d2d(void *vir_dst, const void *vir_src, size_t n, int /*devid*/) { memcpy(vir_dst, vir_src, n); }
#define LLM_WADDR(t)      ((t).pVirAddr)
#define LLM_RADDR(t)      ((const void *)(t).pVirAddr)
#define LLM_DEVID(layer_obj) (0)
#endif
