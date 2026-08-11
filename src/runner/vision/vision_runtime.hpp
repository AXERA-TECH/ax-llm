#pragma once

// Backend-selected vision/audio encoder runner type + device-copy primitives.
// Shared by vision_module.cpp (image/video encoders) and audio_encoder.cpp (audio encoder)
// so both use the same ax_runner_t + host<->device copy under AXCL / on-chip AX650.

#include <cstddef>
#include <cstring>

#ifdef USE_AXCL
#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "utils/axcl_manager.h"
using ax_runner_t = ax_runner_axcl;
static inline void v_h2d(void *phy_dst, const void *src, size_t n, int devid) { axcl_Memcpy(phy_dst, src, n, AXCL_MEMCPY_HOST_TO_DEVICE, devid); }
static inline void v_d2h(void *dst, const void *phy_src, size_t n, int devid) { axcl_Memcpy(dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_HOST, devid); }
#define V_WADDR(t) ((void *)(t).phyAddr)
#define V_RADDR(t) ((const void *)(t).phyAddr)
#else
#include "ax_model_runner/ax_model_runner_ax650.hpp"
using ax_runner_t = ax_runner_ax650;
static inline void v_h2d(void *vir_dst, const void *src, size_t n, int /*devid*/) { memcpy(vir_dst, src, n); }
static inline void v_d2h(void *dst, const void *vir_src, size_t n, int /*devid*/) { memcpy(dst, vir_src, n); }
#define V_WADDR(t) ((t).pVirAddr)
#define V_RADDR(t) ((const void *)(t).pVirAddr)
#endif
