#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "LLM.hpp"              // LLMAttrType
#include "VLMType.hpp"          // VLMType
#include "utils/logger.hpp"     // ALOGE / ALOGI
#include "ax_cmm_utils.hpp"     // get_remaining_cmm_size / estimate_model_mb / mem_guard_allow_load
#ifdef USE_AXCL
#include "utils/axcl_manager.h" // axcl_GetCMMRemain
#endif

// Pre-load memory guard: CMM/DDR accounting, the running (measurement-based) load
// guard extrapolation + its state, and the CMM teardown-balance sentry.
//
// Extracted from LLM::Impl (behavior-preserving). This owns the CMM *measurement*
// and *extrapolation* engine plus its state; the topology-aware orchestration
// (mem_preflight, building the per-layer device list for running_guard_init,
// surfacing user-facing errors via set_last_error) stays in LLM::Impl and calls
// into this. Holds a reference to the owner's LLMAttrType (lives as long as Impl).
class MemGuard
{
public:
    explicit MemGuard(const LLMAttrType &attr) : attr_(attr) {}

    // ---- CMM remaining queries ----
    int device_remaining_cmm_mb(int devid) const
    {
#ifdef USE_AXCL
        return axcl_GetCMMRemain(devid);
#else
        (void)devid;
        return get_remaining_cmm_size();
#endif
    }

    int cmm_total_remaining_mb() const
    {
#ifdef USE_AXCL
        int total = 0;
        for (int d : attr_.dev_ids) { int r = device_remaining_cmm_mb(d); if (r > 0) total += r; }
        return total;
#else
        return device_remaining_cmm_mb(-1);
#endif
    }

    // ---- CMM teardown-balance sentry (automated leak detection) ----
    // Invariant: after Deinit() the model must return all CMM it took, so free CMM
    // should return to the value captured at Init start. A persistent shortfall means
    // a leak. Warns only; disable with AXLLM_CMM_SENTRY=0.
    static bool cmm_sentry_enabled()
    {
        const char *v = std::getenv("AXLLM_CMM_SENTRY");
        return !(v && v[0] == '0');
    }

    // Capture the free-CMM baseline at Init start (no-op when the sentry is disabled).
    void capture_sentry_baseline()
    {
        if (cmm_sentry_enabled()) cmm_sentry_baseline_mb_ = cmm_total_remaining_mb();
    }

    void CheckCmmBalance(const char *tag) const
    {
        if (cmm_sentry_baseline_mb_ < 0) return; // no baseline -> nothing to check
        const int now = cmm_total_remaining_mb();
        if (now <= 0) return;                    // query unavailable (non-AX host) -> skip
        const int leaked = cmm_sentry_baseline_mb_ - now;
        if (leaked > kCmmSentryMarginMb)
            ALOGE("[cmm-sentry] %s: %d MB CMM not reclaimed after teardown (baseline %d MB -> now %d MB) -- suspected leak",
                  tag, leaked, cmm_sentry_baseline_mb_, now);
        else
            ALOGI("[cmm-sentry] %s: CMM balanced (baseline %d MB, now %d MB)", tag, cmm_sentry_baseline_mb_, now);
    }

    // ---- Running (measurement-based) load guard ----
    // The file-size pre-flight under-counts CMM because the engine also allocates
    // each layer's KV/IO buffers at load time. During the layer load we measure
    // ACTUAL per-device CMM consumption and extrapolate to the not-yet-loaded layers
    // + post/encoder tail, aborting *before* an allocation that would breach the floor.
    struct GuardVerdict { bool ok = true; bool warned = false; bool confident = false; std::string what; int projected = 0; int remain = -1; };

    // `devid_of_layer[i]` = device id layer i loads on (Impl builds it via layer_devid_for).
    // Must have attr_.axmodel_num entries when the guard is enabled.
    void running_guard_init(const std::vector<int> &devid_of_layer)
    {
        rl_baseline_mb_.clear();
        rl_total_.clear();
        rl_tail_mb_.clear();
        guard_settled_.clear();
        if (!attr_.mem_guard_enable || attr_.axmodel_num <= 0) return;
        for (int i = 0; i < attr_.axmodel_num && i < (int)devid_of_layer.size(); ++i)
            rl_total_[devid_of_layer[i]] += 1;
        for (const auto &kv : rl_total_)
            rl_baseline_mb_[kv.first] = device_remaining_cmm_mb(kv.first);
        // tail: post head loads on the last layer's device; vision/audio encoders on the first.
        rl_tail_mb_[devid_of_layer[attr_.axmodel_num - 1]] += estimate_model_mb(attr_.filename_post_axmodel);
        if (attr_.vlm_type != VLMType::None)
        {
            const int front_dev = devid_of_layer[0];
            rl_tail_mb_[front_dev] += estimate_model_mb(attr_.filename_image_encoder_axmodel);
            rl_tail_mb_[front_dev] += estimate_model_mb(attr_.filename_audio_encoder_axmodel_5s);
            rl_tail_mb_[front_dev] += estimate_model_mb(attr_.filename_audio_encoder_axmodel_30s);
        }
    }

    bool is_guard_settled(int devid) const { return guard_settled_.count(devid) != 0; }
    void mark_guard_settled(int devid) { guard_settled_[devid] = 1; }

    // Pure (no logging / no shared-state writes -> safe to call from loader threads).
    // Call BEFORE initializing a layer on `devid`; `loaded_on_dev` = layers already
    // loaded on this device. verdict.ok=false means abort.
    GuardVerdict running_guard_eval(int devid, int loaded_on_dev) const
    {
        GuardVerdict v;
        // Test-only seam: force a deterministic mid-load abort after N layers so the
        // teardown / CMM-reclaim path can be regression-tested (tools/test_teardown_cmm_leak.sh).
        if (const char *ta = std::getenv("AXLLM_TEST_ABORT_AFTER_LAYER"))
        {
            const int n = std::atoi(ta);
            if (n >= 0 && loaded_on_dev >= n) { v.ok = false; v.what = "forced test abort (AXLLM_TEST_ABORT_AFTER_LAYER)"; return v; }
        }
        auto get = [](const std::map<int, int> &m, int k) { auto it = m.find(k); return it == m.end() ? 0 : it->second; };
        if (!attr_.mem_guard_enable || loaded_on_dev <= 0) return v;
        auto bit = rl_baseline_mb_.find(devid);
        if (bit == rl_baseline_mb_.end()) return v;
        const int baseline = bit->second;
        const int cur = device_remaining_cmm_mb(devid);
        v.remain = cur;
        if (cur < 0 || baseline < 0) return v;
        const int used = baseline - cur;
        if (used <= 0) return v; // no measurable consumption yet -> can't project
        // Hard safety net: never let a load drive remaining CMM below the floor,
        // regardless of projection or sample count -- this is the real OOM guard.
        if (cur < attr_.mem_guard_floor_mb)
        {
            char whatf[192];
            std::snprintf(whatf, sizeof(whatf), "device %d CMM below floor (remain %d MB < floor %d MB)",
                          devid, cur, attr_.mem_guard_floor_mb);
            v.what = whatf;
            if (attr_.mem_guard_on_unsafe == "warn") { v.warned = true; return v; }
            v.ok = false;
            return v;
        }
        // A single measured layer (loaded==1) is dominated by one-time / shared
        // allocations and grossly over-projects. Require a few samples before trusting
        // the extrapolation; the hard-floor check above still protects meanwhile.
        if (loaded_on_dev < kGuardMinSamples) return v;
        const double per_layer = (double)used / loaded_on_dev;
        const int left = std::max(0, get(rl_total_, devid) - loaded_on_dev);
        const int tail = (int)(get(rl_tail_mb_, devid) * 1.15 + 0.5);
        v.projected = (int)(per_layer * left + 0.5) + tail;
        const int headroom = cur - v.projected;
        if (headroom >= attr_.mem_guard_floor_mb)
        {
            // Comfortably safe -> let the caller stop re-checking (avoids a slow
            // per-layer CMM query for the rest of the load, esp. on AXCL).
            if (headroom >= attr_.mem_guard_floor_mb + kGuardSettleMarginMb) v.confident = true;
            return v;
        }
        char what[192];
        std::snprintf(what, sizeof(what), "device %d CMM (%d more layers @~%.0f MB + tail %d MB, measured)",
                      devid, left, per_layer, tail);
        v.what = what;
        if (attr_.mem_guard_on_unsafe == "warn") { v.warned = true; return v; } // warn -> proceed
        v.ok = false; // abort / prompt -> abort mid-load
        return v;
    }

private:
    const LLMAttrType &attr_;

    int cmm_sentry_baseline_mb_ = -1;              // free CMM captured at Init start (teardown-balance sentry)
    static constexpr int kCmmSentryMarginMb = 16;  // tolerate fragmentation / rounding

    std::map<int, int> rl_baseline_mb_; // per-device remaining CMM before the loop
    std::map<int, int> rl_total_;       // per-device layer count
    std::map<int, int> rl_tail_mb_;     // per-device post/encoder file MB (loaded after layers)
    std::map<int, int> guard_settled_;  // devices where the measured guard confidently passed -> stop re-checking
    static constexpr int kGuardMinSamples = 4;      // layers to load before trusting the extrapolation
    static constexpr int kGuardSettleMarginMb = 512; // headroom above floor at which we stop re-checking
};
