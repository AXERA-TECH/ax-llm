#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/sysinfo.h>
#endif

#include "LLM.hpp"           // LLMAttrType, Content
#include "LLMLayer.hpp"      // LLMLayer, ax_runner_t, llm_h2d/llm_d2h, LLM_RADDR/WADDR/DEVID
#include "KvSlotTypes.hpp"   // KvCacheSlot, KvSlotLocation, kSlotReuseMinPrefix
#include "KvSlotSelect.hpp"  // SlotDecision, decide_slot_by_tokens / decide_slot_by_history
#include "VLMType.hpp"       // VLMType, VLMTypeName
#include "utils/logger.hpp"  // ALOGE / ALOGW / ALOGI

// The KvSlotManager owns the multi-slot prefix-KV state machine (extracted from
// LLM::Impl). It needs a handful of Impl-owned bits -- the per-conversation decode
// state it snapshots per slot + a few topology/CMM helpers -- exposed via this
// narrow interface. Distinct slot_* names deliberately avoid clashing with (and
// thus avoid making virtual) Impl's existing hot-path methods; Impl implements
// each as a one-line forwarder.
struct IKvSlotHost {
    virtual ~IKvSlotHost() = default;
    virtual int  slot_decode_gid_for_layer(int layer_idx, int requested_gid) const = 0;
    virtual bool slot_is_linear_layer(int layer_idx) const = 0;
    virtual int  slot_kv_cache_size_for_layer(int layer_idx) const = 0;
    virtual int  slot_layer_devid_for(int layer_idx) const = 0;
    virtual int  slot_cheap_prefill_capacity() const = 0;
    virtual int  slot_remaining_cmm_mb(int devid) const = 0;
    virtual bool slot_dynamic_layer_load_enabled() const = 0;
    virtual int  slot_decode_grpid() const = 0;
    virtual int  slot_decode_grpids_back() const = 0; // decode_grpids_.empty()?0:back()
    virtual int  slot_precompute_len() const = 0;
    virtual void slot_reset_kv_cache() = 0;
    virtual void slot_capture_decode_state(KvCacheSlot &s) = 0;       // Impl's 7 decode-state members -> slot
    virtual void slot_restore_decode_state(const KvCacheSlot &s) = 0; // slot -> Impl's 7 decode-state members
};

class KvSlotManager {
public:
    KvSlotManager(IKvSlotHost &host, std::vector<LLMLayer> &layers, const LLMAttrType &attr)
        : host_(host), layers_(layers), attr_(attr) {}

    bool multi_slot_enabled() const { return multi_slot_enabled_; }
    bool multi_slot_active_request() const { return multi_slot_active_request_; }

    // ---- Multi-slot prefix KV cache management ----
    // Called from Init() after layers/groups are ready. Estimates how many slots
    // actually fit before allocating, reduces to what fits, and warns when the
    // configured count cannot be satisfied. Safe no-op when slots<=1.
    void init_kv_slots()
    {
        multi_slot_enabled_ = false;
        multi_slot_active_request_ = false;
        kv_slots_.clear();
        kv_active_slot_idx_ = 0;
        kv_slot_lru_counter_ = 0;

        const int want = attr_.kv_cache_slots;
        if (want <= 1) return;

        if (host_.slot_dynamic_layer_load_enabled())
        {
            ALOGW("kv_cache_slots=%d ignored: not supported together with dynamic_load_enable yet", want);
            return;
        }

        std::string loc = attr_.kv_cache_slot_location;
        std::transform(loc.begin(), loc.end(), loc.begin(), ::tolower);
        kv_slot_location_ = (loc == "host" || loc == "ddr") ? KvSlotLocation::Host : KvSlotLocation::Device;
        if (loc != "host" && loc != "ddr" && loc != "device")
            ALOGW("kv_cache_slot_location='%s' unknown; defaulting to 'device'", attr_.kv_cache_slot_location.c_str());

        const std::string vlm_tag = attr_.vlm_type == VLMType::None ? std::string("no") : std::string(VLMTypeName(attr_.vlm_type));
        int granted = want;

        if (kv_slot_location_ == KvSlotLocation::Device)
        {
            // 1) Build per-layer slot metadata and sum per-slot device bytes by device.
            std::map<int, size_t> per_dev_bytes;
            for (int i = 0; i < attr_.axmodel_num; ++i)
            {
                const size_t b = layers_[i].layer.kv_cache_slots_prepare();
                if (b == 0)
                {
                    ALOGE("layer %d has no slottable KV tensors; disabling multi-slot", i);
                    for (int j = 0; j <= i; ++j) layers_[j].layer.kv_cache_slots_set_count(1);
                    return;
                }
                per_dev_bytes[host_.slot_layer_devid_for(i)] += b;
            }

            // 2) Budget: keep a safety margin of free CMM for runtime allocations.
            //    Honor mem_guard_floor_mb when it asks for a larger reserve (never smaller).
            const long long floor_b = attr_.mem_guard_enable ? (long long)attr_.mem_guard_floor_mb * 1024 * 1024 : 0;
            const long long margin = std::max(256LL * 1024 * 1024, floor_b);
            for (const auto &kv : per_dev_bytes)
            {
                const int dev = kv.first;
                const long long per_slot = (long long)kv.second;
                const long long free_b = (long long)host_.slot_remaining_cmm_mb(dev) * 1024LL * 1024LL;
                const long long usable = free_b - margin;
                int fit = 1;
                if (per_slot > 0 && usable > 0) fit = 1 + (int)(usable / per_slot);
                ALOGI("kv slot budget: dev=%d per_slot=%lldMB free=%lldMB margin=256MB -> max_slots=%d",
                      dev, per_slot >> 20, free_b >> 20, fit);
                granted = std::min(granted, fit);
            }
            if (granted < 1) granted = 1;
            if (granted < want)
                ALOGW("⚠ kv_cache_slots=%d requested but device CMM only fits %d; reducing to %d", want, granted, granted);
            if (granted <= 1)
            {
                ALOGW("⚠ kv_cache_slots disabled: device CMM cannot fit even one extra slot");
                for (int i = 0; i < attr_.axmodel_num; ++i) layers_[i].layer.kv_cache_slots_set_count(1);
                return;
            }

            // 3) Allocate the budgeted count; take the min actually achieved
            //    (fragmentation may yield less than the estimate), then trim all
            //    layers to that common count.
            int achieved = granted;
            for (int i = 0; i < attr_.axmodel_num; ++i)
                achieved = std::min(achieved, layers_[i].layer.kv_cache_slots_alloc(granted));
            if (achieved < granted)
                ALOGW("⚠ kv_cache_slots: allocated %d of estimated %d (CMM fragmentation); using %d", achieved, granted, achieved);
            for (int i = 0; i < attr_.axmodel_num; ++i) layers_[i].layer.kv_cache_slots_set_count(achieved);
            granted = achieved;
            if (granted <= 1)
            {
                ALOGW("⚠ kv_cache_slots disabled after allocation (only 1 slot available)");
                return;
            }
        }
        else // Host mode: one device buffer + N host copies; bound by host RAM.
        {
            size_t per_slot = 0;
            const int gid0 = host_.slot_decode_grpids_back();
            for (int i = 0; i < attr_.axmodel_num; ++i)
            {
                const int gid = host_.slot_decode_gid_for_layer(i, gid0);
                per_slot += (size_t)layers_[i].layer.get_input(gid, "K_cache").nSize;
                per_slot += (size_t)layers_[i].layer.get_input(gid, "V_cache").nSize;
            }
            long long free_ram = 0;
#ifndef _WIN32
            struct sysinfo si;
            if (sysinfo(&si) == 0) free_ram = (long long)si.freeram * (long long)si.mem_unit;
#endif
            const long long host_floor_b = attr_.mem_guard_enable ? (long long)attr_.mem_guard_floor_mb * 1024 * 1024 : 0;
            const long long margin = std::max(512LL * 1024 * 1024, host_floor_b);
            int fit = want;
            if (free_ram > 0 && per_slot > 0)
            {
                const long long usable = free_ram - margin;
                fit = 1;
                if (usable > 0) fit = 1 + (int)(usable / (long long)per_slot);
            }
            ALOGI("kv slot budget (host): per_slot<=%zuMB free_ram=%lldMB -> max_slots=%d",
                  per_slot >> 20, free_ram >> 20, fit);
            granted = std::min(want, fit);
            if (granted < 1) granted = 1;
            if (granted < want)
                ALOGW("⚠ kv_cache_slots=%d requested but host RAM only fits ~%d; reducing to %d", want, granted, granted);
            if (granted <= 1)
            {
                ALOGW("⚠ kv_cache_slots disabled: host RAM cannot fit even one extra slot");
                return;
            }
        }

        kv_slots_.resize(granted);
        kv_active_slot_idx_ = 0;
        multi_slot_enabled_ = true;
        if (granted < want)
            ALOGW("multi-slot prefix KV cache: %d/%d slots enabled (location=%s vlm=%s)", granted, want,
                  kv_slot_location_ == KvSlotLocation::Host ? "host" : "device", vlm_tag.c_str());
        else
            ALOGI("multi-slot prefix KV cache enabled: slots=%d location=%s vlm=%s", granted,
                  kv_slot_location_ == KvSlotLocation::Host ? "host" : "device", vlm_tag.c_str());
    }

    // Pick the slot to serve `sel_tokens` (token path). See KvSlotSelect.hpp.
    void select_kv_slot(const std::vector<int> &sel_tokens)
    {
        multi_slot_active_request_ = false;
        if (!multi_slot_enabled_) return;
        multi_slot_active_request_ = true;
        const SlotDecision d = decide_slot_by_tokens(kv_slots_, sel_tokens, host_.slot_cheap_prefill_capacity());
        commit_kv_slot_choice(d.chosen, d.fresh, d.shared, (int)sel_tokens.size());
    }

    // VLM slot selection: match on the chat history (Content) prefix instead.
    void select_kv_slot_by_history(const std::vector<Content> &history)
    {
        multi_slot_active_request_ = false;
        if (!multi_slot_enabled_) return;
        multi_slot_active_request_ = true;
        const SlotDecision d = decide_slot_by_history(kv_slots_, history);
        commit_kv_slot_choice(d.chosen, d.fresh, d.shared, (int)history.size());
    }

    void save_active_kv_slot()
    {
        if (!multi_slot_enabled_) return;
        if (kv_active_slot_idx_ < 0 || kv_active_slot_idx_ >= (int)kv_slots_.size()) return;
        auto &s = kv_slots_[(size_t)kv_active_slot_idx_];
        host_.slot_capture_decode_state(s);
        s.used = (host_.slot_precompute_len() > 0) || !s.last_tokens_ids.empty();
        if (kv_slot_location_ == KvSlotLocation::Host && s.used)
            host_dump_active_kv();
    }

private:
    // Host mode: copy the active slot's device KV into its host store.
    void host_dump_active_kv()
    {
        if (kv_active_slot_idx_ < 0 || kv_active_slot_idx_ >= (int)kv_slots_.size()) return;
        auto &s = kv_slots_[(size_t)kv_active_slot_idx_];
        s.host_k.assign(attr_.axmodel_num, {});
        s.host_v.assign(attr_.axmodel_num, {});
        const int precompute_len = host_.slot_precompute_len();
        for (int m = 0; m < attr_.axmodel_num; ++m)
        {
            auto &lyr = layers_[(size_t)m];
            const int devid = LLM_DEVID(lyr);
            const int gid = host_.slot_decode_gid_for_layer(m, host_.slot_decode_grpid());
            auto &t_k = lyr.layer.get_input(gid, "K_cache");
            auto &t_v = lyr.layer.get_input(gid, "V_cache");
            size_t k_elems, v_elems;
            if (host_.slot_is_linear_layer(m))
            {
                k_elems = (size_t)t_k.nSize / sizeof(unsigned short);
                v_elems = (size_t)t_v.nSize / sizeof(unsigned short);
            }
            else
            {
                const size_t layer_kv = (size_t)host_.slot_kv_cache_size_for_layer(m);
                k_elems = v_elems = (size_t)std::max(0, precompute_len) * layer_kv;
            }
            s.host_k[(size_t)m].resize(k_elems);
            s.host_v[(size_t)m].resize(v_elems);
            if (k_elems) llm_d2h(s.host_k[(size_t)m].data(), LLM_RADDR(t_k), std::min(k_elems * sizeof(unsigned short), (size_t)t_k.nSize), devid);
            if (v_elems) llm_d2h(s.host_v[(size_t)m].data(), LLM_RADDR(t_v), std::min(v_elems * sizeof(unsigned short), (size_t)t_v.nSize), devid);
        }
    }

    // Host mode: load a slot's host KV into the (shared) device decode-group buffer.
    void host_load_kv(int idx)
    {
        auto &s = kv_slots_[(size_t)idx];
        if ((int)s.host_k.size() != attr_.axmodel_num) return; // fresh slot, nothing to load
        for (int m = 0; m < attr_.axmodel_num; ++m)
        {
            auto &lyr = layers_[(size_t)m];
            const int devid = LLM_DEVID(lyr);
            const int gid = host_.slot_decode_gid_for_layer(m, host_.slot_decode_grpid());
            auto &t_k = lyr.layer.get_input(gid, "K_cache");
            auto &t_v = lyr.layer.get_input(gid, "V_cache");
            if (!s.host_k[(size_t)m].empty())
                llm_h2d(LLM_WADDR(t_k), s.host_k[(size_t)m].data(), std::min(s.host_k[(size_t)m].size() * sizeof(unsigned short), (size_t)t_k.nSize), devid);
            if (!s.host_v[(size_t)m].empty())
                llm_h2d(LLM_WADDR(t_v), s.host_v[(size_t)m].data(), std::min(s.host_v[(size_t)m].size() * sizeof(unsigned short), (size_t)t_v.nSize), devid);
        }
    }

    bool activate_kv_slot(int idx)
    {
        if (!multi_slot_enabled_) return false;
        if (idx < 0 || idx >= (int)kv_slots_.size()) return false;
        if (kv_slot_location_ == KvSlotLocation::Device)
        {
            for (int i = 0; i < attr_.axmodel_num; ++i)
            {
                if (layers_[i].layer.kv_cache_slots_activate(idx) != 0)
                {
                    ALOGE("kv_cache_slots_activate(layer=%d slot=%d) failed", i, idx);
                    return false;
                }
            }
        }
        else
        {
            host_load_kv(idx); // copy this slot's host KV onto the single device buffer
        }
        host_.slot_restore_decode_state(kv_slots_[(size_t)idx]);
        kv_active_slot_idx_ = idx;
        return true;
    }

    void commit_kv_slot_choice(int chosen, bool fresh, int shared, int req_size)
    {
        if (chosen != kv_active_slot_idx_)
        {
            save_active_kv_slot();
            activate_kv_slot(chosen);
        }
        if (fresh)
        {
            host_.slot_reset_kv_cache();
            kv_slots_[(size_t)chosen].used = false;
            kv_slots_[(size_t)chosen].host_k.clear();
            kv_slots_[(size_t)chosen].host_v.clear();
        }
        kv_slots_[(size_t)chosen].lru = ++kv_slot_lru_counter_;
        ALOGI("kv slot select: chosen=%d reuse=%d shared=%d req=%d (slots=%zu)",
              chosen, fresh ? 0 : 1, fresh ? 0 : shared, req_size, kv_slots_.size());
    }

    IKvSlotHost &host_;
    std::vector<LLMLayer> &layers_;
    const LLMAttrType &attr_;

    std::vector<KvCacheSlot> kv_slots_;
    int kv_active_slot_idx_ = 0;
    bool multi_slot_enabled_ = false;
    bool multi_slot_active_request_ = false; // this request is served from a slot
    KvSlotLocation kv_slot_location_ = KvSlotLocation::Device;
    uint64_t kv_slot_lru_counter_ = 0;
};
