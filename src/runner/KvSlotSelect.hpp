#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "KvSlotTypes.hpp"    // KvCacheSlot, kSlotReuseMinPrefix, same_history_content
#include "BaseTokenizer.hpp"  // Content

// Pure multi-slot prefix-KV selection policy, extracted verbatim from LLM::Impl's
// select_kv_slot / select_kv_slot_by_history so the (bug-prone) 4-strategy + LRU
// decision can be host-unit-tested with NO engine/NPU (see tools/kv_slot_select_test.cpp).
// The caller applies the decision via commit_kv_slot_choice().

struct SlotDecision {
    int chosen = 0;   // slot index to serve this request from
    bool fresh = true; // true => reset that slot for a full prefill; false => reuse its prefix
    int shared = 0;    // matched prefix length (tokens) / matched history turns
};

// Number of matching leading elements (== diff_token_ids offset semantics).
inline int common_prefix_len(const std::vector<int> &a, const std::vector<int> &b) {
    const int n = (int)std::min(a.size(), b.size());
    int i = 0;
    while (i < n && a[(size_t)i] == b[(size_t)i]) ++i;
    return i;
}

// Token-based selection (text path). Choose the used slot with the longest leading
// token prefix; else a free slot; else salvage any overlap; else evict LRU. If a
// would-be reuse fits the cheap (history-less) prefill group, prefer a fresh prefill
// (faster than reuse via the wider history group). cheap_cap<=0 disables that path.
inline SlotDecision decide_slot_by_tokens(const std::vector<KvCacheSlot> &slots,
                                          const std::vector<int> &sel_tokens,
                                          int cheap_cap) {
    int best = -1, best_off = 0;
    uint64_t best_lru = 0;
    int free_slot = -1;
    int lru_victim = 0;
    uint64_t lru_min = 0;
    bool lru_init = false;
    for (int i = 0; i < (int)slots.size(); ++i) {
        const auto &s = slots[(size_t)i];
        if (!s.used) {
            if (free_slot < 0) free_slot = i;
            continue;
        }
        if (!lru_init || s.lru < lru_min) { lru_min = s.lru; lru_victim = i; lru_init = true; }
        const int off = common_prefix_len(s.last_tokens_ids, sel_tokens);
        if (off > best_off || (off == best_off && s.lru > best_lru)) {
            best_off = off; best = i; best_lru = s.lru;
        }
    }

    SlotDecision d;
    d.shared = best_off;
    if (best >= 0 && best_off >= kSlotReuseMinPrefix) { d.chosen = best; d.fresh = false; }
    else if (free_slot >= 0) { d.chosen = free_slot; d.fresh = true; }
    else if (best >= 0 && best_off > 0) { d.chosen = best; d.fresh = false; }
    else { d.chosen = lru_victim; d.fresh = true; }

    if (!d.fresh && best >= 0 && cheap_cap > 0 && (int)sel_tokens.size() <= cheap_cap)
        d.fresh = true;
    return d;
}

// History-based selection (VLM path: tokenizing VLM history needs the expensive
// vision Prepare, so match on the chat-history Content prefix instead). Used by
// Qwen*-VL / MiniCPM-V etc. Share at least the system turn to reuse.
inline SlotDecision decide_slot_by_history(const std::vector<KvCacheSlot> &slots,
                                           const std::vector<Content> &history) {
    int best = -1, best_common = 0;
    uint64_t best_lru = 0;
    int free_slot = -1, lru_victim = 0;
    uint64_t lru_min = 0;
    bool lru_init = false;
    for (int i = 0; i < (int)slots.size(); ++i) {
        const auto &s = slots[(size_t)i];
        if (!s.used) {
            if (free_slot < 0) free_slot = i;
            continue;
        }
        if (!lru_init || s.lru < lru_min) { lru_min = s.lru; lru_victim = i; lru_init = true; }
        const auto &h = s.last_history_snapshot;
        const int n = (int)std::min(h.size(), history.size());
        int common = 0;
        while (common < n && same_history_content(h[(size_t)common], history[(size_t)common])) ++common;
        if (common > best_common || (common == best_common && s.lru > best_lru)) {
            best_common = common; best = i; best_lru = s.lru;
        }
    }

    SlotDecision d;
    d.shared = best_common;
    if (best >= 0 && best_common >= 1) { d.chosen = best; d.fresh = false; }
    else if (free_slot >= 0) { d.chosen = free_slot; d.fresh = true; }
    else if (best >= 0 && best_common > 0) { d.chosen = best; d.fresh = false; }
    else { d.chosen = lru_victim; d.fresh = true; }
    return d;
}
