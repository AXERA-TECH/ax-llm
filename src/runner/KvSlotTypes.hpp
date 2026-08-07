#pragma once

#include <cstdint>
#include <vector>

#include "BaseTokenizer.hpp"  // Content

// Value types for the multi-slot prefix KV cache, hoisted verbatim out of
// LLM::Impl so the KV-slot selection policy + manager (and their host-side unit
// tests) can live in their own translation units. Pure data -- no engine/backend
// dependency. Impl still owns the *active* instances of this state.

// Snapshot of a linear-attention layer's recurrent K/V state at a token length,
// used to roll the host-side cache back on a partial reuse.
struct LinearStateSnapshot {
    int token_len = 0;
    std::vector<std::vector<unsigned short>> k;
    std::vector<std::vector<unsigned short>> v;
};

// Where a slot's K/V lives: Device (zero-copy pointer switch) or Host (DDR copy).
enum class KvSlotLocation { Device, Host };

// One prefix-KV slot: the host-side context state mirroring a conversation. The
// device-side K/V lives in a per-slot device buffer managed by the backend
// (zero-copy activate); in host mode host_k/host_v hold the swapped-out copy.
struct KvCacheSlot {
    bool used = false;
    std::vector<Content> last_history_snapshot;
    std::vector<int> last_tokens_ids;
    int precompute_len = 0;
    std::vector<LinearStateSnapshot> linear_state_snapshots;
    int cached_mrope_next_pos = -1;
    std::vector<unsigned char> full_cache_valid_slots;
    bool full_cache_has_sparse_slots = false;
    uint64_t lru = 0;
    // Host-mode only: per-layer host copy of this slot's device K/V. Swapped
    // in/out of the single engine KV buffer on activation.
    std::vector<std::vector<unsigned short>> host_k, host_v;
};

// A used slot is reused when it shares at least this many leading tokens with the
// request (covers same-system-prompt requests; avoids clobbering a slot for only
// the few chat-template header tokens every request trivially shares).
inline constexpr int kSlotReuseMinPrefix = 8;

// Two chat-history entries are "the same" iff role, type and payload all match.
// Used for history-prefix reuse (VLM slot selection) and prefix checks.
inline bool same_history_content(const Content &lhs, const Content &rhs) {
    return lhs.role == rhs.role &&
           lhs.type == rhs.type &&
           lhs.data == rhs.data;
}
