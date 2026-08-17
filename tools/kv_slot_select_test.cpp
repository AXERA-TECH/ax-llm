// Host unit test (no NPU/engine) for the multi-slot prefix-KV selection policy.
// Pins the 4-strategy + LRU decision that select_kv_slot / select_kv_slot_by_history
// delegate to. Runs in CI (links only the tokenizer, for the Content type).
#include <cstdio>
#include <string>
#include <vector>

#include "runner/KvSlotSelect.hpp"

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", (msg)); ++failures; } else { std::printf("ok  : %s\n", (msg)); } } while (0)

static KvCacheSlot tok_slot(std::vector<int> toks, uint64_t lru) {
    KvCacheSlot s; s.used = true; s.last_tokens_ids = std::move(toks); s.lru = lru; return s;
}
static KvCacheSlot hist_slot(std::vector<Content> h, uint64_t lru) {
    KvCacheSlot s; s.used = true; s.last_history_snapshot = std::move(h); s.lru = lru; return s;
}

int main() {
    // ---------- token path (text) ----------
    {   // all slots free -> first free, fresh
        std::vector<KvCacheSlot> slots(2);
        auto d = decide_slot_by_tokens(slots, {1, 2, 3}, 0);
        CHECK(d.chosen == 0 && d.fresh, "tok: all-free -> slot0 fresh");
    }
    std::vector<int> base(kSlotReuseMinPrefix + 2);
    for (size_t i = 0; i < base.size(); ++i) base[i] = (int)i;
    {   // long shared prefix (>= kSlotReuseMinPrefix) -> reuse
        std::vector<KvCacheSlot> slots{tok_slot(base, 5), KvCacheSlot{}};
        std::vector<int> req = base; req.push_back(999);
        auto d = decide_slot_by_tokens(slots, req, 0);
        CHECK(!d.fresh && d.chosen == 0 && d.shared == (int)base.size(), "tok: long prefix -> reuse slot0");
    }
    {   // short prefix (< min) but a free slot exists -> take the free slot, fresh
        std::vector<KvCacheSlot> slots{tok_slot({1, 2, 3}, 5), KvCacheSlot{}};
        auto d = decide_slot_by_tokens(slots, {1, 2, 777}, 0);
        CHECK(d.chosen == 1 && d.fresh, "tok: short prefix + free -> free slot fresh");
    }
    {   // slots full, no free, some overlap -> salvage best (reuse)
        std::vector<KvCacheSlot> slots{tok_slot({1, 2, 3}, 5), tok_slot({9, 9}, 6)};
        auto d = decide_slot_by_tokens(slots, {1, 2, 777}, 0);
        CHECK(!d.fresh && d.chosen == 0 && d.shared == 2, "tok: full + overlap -> salvage best");
    }
    {   // slots full, nothing shared -> evict LRU (lowest lru), fresh
        std::vector<KvCacheSlot> slots{tok_slot({1, 2}, 9), tok_slot({3, 4}, 2)};
        auto d = decide_slot_by_tokens(slots, {7, 8}, 0);
        CHECK(d.fresh && d.chosen == 1, "tok: full + no overlap -> LRU victim (slot1)");
    }
    {   // cheap-prefill fast path flips a would-be reuse to fresh
        std::vector<KvCacheSlot> slots{tok_slot(base, 5)};
        std::vector<int> req = base;
        auto d = decide_slot_by_tokens(slots, req, (int)req.size() + 5); // cheap_cap >= req
        CHECK(d.chosen == 0 && d.fresh, "tok: cheap-prefill -> fresh despite prefix");
    }
    {   // prefix tie broken toward higher lru (most-recent)
        std::vector<KvCacheSlot> slots{tok_slot(base, 2), tok_slot(base, 9)};
        std::vector<int> req = base;
        auto d = decide_slot_by_tokens(slots, req, 0);
        CHECK(!d.fresh && d.chosen == 1, "tok: prefix tie -> higher-lru slot");
    }

    // ---------- history path (VLM: Qwen*-VL / MiniCPM-V etc.) ----------
    const Content sys{SYSTEM, TEXT, std::string("sys")};
    const Content u_img{USER, IMAGE, std::string("img1")};
    const Content u_q{USER, TEXT, std::string("q2")};
    {   // share >= 1 history turn -> reuse
        std::vector<KvCacheSlot> slots{hist_slot({sys, u_img}, 5), KvCacheSlot{}};
        auto d = decide_slot_by_history(slots, {sys, u_img, u_q});
        CHECK(!d.fresh && d.chosen == 0 && d.shared == 2, "hist: shared prefix -> reuse (VLM)");
    }
    {   // no shared history + free slot -> fresh free
        std::vector<KvCacheSlot> slots{hist_slot({sys, u_img}, 5), KvCacheSlot{}};
        auto d = decide_slot_by_history(slots, {Content{SYSTEM, TEXT, std::string("other")}});
        CHECK(d.chosen == 1 && d.fresh, "hist: no overlap + free -> free fresh");
    }
    {   // slots full, no overlap -> LRU victim, fresh
        std::vector<KvCacheSlot> slots{hist_slot({sys}, 9), hist_slot({u_q}, 1)};
        auto d = decide_slot_by_history(slots, {Content{USER, TEXT, std::string("z")}});
        CHECK(d.fresh && d.chosen == 1, "hist: full + no overlap -> LRU victim");
    }

    if (failures == 0) { std::printf("\nkv_slot_select_test: ALL PASS\n"); return 0; }
    std::printf("\nkv_slot_select_test: %d FAILURE(S)\n", failures);
    return 1;
}
