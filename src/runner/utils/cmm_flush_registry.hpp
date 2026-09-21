#pragma once
// On-chip (AX650) CMM cache-maintenance registry.
//
// Engine IO tensors live in CACHED CMM memory (AX_SYS_MemAllocCached). The NPU
// is not cache-coherent with the CPU, so any CPU write into an input tensor
// must be cleaned (AX_SYS_MflushCache) before inference. Historically the
// runner flushed EVERY input tensor in FULL before EVERY inference
// (_auto_sync_before_inference) — for LLM decode that means re-flushing the
// multi-MB K_cache/V_cache blocks 2x per layer per token (~300 MB of DC-CVAC
// sweeps per generated token) even though the CPU only wrote one KV row.
//
// This registry lets the CPU-side write helpers (llm_h2d/llm_d2d/llm_memset in
// LLMLayer.hpp) flush exactly the bytes they wrote instead: the AX650 runner
// registers each allocated IO block (vir base, phy base, size); flush_written()
// resolves a written vir range back to its phy range and cleans just that.
// K_cache/V_cache are then excluded from the runner's full auto-flush
// (see ax_runner set_sync_skip_input / LLM_init).
//
// AXCL builds never register anything here (PCIe DMA has no CPU-cache
// interaction), so flush_written() degrades to a no-op lookup miss.
//
// Env escape hatch: AXLLM_LEGACY_FULL_SYNC=1 restores the historical
// full-flush behavior (LLM_init skips the skip-registration).

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <mutex>

#ifndef USE_AXCL
#include <ax_sys_api.h>
#endif

class CmmFlushRegistry
{
public:
    static CmmFlushRegistry &instance()
    {
        static CmmFlushRegistry inst;
        return inst;
    }

    void register_block(void *vir, unsigned long long phy, size_t size)
    {
        if (!vir || !phy || size == 0) return;
        std::lock_guard<std::mutex> lk(mtx_);
        blocks_[reinterpret_cast<std::uintptr_t>(vir)] = Block{phy, size};
    }

    void clear_all()
    {
        std::lock_guard<std::mutex> lk(mtx_);
        blocks_.clear();
    }

    void unregister_block(void *vir)
    {
        if (!vir) return;
        std::lock_guard<std::mutex> lk(mtx_);
        blocks_.erase(reinterpret_cast<std::uintptr_t>(vir));
    }

    // Invalidate the byte range [vir, vir+bytes) before a CPU read, so stale
    // cache lines (e.g. the zero-fill of a freshly allocated KV slot buffer, or
    // rows the NPU rewrote since) never shadow NPU-written data. The historical
    // full pre-inference MflushCache used to paper over this; with K/V_cache
    // excluded from it, every CPU read of those blocks must invalidate first.
    // Unknown ranges are ignored (ordinary host memory).
    void invalidate_read(const void *vir, size_t bytes)
    {
        if (!no_invalidate_) maintain(vir, bytes, false);
    }

    // Clean (flush) the CPU-written byte range [vir, vir+bytes) if it falls in
    // a registered cached CMM block. Unknown ranges (host heap, AXCL shadow
    // buffers) are silently ignored — the caller wrote ordinary host memory.
    void flush_written(const void *vir, size_t bytes) { maintain(vir, bytes, true); }

private:
    // Debug bisection knobs (read once): AXLLM_NO_REGISTRY=1 turns every
    // maintenance call into a no-op; AXLLM_NO_INVALIDATE=1 disables only the
    // read-side invalidate. Diagnostics only — not part of the contract.
    const bool no_registry_ = std::getenv("AXLLM_NO_REGISTRY") != nullptr;
    const bool no_invalidate_ = std::getenv("AXLLM_NO_INVALIDATE") != nullptr;

    void maintain(const void *vir, size_t bytes, bool clean)
    {
        if (no_registry_) return;
#ifndef USE_AXCL
        if (!vir || bytes == 0) return;
        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(vir);
        unsigned long long phy = 0;
        void *vbase = nullptr;
        size_t span = 0;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (blocks_.empty()) return;
            auto it = blocks_.upper_bound(addr);
            if (it == blocks_.begin()) return;
            --it;
            const std::uintptr_t base = it->first;
            const Block &b = it->second;
            if (addr < base || addr >= base + b.size) return;
            const size_t off = (size_t)(addr - base);
            span = std::min(bytes, b.size - off);
            phy = b.phy + off;
            vbase = reinterpret_cast<void *>(addr);
        }
        const AX_S32 mret = clean
            ? AX_SYS_MflushCache((AX_U64)phy, (AX_VOID *)vbase, (AX_U32)span)
            : AX_SYS_MinvalidateCache((AX_U64)phy, (AX_VOID *)vbase, (AX_U32)span);
        if (mret != 0)
        {
            // A silent failure here means the NPU/CPU can see stale KV rows —
            // exactly the nondeterminism this registry exists to prevent.
            fprintf(stderr,
                    "[cmm-flush] %s(phy=0x%llx vir=%p span=%zu) FAILED ret=0x%x\n",
                    clean ? "MflushCache" : "MinvalidateCache",
                    (unsigned long long)phy, vbase, span, (unsigned)mret);
        }
#else
        (void)vir;
        (void)bytes;
        (void)clean;
#endif
    }

    struct Block
    {
        unsigned long long phy;
        size_t size;
    };
    std::map<std::uintptr_t, Block> blocks_;
    std::mutex mtx_;
};
