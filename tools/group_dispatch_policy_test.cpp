#include "GroupDispatchPolicy.hpp"

#include <cassert>
#include <string>
#include <vector>

int main()
{
    using axllm::dispatch_cache_key;
    using axllm::make_capacity_slot_map;

    // The slot map is indexed by capacity-order, then resolves to an index in
    // the layer-local raw-id list.  Non-contiguous/reordered raw IDs must not
    // leak into the mapping itself.
    const std::vector<int> full_attention_gids = {2, 7, 11};
    const std::vector<int> global_decode_caps = {128, 512, 2048};
    const auto full_decode_slots = make_capacity_slot_map(global_decode_caps, {128, 512, 2048});
    assert((full_decode_slots == std::vector<int>{0, 1, 2}));
    assert(full_attention_gids[(size_t)full_decode_slots[1]] == 7);

    // Full-attention prefill keeps the largest graph plus any smaller graph
    // needed for a request.  Oversized requests alias the largest retained
    // graph, as used by the deduplicated max-capacity export.
    const auto full_prefill_slots = make_capacity_slot_map({128, 256, 512, 1024}, {128, 512});
    assert((full_prefill_slots == std::vector<int>{0, 1, 1, 1}));

    // GDN has one recurrent-state chunk group after dedup; all reference
    // prefill slots route to that group (slot 0), independent of context size.
    const auto gdn_prefill_slots = make_capacity_slot_map({128, 256, 512, 1024}, {128});
    assert((gdn_prefill_slots == std::vector<int>{0, 0, 0, 0}));

    // Empty local topology is represented explicitly and never creates an
    // invalid negative vector index at call sites.
    const auto missing_slots = make_capacity_slot_map({128}, {});
    assert((missing_slots == std::vector<int>{-1}));

    const std::string same_abi = "decode=indices:u32[1];mask:bf16[2048];prefill=128;";
    // NPU1/NPU4 are not special-cased here: same graph class and exact ABI
    // permits metadata reuse across those compilation modes.
    assert(dispatch_cache_key(true, 0, "linear_attention", same_abi) ==
           dispatch_cache_key(true, 4, "linear_attention", same_abi));
    // Mixed GDN/full-attention layers must never share mappings even if an
    // accidental subset of their tensor ABI strings matches.
    assert(dispatch_cache_key(true, 0, "linear_attention", same_abi) !=
           dispatch_cache_key(true, 1, "full_attention", same_abi));
    // Runtime opt-out and missing class metadata keep the mapping layer-local.
    assert(dispatch_cache_key(false, 0, "linear_attention", same_abi) !=
           dispatch_cache_key(false, 1, "linear_attention", same_abi));
    assert(dispatch_cache_key(true, 0, "", same_abi) !=
           dispatch_cache_key(true, 1, "", same_abi));

    // Tensor layout or capacity changes invalidate reuse.
    assert(dispatch_cache_key(true, 0, "full_attention", same_abi) !=
           dispatch_cache_key(true, 1, "full_attention", same_abi + "dtype=fp32"));
    return 0;
}
