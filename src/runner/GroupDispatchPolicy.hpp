#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace axllm {

// Build a map from reference-model capacity slots to indices in a layer-local
// list of groups.  The result deliberately contains local vector indices, not
// raw AXModel group ids: those ids are not guaranteed to be contiguous or to
// have the same ordering in every layer.
inline std::vector<int> make_capacity_slot_map(
    const std::vector<int> &requested_capacities,
    const std::vector<int> &local_capacities)
{
    std::vector<int> result;
    result.reserve(requested_capacities.size());
    for (const int requested : requested_capacities)
    {
        int selected = -1;
        // Prefer the smallest local group which can represent this request.
        for (size_t i = 0; i < local_capacities.size(); ++i)
        {
            if (local_capacities[i] >= requested &&
                (selected < 0 || local_capacities[i] < local_capacities[(size_t)selected]))
                selected = static_cast<int>(i);
        }
        // Deduplicated graphs intentionally expose a single group.  GDN's
        // recurrent state and while-loop group can serve all reference prefill
        // slots even though its mask capacity is only the chunk length.  For
        // full-attention layers the single retained group is the maximum-size
        // group, so this is also the safe fallback for capacity aliases.
        if (selected < 0 && !local_capacities.empty())
        {
            selected = static_cast<int>(std::distance(
                local_capacities.begin(),
                std::max_element(local_capacities.begin(), local_capacities.end())));
        }
        result.push_back(selected);
    }
    return result;
}

inline std::string capacity_signature(const std::vector<int> &values)
{
    std::string result;
    for (const int value : values)
    {
        result += std::to_string(value);
        result.push_back(',');
    }
    return result;
}

inline std::string dispatch_cache_key(bool reuse_metadata,
                                     int layer_idx,
                                     const std::string &graph_class,
                                     const std::string &abi_and_capacity_signature)
{
    if (!reuse_metadata || graph_class.empty())
        return "layer:" + std::to_string(layer_idx) + ":" + abi_and_capacity_signature;
    return graph_class + ":" + abi_and_capacity_signature;
}

} // namespace axllm
