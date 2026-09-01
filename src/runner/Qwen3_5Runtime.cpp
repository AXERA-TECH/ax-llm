#include "Qwen3_5Runtime.hpp"

#include <algorithm>
#include <utility>

namespace axllm::qwen3_5 {

void LinearStateStore::drop_after(int token_len) {
  snapshots_.erase(
      std::remove_if(snapshots_.begin(), snapshots_.end(),
                     [token_len](const LinearStateSnapshot &snapshot) {
                       return snapshot.token_len > token_len;
                     }),
      snapshots_.end());
}

int LinearStateStore::best_len(int token_len) const {
  int best = -1;
  for (const auto &snapshot : snapshots_)
    if (snapshot.token_len <= token_len && snapshot.token_len > best)
      best = snapshot.token_len;
  return best;
}

LinearStateSnapshot *LinearStateStore::find(int token_len) {
  for (auto &snapshot : snapshots_)
    if (snapshot.token_len == token_len)
      return &snapshot;
  return nullptr;
}

const LinearStateSnapshot *LinearStateStore::find(int token_len) const {
  for (const auto &snapshot : snapshots_)
    if (snapshot.token_len == token_len)
      return &snapshot;
  return nullptr;
}

Runtime::Runtime(int layer_count, AttentionConfig config) {
  configure(layer_count, std::move(config));
}

void Runtime::configure(int layer_count, AttentionConfig config) {
  layer_count = std::max(0, layer_count);
  sliding_window_ = config.sliding_window;
  linear_layers_.assign(static_cast<size_t>(layer_count), false);
  sliding_layers_.assign(static_cast<size_t>(layer_count), false);

  if (!config.layer_types.empty()) {
    const int count =
        std::min(layer_count, static_cast<int>(config.layer_types.size()));
    for (int i = 0; i < count; ++i) {
      const std::string &type = config.layer_types[static_cast<size_t>(i)];
      // Unknown types intentionally retain the generic full-attention
      // behavior used by older models.
      linear_layers_[static_cast<size_t>(i)] = type == "linear_attention";
      sliding_layers_[static_cast<size_t>(i)] = type == "sliding_attention";
    }
  } else if (config.full_attention_interval > 0) {
    for (int i = 0; i < layer_count; ++i) {
      const bool is_full = ((i + 1) % config.full_attention_interval) == 0;
      linear_layers_[static_cast<size_t>(i)] = !is_full;
    }
  }

  shared_kv_source_layers_.assign(static_cast<size_t>(layer_count), -1);
  if (config.layer_types.empty() || config.num_kv_shared_layers <= 0)
    return;

  const int type_count =
      std::min(layer_count, static_cast<int>(config.layer_types.size()));
  const int first_shared = type_count - config.num_kv_shared_layers;
  if (first_shared <= 0)
    return;

  for (int layer_idx = first_shared; layer_idx < type_count; ++layer_idx) {
    const std::string &type =
        config.layer_types[static_cast<size_t>(layer_idx)];
    for (int prev_idx = first_shared - 1; prev_idx >= 0; --prev_idx) {
      if (config.layer_types[static_cast<size_t>(prev_idx)] == type) {
        shared_kv_source_layers_[static_cast<size_t>(layer_idx)] = prev_idx;
        break;
      }
    }
  }
}

bool Runtime::is_linear_layer(int layer_idx) const {
  return layer_idx >= 0 &&
         layer_idx < static_cast<int>(linear_layers_.size()) &&
         linear_layers_[static_cast<size_t>(layer_idx)];
}

bool Runtime::has_linear_attention_layers() const {
  return std::any_of(linear_layers_.begin(), linear_layers_.end(),
                     [](bool value) { return value; });
}

bool Runtime::is_sliding_attention_layer(int layer_idx) const {
  return sliding_window_ > 0 && layer_idx >= 0 &&
         layer_idx < static_cast<int>(sliding_layers_.size()) &&
         sliding_layers_[static_cast<size_t>(layer_idx)];
}

int Runtime::first_full_layer_idx() const {
  for (int i = 0; i < static_cast<int>(linear_layers_.size()); ++i)
    if (!linear_layers_[static_cast<size_t>(i)])
      return i;
  return -1;
}

int Runtime::shared_kv_source_for_layer(int layer_idx) const {
  if (layer_idx < 0 ||
      layer_idx >= static_cast<int>(shared_kv_source_layers_.size()))
    return -1;
  return shared_kv_source_layers_[static_cast<size_t>(layer_idx)];
}

std::vector<int> Runtime::token_suffix(const std::vector<int> &cached,
                                       const std::vector<int> &requested,
                                       int &common_prefix) {
  const size_t limit = std::min(cached.size(), requested.size());
  common_prefix = 0;
  while (static_cast<size_t>(common_prefix) < limit &&
         cached[static_cast<size_t>(common_prefix)] ==
             requested[static_cast<size_t>(common_prefix)])
    ++common_prefix;
  if (static_cast<size_t>(common_prefix) >= requested.size())
    return {};
  return std::vector<int>(requested.begin() + common_prefix, requested.end());
}

size_t Runtime::cache_bytes(uint32_t n_size) {
  return static_cast<size_t>(n_size);
}

size_t Runtime::cache_u16_slots(uint32_t n_size) {
  const size_t bytes = cache_bytes(n_size);
  return (bytes + sizeof(uint16_t) - 1) / sizeof(uint16_t);
}

bool Runtime::compatible_cache_dtype(int src_dtype, int dst_dtype) {
  // Older AXCL tensor descriptors report zero, which means unknown.
  return src_dtype == 0 || dst_dtype == 0 || src_dtype == dst_dtype;
}

} // namespace axllm::qwen3_5
