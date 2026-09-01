#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace axllm::qwen3_5 {

// Configuration and state policy for Qwen3.5's mixed linear/full attention.
// Device runners stay in LLM.cpp; this class only owns model policy and opaque
// recurrent-state metadata so it is usable by both AX650 and AXCL adapters.
struct AttentionConfig {
  std::vector<std::string> layer_types;
  int full_attention_interval = 0;
  int num_kv_shared_layers = 0;
  int sliding_window = 0;
};

struct LinearStateSnapshot {
  int token_len = 0;
  std::vector<std::vector<unsigned char>> k;
  std::vector<std::vector<unsigned char>> v;
};

class LinearStateStore {
public:
  void clear() { snapshots_.clear(); }
  bool empty() const { return snapshots_.empty(); }
  void drop_after(int token_len);
  int best_len(int token_len) const;
  LinearStateSnapshot *find(int token_len);
  const LinearStateSnapshot *find(int token_len) const;

  std::vector<LinearStateSnapshot> &items() { return snapshots_; }
  const std::vector<LinearStateSnapshot> &items() const { return snapshots_; }

private:
  std::vector<LinearStateSnapshot> snapshots_;
};

class Runtime {
public:
  Runtime() = default;
  Runtime(int layer_count, AttentionConfig config);

  void configure(int layer_count, AttentionConfig config);

  bool is_linear_layer(int layer_idx) const;
  bool has_linear_attention_layers() const;
  bool is_sliding_attention_layer(int layer_idx) const;
  int first_full_layer_idx() const;
  int shared_kv_source_for_layer(int layer_idx) const;

  // Return the uncached suffix and its common-prefix length.  This is the
  // token-level contract used by multi-turn KV reuse and rollback.
  static std::vector<int> token_suffix(const std::vector<int> &cached,
                                       const std::vector<int> &requested,
                                       int &common_prefix);

  static size_t cache_bytes(uint32_t n_size);
  static size_t cache_u16_slots(uint32_t n_size);
  static bool compatible_cache_dtype(int src_dtype, int dst_dtype);

private:
  std::vector<bool> linear_layers_;
  std::vector<bool> sliding_layers_;
  std::vector<int> shared_kv_source_layers_;
  int sliding_window_ = 0;
};

} // namespace axllm::qwen3_5
