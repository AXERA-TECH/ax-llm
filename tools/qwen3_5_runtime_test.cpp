#include "Qwen3_5Runtime.hpp"

#include <cassert>
#include <vector>

int main() {
  using axllm::qwen3_5::AttentionConfig;
  using axllm::qwen3_5::LinearStateSnapshot;
  using axllm::qwen3_5::LinearStateStore;
  using axllm::qwen3_5::Runtime;

  AttentionConfig config;
  config.layer_types = {"linear_attention", "sliding_attention",
                        "full_attention", "linear_attention", "full_attention"};
  config.num_kv_shared_layers = 2;
  config.sliding_window = 128;
  Runtime runtime(5, config);
  assert(runtime.is_linear_layer(0));
  assert(!runtime.is_linear_layer(1));
  assert(!runtime.is_linear_layer(2));
  assert(runtime.is_sliding_attention_layer(1));
  assert(!runtime.is_sliding_attention_layer(2));
  assert(runtime.first_full_layer_idx() == 1);
  assert(runtime.shared_kv_source_for_layer(3) == 0);
  assert(runtime.shared_kv_source_for_layer(4) == 2);

  AttentionConfig interval_config;
  interval_config.full_attention_interval = 4;
  Runtime interval_runtime(6, interval_config);
  assert(interval_runtime.is_linear_layer(0));
  assert(interval_runtime.is_linear_layer(2));
  assert(!interval_runtime.is_linear_layer(3));
  assert(interval_runtime.first_full_layer_idx() == 3);

  std::vector<int> cached = {1, 2, 3};
  std::vector<int> requested = {1, 2, 4, 5};
  int offset = -1;
  const auto suffix = Runtime::token_suffix(cached, requested, offset);
  assert(offset == 2);
  assert((suffix == std::vector<int>{4, 5}));
  assert(Runtime::cache_bytes(17) == 17);
  assert(Runtime::cache_u16_slots(17) == 9);
  assert(Runtime::compatible_cache_dtype(0, 3));
  assert(!Runtime::compatible_cache_dtype(2, 3));

  LinearStateStore store;
  LinearStateSnapshot first;
  first.token_len = 8;
  LinearStateSnapshot second;
  second.token_len = 16;
  store.items().push_back(first);
  store.items().push_back(second);
  assert(store.best_len(15) == 8);
  assert(store.find(16) != nullptr);
  store.drop_after(8);
  assert(store.best_len(16) == 8);
  assert(store.find(16) == nullptr);
  return 0;
}
