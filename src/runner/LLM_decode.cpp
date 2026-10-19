// Auto-split from LLM.cpp/LLMImpl.hpp: generation/decode engine (LLM::Impl::Run + prefill).
#include "LLMImpl.hpp"
#include <filesystem>
#include <fstream>

int LLM::Impl::GenerateKVCachePrefill(std::vector<int> &_token_ids,
                           std::vector<std::vector<unsigned short>> &k_caches,
                           std::vector<std::vector<unsigned short>> &v_caches,
                           int &prefill_precompute_len)
{
    bfloat16 bf16 = -65536.f;
    int input_embed_num = (int)_token_ids.size();
    prefill_precompute_len = input_embed_num;
    k_caches.resize(_attr.axmodel_num);
    v_caches.resize(_attr.axmodel_num);

    int prefill_split_num = (int)ceil((double)input_embed_num / _attr.prefill_token_num);
    ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);

    if (input_embed_num == 0)
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            const int layer_kv = kv_cache_size_for_layer(i);
            k_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
            v_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
        }
        ALOGI("input token num is 0, skip");
        return 0;
    }

    std::vector<int> prefill_grp_list;
    prefill_grp_list.resize(prefill_split_num, -1);
    for (int p = 0; p < prefill_split_num; ++p)
    {
        const int history_len = p * _attr.prefill_token_num;
        const int chunk_tokens = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;
        const int gid = select_prefill_group(history_len, chunk_tokens, false);
        if (gid < 0)
        {
            ALOGE("failed to select prefill group for KV prefill: history_len=%d chunk_tokens=%d", history_len, chunk_tokens);
            return -1;
        }
        prefill_grp_list[p] = gid;
    }

    const int prefill_decode_grpid = choose_decode_gid(std::max(1, input_embed_num));
    decode_grpid = prefill_decode_grpid;
    clear_all_group_kv_cache_tensors();

    std::vector<unsigned short> test_embed(_token_ids.size() * _attr.tokens_embed_size);
    for (size_t i = 0; i < _token_ids.size(); i++) embed_selector.getByIndex(_token_ids[i], test_embed.data() + i * _attr.tokens_embed_size);
    std::vector<unsigned short> linear_mask_tmp;
    bfloat16 bf16_one = 1.0f;

    for (int p = 0; p < prefill_split_num; p++)
    {
        int input_num_token = (p == prefill_split_num - 1) ? input_embed_num - p * _attr.prefill_token_num : _attr.prefill_token_num;
        const int history_len = p * _attr.prefill_token_num;
        const int prefill_grpid = prefill_grp_list[p];
        if (history_len > 0 &&
            !sync_active_prefill_group_from_decode(prefill_grpid, prefill_decode_grpid, history_len))
        {
            ALOGE("KV prefill cache sync failed: group=%d history_len=%d", prefill_grpid, history_len);
            return -1;
        }
        int query_tokens = prefill_query_tokens_by_gid(prefill_grpid);
        if (query_tokens <= 1) query_tokens = _attr.prefill_token_num;
        if (input_num_token > query_tokens)
        {
            ALOGE("prefill group %d query=%d cannot hold chunk_tokens=%d", prefill_grpid, query_tokens, input_num_token);
            return -1;
        }
        int kv_cache_num = prefill_history_capacity_by_gid(prefill_grpid);
        const int kv_from_mask = prefill_history_capacity_by_mask(prefill_grpid);
        if (kv_from_mask >= 0) kv_cache_num = kv_from_mask;
        if (kv_cache_num < 0)
        {
            ALOGE("invalid kv_cache_num=%d for prefill_grpid=%d", kv_cache_num, prefill_grpid);
            return -1;
        }
        std::vector<unsigned short> mask_tmp((size_t)query_tokens * (size_t)(kv_cache_num + query_tokens), bf16.data);
        std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
        size_t copy_tokens = (p == prefill_split_num - 1) ? (size_t)(input_embed_num - p * _attr.prefill_token_num) : (size_t)_attr.prefill_token_num;
        memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));

        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            auto &lyr = llama_layers[m]; int devid = LLM_DEVID(lyr);
            const int layer_prefill_grpid = prefill_gid_for_layer(m, prefill_grpid);
            int layer_query_tokens = prefill_query_tokens_for_layer(m, layer_prefill_grpid);
            if (layer_query_tokens <= 1) layer_query_tokens = _attr.prefill_token_num;
            if (input_num_token > layer_query_tokens)
            {
                ALOGE("prefill layer %d group %d query=%d cannot hold chunk_tokens=%d",
                      m, layer_prefill_grpid, layer_query_tokens, input_num_token);
                return -1;
            }
            auto &t_idx = lyr.layer.get_input(layer_prefill_grpid, "indices");
            unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr; memset(idx_ptr, 0, t_idx.nSize);
            const int idx_elems = (int)(t_idx.nSize / (int)sizeof(unsigned int));
            const int idx_query = std::max(1, layer_query_tokens);
            const int idx_rows = std::max(1, idx_elems / idx_query);
            const int live_last = std::max(0, input_num_token - 1);
            for (int r = 0; r < idx_rows; ++r)
                for (int i = 0; i < idx_query; ++i)
                    idx_ptr[r * idx_query + i] = (unsigned int)(p * _attr.prefill_token_num + std::min(i, live_last));
            llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);
            auto &t_mask = lyr.layer.get_input(layer_prefill_grpid, "mask");
            if (is_linear_layer(m))
            {
                const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                fill_linear_prefill_mask(linear_mask_tmp, elems, input_num_token);
                llm_h2d(LLM_WADDR(t_mask), linear_mask_tmp.data(), linear_mask_tmp.size() * sizeof(unsigned short), devid);
            }
            else
            {
                mask_tmp.assign((size_t)layer_query_tokens * (size_t)(kv_cache_num + layer_query_tokens), bf16.data);
                build_layer_prefill_mask(mask_tmp, kv_cache_num, layer_query_tokens, history_len, input_num_token, m);
                llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
            }
            auto &t_in = lyr.layer.get_input(layer_prefill_grpid, "input"); llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);
            set_kv_block_count_if_present(lyr.layer, layer_prefill_grpid, history_len, devid);
            if (!ensure_layer_loaded(m)) return -1;
            const int kv_prefill_infer_ret = lyr.layer.inference(layer_prefill_grpid);
            if (kv_prefill_infer_ret != 0)
            {
                ALOGW("KV prefill layer %d gid=%d inference failed: 0x%x", m, layer_prefill_grpid, kv_prefill_infer_ret);
                set_last_error("KV prefill layer inference 失败: layer=" + std::to_string(m) +
                               " group=" + std::to_string(layer_prefill_grpid) +
                               " ret=" + std::to_string(kv_prefill_infer_ret));
                return -1;
            }
            const int layer_decode_grpid = decode_gid_for_layer(m, prefill_decode_grpid);
            auto &dec_k  = lyr.layer.get_input(layer_decode_grpid, "K_cache");
            auto &dec_v  = lyr.layer.get_input(layer_decode_grpid, "V_cache");
            const int layer_kv = kv_cache_size_for_layer(m);
            if (is_linear_layer(m))
            {
                (void)dec_k;
                (void)dec_v;
                // Keep only decode groups current.  The next prefill chunk
                // copies the recurrent state into its selected shape bucket
                // on demand.
                if (!sync_linear_input_state_from_group(m, lyr.layer, layer_prefill_grpid, devid, false, false)) return -1;
            }
            else
            {
                auto &out_k  = lyr.layer.get_output(layer_prefill_grpid, "K_cache_out");
                auto &out_v  = lyr.layer.get_output(layer_prefill_grpid, "V_cache_out");
                const int kv_off = history_len * layer_kv;
                const size_t kv_sz = (size_t)input_num_token * (size_t)layer_kv * sizeof(unsigned short);
                llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);

            }
            auto &t_out = lyr.layer.get_output(layer_prefill_grpid, "output"); llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), std::min((size_t)t_out.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);
        }
    }

    for (int i = 0; i < _attr.axmodel_num; i++)
    {
        auto &lyr = llama_layers[i]; int devid = LLM_DEVID(lyr);
        const int layer_decode_grpid = decode_gid_for_layer(i, prefill_decode_grpid);
        auto &t_k = lyr.layer.get_input(layer_decode_grpid, "K_cache");
        auto &t_v = lyr.layer.get_input(layer_decode_grpid, "V_cache");
        if (is_linear_layer(i))
        {
            k_caches[i].resize((size_t)t_k.nSize / sizeof(unsigned short));
            v_caches[i].resize((size_t)t_v.nSize / sizeof(unsigned short));
            llm_d2h(k_caches[i].data(), LLM_RADDR(t_k), t_k.nSize, devid);
            llm_d2h(v_caches[i].data(), LLM_RADDR(t_v), t_v.nSize, devid);
        }
        else
        {
            const int layer_kv = kv_cache_size_for_layer(i);
            k_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
            v_caches[i].resize((size_t)prefill_precompute_len * (size_t)layer_kv);
            const size_t kv_bytes = (size_t)prefill_precompute_len * (size_t)layer_kv * sizeof(unsigned short);
            llm_d2h(k_caches[i].data(), LLM_RADDR(t_k), std::min(kv_bytes, (size_t)t_k.nSize), devid);
            llm_d2h(v_caches[i].data(), LLM_RADDR(t_v), std::min(kv_bytes, (size_t)t_v.nSize), devid);
        }
    }
    return 0;
}

std::string LLM::Impl::Run(std::vector<unsigned short> &test_embed, int output_max_token)
{
    using DecodeClock = std::chrono::steady_clock;
    struct DecodeProfileStats
    {
        uint64_t per_layer_ns = 0;
        uint64_t shared_kv_ns = 0;
        uint64_t prep_ns = 0;
        uint64_t inference_ns = 0;
        uint64_t outcopy_ns = 0;
        uint64_t post_ns = 0;
    };

    b_stop.store(false, std::memory_order_relaxed); std::string final_out;
    utf8_filter = UTF8Filter();
    ChannelSectionFilter channel_filter;
    channel_filter.reset();
    const bool hide_channel_markup = tokenizer_uses_hidden_channel_markup(_attr.tokenizer_type);
    const bool debug_prefill =
        std::getenv("AXLLM_DEBUG_PREFILL") ||
        std::getenv("AXLLM_DEBUG_LAYER0_IO") ||
        std::getenv("AXLLM_DEBUG_DECODE_TRACE");
    // AX650 group outputs are not safely consumable through a plain host
    // memcpy on every board/firmware combination.  An explicit D2H/D2D
    // transaction creates the required output barrier and made repeated
    // full-attention decode runs bit-stable.  Keep the old path available for
    // binary search and performance regression testing.
    const bool decode_explicit_output_d2h =
        std::getenv("AXLLM_DISABLE_DECODE_EXPLICIT_D2H") == nullptr ||
        std::string(std::getenv("AXLLM_DISABLE_DECODE_EXPLICIT_D2H")) != "1";
    std::string streamed_visible_text;
    std::string streamed_raw_visible_text;
    auto emit_stream_chunk = [&](const std::string &chunk, float tps)
    {
        if (!_attr.runing_callback || chunk.empty())
            return;

        streamed_raw_visible_text += chunk;
        const std::string visible_text = sanitize_utf8_text(streamed_raw_visible_text);
        if (visible_text.size() <= streamed_visible_text.size())
            return;
        if (visible_text.compare(0, streamed_visible_text.size(), streamed_visible_text) != 0)
        {
            ALOGW("streamed visible text prefix mismatch, suppressing incremental emit");
            streamed_visible_text = visible_text;
            return;
        }

        const std::string delta = visible_text.substr(streamed_visible_text.size());
        if (!delta.empty())
        {
            streamed_visible_text = visible_text;
            _attr.runing_callback(delta, tps, _attr.reserve);
        }
    };
    auto dump_hidden_trace = [&](const char *tag, int step, const unsigned short *hidden, int n)
    {
        if (!std::getenv("AXLLM_DEBUG_DECODE_TRACE") || !tag || !hidden || n <= 0)
            return;
        float min_v = 0.0f, max_v = 0.0f, mean_abs = 0.0f;
        summarize_bf16_buffer(hidden, n, min_v, max_v, mean_abs);
        ALOGI("DTRACE step=%d %s n=%d hash=0x%llx min=%.6f max=%.6f mean_abs=%.6f",
              step,
              tag,
              n,
              (unsigned long long)hash_u16_buffer(hidden, n),
              min_v,
              max_v,
              mean_abs);
    };
    auto dump_decode_trace = [&](int step, const unsigned short *logits, int n, int chosen_token)
    {
        if (!std::getenv("AXLLM_DEBUG_DECODE_TRACE") || !logits || n <= 0)
            return;
        const auto topk = topk_bfloat16(const_cast<unsigned short *>(logits), n, std::min(5, n));
        std::string topk_str = "[";
        for (size_t i = 0; i < topk.size(); ++i)
        {
            const int token = topk[i].first;
            const float val = topk[i].second;
            if (i > 0) topk_str += ",";
            topk_str += "(" + std::to_string(token) + ":" + std::to_string(val) + ")";
        }
        topk_str += "]";
        ALOGI("DTRACE step=%d post_logits n=%d hash=0x%llx topk=%s",
              step,
              n,
              (unsigned long long)hash_u16_buffer(logits, n),
              topk_str.c_str());
        ALOGI("DTRACE step=%d token=%d piece='%s'",
              step,
              chosen_token,
              safe_decode_token(tokenizer, chosen_token).c_str());
    };
    const char *prefill_dump_dir_text = std::getenv("AXLLM_DEBUG_PREFILL_DUMP");
    const std::filesystem::path prefill_dump_dir(
        prefill_dump_dir_text ? prefill_dump_dir_text : "/tmp/axllm_prefill_dump"
    );
    auto dump_prefill_bf16 = [&](const std::string &name, const ax_runner_tensor_t &tensor) {
        if (!prefill_dump_dir_text || !tensor.pVirAddr || tensor.nSize == 0) return;
        std::error_code ec;
        std::filesystem::create_directories(prefill_dump_dir, ec);
        if (ec) return;
        std::ofstream out(prefill_dump_dir / (name + ".bf16"), std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(tensor.pVirAddr), tensor.nSize);
    };
    const char *decode_dump_dir_text = std::getenv("AXLLM_DEBUG_DECODE_DUMP");
    const std::filesystem::path decode_dump_dir(
        decode_dump_dir_text ? decode_dump_dir_text : "/tmp/axllm_decode_dump"
    );
    const int decode_dump_start_token = decode_dump_dir_text
        ? std::max(0, atoi(std::getenv("AXLLM_DEBUG_DECODE_DUMP_START_TOKEN") ? std::getenv("AXLLM_DEBUG_DECODE_DUMP_START_TOKEN") : "0"))
        : 0;
    auto dump_debug_tensor = [&](const std::string &name, const ax_runner_tensor_t &tensor) {
        size_t token_pos = name.find("token_");
        int dump_token = token_pos == std::string::npos ? 0 : std::atoi(name.c_str() + token_pos + 6);
        if (!decode_dump_dir_text || dump_token < decode_dump_start_token || !tensor.pVirAddr || tensor.nSize == 0) return;
        std::error_code ec;
        std::filesystem::create_directories(decode_dump_dir, ec);
        if (ec) return;
        std::ofstream out(decode_dump_dir / (name + ".bin"), std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(tensor.pVirAddr), tensor.nSize);
    };
    auto dump_debug_device_tensor = [&](const std::string &name, const ax_runner_tensor_t &tensor, int devid) {
        size_t token_pos = name.find("token_");
        int dump_token = token_pos == std::string::npos ? 0 : std::atoi(name.c_str() + token_pos + 6);
        if (!decode_dump_dir_text || dump_token < decode_dump_start_token || tensor.nSize == 0) return;
        std::vector<char> host(tensor.nSize);
        llm_d2h(host.data(), LLM_RADDR(tensor), tensor.nSize, devid);
        std::error_code ec;
        std::filesystem::create_directories(decode_dump_dir, ec);
        if (ec) return;
        std::ofstream out(decode_dump_dir / (name + ".bin"), std::ios::binary | std::ios::trunc);
        out.write(host.data(), host.size());
    };
    bfloat16 bf16 = -65536.f;
    bfloat16 bf16_one = 1.0f;

    int max_decode_cap = _attr.max_token_len;
    if (!decode_max_token_len_grp_.empty()) max_decode_cap = std::max(max_decode_cap, decode_max_token_len_grp_.back());
    if (max_decode_cap <= 0) max_decode_cap = _attr.kv_cache_num;
    std::vector<unsigned short> decode_mask((size_t)max_decode_cap + 1, bf16.data);
    std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
    std::vector<int> token_ids;
    int input_embed_num  = (int)(test_embed.size() / _attr.tokens_embed_size);
    int prefill_split_num = (int)ceil((double)input_embed_num / _attr.prefill_token_num);
    ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
    timer t_cost, ttft_timer, decode_timer; ttft_timer.start();
    bool decode_timer_started = false;
    // Reset per-run performance metrics; assigned once measured below.
    last_run_ttft_ms_ = -1.0f;
    last_run_decode_tps_ = -1.0f;
    last_run_prefill_tps_ = -1.0f;
    last_run_prefill_tokens_ = -1;

    std::vector<int> prefill_grp_list;
    const int default_prefill_gid = prefill_grpids_.empty() ? 1 : prefill_grpids_.front();
    prefill_grp_list.resize(prefill_split_num, default_prefill_gid);
    for (int p = 0; p < prefill_split_num; ++p) {
        const int history_len = precompute_len + p * _attr.prefill_token_num;
        const int chunk_tokens = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;
        const bool prefer_symbolic_group = has_vision_state && history_len > 0;
        int g = select_prefill_group(history_len, chunk_tokens, prefer_symbolic_group);
        if (g < 0)
        {
            ALOGE("failed to select prefill group for history_len=%d chunk_tokens=%d", history_len, chunk_tokens);
            set_last_error("prefill group 选择失败: history_len=" + std::to_string(history_len) +
                           " chunk_tokens=" + std::to_string(chunk_tokens));
            return final_out;
        }
        prefill_grp_list[p] = g;
    }

    const bool use_per_layer_input = gemma4_per_layer_helper.enabled();
    const int per_layer_hidden = gemma4_per_layer_helper.hidden_size_per_layer_input();
    const auto env_flag_enabled = [](const char *name) {
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return false;
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "no") != 0;
    };
    const bool prefill_io_alias_enabled = env_flag_enabled("AXLLM_PREFILL_IO_ALIAS");
    const bool decode_profile_enabled = env_flag_enabled("AXLLM_PROFILE_DECODE");
    const bool prefill_chunk_profile_enabled = env_flag_enabled("AXLLM_PROFILE_PREFILL_CHUNKS");
    DecodeProfileStats decode_profile{};
    std::vector<unsigned short> prefill_per_layer_inputs;
    std::vector<unsigned short> decode_per_layer_input;
    std::vector<unsigned short> per_layer_tmp;

    if (use_per_layer_input)
    {
        if ((int)run_input_token_ids.size() != input_embed_num)
        {
            ALOGE("Gemma4 requires run_input_token_ids for prefill: ids=%zu embeds=%d",
                  run_input_token_ids.size(),
                  input_embed_num);
            return final_out;
        }

        std::vector<int> per_layer_token_ids = run_input_token_ids;
        std::vector<unsigned short> per_layer_embed = test_embed;
        scale_all_embeds_inplace(per_layer_embed.data(), input_embed_num);
        if (has_vision_state && !vision_state.pos2vision.empty())
        {
            for (int i = 0; i < input_embed_num; ++i)
            {
                const int abs_pos = (active_token_pos_start >= 0) ? (active_token_pos_start + i) : (precompute_len + i);
                if ((size_t)abs_pos < vision_state.pos2vision.size() && vision_state.pos2vision[(size_t)abs_pos] >= 0)
                {
                    per_layer_token_ids[(size_t)i] = gemma4_per_layer_helper.pad_token_id();
                }
            }
        }

        if (!gemma4_per_layer_helper.Compute(per_layer_token_ids,
                                             per_layer_embed.data(),
                                             input_embed_num,
                                             _attr.tokens_embed_size,
                                             prefill_per_layer_inputs))
        {
            ALOGE("Gemma4 prefill per-layer input compute failed");
            return final_out;
        }
    }

    for (int p = 0; p < prefill_split_num; p++)
    {
        if (b_stop.load(std::memory_order_relaxed)) break;
        const auto prefill_chunk_t0 = prefill_chunk_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
        uint64_t prefill_layer_inference_ns = 0;
        int prefill_profiled_layers = 0;
        int input_num_token = (p == prefill_split_num - 1) ? input_embed_num - p * _attr.prefill_token_num : _attr.prefill_token_num;
        const int history_len = precompute_len + p * _attr.prefill_token_num;
        const int token_start_pos = (active_token_pos_start >= 0)
                                        ? active_token_pos_start + p * _attr.prefill_token_num
                                        : history_len;
        const int rope_start_pos = (active_prefill_pos_start >= 0)
                                       ? active_prefill_pos_start + p * _attr.prefill_token_num
                                       : history_len;

        const int prefill_grpid = prefill_grp_list[p];
        int query_tokens = prefill_query_tokens_by_gid(prefill_grpid);
        if (query_tokens <= 1) query_tokens = _attr.prefill_token_num;
        const auto log_prefill_chunk_profile = [&](const char *status, int failed_layer, const char *stage) {
            if (!prefill_chunk_profile_enabled) return;
            const double total_ms = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - prefill_chunk_t0).count() / 1e6;
            const double inference_ms = (double)prefill_layer_inference_ns / 1e6;
            ALOGN("prefill chunk profile: status=%s chunk=%d/%d history=%d tokens=%d gid=%d query=%d layers=%d total=%.3f ms layer_inference_sum=%.3f ms failed_layer=%d stage=%s\n",
                  status,
                  p + 1,
                  prefill_split_num,
                  history_len,
                  input_num_token,
                  prefill_grpid,
                  query_tokens,
                  prefill_profiled_layers,
                  total_ms,
                  inference_ms,
                  failed_layer,
                  stage);
        };
        // ALOGI("prefill group for chunk %d: %d", p, prefill_grpid);
        // Prefill groups are shape buckets and are synchronized lazily.  The
        // canonical decode group contains the prefix produced by previous
        // chunks (or SetKVCache/prefix reuse); copy only the bucket selected
        // for this chunk before its graph reads K/V.  Without this call a
        // bucket switch could consume stale/zero history even though the
        // decode cache itself was correct.
        if (history_len > 0 &&
            !sync_active_prefill_group_from_decode(prefill_grpid, decode_grpid, history_len))
        {
            ALOGE("prefill cache sync failed: group=%d history_len=%d", prefill_grpid, history_len);
            log_prefill_chunk_profile("failed", -1, "cache_sync");
            set_last_error("模型运行失败，请重新尝试。");
            ResetKVCache();
            return final_out;
        }
        if (input_num_token > query_tokens)
        {
            ALOGE("prefill group %d query=%d cannot hold chunk_tokens=%d", prefill_grpid, query_tokens, input_num_token);
            log_prefill_chunk_profile("failed", -1, "group_capacity");
            set_last_error("prefill chunk 超出 group 容量: group=" + std::to_string(prefill_grpid) +
                           " query=" + std::to_string(query_tokens) +
                           " tokens=" + std::to_string(input_num_token));
            return final_out;
        }
        int kv_cache_num_p = prefill_history_capacity_for_layer_group(cache_ref_full_layer_idx, prefill_grpid);
        if (kv_cache_num_p < 0) kv_cache_num_p = prefill_history_capacity_by_gid(prefill_grpid);
        ALOGI("prefill chunk p=%d history_len=%d grpid=%d kv_cache_num=%d input_tokens=%d",
              p, history_len, prefill_grpid, kv_cache_num_p, input_num_token);

        std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
        std::vector<unsigned short> mask_tmp;
        std::vector<unsigned short> linear_mask_tmp;
        if (use_per_layer_input) per_layer_tmp.assign((size_t)_attr.prefill_token_num * (size_t)per_layer_hidden, 0);

        std::map<std::string, std::pair<int, int>> prefill_io_anchors;
        std::vector<std::pair<int, int>> prefill_aliased_layers;
        auto restore_prefill_input_aliases = [&prefill_aliased_layers, &layers = llama_layers](void *) {
            for (auto it = prefill_aliased_layers.rbegin(); it != prefill_aliased_layers.rend(); ++it)
                (void)layers[(size_t)it->first].layer.restore_group_input_aliases(it->second);
        };
        std::unique_ptr<void, decltype(restore_prefill_input_aliases)> prefill_alias_guard(
            reinterpret_cast<void *>(1), restore_prefill_input_aliases);

        size_t copy_tokens = (p == prefill_split_num - 1) ? (size_t)(input_embed_num - p * _attr.prefill_token_num) : (size_t)_attr.prefill_token_num;
        memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));
        scale_prefill_text_embeds_inplace(embed_tmp.data(), input_num_token, token_start_pos);
        if (std::getenv("AXLLM_DEBUG_DECODE_TRACE") && p == 0 && input_num_token > 0)
        {
            const unsigned short *last_token_embed =
                embed_tmp.data() + (size_t)(input_num_token - 1) * (size_t)_attr.tokens_embed_size;
            dump_hidden_trace("prefill_input_last_embed", -1, last_token_embed, _attr.tokens_embed_size);
        }

        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;
            auto &lyr   = llama_layers[m]; int devid = LLM_DEVID(lyr);
            const int layer_prefill_grpid = prefill_gid_for_layer(m, prefill_grpid);
            int layer_query_tokens = prefill_query_tokens_for_layer(m, layer_prefill_grpid);
            if (layer_query_tokens <= 1) layer_query_tokens = _attr.prefill_token_num;
            if (input_num_token > layer_query_tokens)
            {
                ALOGE("prefill layer %d group %d query=%d cannot hold chunk_tokens=%d",
                      m, layer_prefill_grpid, layer_query_tokens, input_num_token);
                log_prefill_chunk_profile("failed", m, "layer_capacity");
                set_last_error("prefill layer chunk 超出 group 容量: layer=" + std::to_string(m) +
                               " group=" + std::to_string(layer_prefill_grpid) +
                               " query=" + std::to_string(layer_query_tokens) +
                               " tokens=" + std::to_string(input_num_token));
                return final_out;
            }
            int layer_kv_cache_num = prefill_history_capacity_for_layer_group(m, layer_prefill_grpid);
            if (layer_kv_cache_num < 0) layer_kv_cache_num = kv_cache_num_p;
            if (layer_kv_cache_num < history_len && !is_linear_layer(m))
            {
                ALOGE("prefill layer %d group %d history_len=%d exceeds layer history cap=%d",
                      m,
                      layer_prefill_grpid,
                      history_len,
                      layer_kv_cache_num);
                return final_out;
            }
            if (layer_prefill_grpid != prefill_grpid)
            {
                ALOGD("prefill layer group remap: layer=%d global_gid=%d layer_gid=%d history_cap=%d",
                      m,
                      prefill_grpid,
                      layer_prefill_grpid,
                      layer_kv_cache_num);
            }
            auto &t_idx = lyr.layer.get_input(layer_prefill_grpid, "indices");
            auto &t_mask = lyr.layer.get_input(layer_prefill_grpid, "mask");
            const std::string prefill_io_signature =
                layer_graph_class_for(m) + "|" +
                (is_linear_layer(m) ? "linear" : "attention") + "|" +
                (is_sliding_attention_layer(m) ? "sliding" : "regular") + "|" +
                std::to_string(layer_query_tokens) + "|" +
                std::to_string(layer_kv_cache_num) + "|" +
                std::to_string(_attr.sliding_window) + "|" +
                std::to_string(t_idx.nSize) + "|" +
                std::to_string(t_mask.nSize);

            bool prefill_inputs_aliased = false;
            if (prefill_io_alias_enabled)
            {
                const auto anchor_it = prefill_io_anchors.find(prefill_io_signature);
                if (anchor_it != prefill_io_anchors.end())
                {
                    auto &anchor_layer = llama_layers[(size_t)anchor_it->second.first];
                    const int anchor_gid = anchor_it->second.second;
                    auto &anchor_idx = anchor_layer.layer.get_input(anchor_gid, "indices");
                    auto &anchor_mask = anchor_layer.layer.get_input(anchor_gid, "mask");
                    if (anchor_layer.layer.get_devid() == lyr.layer.get_devid() &&
                        anchor_idx.nSize == t_idx.nSize && anchor_mask.nSize == t_mask.nSize &&
                        anchor_idx.phyAddr != 0 && anchor_mask.phyAddr != 0)
                    {
                        const int idx_alias_ret = lyr.layer.alias_group_input(
                            layer_prefill_grpid, "indices", anchor_idx.pVirAddr, anchor_idx.phyAddr);
                        const int mask_alias_ret = lyr.layer.alias_group_input(
                            layer_prefill_grpid, "mask", anchor_mask.pVirAddr, anchor_mask.phyAddr);
                        prefill_inputs_aliased = idx_alias_ret == 0 && mask_alias_ret == 0;
                        if (!prefill_inputs_aliased)
                        {
                            (void)lyr.layer.restore_group_input_aliases(layer_prefill_grpid);
                        }
                        else
                        {
                            prefill_aliased_layers.emplace_back(m, layer_prefill_grpid);
                        }
                    }
                }
            }

            if (!prefill_inputs_aliased)
            {
                prefill_io_anchors.emplace(prefill_io_signature, std::make_pair(m, layer_prefill_grpid));
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr; memset(idx_ptr, 0, t_idx.nSize);
                {
                    const int seq_start_pos = rope_start_pos;
                    const int idx_elems = (int)(t_idx.nSize / (int)sizeof(unsigned int));
                    int idx_rows = idx_elems / layer_query_tokens;
                    if (idx_rows <= 0) idx_rows = 1;
                    const bool use_pos_ids = has_vision_state &&
                                             idx_rows >= 3 &&
                                             vision_state.position_ids.size() >= 3 &&
                                             !vision_state.position_ids.empty();

                    for (int r = 0; r < idx_rows; ++r)
                    {
                        // ALOGI("r %d, idx_rows: %d", r, idx_rows);
                        // Fixed-query graphs execute padded rows too.  Keep their
                        // position indices in range by repeating the last live
                        // token instead of leaving zero/uninitialized positions.
                        // These rows are masked from all real query rows and are
                        // not copied to the host KV/output.
                        const int live_last = std::max(0, input_num_token - 1);
                        for (int j = 0; j < layer_query_tokens; ++j)
                        {
                            const int live_j = std::min(j, live_last);
                            unsigned int v = (unsigned int)(seq_start_pos + live_j);
                            if (use_pos_ids)
                            {
                                if ((size_t)r < vision_state.position_ids.size())
                                {
                                    const auto &row = vision_state.position_ids[r];
                                    if ((size_t)(token_start_pos + live_j) < row.size())
                                        v = (unsigned int)row[token_start_pos + live_j];
                                }
                            }
                            idx_ptr[r * layer_query_tokens + j] = v;
                        }
                    }
                }
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    fill_linear_prefill_mask(linear_mask_tmp, elems, input_num_token);
                    llm_h2d(LLM_WADDR(t_mask), linear_mask_tmp.data(), linear_mask_tmp.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    mask_tmp.assign((size_t)layer_query_tokens * (size_t)(layer_kv_cache_num + layer_query_tokens), bf16.data);
                    if (use_sparse_full_cache_mask())
                        build_sparse_layer_prefill_mask(mask_tmp, layer_kv_cache_num, layer_query_tokens, history_len, input_num_token, m);
                    else
                        build_layer_prefill_mask(mask_tmp, layer_kv_cache_num, layer_query_tokens, history_len, input_num_token, m);
                    llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
                }
            }
            auto &t_in = lyr.layer.get_input(layer_prefill_grpid, "input");
            if (std::getenv("AXLLM_ZERO_LAYER0_PREFILL_INPUT") && m == 0 && p == 0)
            {
                std::vector<unsigned short> zero_embed(embed_tmp.size(), 0);
                llm_h2d(LLM_WADDR(t_in), zero_embed.data(), std::min((size_t)t_in.nSize, zero_embed.size() * sizeof(unsigned short)), devid);
            }
            else
            {
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);
            }
            set_kv_block_count_if_present(lyr.layer, layer_prefill_grpid, history_len, devid);
            const int shared_src = shared_kv_source_for_layer(m);
            if (shared_src >= 0)
            {
                auto &src_layer = llama_layers[(size_t)shared_src];
                const int src_decode_grpid = decode_gid_for_layer(shared_src, decode_grpid);
                auto &src_k = src_layer.layer.get_input(src_decode_grpid, "K_cache");
                auto &src_v = src_layer.layer.get_input(src_decode_grpid, "V_cache");
                auto &dst_k = lyr.layer.get_input(layer_prefill_grpid, "K_cache");
                auto &dst_v = lyr.layer.get_input(layer_prefill_grpid, "V_cache");
                const int layer_kv = kv_cache_size_for_layer(m);
                copy_shared_prefill_cache(dst_k,
                                          src_k,
                                          layer_kv,
                                          history_len,
                                          layer_kv_cache_num,
                                          history_len,
                                          input_num_token,
                                          devid);
                copy_shared_prefill_cache(dst_v,
                                          src_v,
                                          layer_kv,
                                          history_len,
                                          layer_kv_cache_num,
                                          history_len,
                                          input_num_token,
                                          devid);
            }
            if (use_per_layer_input)
            {
                const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, layer_prefill_grpid, "per_layer_input");
                if (!t_per_layer)
                {
                    ALOGE("Gemma4 decoder layer %d is missing per_layer_input for group %d", m, layer_prefill_grpid);
                    return final_out;
                }
                std::fill(per_layer_tmp.begin(), per_layer_tmp.end(), 0);
                for (int j = 0; j < input_num_token; ++j)
                {
                    const size_t src_off = (((size_t)p * (size_t)_attr.prefill_token_num + (size_t)j) * (size_t)_attr.axmodel_num + (size_t)m) * (size_t)per_layer_hidden;
                    const size_t dst_off = (size_t)j * (size_t)per_layer_hidden;
                    std::memcpy(per_layer_tmp.data() + dst_off,
                                prefill_per_layer_inputs.data() + src_off,
                                (size_t)per_layer_hidden * sizeof(unsigned short));
                }
                llm_h2d(LLM_WADDR(*t_per_layer),
                        per_layer_tmp.data(),
                        std::min((size_t)t_per_layer->nSize, per_layer_tmp.size() * sizeof(unsigned short)),
                        devid);
            }
            if (!ensure_layer_loaded(m)) return final_out;
            if (debug_prefill)
            {
                ALOGI("prefill layer %d begin: gid=%d devid=%d input_tokens=%d history_len=%d layer_kv_cache_num=%d",
                      m,
                      layer_prefill_grpid,
                      devid,
                      input_num_token,
                      history_len,
                      layer_kv_cache_num);
            }
            if (m == 0 && p == 0)
                dump_selected_prefill_tensors(lyr.layer, layer_prefill_grpid, -1);
            const bool linear_prefill = is_linear_layer(m);
            const int layer_decode_grpid = decode_gid_for_layer(m, decode_grpid);
            const auto layer_infer_t0 = prefill_chunk_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
            const int infer_ret = lyr.layer.inference(layer_prefill_grpid);
            if (prefill_chunk_profile_enabled)
            {
                prefill_layer_inference_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - layer_infer_t0).count();
                if (infer_ret == 0) ++prefill_profiled_layers;
            }
            if (infer_ret != 0)
            {
                ALOGW("prefill layer %d gid=%d inference failed: 0x%x", m, layer_prefill_grpid, infer_ret);
                log_prefill_chunk_profile("failed", m, "inference");
                set_last_error("prefill layer inference 失败: chunk=" + std::to_string(p + 1) + "/" +
                               std::to_string(prefill_split_num) + " layer=" + std::to_string(m) +
                               " group=" + std::to_string(layer_prefill_grpid) +
                               " ret=" + std::to_string(infer_ret));
                ResetKVCache();
                return final_out;
            }
            if (debug_prefill) ALOGI("prefill layer %d inference done", m);
            if (std::getenv("AXLLM_DEBUG_LAYER0_IO") && m == 0 && p == 0)
            {
                try
                {
                    const auto &t_out0 = lyr.layer.get_output(layer_prefill_grpid, "output");
                    const int n = (int)(t_out0.nSize / sizeof(unsigned short));
                    if (n > 0 && t_out0.pVirAddr)
                    {
                        float min_v = 0.0f, max_v = 0.0f, mean_abs = 0.0f;
                        summarize_bf16_buffer((const unsigned short *)t_out0.pVirAddr, n, min_v, max_v, mean_abs);
                        ALOGI("DBGIO step=0 gid=%d output hash=0x%llx min=%.6f max=%.6f mean_abs=%.6f",
                              layer_prefill_grpid,
                              (unsigned long long)hash_u16_buffer((const unsigned short *)t_out0.pVirAddr, n),
                              min_v,
                              max_v,
                              mean_abs);
                    }
                }
                catch (const std::exception &)
                {
                }
            }
            auto &dec_k = lyr.layer.get_input(layer_decode_grpid, "K_cache");
            auto &dec_v = lyr.layer.get_input(layer_decode_grpid, "V_cache");
            if (linear_prefill)
            {
                (void)dec_k;
                (void)dec_v;
                // Recurrent state is canonicalized in decode groups; avoid
                // an O(number-of-prefill-buckets) copy after every chunk.
                if (!sync_linear_input_state_from_group(m, lyr.layer, layer_prefill_grpid, devid, false, false))
                {
                    set_last_error("模型运行失败，请重新尝试。");
                    ResetKVCache();
                    return final_out;
                }
            }
            else
            {
                auto &out_k = lyr.layer.get_output(layer_prefill_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(layer_prefill_grpid, "V_cache_out");
                const int layer_kv = kv_cache_size_for_layer(m);
                int kv_off = history_len * layer_kv;
                size_t kv_sz = (size_t)input_num_token * (size_t)layer_kv * sizeof(unsigned short);
                llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
            }

            auto &t_out = lyr.layer.get_output(layer_prefill_grpid, "output");
            dump_prefill_bf16(
                "chunk" + std::to_string(p) + "_layer" + std::to_string(m) + "_output",
                t_out
            );
            llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), embed_tmp.size() * sizeof(unsigned short), devid);
            if (debug_prefill) ALOGI("prefill layer %d output d2h done", m);
            if (std::getenv("AXLLM_DEBUG_DECODE_TRACE") &&
                p == prefill_split_num - 1 &&
                input_num_token > 0 &&
                (m == 0 || m == _attr.axmodel_num - 1))
            {
                const unsigned short *last_token_hidden =
                    embed_tmp.data() + (size_t)(input_num_token - 1) * (size_t)_attr.tokens_embed_size;
                dump_hidden_trace(m == 0 ? "prefill_layer0_last_hidden" : "prefill_last_layer_hidden",
                                  m,
                                  last_token_hidden,
                                  _attr.tokens_embed_size);
            }

            if (has_vision_state &&
                !vision_state.deepstack_features.empty() &&
                (size_t)m < vision_state.deepstack_features.size() &&
                !vision_state.deepstack_features[m].empty())
            {
                const int start_pos = token_start_pos;
                const int emb_sz = _attr.tokens_embed_size;
                const auto &feat = vision_state.deepstack_features[m];

                for (int j = 0; j < input_num_token; ++j)
                {
                    const int abs_pos = start_pos + j;
                    if ((size_t)abs_pos >= vision_state.pos2vision.size()) continue;
                    const int vidx = vision_state.pos2vision[abs_pos];
                    if (vidx < 0) continue;

                    const float *fv = feat.data() + (size_t)vidx * (size_t)emb_sz;
                    unsigned short *ev = embed_tmp.data() + (size_t)j * (size_t)emb_sz;

                    for (int di = 0; di < emb_sz; ++di)
                    {
                        unsigned int tmp_bf16 = ((unsigned int)ev[di]) << 16;
                        float fp32 = *reinterpret_cast<float *>(&tmp_bf16);
                        ev[di] = fp32_to_bfloat16_rne(fp32 + fv[di]);
                    }
                }
            }
        }

        if (debug_prefill && !prefill_aliased_layers.empty())
        {
            ALOGI("prefill IO alias: chunk=%d reused_layers=%zu/%d",
                  p,
                  prefill_aliased_layers.size(),
                  _attr.axmodel_num);
        }

        mark_full_cache_slots(history_len, input_num_token);
        capture_linear_state_snapshot(history_len + input_num_token);
        if (p == prefill_split_num - 1)
            memcpy(embed.data(), embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size, _attr.tokens_embed_size * sizeof(unsigned short));
        log_prefill_chunk_profile("ok", -1, "complete");
    }

    int next_token = -1; t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
    bool b_hit_eos = false;
    if (use_per_layer_input)
        gemma4_per_layer_helper.reset_decode_stats(decode_profile_enabled);
    int last_shared_sync_decode_grpid = -1;
    // Compute the decode rope/kv start positions up front so the VLM-positions
    // diagnostic prints BEFORE the first streamed token below, instead of
    // landing mid-line right after it (it used to be logged just before the
    // decode loop, after the first token was already emitted). These values are
    // invariant across the post stage, so the decode loop reuses them as-is.
    const unsigned int dense_decode_start = (unsigned int)(precompute_len + input_embed_num);
    unsigned int decode_start = dense_decode_start;
    if (has_vision_state && vision_state.decode_start > 0) decode_start = (unsigned int)vision_state.decode_start;
    else if (active_prefill_pos_start >= 0) decode_start = (unsigned int)(active_prefill_pos_start + input_embed_num);
    if (decode_start != dense_decode_start)
        ALOGI("VLM decode positions: rope_start=%u dense_kv_start=%u", decode_start, dense_decode_start);
    {
        auto &t_in = llama_post.get_input("input");
        if (debug_prefill) ALOGI("post stage begin: input_bytes=%u device=%d", t_in.nSize, LLM_DEVID(llama_layers.back()));
        dump_hidden_trace("post_input_hidden", 0, embed.data(), _attr.tokens_embed_size);
        llm_h2d(LLM_WADDR(t_in), embed.data(), embed.size() * sizeof(unsigned short), LLM_DEVID(llama_layers.back()));
        if (debug_prefill) ALOGI("post stage h2d done");
        const int post_infer_ret = llama_post.inference();
        if (post_infer_ret != 0)
        {
            ALOGW("post layer inference failed: ret=0x%x", post_infer_ret);
            set_last_error("post layer inference 失败: ret=" + std::to_string(post_infer_ret));
            ResetKVCache();
            return final_out;
        }
        if (debug_prefill) ALOGI("post stage inference done");
        auto &t_out = llama_post.get_output("output");
        llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
        if (debug_prefill) ALOGI("post stage d2h done: output_bytes=%u", t_out.nSize);
        unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
        next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
        dump_decode_trace(0, post_out, _attr.tokens_embed_num, next_token);
        const float ttft_text_ms = ttft_timer.cost();
        const float ttft_e2e_ms = CurrentRequestElapsedMs();
        if (ttft_e2e_ms >= 0.0f)
        {
            const float ttft_prepare_ms = std::max(0.0f, ttft_e2e_ms - ttft_text_ms);
            ALOGI("ttft: %.2f ms (prepare=%.2f ms, text=%.2f ms)", ttft_e2e_ms, ttft_prepare_ms, ttft_text_ms);
            ALOGI("ttft_prepare: %.2f ms", ttft_prepare_ms);
            ALOGI("ttft_text: %.2f ms", ttft_text_ms);
        }
        else
        {
            ALOGI("ttft: %.2f ms", ttft_text_ms);
        }
        // Store the same value reported by the "ttft:" log line above so the
        // OpenAI usage object matches on-device measurements.
        last_run_ttft_ms_ = (ttft_e2e_ms >= 0.0f) ? ttft_e2e_ms : ttft_text_ms;
        // Prefill throughput: prompt tokens processed this run over the time to
        // produce the first token (ttft_text_ms ~= prefill compute time, since the
        // first token comes straight out of the last prefill forward pass).
        if (ttft_text_ms > 0.0f && input_embed_num > 0)
            last_run_prefill_tps_ = (float)input_embed_num / (ttft_text_ms / 1000.0f);
        // Tokens actually prefilled this turn (prompt minus KV-cache-reused prefix).
        last_run_prefill_tokens_ = input_embed_num;
        if (debug_prefill) ALOGI("first decode token: id=%d", next_token);
        if (next_token < 0 || next_token >= _attr.tokens_embed_num)
        {
            ALOGE("first decode token out of range: token=%d vocab=%d", next_token, _attr.tokens_embed_num);
            return final_out;
        }
        b_hit_eos = tokenizer->is_stop(next_token);
        if (b_hit_eos)
        {
            ALOGW("first decode token hit stop: token=%d piece='%s' precompute_len=%d input_tokens=%d",
                  next_token,
                  safe_decode_token(tokenizer, next_token).c_str(),
                  precompute_len,
                  input_embed_num);
        }
        if (!b_hit_eos)
        {
            token_ids.push_back(next_token);
            decode_timer.start();
            decode_timer_started = true;
            if (_attr.runing_callback)
            {
                auto str = safe_decode_token(tokenizer, next_token);
                if (hide_channel_markup) str = channel_filter.filter(str);
                emit_stream_chunk(str, -1);
            }
            if (output_max_token > 0 && (int)token_ids.size() >= output_max_token)
            {
                b_hit_eos = true;
            }
        }
    }

    t_cost.start();
    const int logical_max_context = logical_context_limit();
    for (unsigned int decode_pos = decode_start, kv_slot = dense_decode_start;
         !b_hit_eos && decode_pos < (unsigned int)logical_max_context && kv_slot < (unsigned int)logical_max_context;
         ++decode_pos, ++kv_slot)
    {
        if (b_stop.load(std::memory_order_relaxed)) break;
        bool need_full_shared_sync = false;
        {
            const int want_gid = choose_decode_gid((int)kv_slot + 1);
            if (want_gid != decode_grpid)
            {
                // ALOGI("switch decode_grpid: %d -> %d (kv_ctx=%u rope_pos=%u)", decode_grpid, want_gid, kv_slot + 1, decode_pos);
                if (!sync_device_kv_cache_from_decode(decode_grpid, want_gid, (int)kv_slot, false))
                {
                    set_last_error("模型运行失败，请重新尝试。");
                    ResetKVCache();
                    return final_out;
                }
                decode_grpid = want_gid;
            }
            need_full_shared_sync = (decode_grpid != last_shared_sync_decode_grpid);
        }
        embed_selector.getByIndex(next_token, embed);
        scale_all_embeds_inplace(embed.data(), 1);
        if (use_per_layer_input)
        {
            const auto t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
            if (!gemma4_per_layer_helper.ComputeSingle(next_token,
                                                       embed.data(),
                                                       _attr.tokens_embed_size,
                                                       decode_per_layer_input))
            {
                ALOGE("Gemma4 decode per-layer input compute failed");
                return final_out;
            }
            if (decode_profile_enabled)
                decode_profile.per_layer_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - t0).count();
        }

#ifdef USE_AXCL
        {
            const int layer0_decode_grpid = decode_gid_for_layer(0, decode_grpid);
            auto &l0_in = llama_layers[0].layer.get_input(layer0_decode_grpid, "input");
            llm_h2d(LLM_WADDR(l0_in), embed.data(), l0_in.nSize, llama_layers[0].layer.get_devid());
        }
        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break; auto &lyr = llama_layers[m]; int devid = lyr.layer.get_devid();
            const int layer_decode_grpid = decode_gid_for_layer(m, decode_grpid);
            auto &in_k = lyr.layer.get_input(layer_decode_grpid, "K_cache");
            auto &in_v = lyr.layer.get_input(layer_decode_grpid, "V_cache");
            const int shared_src = shared_kv_source_for_layer(m);
            if (shared_src >= 0)
            {
                auto &src_layer = llama_layers[(size_t)shared_src];
                const int src_decode_grpid = decode_gid_for_layer(shared_src, decode_grpid);
                auto &src_k = src_layer.layer.get_input(src_decode_grpid, "K_cache");
                auto &src_v = src_layer.layer.get_input(src_decode_grpid, "V_cache");
                const int layer_kv = kv_cache_size_for_layer(m);
                const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                const size_t visible_past = std::min((size_t)kv_slot, dst_tokens > 0 ? (dst_tokens - 1) : 0);
                llm_memset(LLM_WADDR(in_k), 0, in_k.nSize, devid);
                llm_memset(LLM_WADDR(in_v), 0, in_v.nSize, devid);
                if (visible_past > 0)
                {
                    const size_t past_bytes = visible_past * (size_t)layer_kv * sizeof(unsigned short);
                    llm_d2d(LLM_WADDR(in_k), LLM_RADDR(src_k), std::min(past_bytes, (size_t)src_k.nSize), devid);
                    llm_d2d(LLM_WADDR(in_v), LLM_RADDR(src_v), std::min(past_bytes, (size_t)src_v.nSize), devid);
                }
                if (dst_tokens > 0)
                {
                    const size_t cur_off_bytes = (size_t)kv_slot * (size_t)layer_kv * sizeof(unsigned short);
                    const size_t dst_off_bytes = (dst_tokens - 1) * (size_t)layer_kv * sizeof(unsigned short);
                    llm_d2d((unsigned char *)LLM_WADDR(in_k) + dst_off_bytes,
                            (const unsigned char *)LLM_RADDR(src_k) + cur_off_bytes,
                            sizeof(unsigned short) * (size_t)layer_kv, devid);
                    llm_d2d((unsigned char *)LLM_WADDR(in_v) + dst_off_bytes,
                            (const unsigned char *)LLM_RADDR(src_v) + cur_off_bytes,
                            sizeof(unsigned short) * (size_t)layer_kv, devid);
                }
            }
            auto &t_idx = lyr.layer.get_input(layer_decode_grpid, "indices"); llm_h2d(LLM_WADDR(t_idx), &decode_pos, sizeof(decode_pos), devid);
            set_kv_block_count_if_present(lyr.layer, layer_decode_grpid, (int)kv_slot, devid);
            auto &t_mask= lyr.layer.get_input(layer_decode_grpid, "mask");
            if (is_linear_layer(m))
            {
                const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                std::vector<unsigned short> linear_decode_mask(elems, bf16_one.data);
                if (linear_decode_mask.empty()) linear_decode_mask.push_back(bf16_one.data);
                llm_h2d(LLM_WADDR(t_mask), linear_decode_mask.data(), linear_decode_mask.size() * sizeof(unsigned short), devid);
            }
            else
            {
                const int mask_elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
                if (use_sparse_full_cache_mask())
                    build_sparse_layer_decode_mask(decode_mask, mask_elems, (int)kv_slot, m);
                else
                    build_layer_decode_mask(decode_mask, mask_elems, (int)kv_slot, m);
                llm_h2d(LLM_WADDR(t_mask),
                        decode_mask.data(),
                        std::min((size_t)t_mask.nSize, (size_t)mask_elems * sizeof(unsigned short)),
                        devid);
            }
            if (use_per_layer_input)
            {
                const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, layer_decode_grpid, "per_layer_input");
                if (!t_per_layer)
                {
                    ALOGE("Gemma4 decoder layer %d is missing decode per_layer_input", m);
                    return final_out;
                }
                const size_t src_off = (size_t)m * (size_t)per_layer_hidden;
                llm_h2d(LLM_WADDR(*t_per_layer),
                        decode_per_layer_input.data() + src_off,
                        std::min((size_t)t_per_layer->nSize, (size_t)per_layer_hidden * sizeof(unsigned short)),
                        devid);
            }
            if (!ensure_layer_loaded(m)) return final_out;
            const int decode_infer_ret = lyr.layer.inference(layer_decode_grpid);
            if (decode_infer_ret != 0)
            {
                ALOGW("decode layer %d gid=%d inference failed: 0x%x", m, layer_decode_grpid, decode_infer_ret);
                set_last_error("decode layer inference 失败: layer=" + std::to_string(m) +
                               " group=" + std::to_string(layer_decode_grpid) +
                               " kv_slot=" + std::to_string(kv_slot) +
                               " ret=" + std::to_string(decode_infer_ret));
                ResetKVCache();
                return final_out;
            }
            if (is_linear_layer(m))
            {
                (void)in_k;
                (void)in_v;
                if (!sync_linear_input_state_from_group(m, lyr.layer, layer_decode_grpid, devid, false))
                {
                    set_last_error("模型运行失败，请重新尝试。");
                    ResetKVCache();
                    return final_out;
                }
            }
            else
            {
                auto &out_k = lyr.layer.get_output(layer_decode_grpid, "K_cache_out"); auto &out_v = lyr.layer.get_output(layer_decode_grpid, "V_cache_out");
                const int layer_kv = kv_cache_size_for_layer(m);
                if (shared_src >= 0)
                {
                    const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                    if (dst_tokens > 0)
                    {
                        const size_t cur_off = (size_t)kv_slot * (size_t)layer_kv;
                        const size_t tail_off = (dst_tokens - 1) * (size_t)layer_kv;
                        const size_t copy_bytes = sizeof(unsigned short) * (size_t)layer_kv;
                        // Shared-KV layers consume the source layer's KV. Preserve that
                        // source KV in the past slot; their own K_cache_out is not the
                        // cache that should be visible to the next decode token.
                        llm_d2d((unsigned short *)LLM_WADDR(in_k) + cur_off,
                                (unsigned short *)LLM_RADDR(in_k) + tail_off,
                                copy_bytes,
                                devid);
                        llm_d2d((unsigned short *)LLM_WADDR(in_v) + cur_off,
                                (unsigned short *)LLM_RADDR(in_v) + tail_off,
                                copy_bytes,
                                devid);
                    }
                }
                else
                {
                    llm_d2d((unsigned short *)LLM_WADDR(in_k) + kv_slot * layer_kv, LLM_RADDR(out_k), std::min((size_t)out_k.nSize, (size_t)layer_kv * sizeof(unsigned short)), devid);
                    llm_d2d((unsigned short *)LLM_WADDR(in_v) + kv_slot * layer_kv, LLM_RADDR(out_v), std::min((size_t)out_v.nSize, (size_t)layer_kv * sizeof(unsigned short)), devid);
                }
            }
            auto &cur_out = lyr.layer.get_output(layer_decode_grpid, "output");
            if (m == _attr.axmodel_num - 1)
            {
                auto &post_in = llama_post.get_input("input");
                if (llama_post.get_devid() == devid) { llm_d2d(LLM_WADDR(post_in), LLM_RADDR(cur_out), post_in.nSize, devid); }
                else { llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid); llm_h2d(LLM_WADDR(post_in), cur_out.pVirAddr, post_in.nSize, llama_post.get_devid()); }
            }
            else
            {
                const int next_decode_grpid = decode_gid_for_layer(m + 1, decode_grpid);
                auto &next_in = llama_layers[m + 1].layer.get_input(next_decode_grpid, "input"); int next_devid = llama_layers[m + 1].layer.get_devid();
                if (next_devid == devid) { llm_d2d(LLM_WADDR(next_in), LLM_RADDR(cur_out), next_in.nSize, devid); }
                else { llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid); llm_h2d(LLM_WADDR(next_in), cur_out.pVirAddr, next_in.nSize, next_devid); }
            }
        }
        if (use_sparse_full_cache_mask()) mark_full_cache_slot((int)kv_slot);
        const int post_infer_ret = llama_post.inference();
        if (post_infer_ret != 0)
        {
            ALOGW("post layer inference failed: ret=0x%x", post_infer_ret);
            set_last_error("post layer inference 失败: ret=" + std::to_string(post_infer_ret));
            ResetKVCache();
            return final_out;
        }
        {
            auto &t_out = llama_post.get_output("output"); llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
        }
#else // AX650
        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break; auto &lyr = llama_layers[m]; int devid = lyr.layer.get_devid();
            const int layer_decode_grpid = decode_gid_for_layer(m, decode_grpid);
            auto &in_k = lyr.layer.get_input(layer_decode_grpid, "K_cache"); auto *in_k_ptr = (unsigned short *)in_k.pVirAddr;
            auto &in_v = lyr.layer.get_input(layer_decode_grpid, "V_cache"); auto *in_v_ptr = (unsigned short *)in_v.pVirAddr;
            const int shared_src = shared_kv_source_for_layer(m);
            const auto shared_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
            if (shared_src >= 0)
            {
                auto &src_layer = llama_layers[(size_t)shared_src];
                const int src_decode_grpid = decode_gid_for_layer(shared_src, decode_grpid);
                auto &src_k = src_layer.layer.get_input(src_decode_grpid, "K_cache");
                auto &src_v = src_layer.layer.get_input(src_decode_grpid, "V_cache");
                const int layer_kv = kv_cache_size_for_layer(m);
                const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                if (need_full_shared_sync)
                {
                    const size_t visible_past = std::min((size_t)kv_slot, dst_tokens > 0 ? (dst_tokens - 1) : 0);
                    memset(in_k.pVirAddr, 0, in_k.nSize);
                    memset(in_v.pVirAddr, 0, in_v.nSize);
                    if (visible_past > 0)
                    {
                        const size_t past_bytes = visible_past * (size_t)layer_kv * sizeof(unsigned short);
                        memcpy(in_k.pVirAddr, src_k.pVirAddr, std::min(past_bytes, (size_t)src_k.nSize));
                        memcpy(in_v.pVirAddr, src_v.pVirAddr, std::min(past_bytes, (size_t)src_v.nSize));
                    }
                }
                if (dst_tokens > 0)
                {
                    const size_t cur_off = (size_t)kv_slot * (size_t)layer_kv;
                    const size_t dst_off = (dst_tokens - 1) * (size_t)layer_kv;
                    memcpy(in_k_ptr + dst_off, (const unsigned short *)src_k.pVirAddr + cur_off, sizeof(unsigned short) * (size_t)layer_kv);
                    memcpy(in_v_ptr + dst_off, (const unsigned short *)src_v.pVirAddr + cur_off, sizeof(unsigned short) * (size_t)layer_kv);
                }
            }
            if (decode_profile_enabled)
                decode_profile.shared_kv_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - shared_t0).count();
            const auto prep_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
            auto &t_idx = lyr.layer.get_input(layer_decode_grpid, "indices"); memcpy(t_idx.pVirAddr, &decode_pos, sizeof(decode_pos));
            set_kv_block_count_if_present(lyr.layer, layer_decode_grpid, (int)kv_slot, devid);
            auto &t_mask= lyr.layer.get_input(layer_decode_grpid, "mask");
            if (is_linear_layer(m))
            {
                const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                std::vector<unsigned short> linear_decode_mask(elems, bf16_one.data);
                if (linear_decode_mask.empty()) linear_decode_mask.push_back(bf16_one.data);
                memcpy(t_mask.pVirAddr, linear_decode_mask.data(), std::min((size_t)t_mask.nSize, linear_decode_mask.size() * sizeof(unsigned short)));
            }
            else
            {
                const int mask_elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
                if (use_sparse_full_cache_mask())
                    build_sparse_layer_decode_mask(decode_mask, mask_elems, (int)kv_slot, m);
                else
                    build_layer_decode_mask(decode_mask, mask_elems, (int)kv_slot, m);
                memcpy(t_mask.pVirAddr,
                       decode_mask.data(),
                       std::min((size_t)t_mask.nSize, (size_t)mask_elems * sizeof(unsigned short)));
            }
            if (use_per_layer_input)
            {
                const ax_runner_tensor_t *t_per_layer = try_get_group_input_tensor(lyr.layer, layer_decode_grpid, "per_layer_input");
                if (!t_per_layer)
                {
                    ALOGE("Gemma4 decoder layer %d is missing decode per_layer_input", m);
                    return final_out;
                }
                const size_t src_off = (size_t)m * (size_t)per_layer_hidden;
                memcpy(t_per_layer->pVirAddr,
                       decode_per_layer_input.data() + src_off,
                       std::min((size_t)t_per_layer->nSize, (size_t)per_layer_hidden * sizeof(unsigned short)));
            }
            auto &t_in  = lyr.layer.get_input(layer_decode_grpid, "input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            dump_debug_device_tensor(
                "token_" + std::to_string(token_ids.size()) +
                "_layer_" + std::to_string(m) + "_pre_K_state",
                in_k,
                devid
            );
            dump_debug_device_tensor(
                "token_" + std::to_string(token_ids.size()) +
                "_layer_" + std::to_string(m) + "_pre_V_state",
                in_v,
                devid
            );
            dump_debug_tensor(
                "token_" + std::to_string(token_ids.size()) +
                "_layer_" + std::to_string(m) + "_mask",
                t_mask
            );
            dump_debug_tensor(
                "token_" + std::to_string(token_ids.size()) +
                "_layer_" + std::to_string(m) + "_indices",
                t_idx
            );
            if (decode_profile_enabled)
                decode_profile.prep_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - prep_t0).count();
            const auto infer_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
            if (!ensure_layer_loaded(m)) return final_out;
            const int decode_infer_ret = lyr.layer.inference(layer_decode_grpid);
            if (decode_infer_ret != 0)
            {
                ALOGW("decode layer %d gid=%d inference failed: 0x%x", m, layer_decode_grpid, decode_infer_ret);
                set_last_error("decode layer inference 失败: layer=" + std::to_string(m) +
                               " group=" + std::to_string(layer_decode_grpid) +
                               " kv_slot=" + std::to_string(kv_slot) +
                               " ret=" + std::to_string(decode_infer_ret));
                ResetKVCache();
                return final_out;
            }
            if (decode_profile_enabled)
                decode_profile.inference_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - infer_t0).count();
            const auto out_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
            if (is_linear_layer(m))
            {
                (void)in_k;
                (void)in_v;
                if (!sync_linear_input_state_from_group(m, lyr.layer, layer_decode_grpid, 0, false))
                {
                    set_last_error("模型运行失败，请重新尝试。");
                    ResetKVCache();
                    return final_out;
                }
            }
            else
            {
                auto &out_k = lyr.layer.get_output(layer_decode_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(layer_decode_grpid, "V_cache_out");
                const int layer_kv = kv_cache_size_for_layer(m);
                if (shared_src >= 0)
                {
                    const size_t dst_tokens = (size_t)in_k.nSize / sizeof(unsigned short) / (size_t)std::max(1, layer_kv);
                    if (dst_tokens > 0)
                    {
                        const size_t cur_off = (size_t)kv_slot * (size_t)layer_kv;
                        const size_t tail_off = (dst_tokens - 1) * (size_t)layer_kv;
                        // See the AXCL branch above: shared layers must retain the
                        // source layer KV in their visible past cache.
                        const size_t copy_bytes = sizeof(unsigned short) * (size_t)layer_kv;
                        if (decode_explicit_output_d2h)
                        {
                            llm_d2d(in_k_ptr + cur_off, in_k_ptr + tail_off, copy_bytes, devid);
                            llm_d2d(in_v_ptr + cur_off, in_v_ptr + tail_off, copy_bytes, devid);
                        }
                        else
                        {
                            memcpy(in_k_ptr + cur_off, in_k_ptr + tail_off, copy_bytes);
                            memcpy(in_v_ptr + cur_off, in_v_ptr + tail_off, copy_bytes);
                        }
                    }
                }
                else
                {
                    const size_t copy_bytes = sizeof(unsigned short) * (size_t)layer_kv;
                    if (decode_explicit_output_d2h)
                    {
                        llm_d2d(in_k_ptr + kv_slot * layer_kv, LLM_RADDR(out_k), copy_bytes, devid);
                        llm_d2d(in_v_ptr + kv_slot * layer_kv, LLM_RADDR(out_v), copy_bytes, devid);
                    }
                    else
                    {
                        memcpy(in_k_ptr + kv_slot * layer_kv, out_k.pVirAddr, copy_bytes);
                        memcpy(in_v_ptr + kv_slot * layer_kv, out_v.pVirAddr, copy_bytes);
                    }
                }
            }
            auto &debug_decode_out = lyr.layer.get_output(layer_decode_grpid, "output");
            dump_debug_device_tensor(
                "token_" + std::to_string(token_ids.size()) +
                "_layer_" + std::to_string(m) + "_output",
                debug_decode_out,
                devid
            );
            dump_debug_tensor(
                "token_" + std::to_string(token_ids.size()) +
                "_layer_" + std::to_string(m) + "_K_state",
                in_k
            );
            dump_debug_tensor(
                "token_" + std::to_string(token_ids.size()) +
                "_layer_" + std::to_string(m) + "_V_state",
                in_v
            );
            auto &t_out= lyr.layer.get_output(layer_decode_grpid, "output");
            if (decode_explicit_output_d2h)
            {
                llm_d2h(embed.data(), LLM_RADDR(t_out), embed.size() * sizeof(unsigned short), devid);
            }
            else
            {
                memcpy(embed.data(), t_out.pVirAddr, embed.size() * sizeof(unsigned short));
            }
            if (decode_profile_enabled)
                decode_profile.outcopy_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - out_t0).count();
        }
        if (use_sparse_full_cache_mask()) mark_full_cache_slot((int)kv_slot);
        const auto post_t0 = decode_profile_enabled ? DecodeClock::now() : DecodeClock::time_point{};
        auto &t_in = llama_post.get_input("input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
        dump_hidden_trace("post_input_hidden", (int)token_ids.size(), embed.data(), _attr.tokens_embed_size);
        const int post_infer_ret = llama_post.inference();
        if (post_infer_ret != 0)
        {
            ALOGW("post layer inference failed: ret=0x%x", post_infer_ret);
            set_last_error("post layer inference 失败: ret=" + std::to_string(post_infer_ret));
            ResetKVCache();
            return final_out;
        }
        auto &t_out = llama_post.get_output("output");
        unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
        dump_debug_tensor("token_" + std::to_string(token_ids.size()) + "_post_logits", t_out);
        dump_decode_trace((int)token_ids.size(), post_out, _attr.tokens_embed_num, next_token);
        if (decode_profile_enabled)
            decode_profile.post_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(DecodeClock::now() - post_t0).count();
        last_shared_sync_decode_grpid = decode_grpid;
#endif

        if (tokenizer->is_stop(next_token)) { b_hit_eos = true; break; }
        token_ids.push_back(next_token);
        if (_attr.runing_callback)
        {
            float tps = -1.0f;
            if (decode_timer_started)
            {
                const int decode_tokens = std::max(0, (int)token_ids.size() - 1);
                const float decode_ms = decode_timer.cost();
                if (decode_tokens > 0 && decode_ms > 0.0f)
                    tps = decode_tokens / (decode_ms / 1000.0f);
            }
            auto  str   = safe_decode_token(tokenizer, next_token);
            if (hide_channel_markup) str = channel_filter.filter(str);
            emit_stream_chunk(str, tps);
        }
        if (output_max_token > 0 && (int)token_ids.size() >= output_max_token) { b_hit_eos = true; break; }
        if (_attr.runing_callback == nullptr) update_cqdm(&cqdm, kv_slot, "token", "");
    }

    const int generated_token_count = (int)token_ids.size();
    last_run_generated_token_ids = token_ids;
    final_out = tokenizer->decode(token_ids);
    if (hide_channel_markup)
    {
        final_out = strip_hidden_channel_sections(final_out);
    }
    final_out = sanitize_utf8_text(final_out);
    if (_attr.runing_callback)
    {
        std::string tail;
        if (hide_channel_markup)
            tail = channel_filter.flush();
        emit_stream_chunk(tail, -1);

        if (final_out.size() > streamed_visible_text.size() &&
            final_out.compare(0, streamed_visible_text.size(), streamed_visible_text) == 0)
        {
            _attr.runing_callback(final_out.substr(streamed_visible_text.size()), -1, _attr.reserve);
        }
    }
    printf("\n\n"); fflush(stdout);
    float avg_decode_tps = 0.0f;
    if (decode_timer_started)
    {
        const int decode_tokens = std::max(0, (int)token_ids.size() - 1);
        const float decode_ms = decode_timer.cost();
        if (decode_tokens > 0 && decode_ms > 0.0f)
            avg_decode_tps = decode_tokens / (decode_ms / 1000.0f);
    }
    last_run_decode_tps_ = avg_decode_tps;
    if (decode_profile_enabled)
    {
        const int decode_tokens = std::max(0, (int)token_ids.size() - 1);
        if (decode_tokens > 0)
        {
            const auto avg_ms = [decode_tokens](uint64_t ns) {
                return (double)ns / ((double)decode_tokens * 1e6);
            };
            ALOGN("decode profile avg/token: per_layer=%.3f ms, shared_kv=%.3f ms, prep=%.3f ms, inference=%.3f ms, outcopy=%.3f ms, post=%.3f ms\n",
                  avg_ms(decode_profile.per_layer_ns),
                  avg_ms(decode_profile.shared_kv_ns),
                  avg_ms(decode_profile.prep_ns),
                  avg_ms(decode_profile.inference_ns),
                  avg_ms(decode_profile.outcopy_ns),
                  avg_ms(decode_profile.post_ns));
            if (use_per_layer_input)
            {
                ALOGN("decode profile cache: hits=%llu misses=%llu\n",
                      (unsigned long long)gemma4_per_layer_helper.decode_cache_hits(),
                      (unsigned long long)gemma4_per_layer_helper.decode_cache_misses());
            }
        }
    }
    ALOGN("hit eos,decode avg %.2f token/s\n", avg_decode_tps);
    if (generated_token_count >= 0)
    {
        precompute_len = dense_decode_start + generated_token_count;
        if ((has_vision_state && !vision_state.position_ids.empty()) || active_prefill_pos_start >= 0)
            cached_mrope_next_pos = (int)decode_start + generated_token_count;
        else
            cached_mrope_next_pos = -1;
    }
    active_prefill_pos_start = -1;
    active_token_pos_start = -1;
    return final_out;
}

std::vector<Content> LLM::Impl::Run(std::vector<Content> history, int output_max_token)
{
    return Run(std::move(history), {}, output_max_token);
}

std::vector<Content> LLM::Impl::Run(std::vector<Content> history,
                                    const std::vector<::MediaInputs> &media_inputs,
                                    int output_max_token,
                                    std::vector<::MediaInputs> *normalized_media_inputs)
{
    clear_last_error();
    has_vision_state = false;
    std::vector<::MediaInputs> effective_media_inputs = media_inputs;
    bool video_history_isolated = false;
    bool cache_invalidated_after_generation = false;

    // Multi-slot prefix KV cache: pick the slot sharing the longest prefix with
    // this request; misses evict the LRU slot. Must run before the single-context
    // history-prefix reset below so the chosen slot's snapshot is in place when
    // the existing reuse logic runs. Text LLM matches on tokens; VLM matches on
    // chat history (tokenizing VLM history needs the expensive vision Prepare).
    if (kv_mgr_.multi_slot_enabled())
    {
        const bool slot_ready = _attr.vlm_type == VLMType::None
                                    ? kv_mgr_.select_kv_slot(tokenizer->encode(history))
                                    : kv_mgr_.select_kv_slot_by_history(history);
        if (!slot_ready) return history;
    }

    if (axllm::media_history::current_request_has_video(history, effective_media_inputs))
    {
        const size_t old_history_size = history.size();
        const size_t old_media_size = effective_media_inputs.size();
        if (axllm::media_history::isolate_current_video(history, effective_media_inputs))
        {
            ALOGW("video history is isolated to current user turn: old_history=%zu new_history=%zu old_media_inputs=%zu new_media_inputs=%zu",
                  old_history_size,
                  history.size(),
                  old_media_size,
                  effective_media_inputs.size());
            ResetKVCache();
            video_history_isolated = true;
        }
    }

    std::vector<int> new_tokens;
    std::string response_prefix;
    bool used_cached_text_turn = false;
    bool used_cached_media_turn = false;
    size_t append_start = last_history_snapshot.size();
    if (!last_history_snapshot.empty() && !is_history_prefix(last_history_snapshot, history))
    {
        // Non-append TEXT histories (same system prompt, different question) can share a
        // long common token prefix -- let token-level prefix reuse keep it instead of
        // wiping the KV and recomputing the shared prefix every time. For VLM, keep the
        // single-context reset: image placeholder token IDs are identical across different
        // images, so a token-level prefix match would wrongly reuse a previous image's KV.
        if (!kv_mgr_.multi_slot_active_request() && _attr.vlm_type != VLMType::None)
        {
            ALOGW("raw history not append. force ResetKVCache before request processing.");
            ResetKVCache();
        }
        append_start = 0;
    }

    const std::string tokenizer_key_for_cache = key_of(_attr.tokenizer_type);
    const bool is_minicpm_v46_cache =
        tokenizer_key_for_cache == "minicpmv46" ||
        tokenizer_key_for_cache == "minicpmv46vl";
    const bool allow_cached_text_turn =
        supports_cached_im_chat_turn_tokens() &&
        (!is_minicpm_v46_cache ||
         std::getenv("AXLLM_ENABLE_MINICPMV46_TEXT_KV_CACHE") != nullptr);
    const bool no_new_media_input = !has_new_media_input(effective_media_inputs, append_start);
    const bool cached_text_turn_requested = vision && vision->enabled() &&
                                            allow_cached_text_turn &&
                                            !last_history_snapshot.empty() &&
                                            precompute_len > 0 &&
                                            no_new_media_input &&
                                            history.size() > append_start &&
                                            appended_history_is_text_only_user_turn(history, append_start);
    const bool cached_media_turn_requested = vision && vision->enabled() &&
                                             supports_cached_im_chat_turn_tokens() &&
                                             !last_history_snapshot.empty() &&
                                             precompute_len > 0 &&
                                             !no_new_media_input &&
                                             history.size() > append_start &&
                                             appended_history_is_user_turn_with_media(history, append_start);
    const bool raw_history_appended = !last_history_snapshot.empty() && is_history_prefix(last_history_snapshot, history);
    const bool raw_history_rollback = !last_history_snapshot.empty() && is_history_prefix(history, last_history_snapshot);
    const bool raw_history_modified = !last_history_snapshot.empty() && !raw_history_appended && !raw_history_rollback;
    if (raw_history_modified)
    {
        if (cached_text_turn_requested || cached_media_turn_requested)
        {
            set_last_error("检测到历史被修改，无法复用已有 KV。请保持返回的 history 继续追加，或 /reset 后重新开始。");
            ALOGE("cached VLM turn cannot reuse KV because history is not append; refuse full recompute");
            return history;
        }
        // Non-append histories can still share a long common token prefix (e.g. same system prompt but
        // different user question). Leave KV intact here and let token-level prefix reuse decide later.
        ALOGI("raw history modified (not append / not rollback). try token-level KV prefix reuse.");
    }
    else if (raw_history_rollback)
    {
        ALOGI("raw history rollback detected: prev_contents=%zu new_contents=%zu, try KV reuse",
              last_history_snapshot.size(),
              history.size());
    }

    const bool history_appended = raw_history_appended && history.size() > append_start;

    if (cached_text_turn_requested)
    {
        if (!history_appended || !build_cached_im_chat_text_turn_tokens(history, append_start, new_tokens))
        {
            set_last_error("无法复用已有图像 KV，已拒绝重新编码历史。请 /reset 后重新开始。");
            ALOGE("failed to build cached text-only turn tokens; refuse vision Prepare/full recompute");
            return history;
        }
        used_cached_text_turn = true;
        ALOGI("reuse cached KV for text-only turn: cached_tokens=%zu append_contents=%zu input_tokens=%zu, skip vision Prepare",
              last_tokens_ids.size(),
              history.size() - append_start,
              new_tokens.size() - last_tokens_ids.size());
    }
    else if (cached_media_turn_requested)
    {
        std::string verr;
        if (!history_appended || !build_cached_im_chat_media_turn_tokens(history, effective_media_inputs, append_start, new_tokens, vision_state, verr))
        {
            set_last_error("无法增量处理新增多模态输入，请 /reset 后重新开始。");
            ALOGE("failed to build cached media turn tokens: %s", verr.c_str());
            return history;
        }
        has_vision_state = true;
        used_cached_media_turn = true;
        ALOGI("reuse cached KV for media turn: cached_tokens=%zu append_contents=%zu input_tokens=%zu",
              last_tokens_ids.size(),
              history.size() - append_start,
              new_tokens.size() - last_tokens_ids.size());
    }

    if (!used_cached_text_turn && !used_cached_media_turn && vision && vision->enabled())
    {
        // If caller provides media, we will fill num_media/num_media_tokens and build injection state.
        if (!effective_media_inputs.empty())
        {
            std::vector<vision::MediaInputs> vmins;
            vmins.reserve(effective_media_inputs.size());
            for (const auto &m : effective_media_inputs) vmins.push_back({m.content_index, m.uris});

            vision::PromptBudget budget;
            budget.last_tokens = last_tokens_ids;
            budget.precompute_len = precompute_len;
            budget.prefill_token_num = _attr.prefill_token_num;
            const int max_cap = logical_context_limit();
            budget.max_total_tokens = max_cap;
            budget.max_history_tokens = (_attr.prefill_token_num > 0)
                                            ? std::max(0, max_cap - _attr.prefill_token_num)
                                            : std::max(0, max_cap);
            // This is a logical context budget, not a count of full regular
            // chunks. Adaptive query profiles handle a short final chunk, so
            // retaining the complete remainder avoids rejecting valid prompts.
            int remaining = std::max(0, max_cap - precompute_len);
            budget.max_tail_tokens = remaining;

            std::vector<Content> prepared_history;
            std::vector<int> input_ids;
            vision::RunState st;
            vision::PrepareMetadata prepare_meta;
            std::string verr;
            if (!vision->Prepare(history, vmins, &budget, prepared_history, input_ids, st, verr, true, &prepare_meta))
            {
                if (verr.find("exceeds current prefill budget") != std::string::npos ||
                    verr.find("exceeds current history budget") != std::string::npos ||
                    verr.find("history budget") != std::string::npos) {
                    set_context_limit_error();
                } else {
                    set_last_error("多模态输入处理失败，请重新开始会话后再试一次。");
                }
                ALOGE("vision.Prepare failed: %s", verr.c_str());
                return history;
            }
            if (prepare_meta.auto_reset_for_video)
            {
                ALOGW("video request auto reset history for quality: current_frames=%d fresh_frames=%d",
                      prepare_meta.current_video_frames,
                      prepare_meta.fresh_video_frames);
                ResetKVCache();
                response_prefix = video_quality_reset_notice();
                if (_attr.runing_callback)
                {
                    _attr.runing_callback(response_prefix, -1, _attr.reserve);
                }
            }
            history = std::move(prepared_history);
            new_tokens = std::move(input_ids);
            vision_state = std::move(st);
            has_vision_state = true;
        }
        else
        {
            // If history contains multimodal content, the caller must provide media inputs.
            bool need_media = false;
            for (const auto &c : history) if (c.type == IMAGE || c.type == VIDEO || c.type == AUDIO) { need_media = true; break; }
            if (need_media)
            {
                ALOGE("vlm_type=%s/%d enabled but media_inputs is empty",
                      std::string(VLMTypeName(_attr.vlm_type)).c_str(),
                      (int)_attr.vlm_type);
            }
            new_tokens = tokenizer->encode(history);
        }
    }
    else if (!used_cached_text_turn && !used_cached_media_turn)
    {
        new_tokens = tokenizer->encode(history);
    }

    last_run_prompt_token_num_ = (int)new_tokens.size();
    int offset = 0;
    auto tokens_diff = diff_token_ids(last_tokens_ids, new_tokens, offset);
    const bool token_appended = (offset == (int)last_tokens_ids.size() && (int)new_tokens.size() >= (int)last_tokens_ids.size());
    const bool token_rollback = (offset == (int)new_tokens.size() && (int)new_tokens.size() <= (int)last_tokens_ids.size());
    const bool token_prefix_reuse = (!token_appended && !token_rollback &&
                                     offset > 0 &&
                                     precompute_len > 0 &&
                                     offset <= precompute_len);
    bool not_append = !(token_appended || token_rollback || token_prefix_reuse);
    if (not_append)
    {
        if (used_cached_text_turn)
        {
            set_last_error("KV 复用失败，已拒绝重新编码历史。请 /reset 后重新开始。");
            ALOGE("cached text-only turn token prefix mismatch: offset=%d cached_tokens=%zu new_tokens=%zu; refuse full recompute",
                  offset,
                  last_tokens_ids.size(),
                  new_tokens.size());
            return history;
        }
        if (used_cached_media_turn)
        {
            set_last_error("多模态 KV 复用失败，已拒绝重新编码历史。请 /reset 后重新开始。");
            ALOGE("cached media turn token prefix mismatch: offset=%d cached_tokens=%zu new_tokens=%zu; refuse full recompute",
                  offset,
                  last_tokens_ids.size(),
                  new_tokens.size());
            return history;
        }
        ALOGW("history diverged with no reusable prefix. force ResetKVCache and recompute.");
        ResetKVCache();
        tokens_diff = new_tokens;
        offset = 0;
    }
    else if (token_rollback)
    {
        const int new_len = (int)new_tokens.size();
        const int prev_tokens = (int)last_tokens_ids.size();
        const int prev_kv = precompute_len;
        int keep = new_len;

        if (has_linear_attention_layers())
        {
            const int target = tokens_diff.empty() ? std::max(0, new_len - 1) : new_len;
            keep = best_linear_state_snapshot_len(target);
            if (target > 0 && keep < 0)
            {
                ALOGW("token rollback has no linear state snapshot <= %d. force ResetKVCache and recompute.", target);
                ResetKVCache();
                tokens_diff = new_tokens;
                offset = 0;
                not_append = false;
            }
            else
            {
                keep = std::max(0, keep);
                if (!restore_linear_state_snapshot_to_device(keep))
                {
                    ALOGW("failed to restore linear state snapshot at %d. force ResetKVCache and recompute.", keep);
                    ResetKVCache();
                    tokens_diff = new_tokens;
                    offset = 0;
                    not_append = false;
                }
                else
                {
                    drop_linear_state_snapshots_after(keep);
                    if (prev_tokens != keep) last_tokens_ids.resize((size_t)keep);
                    if (precompute_len > keep) precompute_len = keep;
                    if (cached_mrope_next_pos >= keep) cached_mrope_next_pos = -1;
                    offset = keep;
                    tokens_diff.assign(new_tokens.begin() + keep, new_tokens.end());
                    ALOGI("token rollback: reuse KV prefix tokens=%d via linear snapshot (prev_tokens=%d prev_kv=%d) recompute_suffix=%zu",
                          keep,
                          prev_tokens,
                          prev_kv,
                          tokens_diff.size());
                }
            }
        }
        else
        {
            if (prev_tokens != keep)
            {
                last_tokens_ids.resize((size_t)keep);
            }
            if (precompute_len > keep)
            {
                precompute_len = keep;
            }
            if (cached_mrope_next_pos >= keep)
            {
                cached_mrope_next_pos = -1;
            }

            ALOGI("token rollback: reuse KV prefix tokens=%d (prev_tokens=%d prev_kv=%d)",
                  keep,
                  prev_tokens,
                  prev_kv);
        }
    }
    else if (token_prefix_reuse)
    {
        int keep = offset;
        const int prev_tokens = (int)last_tokens_ids.size();
        const int prev_kv = precompute_len;

        if (has_linear_attention_layers())
        {
            // Cap the reuse target to the max prefill history capacity before snapping to
            // a linear-state snapshot, so the resulting prefix fits a single prefill group.
            const int max_hist_lin = max_prefill_history_cap();
            const int want_lin = (max_hist_lin > 0 && offset > max_hist_lin) ? max_hist_lin : offset;
            keep = best_linear_state_snapshot_len(want_lin);
            if (keep < 0)
            {
                ALOGW("token prefix reuse has no linear state snapshot <= %d. force ResetKVCache and recompute.", offset);
                ResetKVCache();
                tokens_diff = new_tokens;
                offset = 0;
                not_append = false;
            }
            else if (!restore_linear_state_snapshot_to_device(keep))
            {
                ALOGW("failed to restore linear state snapshot at %d. force ResetKVCache and recompute.", keep);
                ResetKVCache();
                tokens_diff = new_tokens;
                offset = 0;
                not_append = false;
            }
            else
            {
                drop_linear_state_snapshots_after(keep);
                if (prev_tokens != keep) last_tokens_ids.resize((size_t)keep);
                if (precompute_len > keep) precompute_len = keep;
                if (cached_mrope_next_pos >= keep) cached_mrope_next_pos = -1;
                offset = keep;
                tokens_diff.assign(new_tokens.begin() + keep, new_tokens.end());
                ALOGI("token prefix reuse: reuse KV prefix tokens=%d via linear snapshot (prev_tokens=%d prev_kv=%d) recompute_suffix=%zu",
                      keep,
                      prev_tokens,
                      prev_kv,
                      tokens_diff.size());
            }
        }
        else
        {
            // Reuse whole prefill chunks only: round the reused prefix down to a
            // prefill-chunk boundary, and cap to what a single prefill can attend to as
            // history (max_prefill_history_cap, itself chunk-aligned). Recompute the
            // remaining tokens (the partial last chunk + the divergent suffix).
            const int step = std::max(1, _attr.prefill_token_num);
            const int cap = max_prefill_history_cap();
            int aligned = (keep / step) * step;
            if (cap > 0 && aligned > cap) aligned = (cap / step) * step;
            if (aligned != keep)
            {
                ALOGW("token prefix reuse: chunk-align prefix %d -> %d (step=%d cap=%d)", keep, aligned, step, cap);
                keep = aligned;
                offset = keep;
                tokens_diff.assign(new_tokens.begin() + keep, new_tokens.end());
            }
            if (prev_tokens != keep)
            {
                last_tokens_ids.resize((size_t)keep);
            }
            if (precompute_len > keep)
            {
                precompute_len = keep;
            }
            if (cached_mrope_next_pos >= keep)
            {
                cached_mrope_next_pos = -1;
            }

            ALOGI("token prefix reuse: reuse KV prefix tokens=%d (prev_tokens=%d prev_kv=%d) recompute_suffix=%zu",
                  keep,
                  prev_tokens,
                  prev_kv,
                  tokens_diff.size());
        }
    }
    if (tokens_diff.empty())
    {
        if (used_cached_text_turn)
        {
            set_last_error("KV 复用失败：当前追问没有新增 token。已拒绝重新编码历史。");
            ALOGE("cached text-only turn has empty token diff; refuse full recompute");
            return history;
        }
        if (used_cached_media_turn)
        {
            set_last_error("多模态 KV 复用失败：当前追问没有新增 token。已拒绝重新编码历史。");
            ALOGE("cached media turn has empty token diff; refuse full recompute");
            return history;
        }
        if (!new_tokens.empty())
        {
            // Identical re-query: the reused prefix is the whole previous input minus one
            // token. Chunk-align + cap it to the prefill history capacity so it fits a
            // single prefill group and can be reused (otherwise SetKVCache would fail and
            // fall back to a full recompute). Linear models keep the exact value and rely
            // on the fallback (their state is snapshotted, not chunk-addressable).
            int keep = (int)new_tokens.size() - 1;
            if (!has_linear_attention_layers())
            {
                const int step = std::max(1, _attr.prefill_token_num);
                const int maxh = max_prefill_history_cap();
                if (maxh > 0 && keep > maxh) keep = maxh;
                keep = (keep / step) * step;
                if (keep < 0) keep = 0;
            }
            precompute_len = keep;
            offset = keep;
            tokens_diff.assign(new_tokens.begin() + keep, new_tokens.end());
            if ((int)last_tokens_ids.size() > keep) last_tokens_ids.resize((size_t)keep);
            ALOGI("identical re-query: reuse KV prefix tokens=%d recompute_suffix=%zu", keep, tokens_diff.size());
        }
        else { ResetKVCache(); precompute_len = 0; }
    }
    if (!not_append && offset != precompute_len && precompute_len > 0)
    {
        if (used_cached_text_turn || used_cached_media_turn)
        {
            set_last_error("KV 复用失败：token 前缀长度与 KV 长度不一致。已拒绝重新编码历史。");
            ALOGE("cached turn token/KV mismatch: token_offset=%d precompute_len=%d; refuse full recompute",
                  offset,
                  precompute_len);
            return history;
        }
        ALOGW("token prefix/KV length mismatch: token_offset=%d precompute_len=%d, recompute full history",
              offset,
              precompute_len);
        ResetKVCache();
        tokens_diff = new_tokens;
        offset = 0;
    }
    int kv_ret = SetKVCache(k_caches, v_caches, precompute_len, (int)tokens_diff.size());
    if (kv_ret != 0 && precompute_len > 0 && !used_cached_text_turn && !used_cached_media_turn)
    {
        // The reused prefix could not be set up in a single prefill group. This is NOT a
        // real context overflow -- fall back to a full chunked recompute instead of
        // surfacing a misleading "context exceeded" error to the user.
        ALOGW("prefix-reuse SetKVCache failed (precompute_len=%d suffix=%zu); fall back to full recompute",
              precompute_len, (int)tokens_diff.size());
        clear_last_error();
        ResetKVCache();
        tokens_diff = new_tokens;
        offset = 0;
        precompute_len = 0;
        kv_ret = SetKVCache(k_caches, v_caches, 0, (int)tokens_diff.size());
    }
    if (kv_ret != 0)
    {
        ALOGE("SetKVCache failed");
        return history;
    }
    active_token_pos_start = offset;
    active_prefill_pos_start = -1;
    if (has_vision_state && !vision_state.position_ids.empty() && offset != precompute_len)
    {
        ALOGI("VLM token/KV offset mismatch: token_offset=%d dense_kv_start=%d input_tokens=%zu",
              offset,
              precompute_len,
              tokens_diff.size());
    }
    if (!(has_vision_state && !vision_state.position_ids.empty()) && cached_mrope_next_pos >= 0 && precompute_len > 0)
    {
        active_prefill_pos_start = cached_mrope_next_pos;
        ALOGI("VLM cached mRoPE positions: prefill_rope_start=%d dense_kv_start=%d input_tokens=%zu",
              active_prefill_pos_start,
              precompute_len,
              tokens_diff.size());
    }
    std::vector<unsigned short> out_embed(tokens_diff.size() * _attr.tokens_embed_size);
    for (size_t i = 0; i < tokens_diff.size(); i++)
    {
        const int abs_pos = offset + (int)i;
        if (has_vision_state && (size_t)abs_pos < vision_state.pos2vision.size())
        {
            int vidx = vision_state.pos2vision[abs_pos];
            if (vidx >= 0)
            {
                memcpy(out_embed.data() + i * _attr.tokens_embed_size,
                       vision_state.vision_embed.data() + (size_t)vidx * _attr.tokens_embed_size,
                       (size_t)_attr.tokens_embed_size * sizeof(unsigned short));
                continue;
            }
        }
        embed_selector.getByIndex(tokens_diff[i], out_embed.data() + i * _attr.tokens_embed_size);
    }
    if (std::getenv("AXLLM_DEBUG_EMBED_TOKEN_IDS"))
    {
        std::string s;
        s.reserve(tokens_diff.size() * 8);
        for (size_t i = 0; i < tokens_diff.size(); ++i)
        {
            if (i) s.push_back(',');
            s += std::to_string(tokens_diff[i]);
        }
        ALOGI("Run history tokens_diff(len=%zu): %s", tokens_diff.size(), s.c_str());
    }
    std::vector<int> cached_prefix_tokens;
    if (precompute_len > 0)
    {
        const size_t prefix_len = std::min((size_t)precompute_len, last_tokens_ids.size());
        cached_prefix_tokens.assign(last_tokens_ids.begin(), last_tokens_ids.begin() + prefix_len);
        if ((int)cached_prefix_tokens.size() != precompute_len)
        {
            ALOGW("cached token prefix size mismatch before run: tokens=%zu precompute_len=%d",
                  cached_prefix_tokens.size(),
                  precompute_len);
        }
    }
    last_run_generated_token_ids.clear();
    run_input_token_ids = tokens_diff;
    auto reply = Run(out_embed, output_max_token);
    run_input_token_ids.clear();
    if (!response_prefix.empty())
    {
        reply = response_prefix + reply;
    }
    history.push_back({ASSISTANT, TEXT, reply});
    last_history_snapshot = history;
    last_tokens_ids = std::move(cached_prefix_tokens);
    last_tokens_ids.insert(last_tokens_ids.end(), tokens_diff.begin(), tokens_diff.end());
    last_tokens_ids.insert(last_tokens_ids.end(), last_run_generated_token_ids.begin(), last_run_generated_token_ids.end());
    // Chat-template normalization can make the returned history encode to a
    // different token stream than the raw stream consumed by the device. For
    // VLM follow-ups preserve that raw prefix (the vision state is authoritative);
    // text-only linear-attention models conservatively invalidate reuse.
    const std::vector<int> canonical_history_tokens = tokenizer->encode(history);
    int cached_prefix_len = 0;
    (void)axllm::qwen3_5::Runtime::token_suffix(last_tokens_ids, canonical_history_tokens, cached_prefix_len);
    if (has_linear_attention_layers() && !last_tokens_ids.empty() &&
        cached_prefix_len != static_cast<int>(last_tokens_ids.size()))
    {
        const bool preserve_vlm_raw_cache = vision && vision->enabled() && !video_history_isolated;
        if (preserve_vlm_raw_cache)
        {
            ALOGW("history/token cache prefix mismatch after generation; preserve raw VLM token/KV prefix for follow-up");
        }
        else
        {
            ALOGW("history/token cache prefix mismatch after generation; invalidate KV reuse");
            ResetKVCache();
            last_tokens_ids.clear();
            cache_invalidated_after_generation = true;
        }
        last_history_snapshot = history;
    }
    if (video_history_isolated)
    {
        ALOGW("drop KV cache after isolated video-history request");
        ResetKVCache();
    }
    else if (cache_invalidated_after_generation)
    {
        ALOGI("skip KV cache snapshot after invalidation; next request will recompute");
    }
    else
    {
        GetKVCache(k_caches, v_caches, precompute_len);
        if ((int)last_tokens_ids.size() != precompute_len)
        {
            ALOGW("exact cached token prefix length differs from KV: tokens=%zu precompute_len=%d generated=%zu input=%zu",
                  last_tokens_ids.size(),
                  precompute_len,
                  last_run_generated_token_ids.size(),
                  tokens_diff.size());
        }
    }

    has_vision_state = false;
    vision_state = {};

    // Persist the updated working state back into the active slot so the next
    // request can match/continue it. Device KV already lives in the slot's
    // own buffer (zero copy).
    kv_mgr_.save_active_kv_slot();

    if (normalized_media_inputs)
        *normalized_media_inputs = effective_media_inputs;

    return history;
}
