// Auto-split from LLM.cpp: embedding-path method bodies (LLM::Impl::Embed*).
#include "LLMImpl.hpp"

bool LLM::Impl::EmbedTokens(const std::vector<int> &token_ids, std::vector<float> &out_embedding)
{
    b_stop.store(false, std::memory_order_relaxed);

    if (token_ids.empty())
    {
        out_embedding.clear();
        return true;
    }

    // Debug: print token ids for alignment checks (opt-in via env var).
    // Useful when aligning embedding outputs against Python reference scripts.
    if (std::getenv("AXLLM_DEBUG_EMBED_TOKEN_IDS"))
    {
        std::string s;
        s.reserve(token_ids.size() * 8);
        for (size_t i = 0; i < token_ids.size(); ++i)
        {
            if (i) s.push_back(',');
            s += std::to_string(token_ids[i]);
        }
        ALOGI("EmbedTokens token_ids(len=%zu): %s", token_ids.size(), s.c_str());
    }

    const int input_embed_num = (int)token_ids.size();
    if (_attr.prefill_token_num <= 0 || _attr.tokens_embed_size <= 0 || _attr.kv_cache_size <= 0)
    {
        ALOGE("LLM embedding not initialized correctly (prefill_token_num/embed_size/kv_cache_size)");
        return false;
    }

    const int prefill_split_num = (int)std::ceil((double)input_embed_num / (double)_attr.prefill_token_num);

    // Stateless embedding: clear all KV caches once to avoid group-specific assumptions (e.g. gid=1 may have history_cap=0).
    clear_all_group_kv_cache_tensors();

    std::vector<int> prefill_grp_list;
    prefill_grp_list.resize(prefill_split_num, -1);
    for (int p = 0; p < prefill_split_num; ++p)
    {
        const int history_len = p * _attr.prefill_token_num;
        const int chunk_tokens = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;
        const bool prefer_symbolic_group = has_vision_state && history_len > 0;
        const int gid = select_prefill_group(history_len, chunk_tokens, prefer_symbolic_group);
        if (gid < 0)
        {
            ALOGE("failed to select prefill group for embedding: history_len=%d chunk_tokens=%d", history_len, chunk_tokens);
            return false;
        }
        prefill_grp_list[p] = gid;
    }

    std::vector<unsigned short> all_embed((size_t)input_embed_num * (size_t)_attr.tokens_embed_size);
    for (int i = 0; i < input_embed_num; i++)
    {
        if (has_vision_state &&
            (size_t)i < vision_state.pos2vision.size() &&
            vision_state.pos2vision[(size_t)i] >= 0 &&
            !vision_state.vision_embed.empty())
        {
            const int vidx = vision_state.pos2vision[(size_t)i];
            const size_t src_off = (size_t)vidx * (size_t)_attr.tokens_embed_size;
            if (src_off + (size_t)_attr.tokens_embed_size <= vision_state.vision_embed.size())
            {
                std::memcpy(all_embed.data() + (size_t)i * (size_t)_attr.tokens_embed_size,
                            vision_state.vision_embed.data() + src_off,
                            (size_t)_attr.tokens_embed_size * sizeof(unsigned short));
                continue;
            }
        }
        embed_selector.getByIndex((unsigned int)token_ids[(size_t)i], all_embed.data() + (size_t)i * (size_t)_attr.tokens_embed_size);
    }

    std::vector<unsigned short> last_hidden((size_t)_attr.tokens_embed_size, 0);
    std::vector<unsigned short> embed_tmp((size_t)_attr.prefill_token_num * (size_t)_attr.tokens_embed_size, 0);
    std::vector<unsigned short> mask_tmp;
    std::vector<unsigned short> linear_mask_tmp;
    bfloat16 bf16_one = 1.0f;

    for (int p = 0; p < prefill_split_num; p++)
    {
        if (b_stop.load(std::memory_order_relaxed)) break;

        const int history_len = p * _attr.prefill_token_num;
        const int input_num_token = (p == prefill_split_num - 1) ? (input_embed_num - p * _attr.prefill_token_num) : _attr.prefill_token_num;

        const int prefill_grpid = prefill_grp_list[p];
        int kv_cache_num = prefill_history_capacity_by_gid(prefill_grpid);
        const int kv_from_mask = prefill_history_capacity_by_mask(prefill_grpid);
        if (kv_from_mask >= 0) kv_cache_num = kv_from_mask;
        if (kv_cache_num < 0)
        {
            ALOGE("invalid kv_cache_num=%d for prefill_grpid=%d", kv_cache_num, prefill_grpid);
            return false;
        }

        mask_tmp.assign((size_t)_attr.prefill_token_num * (size_t)(kv_cache_num + _attr.prefill_token_num), bfloat16(-65536.f).data);
        build_prefill_mask(mask_tmp, kv_cache_num, _attr.prefill_token_num, history_len, input_num_token);

        std::fill(embed_tmp.begin(), embed_tmp.end(), 0);
        const size_t copy_tokens = (size_t)input_num_token;
        std::memcpy(embed_tmp.data(),
                    all_embed.data() + (size_t)history_len * (size_t)_attr.tokens_embed_size,
                    copy_tokens * (size_t)_attr.tokens_embed_size * sizeof(unsigned short));

        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;
            auto &lyr = llama_layers[m];
            const int devid = LLM_DEVID(lyr);
            const int layer_prefill_grpid = prefill_gid_for_layer(m, prefill_grpid);

            // indices
            const auto &t_idx = lyr.layer.get_input(layer_prefill_grpid, "indices");
            unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr;
            std::memset(idx_ptr, 0, (size_t)t_idx.nSize);
            {
                const int start_pos = history_len;
                const int idx_elems = (int)(t_idx.nSize / (int)sizeof(unsigned int));
                int idx_rows = _attr.prefill_token_num > 0 ? (idx_elems / _attr.prefill_token_num) : 1;
                if (idx_rows <= 0) idx_rows = 1;

                const bool use_pos_ids = has_vision_state &&
                                         idx_rows >= 3 &&
                                         vision_state.position_ids.size() >= 3 &&
                                         !vision_state.position_ids.empty();

                // Some models (e.g. Qwen-VL mRoPE) use multi-row indices. For multimodal embedding,
                // use vision_state.position_ids when available; otherwise fill sequential positions.
                for (int r = 0; r < idx_rows; ++r)
                {
                    for (int j = 0; j < input_num_token; ++j)
                    {
                        unsigned int v = (unsigned int)(start_pos + j);
                        if (use_pos_ids)
                        {
                            if ((size_t)r < vision_state.position_ids.size())
                            {
                                const auto &row = vision_state.position_ids[(size_t)r];
                                const int abs_pos = start_pos + j;
                                if ((size_t)abs_pos < row.size())
                                {
                                    v = (unsigned int)row[(size_t)abs_pos];
                                }
                            }
                        }
                        idx_ptr[(size_t)r * (size_t)_attr.prefill_token_num + (size_t)j] = v;
                    }
                }
            }
            llm_h2d(LLM_WADDR(t_idx), idx_ptr, (size_t)t_idx.nSize, devid);

            // mask
            const auto &t_mask = lyr.layer.get_input(layer_prefill_grpid, "mask");
            if (is_linear_layer(m))
            {
                const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                fill_linear_prefill_mask(linear_mask_tmp, elems, input_num_token);
                llm_h2d(LLM_WADDR(t_mask), linear_mask_tmp.data(), linear_mask_tmp.size() * sizeof(unsigned short), devid);
            }
            else
            {
                llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
            }

            // input
            const auto &t_in = lyr.layer.get_input(layer_prefill_grpid, "input");
            llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);

            // inference
            if (!ensure_layer_loaded(m)) return false;
            lyr.layer.inference(layer_prefill_grpid);

            // KV cache update
            const auto &in_k  = lyr.layer.get_input(layer_prefill_grpid, "K_cache");
            const auto &in_v  = lyr.layer.get_input(layer_prefill_grpid, "V_cache");
            if (is_linear_layer(m))
            {
                (void)in_k;
                (void)in_v;
                sync_linear_input_state_from_group(m, lyr.layer, layer_prefill_grpid, devid, true);
            }
            else
            {
                const auto &out_k = lyr.layer.get_output(layer_prefill_grpid, "K_cache_out");
                const auto &out_v = lyr.layer.get_output(layer_prefill_grpid, "V_cache_out");
                const int layer_kv = kv_cache_size_for_layer(m);
                const size_t layer_kv_off = (size_t)history_len * (size_t)layer_kv;
                const size_t layer_kv_sz = (size_t)input_num_token * (size_t)layer_kv * sizeof(unsigned short);
                llm_d2d((unsigned short *)LLM_WADDR(in_k) + layer_kv_off, LLM_RADDR(out_k), layer_kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(in_v) + layer_kv_off, LLM_RADDR(out_v), layer_kv_sz, devid);
            }

            // output -> embed_tmp for next layer
            const auto &t_out = lyr.layer.get_output(layer_prefill_grpid, "output");
            llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), std::min((size_t)t_out.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);

            // Qwen3-VL deepstack injection (vision tokens only).
            if (has_vision_state &&
                !vision_state.deepstack_features.empty() &&
                (size_t)m < vision_state.deepstack_features.size() &&
                !vision_state.deepstack_features[m].empty())
            {
                const int start_pos = history_len;
                const int emb_sz = _attr.tokens_embed_size;
                const auto &feat = vision_state.deepstack_features[m];
                for (int j = 0; j < input_num_token; ++j)
                {
                    const int abs_pos = start_pos + j;
                    if ((size_t)abs_pos >= vision_state.pos2vision.size()) continue;
                    const int vidx = vision_state.pos2vision[(size_t)abs_pos];
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

        if (p == prefill_split_num - 1)
        {
            const int local_last = input_embed_num - history_len - 1;
            if (local_last >= 0 && local_last < _attr.prefill_token_num)
            {
                std::memcpy(last_hidden.data(),
                            embed_tmp.data() + (size_t)local_last * (size_t)_attr.tokens_embed_size,
                            (size_t)_attr.tokens_embed_size * sizeof(unsigned short));
            }
        }
    }

    // Prefer post-process normalized output when available (e.g. Qwen3-VL-Embedding post exposes `output_norm`).
    // Otherwise fall back to the last hidden state directly.
    std::vector<unsigned short> embed_bf16 = last_hidden;

    const ax_runner_tensor_t *t_post_out = nullptr;
    const ax_runner_tensor_t *t_out_norm = try_get_output_tensor(llama_post, "output_norm");
    if (t_out_norm && (size_t)t_out_norm->nSize == (size_t)_attr.tokens_embed_size * sizeof(unsigned short))
    {
        t_post_out = t_out_norm;
    }
    else
    {
        const ax_runner_tensor_t *t_out = try_get_output_tensor(llama_post, "output");
        if (t_out && (size_t)t_out->nSize == (size_t)_attr.tokens_embed_size * sizeof(unsigned short))
        {
            t_post_out = t_out;
        }
    }

    if (t_post_out)
    {
        const auto &t_in = llama_post.get_input("input");
        llm_h2d(LLM_WADDR(t_in),
                last_hidden.data(),
                std::min((size_t)t_in.nSize, last_hidden.size() * sizeof(unsigned short)),
                llama_post.get_devid());
        llama_post.inference();

        std::vector<unsigned short> post_out((size_t)_attr.tokens_embed_size);
        llm_d2h(post_out.data(),
                LLM_RADDR(*t_post_out),
                std::min((size_t)t_post_out->nSize, post_out.size() * sizeof(unsigned short)),
                llama_post.get_devid());
        embed_bf16 = std::move(post_out);
    }

    out_embedding.resize((size_t)_attr.tokens_embed_size);
    for (int i = 0; i < _attr.tokens_embed_size; i++)
    {
        out_embedding[(size_t)i] = bfloat16(embed_bf16[(size_t)i]).fp32();
    }
    out_embedding = l2norm(std::move(out_embedding));
    return true;
}

bool LLM::Impl::EmbedHistory(const std::vector<Content> &history_in,
                  const std::vector<::MediaInputs> &media_inputs,
                  std::vector<float> &out_embedding)
{
    if (!tokenizer)
    {
        ALOGE("LLM not initialized");
        return false;
    }

    has_vision_state = false;
    vision_state = {};

    std::vector<int> token_ids;
    if (vision && vision->enabled())
    {
        if (!media_inputs.empty())
        {
            std::vector<vision::MediaInputs> vmins;
            vmins.reserve(media_inputs.size());
            for (const auto &m : media_inputs) vmins.push_back({m.content_index, m.uris});

            std::vector<Content> prepared_history;
            vision::RunState st;
            std::string verr;
            if (!vision->Prepare(history_in, vmins, nullptr, prepared_history, token_ids, st, verr, false))
            {
                ALOGE("vision.Prepare failed: %s", verr.c_str());
                return false;
            }
            (void)prepared_history;
            vision_state = std::move(st);
            has_vision_state = true;
        }
        else
        {
            token_ids = tokenizer->encode(history_in);
        }
    }
    else
    {
        token_ids = tokenizer->encode(history_in);
    }

    // Align with Python reference implementations for Qwen3/Qwen3-VL embedding models:
    // they include the final pad/end-of-text token (151643) as the pooled position.
    if (embedding_append_eos && embedding_eos_token_id >= 0)
    {
        token_ids.push_back(embedding_eos_token_id);
    }

    const bool ok = EmbedTokens(token_ids, out_embedding);

    has_vision_state = false;
    vision_state = {};

    return ok;
}

bool LLM::Impl::EmbedText(const std::string &text, std::vector<float> &out_embedding)
{
    if (!tokenizer)
    {
        ALOGE("LLM not initialized");
        return false;
    }
    std::vector<int> token_ids = tokenizer->encode(text);
    if (embedding_append_eos && embedding_eos_token_id >= 0) token_ids.push_back(embedding_eos_token_id);

    if (_attr.max_token_len > 0 && (int)token_ids.size() > _attr.max_token_len)
    {
        token_ids.resize((size_t)_attr.max_token_len);
        if (embedding_append_eos && !token_ids.empty()) token_ids.back() = embedding_eos_token_id;
    }

    return EmbedTokens(token_ids, out_embedding);
}

bool LLM::Impl::EmbedBatch(const std::vector<std::string> &inputs, std::vector<std::vector<float>> &out_embeddings)
{
    out_embeddings.clear();
    out_embeddings.reserve(inputs.size());
    for (const auto &s : inputs)
    {
        std::vector<float> e;
        if (!EmbedText(s, e)) return false;
        out_embeddings.push_back(std::move(e));
    }
    return true;
}

