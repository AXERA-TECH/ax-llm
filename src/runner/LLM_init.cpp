// Auto-split from LLM.cpp/LLMImpl.hpp: init + layer-topology method bodies (LLM::Impl::*).
#include "LLMImpl.hpp"

void LLM::Impl::init_layer_groups()
{
    layer_decode_grpids_.assign((size_t)_attr.axmodel_num, {});
    layer_prefill_grpids_.assign((size_t)_attr.axmodel_num, {});
    for (int i = 0; i < _attr.axmodel_num; ++i)
    {
        layer_decode_grpids_[(size_t)i] = detect_decode_grpids(llama_layers[(size_t)i].layer);
        layer_prefill_grpids_[(size_t)i] = detect_prefill_grpids(llama_layers[(size_t)i].layer);
        const auto &decode_gids = layer_decode_grpids_[(size_t)i];
        const auto &gids = layer_prefill_grpids_[(size_t)i];
        if (decode_gids.empty())
        {
            ALOGW("layer %d has no decode groups detected", i);
        }
        else if (decode_gids.size() < decode_grpids_.size())
        {
            ALOGI("layer %d decode groups=%zu ref=%zu, reuse gid %d for later decode shapes",
                  i,
                  decode_gids.size(),
                  decode_grpids_.size(),
                  decode_gids.back());
        }
        if (gids.empty())
        {
            ALOGW("layer %d has no prefill groups detected", i);
            continue;
        }
        if (gids.size() < prefill_grpids_.size())
        {
            ALOGI("layer %d prefill groups=%zu ref=%zu, reuse gid %d for later prefill chunks",
                  i,
                  gids.size(),
                  prefill_grpids_.size(),
                  gids.back());
        }
    }
}

void LLM::Impl::init_shared_kv_source_layers()
{
    shared_kv_source_layers.assign(_attr.axmodel_num, -1);

    if (_attr.layer_types.empty() || _attr.num_kv_shared_layers <= 0)
        return;

    const int num_layers = std::min(_attr.axmodel_num, (int)_attr.layer_types.size());
    const int first_shared_layer = num_layers - _attr.num_kv_shared_layers;
    if (first_shared_layer <= 0)
        return;

    std::vector<std::string> prev_layers(_attr.layer_types.begin(), _attr.layer_types.begin() + first_shared_layer);
    for (int layer_idx = first_shared_layer; layer_idx < num_layers; ++layer_idx)
    {
        const std::string &layer_type = _attr.layer_types[(size_t)layer_idx];
        for (int prev_idx = (int)prev_layers.size() - 1; prev_idx >= 0; --prev_idx)
        {
            if (prev_layers[(size_t)prev_idx] == layer_type)
            {
                shared_kv_source_layers[(size_t)layer_idx] = prev_idx;
                break;
            }
        }
    }
}

bool LLM::Impl::init_groups_from_model(ax_runner_t &ref_layer)
{
    const int group_count = ref_layer.get_num_input_groups();
    if (group_count <= 0)
    {
        ALOGE("invalid group_count=%d", group_count);
        return false;
    }

    std::vector<int> decode_gids;
    std::vector<int> prefill_gids;
    decode_gids.reserve((size_t)group_count);
    prefill_gids.reserve((size_t)group_count);

    for (int gid = 0; gid < group_count; ++gid)
    {
        try
        {
            const auto &t_idx = ref_layer.get_input(gid, "indices");
            const int idx_elems = (int)((size_t)t_idx.nSize / sizeof(unsigned int));
            if (idx_elems == 1) decode_gids.push_back(gid);
            else prefill_gids.push_back(gid);
        }
        catch (const std::exception &)
        {
            // Skip groups without indices tensor.
        }
    }

    if (decode_gids.empty())
    {
        ALOGE("no decode groups detected");
        return false;
    }
    decode_only_prefill_mode_ = false;
    if (prefill_gids.empty())
    {
        decode_only_prefill_mode_ = true;
        ALOGW("no native prefill groups detected, fallback to decode-only prefill mode");
    }

    // Detect prefill_token_num from the first prefill group.
    // Prefer `indices` last-dim (most consistent across models/backends). Fall back to `mask` shape.
    int detected_prefill_token_num = 0;
    if (decode_only_prefill_mode_)
    {
        detected_prefill_token_num = 1;
    }
    else
    {
        const int gid = prefill_gids.front();
        try
        {
            const auto &t_idx = ref_layer.get_input(gid, "indices");
            if (!t_idx.vShape.empty()) detected_prefill_token_num = (int)t_idx.vShape.back();
        }
        catch (const std::exception &)
        {
        }
        // Guard against picking the batch dimension (often 1).
        if (detected_prefill_token_num <= 1)
        {
            try
            {
                const auto &t_mask = ref_layer.get_input(gid, "mask");
                if (!t_mask.vShape.empty())
                {
                    if (t_mask.vShape.size() >= 2 && t_mask.vShape[0] == 1)
                        detected_prefill_token_num = (int)t_mask.vShape[1];
                    else
                        detected_prefill_token_num = (int)t_mask.vShape[0];
                }
            }
            catch (const std::exception &)
            {
            }
        }
        if (detected_prefill_token_num <= 1) detected_prefill_token_num = _attr.prefill_token_num;
    }

    _attr.prefill_token_num = detected_prefill_token_num;
    ALOGI("prefill_token_num : %d", _attr.prefill_token_num);

    // Build decode groups: capacity from mask length (max_token_len), fallback to K_cache tokens.
    std::vector<std::pair<int, int>> decode_pairs; // (cap, gid)
    for (const int gid : decode_gids)
    {
        int cap = -1;
        try
        {
            const auto &t_mask = ref_layer.get_input(gid, "mask");
            const int elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
            if (elems > 0) cap = elems - 1;
        }
        catch (const std::exception &)
        {
        }
        if (cap < 0)
        {
            try
            {
                const auto &t_k = ref_layer.get_input(gid, "K_cache");
                const size_t denom = (size_t)_attr.kv_cache_size * sizeof(unsigned short);
                if (denom > 0) cap = (int)((size_t)t_k.nSize / denom);
            }
            catch (const std::exception &)
            {
            }
        }
        if (cap > 0) decode_pairs.push_back({cap, gid});
    }

    // Build prefill groups:
    // - total capacity: number of tokens handled by the prefill group (`history + chunk`)
    // - history capacity: visible cached tokens before the current chunk
    // - symbolic capacity: raw `K_cache` shape, useful for some multimodal models
    struct PrefillGroupInfo {
        int total_cap = -1;
        int history_cap = -1;
        int symbolic_cap = -1;
        int gid = -1;
    };
    std::vector<PrefillGroupInfo> prefill_infos;
    const auto &prefill_scan_gids = decode_only_prefill_mode_ ? decode_gids : prefill_gids;
    for (const int gid : prefill_scan_gids)
    {
        int total_cap = -1;
        int history_cap = -1;
        int symbolic_cap = -1;
        if (decode_only_prefill_mode_)
        {
            int decode_cap = -1;
            try
            {
                const auto &t_mask = ref_layer.get_input(gid, "mask");
                const int elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
                if (elems > 0) decode_cap = elems - 1;
            }
            catch (const std::exception &)
            {
            }
            if (decode_cap < 0)
            {
                try
                {
                    const auto &t_k = ref_layer.get_input(gid, "K_cache");
                    const size_t denom = (size_t)_attr.kv_cache_size * sizeof(unsigned short);
                    if (denom > 0) decode_cap = (int)((size_t)t_k.nSize / denom);
                }
                catch (const std::exception &)
                {
                }
            }
            if (decode_cap >= 0)
            {
                history_cap = decode_cap;
                total_cap = decode_cap + 1;
                symbolic_cap = decode_cap;
                prefill_infos.push_back({total_cap, history_cap, symbolic_cap, gid});
            }
            continue;
        }
        try
        {
            const auto &t_mask = ref_layer.get_input(gid, "mask");
            const int elems = (int)((size_t)t_mask.nSize / sizeof(unsigned short));
            if (_attr.prefill_token_num > 0 && elems > 0 && (elems % _attr.prefill_token_num) == 0)
            {
                const int cols = elems / _attr.prefill_token_num;
                if (cols >= _attr.prefill_token_num)
                {
                    total_cap = cols;
                    history_cap = cols - _attr.prefill_token_num;
                }
            }
        }
        catch (const std::exception &)
        {
        }
        try
        {
            const auto &t_k = ref_layer.get_input(gid, "K_cache");
            if (t_k.vShape.size() >= 2)
            {
                symbolic_cap = (int)t_k.vShape[1];
            }
            if (symbolic_cap <= 0)
            {
                const size_t denom = (size_t)_attr.kv_cache_size * sizeof(unsigned short);
                if (denom > 0) symbolic_cap = (int)((size_t)t_k.nSize / denom);
            }
        }
        catch (const std::exception &)
        {
        }
        if (total_cap < 0 && symbolic_cap >= 0)
        {
            total_cap = std::max(_attr.prefill_token_num, symbolic_cap + _attr.prefill_token_num);
            history_cap = std::max(0, total_cap - _attr.prefill_token_num);
        }
        if (total_cap >= 0)
        {
            prefill_infos.push_back({total_cap, history_cap, symbolic_cap, gid});
        }
    }

    if (decode_pairs.empty() || prefill_infos.empty())
    {
        ALOGE("failed to parse groups (decode=%zu prefill=%zu)", decode_pairs.size(), prefill_infos.size());
        return false;
    }

    auto dedup_sorted = [](std::vector<std::pair<int, int>> &pairs) {
        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end(),
                                [](const auto &a, const auto &b) { return a.first == b.first; }),
                    pairs.end());
    };
    dedup_sorted(decode_pairs);
    std::sort(prefill_infos.begin(), prefill_infos.end(), [](const PrefillGroupInfo &a, const PrefillGroupInfo &b) {
        if (a.total_cap != b.total_cap) return a.total_cap < b.total_cap;
        return a.gid < b.gid;
    });
    prefill_infos.erase(std::unique(prefill_infos.begin(), prefill_infos.end(),
                                    [](const PrefillGroupInfo &a, const PrefillGroupInfo &b) {
                                        return a.total_cap == b.total_cap;
                                    }),
                        prefill_infos.end());

    auto shape_to_str = [](const std::vector<unsigned int> &shape) -> std::string {
        std::string s;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (i) s.push_back('x');
            s += std::to_string(shape[i]);
        }
        if (s.empty()) s = "(none)";
        return s;
    };

    // Extra debug: print per-group KV cache tensor sizing for sanity checks.
    for (const auto &p : decode_pairs)
    {
        const int cap = p.first;
        const int gid = p.second;
        try
        {
            const auto &t_mask = ref_layer.get_input(gid, "mask");
            const auto &t_k = ref_layer.get_input(gid, "K_cache");
            const auto &t_v = ref_layer.get_input(gid, "V_cache");
            ALOGD("decode gid=%d cap=%d mask=%s(%zuB) K_cache=%s(%zuB) V_cache=%s(%zuB)",
                  gid,
                  cap,
                  shape_to_str(t_mask.vShape).c_str(),
                  (size_t)t_mask.nSize,
                  shape_to_str(t_k.vShape).c_str(),
                  (size_t)t_k.nSize,
                  shape_to_str(t_v.vShape).c_str(),
                  (size_t)t_v.nSize);
        }
        catch (const std::exception &)
        {
        }
    }
    for (const auto &info : prefill_infos)
    {
        const int gid = info.gid;
        try
        {
            const auto &t_mask = ref_layer.get_input(gid, "mask");
            const auto &t_k = ref_layer.get_input(gid, "K_cache");
            const auto &t_v = ref_layer.get_input(gid, "V_cache");
            ALOGD("prefill gid=%d total_cap=%d history_cap=%d symbolic_cap=%d mask=%s(%zuB) K_cache=%s(%zuB) V_cache=%s(%zuB)",
                  gid,
                  info.total_cap,
                  info.history_cap,
                  info.symbolic_cap,
                  shape_to_str(t_mask.vShape).c_str(),
                  (size_t)t_mask.nSize,
                  shape_to_str(t_k.vShape).c_str(),
                  (size_t)t_k.nSize,
                  shape_to_str(t_v.vShape).c_str(),
                  (size_t)t_v.nSize);
        }
        catch (const std::exception &)
        {
        }
    }

    decode_grpids_.clear();
    decode_max_token_len_grp_.clear();
    for (const auto &p : decode_pairs)
    {
        decode_max_token_len_grp_.push_back(p.first);
        decode_grpids_.push_back(p.second);
    }

    prefill_grpids_.clear();
    _attr.prefill_max_kv_cache_num_grp.clear();
    prefill_history_kv_cache_num_grp.clear();
    prefill_symbolic_kv_cache_num_grp.clear();
    for (const auto &info : prefill_infos)
    {
        _attr.prefill_max_kv_cache_num_grp.push_back(info.total_cap);
        prefill_history_kv_cache_num_grp.push_back(std::max(0, info.history_cap));
        prefill_symbolic_kv_cache_num_grp.push_back(std::max(0, info.symbolic_cap));
        prefill_grpids_.push_back(info.gid);
    }

    // Default to the largest groups.
    decode_grpid = decode_grpids_.back();
    _attr.prefill_grpid = prefill_grpids_.back();
    _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp.back();

    // Canonical context capacity should follow the largest decode/prefill group.
    // Some models provide multiple decode groups (2k/4k/8k/16k), and group 0 may
    // not be the largest. Use discovered capacities to drive global limits.
    const int max_decode = decode_max_token_len_grp_.back();
    const int max_prefill = _attr.prefill_max_kv_cache_num_grp.back();
    const int canonical_cap = std::max(max_decode, max_prefill);
    if (canonical_cap > 0)
    {
        _attr.max_token_len = canonical_cap;
        _attr.kv_cache_num = canonical_cap;
    }

    // Print group summary for debugging.
    for (size_t i = 0; i < decode_grpids_.size(); ++i)
    {
        ALOGI("decode grp: %zu, gid: %d, max_token_len : %d", i, decode_grpids_[i], decode_max_token_len_grp_[i]);
    }
    for (size_t i = 0; i < prefill_grpids_.size(); ++i)
    {
        ALOGI("prefill grp: %zu, gid: %d, history_cap: %d, total_cap: %d, symbolic_cap: %d",
              i,
              prefill_grpids_[i],
              prefill_history_kv_cache_num_grp[i],
              _attr.prefill_max_kv_cache_num_grp[i],
              prefill_symbolic_kv_cache_num_grp[i]);
    }
    ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
    return true;
}

void LLM::Impl::init_layer_types()
{
    layer_is_linear_attn.assign(_attr.axmodel_num, false);
    if (!_attr.layer_types.empty())
    {
        const int num_layers = std::min(_attr.axmodel_num, (int)_attr.layer_types.size());
        for (int i = 0; i < num_layers; ++i)
        {
            const std::string &layer_type = _attr.layer_types[(size_t)i];
            if (layer_type == "linear_attention")
            {
                layer_is_linear_attn[(size_t)i] = true;
            }
            else if (layer_type == "full_attention" || layer_type == "sliding_attention")
            {
                // Gemma4 uses `sliding_attention` to distinguish local/global attention style,
                // not the legacy runtime's "linear attention" cache/mask path.
                layer_is_linear_attn[(size_t)i] = false;
            }
            else
            {
                ALOGW("unknown layer_type[%d]=%s, fallback to full-attention", i, layer_type.c_str());
                layer_is_linear_attn[(size_t)i] = false;
            }
        }
        return;
    }

    const int interval = _attr.full_attention_interval;
    if (interval <= 0) return;
    for (int i = 0; i < _attr.axmodel_num; ++i)
    {
        const bool is_full = (((i + 1) % interval) == 0);
        layer_is_linear_attn[(size_t)i] = !is_full;
    }
}

#ifdef USE_AXCL
std::vector<int> LLM::Impl::distributeModels(int cardCount, int modelCount)
{
    std::vector<int> assign(modelCount, 0);
    if (cardCount <= 0 || modelCount <= 0) return assign;
    int base = modelCount / cardCount;
    int rem  = modelCount % cardCount;
    int idx  = 0;
    for (int c = 0; c < cardCount; c++) {
        int cnt = base + (c < rem ? 1 : 0);
        for (int i = 0; i < cnt; i++) assign[idx++] = c;
    }
    return assign;
}
#endif

bool LLM::Impl::Init(LLMAttrType attr)
{
    ALOGI("LLM init start");
    this->_attr = attr;
    deinited_ = false;
    mem_guard_.capture_sentry_baseline();
    dynamic_layer_load_enabled_ = _attr.dynamic_load_enable;
    dynamic_layer_pool_size_ = _attr.dynamic_load_pool_size;
    if (dynamic_layer_load_enabled_ && dynamic_layer_pool_size_ <= 0)
        dynamic_layer_pool_size_ = 2;
    if (dynamic_layer_pool_size_ > 0)
        dynamic_layer_pool_size_ = std::min(dynamic_layer_pool_size_, std::max(1, _attr.axmodel_num));
    dynamic_layer_loaded_.assign((size_t)std::max(0, _attr.axmodel_num), 0);
    dynamic_layer_lru_.clear();
    dynamic_layer_devids_.clear();
    embedding_profile_for_tokenizer(_attr.tokenizer_type, embedding_append_eos, embedding_eos_token_id);
    init_layer_types();
    init_shared_kv_source_layers();
    cache_ref_full_layer_idx = first_full_layer_idx();
    if (cache_ref_full_layer_idx < 0) cache_ref_full_layer_idx = 0;
    if (_attr.full_attention_interval > 0)
    {
        ALOGI("mixed attention enabled: full_attention_interval=%d ref_full_layer_idx=%d",
              _attr.full_attention_interval,
              cache_ref_full_layer_idx);
    }
    if (!_attr.layer_types.empty() && _attr.num_kv_shared_layers > 0)
    {
        ALOGI("shared kv enabled: num_kv_shared_layers=%d", _attr.num_kv_shared_layers);
    }
    if (!_attr.layer_types.empty())
    {
        int sliding_count = 0;
        int full_count = 0;
        int linear_count = 0;
        for (const auto &layer_type : _attr.layer_types)
        {
            if (layer_type == "sliding_attention")
                sliding_count++;
            else if (layer_type == "full_attention")
                full_count++;
            else if (layer_type == "linear_attention")
                linear_count++;
        }
        ALOGI("attention config: layers=%zu sliding=%d full=%d linear=%d sliding_window=%d ref_full_layer_idx=%d",
              _attr.layer_types.size(),
              sliding_count,
              full_count,
              linear_count,
              _attr.sliding_window,
              cache_ref_full_layer_idx);
    }

#ifdef USE_AXCL
    // AXCL init may spawn worker threads that print logs. Do it before the progress bar starts.
    for (auto &devid : _attr.dev_ids)
    {
        if (axcl_Init(devid) != 0)
        {
            ALOGE("axcl_Init(%d) failed", devid);
            return false;
        }
    }
#endif

    t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
    tokenizer = create_tokenizer(this->_attr.tokenizer_type);
    if (!tokenizer) { ALOGE("create_tokenizer(%s) failed", this->_attr.tokenizer_type.c_str()); return false; }
    if (!tokenizer->load(attr.url_tokenizer_model)) { ALOGE("tokenizer.init(%s) failed", attr.url_tokenizer_model.c_str()); return false; }
    tokenizer->set_think_in_prompt(!tokenizer_uses_hidden_channel_markup(this->_attr.tokenizer_type));
    tokenizer->set_generation_thinking_mode(this->_attr.generation_thinking_mode);
    if (this->_attr.generation_thinking_mode != ThinkingMode::Unspecified && !tokenizer->supports_thinking_toggle())
        ALOGW("[thinking] tokenizer_type='%s' does not honor thinking_mode/enable_thinking; the setting is ignored "
              "(currently supported: Qwen3 family, MiniCPM5)", this->_attr.tokenizer_type.c_str());
    update_cqdm(&cqdm, 0, "count", "tokenizer init ok");

#ifdef USE_AXCL
    llama_layers.resize(attr.axmodel_num);
    auto dev_assign = distributeModels((int)_attr.dev_ids.size(), attr.axmodel_num);
    std::vector<int> rets(dynamic_layer_load_enabled() ? 0 : attr.axmodel_num, 0);
    dynamic_layer_devids_.assign((size_t)attr.axmodel_num, _attr.dev_ids.empty() ? 0 : _attr.dev_ids.front());

    // Prepare filenames first (thread-safe, no I/O).
    for (int i = 0; i < attr.axmodel_num; i++)
    {
        char path[1024];
        std::snprintf(path, sizeof(path), attr.template_filename_axmodel.c_str(), i);
        llama_layers[i].filename = path;
        const int dev_idx = (dev_assign.empty() ? 0 : dev_assign[i]);
        if (dev_idx >= 0 && (size_t)dev_idx < _attr.dev_ids.size())
            dynamic_layer_devids_[(size_t)i] = _attr.dev_ids[(size_t)dev_idx];
    }

    if (!mem_preflight(dynamic_layer_devids_)) return false;
    running_guard_init();

    if (dynamic_layer_load_enabled())
    {
        if (_attr.dev_ids.empty())
        {
            ALOGE("dynamic layer loading requires at least one AXCL device");
            return false;
        }
        if (_attr.dev_ids.size() != 1)
        {
            ALOGE("dynamic layer loading only supports single AXCL device for now (dev_ids=%zu)", _attr.dev_ids.size());
            return false;
        }

        const int devid = _attr.dev_ids.front();
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            if (!running_guard_check(devid, i)) return false;
            const int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), devid);
            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
            // Keep IO (KV etc), but unload weights/handle to save CMM.
            llama_layers[i].layer.unload_handle_keep_io();

            char path[256];
            std::snprintf(path, sizeof(path), "init %d axmodel io ok,remain_cmm(%d MB)", i, axcl_GetCMMRemain(devid));
            update_cqdm(&cqdm, i + 1, "count", path);
        }

        // Start with an empty residency pool (handles will be loaded on-demand).
        std::fill(dynamic_layer_loaded_.begin(), dynamic_layer_loaded_.end(), 0);
        dynamic_layer_lru_.clear();

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), devid);
        if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
        char path[1024];
        sprintf(path, "init post axmodel ok,remain_cmm(%d MB)", axcl_GetCMMRemain(devid));
        update_cqdm(&cqdm, attr.axmodel_num + 1, "count", path);
    }
    else
    {
    // Load models in parallel across devices (per-device sequential), while the main thread updates progress.
    struct LoadResult {
        int idx = -1;
        int ret = -1;
        int devid = -1;
        int remain_mb = -1;
        bool skipped = false;   // not loaded (a prior guard abort stopped the run)
        int guard = 0;          // 0=ok, 1=warn(loaded anyway), 2=mem-guard abort(not loaded)
        std::string guard_what;
        std::string msg;
    };

    std::vector<std::vector<int>> models_per_dev(_attr.dev_ids.size());
    for (int i = 0; i < attr.axmodel_num; ++i)
    {
        const int dev_idx = (dev_assign.empty() ? 0 : dev_assign[i]);
        if (dev_idx >= 0 && (size_t)dev_idx < models_per_dev.size()) models_per_dev[(size_t)dev_idx].push_back(i);
    }

    std::mutex q_mu;
    std::condition_variable q_cv;
    std::queue<LoadResult> q;
    std::atomic<bool> load_abort{false};  // measured guard refused -> stop all device threads
    std::vector<std::thread> loaders;
    loaders.reserve(_attr.dev_ids.size());

    for (size_t dev_idx = 0; dev_idx < _attr.dev_ids.size(); ++dev_idx)
    {
        const int devid = _attr.dev_ids[dev_idx];
        loaders.emplace_back([&, dev_idx, devid]() {
            int loaded_on_dev = 0;
            for (const int i : models_per_dev[dev_idx])
            {
                LoadResult r;
                r.idx = i;
                r.devid = devid;
                if (load_abort.load(std::memory_order_relaxed))
                {
                    r.skipped = true; // another layer/device already breached the budget
                }
                else
                {
                    // Measured-extrapolation guard (pure -> thread-safe). On abort,
                    // stop this and the other device threads before over-allocating.
                    const auto v = mem_guard_.running_guard_eval(devid, loaded_on_dev);
                    if (!v.ok)
                    {
                        load_abort.store(true, std::memory_order_relaxed);
                        r.guard = 2;
                        r.guard_what = v.what;
                        r.remain_mb = v.remain;
                        r.skipped = true;
                    }
                    else
                    {
                        const int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), devid);
                        const int remain = axcl_GetCMMRemain(devid);
                        ++loaded_on_dev;
                        char buf[256];
                        std::snprintf(buf, sizeof(buf), "init %d axmodel ok,devid(%d) remain_cmm(%d MB)", i, devid, remain);
                        r.ret = ret;
                        r.remain_mb = remain;
                        r.msg = buf;
                        if (v.warned) { r.guard = 1; r.guard_what = v.what; }
                    }
                }

                {
                    std::lock_guard<std::mutex> lk(q_mu);
                    q.push(std::move(r));
                }
                q_cv.notify_one();
            }
        });
    }

    int progress_step = 1;
    int finished = 0;
    bool guard_aborted = false;
    while (finished < attr.axmodel_num)
    {
        LoadResult r;
        {
            std::unique_lock<std::mutex> lk(q_mu);
            q_cv.wait(lk, [&]() { return !q.empty(); });
            r = std::move(q.front());
            q.pop();
        }
        finished++;
        if (r.guard == 2)
        {
            guard_aborted = true;
            ALOGE("mem-guard aborted mid-load: %s (remain %d MB)", r.guard_what.c_str(), r.remain_mb);
            set_last_error("CMM 不足，已中止加载(实测外推): " + r.guard_what +
                           "（可调小模型/释放显存，或在 config.json 设 mem_guard_enable=false）");
            continue;
        }
        if (r.skipped) continue; // a prior abort stopped this layer
        if (r.guard == 1)
            ALOGW("[mem-guard] WARN(measured): %s (remain %d MB) -> continuing", r.guard_what.c_str(), r.remain_mb);
        if (r.idx >= 0 && r.idx < attr.axmodel_num) rets[r.idx] = r.ret;
        update_cqdm(&cqdm, progress_step++, "count", r.msg.c_str());
    }

    for (auto &t : loaders)
    {
        if (t.joinable()) t.join();
    }

    if (guard_aborted) return false;

    for (int i = 0; i < attr.axmodel_num; i++) { if (rets[i] != 0) { ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str()); return false; } }
    {
        int post_devid = llama_layers.back().layer.get_devid();
        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), post_devid);
        if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
        char path[1024];
        sprintf(path, "init post axmodel ok,remain_cmm(%d MB)", axcl_GetCMMRemain(post_devid));
        update_cqdm(&cqdm, attr.axmodel_num + 1, "count", path);
    }
    }
#else
    llama_layers.resize(attr.axmodel_num);
    char axmodel_path[1024];
    for (int i = 0; i < attr.axmodel_num; i++)
    {
        sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
        llama_layers[i].filename = axmodel_path;
    }
    if (!mem_preflight({})) return false;
    running_guard_init();
    if (dynamic_layer_load_enabled())
    {
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            if (!running_guard_check(-1, i)) return false;
            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), -1);
            if (ret != 0) { ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str()); return false; }
            llama_layers[i].layer.set_auto_sync_before_inference(true);
            llama_layers[i].layer.set_auto_sync_after_inference(true);
            llama_layers[i].layer.unload_handle_keep_io();
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel io ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 1, "count", axmodel_path);
        }

        // Start with an empty residency pool (handles will be loaded on-demand).
        std::fill(dynamic_layer_loaded_.begin(), dynamic_layer_loaded_.end(), 0);
        dynamic_layer_lru_.clear();

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), -1);
        if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
        llama_post.set_auto_sync_before_inference(true);
        llama_post.set_auto_sync_after_inference(true);
        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 1, "count", axmodel_path);
    }
    else
    {
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            if (!running_guard_check(-1, i)) return false;
            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), -1);
            if (ret != 0) { ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str()); return false; }
            llama_layers[i].layer.set_auto_sync_before_inference(true);
            llama_layers[i].layer.set_auto_sync_after_inference(true);
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 1, "count", axmodel_path);
        }
        {
            int ret = llama_post.init(attr.filename_post_axmodel.c_str(), -1);
            if (ret != 0) { ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str()); return false; }
            llama_post.set_auto_sync_before_inference(true);
            llama_post.set_auto_sync_after_inference(true);
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
            update_cqdm(&cqdm, attr.axmodel_num + 1, "count", axmodel_path);
        }
    }
#endif
    axllm::Logger::finish_inplace_line();
    {
        auto &ref_layer = llama_layers[(size_t)cache_ref_full_layer_idx].layer;
        _attr.max_token_len = ref_layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
        ALOGI("max_token_len : %d", _attr.max_token_len);
        _attr.kv_cache_size = ref_layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
        _attr.kv_cache_num  = ref_layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
        ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
        if (_attr.max_token_len > _attr.kv_cache_num) { ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num); return false; }
        if (!init_groups_from_model(ref_layer)) return false;
        init_layer_groups();
        if (std::getenv("AXLLM_DEBUG_LAYER0_IO"))
        {
            const int layer0_decode_grpid = decode_gid_for_layer(0, decode_grpid);
            dump_group_tensor_layout(llama_layers[0].layer, layer0_decode_grpid, "layer0_decode");
            if (!prefill_grpids_.empty())
            {
                const int layer0_prefill_grpid = prefill_gid_for_layer(0, prefill_grpids_[0]);
                dump_group_tensor_layout(llama_layers[0].layer, layer0_prefill_grpid, "layer0_prefill");
            }
        }
        layer_kv_cache_sizes.assign(_attr.axmodel_num, _attr.kv_cache_size);
        for (int i = 0; i < _attr.axmodel_num; ++i)
        {
            const int layer_decode_grpid = decode_gid_for_layer(i, decode_grpid);
            auto &layer_k = llama_layers[(size_t)i].layer.get_input(layer_decode_grpid, "K_cache");
            const int layer_kv_size = (int)(layer_k.nSize / sizeof(unsigned short) / std::max(1, _attr.kv_cache_num));
            if (layer_kv_size > 0)
                layer_kv_cache_sizes[(size_t)i] = layer_kv_size;
        }
    }

    // embed file check → then init
    {
        const int layer0_decode_grpid = decode_gid_for_layer(0, decode_grpid);
        auto &t_in0 = llama_layers[0].layer.get_input(layer0_decode_grpid, "input");
        int model_embed_sz = t_in0.nSize / (int)sizeof(unsigned short);
        if (model_embed_sz != _attr.tokens_embed_size)
        {
            ALOGE("tokens_embed_size mismatch: config(%d) != model(%d). Please fix config or embed file.", _attr.tokens_embed_size, model_embed_sz);
            return false;
        }
        if (!embed_selector.Init(_attr.filename_tokens_embed, _attr.tokens_embed_num, _attr.tokens_embed_size, _attr.b_use_mmap_load_embed))
        {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", _attr.filename_tokens_embed.c_str(), _attr.tokens_embed_num, _attr.tokens_embed_size);
            return false;
        }
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", "embed_selector init ok");
    }
    axllm::Logger::finish_inplace_line();

    if (_attr.hidden_size_per_layer_input > 0)
    {
        if (!gemma4_per_layer_helper.Init(_attr.tokens_embed_num,
                                          _attr.tokens_embed_size,
                                          _attr.axmodel_num,
                                          _attr.hidden_size_per_layer_input,
                                          _attr.pad_token_id,
                                          _attr.rms_norm_eps,
                                          _attr.filename_tokens_embed_per_layer,
                                          _attr.filename_per_layer_model_projection,
                                          _attr.filename_per_layer_projection_norm))
        {
            ALOGE("Gemma4 per-layer helper init failed");
            return false;
        }
    }

    // Optional VLM vision encoder (runtime controlled by attr.vlm_type).
    has_vision_state = false;
    if (_attr.vlm_type != VLMType::None)
    {
        vision.reset(new vision::VisionModule());
        std::string verr;
        int vdevid =
#ifdef USE_AXCL
            llama_layers[0].layer.get_devid();
#else
            -1;
#endif

        if (!vision->Init(_attr.vlm_type,
                          _attr.filename_image_encoder_axmodel,
                          _attr.vision_cache_dir,
                          _attr.tokens_embed_size,
                          vdevid,
                          tokenizer,
                          _attr.vision_width,
                          _attr.vision_height,
                          _attr.vision_temporal_patch_size,
                          _attr.vision_spatial_merge_size,
                          _attr.vision_patch_size,
                          _attr.vision_fps,
                          _attr.vision_tokens_per_second,
                          _attr.vision_num_frames,
                          _attr.vision_do_sample_frames,
                          _attr.filename_audio_encoder_axmodel_5s,
                          _attr.filename_audio_encoder_axmodel_30s,
                          _attr.audio_tokens_per_second,
                          _attr.audio_time_marker_every_seconds,
                          _attr.audio_time_markers_enabled,
                          verr))
        {
            ALOGE("vision.Init(vlm_type=%s/%d) failed: %s",
                  std::string(VLMTypeName(_attr.vlm_type)).c_str(),
                  (int)_attr.vlm_type,
                  verr.c_str());
            return false;
        }
    }

    if (!this->_attr.post_config_path.empty())
    {
        if (!postprocess.load_config(this->_attr.post_config_path))
        {
            ALOGW("load postprocess config(%s) failed", this->_attr.post_config_path.c_str());
        }
    }
    postprocess.set_pad_token_id(_attr.pad_token_id);

    kv_mgr_.init_kv_slots();

    ALOGI("LLM init ok");
    return true;
}

void LLM::Impl::Deinit()
{
    if (deinited_) return;
    deinited_ = true;
    for (size_t i = 0; i < llama_layers.size(); i++) llama_layers[i].layer.deinit();
    llama_post.deinit();
    embed_selector.Deinit();
    gemma4_per_layer_helper.Deinit();
    if (vision) vision->Deinit();
    mem_guard_.CheckCmmBalance("Deinit"); // automated teardown-balance self-check
#ifdef USE_AXCL
    for (auto &devid : _attr.dev_ids) axcl_Exit(devid);
#endif
}
