#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include "utils/bfloat16.hpp"
#include "utils/json.hpp"
#include "utils/sample_log.h"
#include "SamplingRequest.hpp"

class LLMPostprocess
{
private:
    struct RequestSamplingOverride : SamplingRequest
    {
        bool force_greedy = false;
    };

    struct EffectiveSampling
    {
        bool enable_temperature = false;
        float temperature = 1.0f;
        bool enable_repetition_penalty = false;
        float repetition_penalty = 1.0f;
        int penalty_window = 20;
        bool repetition_penalty_hf = false; // request-level: HF rule over all generated tokens
        bool enable_frequency_penalty = false;
        float frequency_penalty = 0.0f;
        bool enable_presence_penalty = false;
        float presence_penalty = 0.0f;
        bool enable_diversity_penalty = false;
        std::vector<int> common_phrases;
        float diversity_penalty = 1.0f;
        bool enable_top_p_sampling = false;
        float top_p = 1.0f;
        bool enable_top_k_sampling = false;
        int top_k = 1;
        float min_p = 0.0f;
        bool greedy = false;
    };

    // Sampling RNG for the decoding thread; set_request_sampling_override() seeds it from a
    // request seed so sampled outputs are reproducible.
    static std::mt19937_64 &rng()
    {
        static thread_local std::mt19937_64 gen(std::random_device{}());
        return gen;
    }

    static thread_local RequestSamplingOverride &request_override()
    {
        static thread_local RequestSamplingOverride override_cfg;
        return override_cfg;
    }

    // 	控制随机性
    void apply_temperature(std::vector<float> &logits, float temperature)
    {
        for (float &logit : logits)
        {
            logit /= temperature;
        }
    }

    // 防止重复
    // HF / vLLM repetition penalty: every token seen so far has its logit divided by `penalty`
    // when positive and multiplied by it when negative (once per distinct token).
    void apply_repetition_penalty(std::vector<float> &logits, const std::vector<int> &history, float penalty)
    {
        if (penalty == 1.0f || penalty <= 0.0f) return;
        std::unordered_set<int> seen(history.begin(), history.end());
        for (int token : seen)
        {
            if (token < 0 || token >= (int)logits.size()) continue;
            logits[token] = logits[token] < 0 ? logits[token] * penalty : logits[token] / penalty;
        }
    }

    void apply_repetition_penalty(std::vector<float> &logits,
                                  const std::vector<int> &generated_tokens,
                                  float repetition_penalty,
                                  int penalty_window)
    {
        if (repetition_penalty == 1.0f || generated_tokens.empty())
        {
            return; // 如果 penalty = 1.0 或者没有生成 token，则不进行修改
        }

        int start_idx = std::max(0, (int)generated_tokens.size() - penalty_window);
        std::unordered_set<int> recent_tokens(generated_tokens.begin() + start_idx, generated_tokens.end());

        for (int token : recent_tokens)
        {
            if (token < 0 || token >= logits.size())
                continue;

            if (logits[token] > 0)
            {
                logits[token] /= std::sqrt(repetition_penalty);
            }
            else
            {
                logits[token] *= std::sqrt(repetition_penalty);
            }
        }
    }

    // OpenAI-style frequency & presence penalties (additive on logits), applied
    // over the same recent window as repetition_penalty (penalty_window; <=0 means
    // the whole generated history):
    //   logit[t] -= frequency_penalty * count(t)   (per occurrence)
    //   logit[t] -= presence_penalty               (once if t appeared at all)
    void apply_frequency_presence_penalty(std::vector<float> &logits,
                                          const std::vector<int> &generated_tokens,
                                          float frequency_penalty,
                                          float presence_penalty,
                                          int penalty_window)
    {
        if ((frequency_penalty == 0.0f && presence_penalty == 0.0f) || generated_tokens.empty())
            return;

        int start_idx = penalty_window > 0
                            ? std::max(0, (int)generated_tokens.size() - penalty_window)
                            : 0;
        std::unordered_map<int, int> counts;
        for (int i = start_idx; i < (int)generated_tokens.size(); ++i)
        {
            const int t = generated_tokens[i];
            if (t >= 0 && t < (int)logits.size())
                counts[t]++;
        }
        for (const auto &kv : counts)
        {
            logits[(size_t)kv.first] -= frequency_penalty * (float)kv.second;
            logits[(size_t)kv.first] -= presence_penalty;
        }
    }

    static int argmax_index(const std::vector<float> &logits)
    {
        if (logits.empty())
            return 0;
        return (int)std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    }

    int argmax_index_bf16(const unsigned short *logits, int n) const
    {
        if (!logits || n <= 0)
            return 0;

        int max_index = -1;
        float max_value = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i)
        {
            if (pad_token_id >= 0 && i == pad_token_id)
                continue;
            const float value = bfloat16(logits[i]).fp32();
            if (max_index < 0 || value > max_value)
            {
                max_value = value;
                max_index = i;
            }
        }
        return max_index >= 0 ? max_index : 0;
    }

    // 增强多样性
    void apply_diversity_penalty(std::vector<float> &logits, const std::vector<int> &common_phrases, float penalty)
    {
        for (int token : common_phrases)
        {
            if (token < logits.size())
            {
                logits[token] *= penalty;
            }
        }
    }

    // Softmax function
    std::vector<float> softmax(const std::vector<float> &logits)
    {
        std::vector<float> probs(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;

        for (size_t i = 0; i < logits.size(); ++i)
        {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }

        for (float &p : probs)
        {
            p /= sum;
        }

        return probs;
    }

    bool enable_temperature = false;
    float temperature = 1.0f;

    bool enable_repetition_penalty = false;
    float repetition_penalty = 1.0f;
    int penalty_window = 20;

    bool enable_frequency_penalty = false;
    float frequency_penalty = 0.0f;
    bool enable_presence_penalty = false;
    float presence_penalty = 0.0f;

    bool enable_diversity_penalty = false;
    std::vector<int> common_phrases;
    float diversity_penalty = 1.0f;

    bool enable_top_p_sampling = false;
    float top_p = 1.0f;

    bool enable_top_k_sampling = false;
    int top_k = 1;
    int pad_token_id = -1;

    EffectiveSampling resolve_effective_sampling() const
    {
        EffectiveSampling eff;
        const auto &ov = request_override();

        eff.enable_temperature = enable_temperature;
        eff.temperature = temperature;
        if (ov.has_temperature)
        {
            eff.enable_temperature = true;
            eff.temperature = ov.temperature;
        }

        eff.enable_repetition_penalty = enable_repetition_penalty;
        eff.repetition_penalty = repetition_penalty;
        eff.penalty_window = penalty_window;
        eff.enable_frequency_penalty = enable_frequency_penalty;
        eff.frequency_penalty = frequency_penalty;
        eff.enable_presence_penalty = enable_presence_penalty;
        eff.presence_penalty = presence_penalty;
        // Per-request overrides (OpenAI API). A value of 0 disables that penalty.
        if (ov.has_frequency_penalty)
        {
            eff.enable_frequency_penalty = (ov.frequency_penalty != 0.0f);
            eff.frequency_penalty = ov.frequency_penalty;
        }
        if (ov.has_presence_penalty)
        {
            eff.enable_presence_penalty = (ov.presence_penalty != 0.0f);
            eff.presence_penalty = ov.presence_penalty;
        }
        eff.enable_diversity_penalty = enable_diversity_penalty;
        eff.common_phrases = common_phrases;
        eff.diversity_penalty = diversity_penalty;

        eff.enable_top_p_sampling = enable_top_p_sampling;
        eff.top_p = top_p;
        if (ov.has_top_p)
        {
            eff.enable_top_p_sampling = true;
            eff.top_p = ov.top_p;
        }

        eff.enable_top_k_sampling = enable_top_k_sampling;
        eff.top_k = top_k;
        if (ov.has_top_k)
        {
            eff.enable_top_k_sampling = ov.top_k > 0;
            eff.top_k = ov.top_k;
        }
        if (ov.has_min_p) eff.min_p = ov.min_p;
        if (ov.has_repetition_penalty)
        {
            eff.enable_repetition_penalty = ov.repetition_penalty != 1.0f;
            eff.repetition_penalty = ov.repetition_penalty;
            eff.repetition_penalty_hf = true;
        }

        if (ov.force_greedy)
        {
            eff.greedy = true;
            eff.enable_temperature = false;
            eff.enable_top_p_sampling = false;
            eff.enable_top_k_sampling = false;
            return eff;
        }

        eff.greedy = eff.enable_temperature && eff.temperature <= 0.0f;
        if (eff.greedy)
        {
            eff.enable_temperature = false;
            eff.enable_top_p_sampling = false;
            eff.enable_top_k_sampling = false;
        }
        else
        {
            if (eff.top_p >= 1.0f) eff.enable_top_p_sampling = false;
            if (eff.top_k < 1) eff.enable_top_k_sampling = false;
            if (!(eff.min_p > 0.0f && eff.min_p < 1.0f)) eff.min_p = 0.0f;
        }

        return eff;
    }

    // Multinomial sample over the full softmax distribution (logits already
    // temperature-scaled). Prevents "temperature enabled but no top_p/top_k"
    // from silently collapsing to greedy argmax.
    // Sample from softmax(logits) restricted by top_k (<= 0: off), then top_p (>= 1: off), then
    // min_p (<= 0: off), in that order (as vLLM / HF do when several are set).
    int sample_filtered(const std::vector<float> &logits, int top_k, float top_p, float min_p)
    {
        if (logits.empty())
            return 0;
        std::vector<int> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        const auto by_logit = [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; };
        if (top_k > 0 && top_k < (int)idx.size())
        {
            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), by_logit);
            idx.resize((size_t)top_k);
        }
        else if (top_p < 1.0f || min_p > 0.0f)
        {
            std::sort(idx.begin(), idx.end(), by_logit);
        }
        const float max_logit = logits[(size_t)*std::max_element(idx.begin(), idx.end(), [&](int a, int b) { return logits[(size_t)a] < logits[(size_t)b]; })];
        std::vector<double> probs(idx.size());
        double sum = 0.0;
        for (size_t i = 0; i < idx.size(); ++i)
        {
            probs[i] = std::exp((double)logits[(size_t)idx[i]] - (double)max_logit);
            sum += probs[i];
        }
        size_t keep = idx.size();
        if (top_p < 1.0f && sum > 0.0)
        {
            double cum = 0.0;
            for (size_t i = 0; i < idx.size(); ++i)
            {
                cum += probs[i] / sum;
                if (cum >= top_p) { keep = i + 1; break; }
            }
        }
        if (min_p > 0.0f && !probs.empty())
        {
            // idx is sorted (top_p/min_p imply sorting), so probs[0] is the max.
            const double floor_p = (double)min_p * probs[0];
            size_t k = 1;
            while (k < keep && probs[k] >= floor_p) ++k;
            keep = k;
        }
        std::discrete_distribution<size_t> dist(probs.begin(), probs.begin() + (std::ptrdiff_t)keep);
        return idx[dist(rng())];
    }

    int apply_logits(std::vector<float> &logits, const std::vector<int> &history, const EffectiveSampling &eff)
    {
        if (eff.enable_temperature)
            apply_temperature(logits, eff.temperature);
        if (eff.enable_repetition_penalty)
        {
            if (eff.repetition_penalty_hf)
                apply_repetition_penalty(logits, history, eff.repetition_penalty);
            else
                apply_repetition_penalty(logits, history, eff.repetition_penalty, eff.penalty_window);
        }
        if (eff.enable_frequency_penalty || eff.enable_presence_penalty)
            apply_frequency_presence_penalty(logits, history,
                                             eff.enable_frequency_penalty ? eff.frequency_penalty : 0.0f,
                                             eff.enable_presence_penalty ? eff.presence_penalty : 0.0f,
                                             eff.penalty_window);
        if (eff.enable_diversity_penalty)
            apply_diversity_penalty(logits, eff.common_phrases, eff.diversity_penalty);
        if (pad_token_id >= 0 && pad_token_id < (int)logits.size())
            logits[(size_t)pad_token_id] = -std::numeric_limits<float>::infinity();

        if (eff.enable_top_p_sampling || eff.enable_top_k_sampling || eff.min_p > 0.0f || eff.enable_temperature)
            return sample_filtered(logits,
                                   eff.enable_top_k_sampling ? eff.top_k : 0,
                                   eff.enable_top_p_sampling ? eff.top_p : 1.0f,
                                   eff.min_p);
        return argmax_index(logits);
    }

public:
    LLMPostprocess() {}

    void set_request_sampling_override(const SamplingRequest &req)
    {
        auto &ov = request_override();
        static_cast<SamplingRequest &>(ov) = req;
        // Penalties are not samplers: a request carrying only penalties still runs greedy
        // (penalties are applied to the argmax) unless it also asks for a sampler.
        ov.force_greedy = !req.has_temperature && !req.has_top_p && !req.has_top_k && !req.has_min_p;
        if (req.has_seed) rng().seed(req.seed);
    }

    void clear_request_sampling_override()
    {
        if (request_override().has_seed) rng().seed(std::random_device{}());
        request_override() = {};
    }

    void set_temperature(bool enable, float temperature)
    {
        enable_temperature = enable;
        this->temperature = temperature;
    }

    void set_repetition_penalty(bool enable, float penalty)
    {
        enable_repetition_penalty = enable;
        this->repetition_penalty = penalty;
    }

    void set_frequency_penalty(bool enable, float penalty)
    {
        enable_frequency_penalty = enable;
        this->frequency_penalty = penalty;
    }

    void set_presence_penalty(bool enable, float penalty)
    {
        enable_presence_penalty = enable;
        this->presence_penalty = penalty;
    }

    void set_diversity_penalty(bool enable, const std::vector<int> &common_phrases, float penalty)
    {
        enable_diversity_penalty = enable;
        this->common_phrases = common_phrases;
        this->diversity_penalty = penalty;
    }

    void set_top_p_sampling(bool enable, float top_p)
    {
        enable_top_k_sampling = false;
        enable_top_p_sampling = enable;
        this->top_p = top_p;
    }

    void set_top_k_sampling(bool enable, int top_k)
    {
        enable_top_p_sampling = false;
        enable_top_k_sampling = enable;
        this->top_k = top_k;
    }

    void set_pad_token_id(int token_id)
    {
        pad_token_id = token_id;
    }

    bool load_config(std::string config_path)
    {
        std::ifstream config_file(config_path);
        if (!config_file.is_open())
        {
            ALOGE("config file(%s) open failed", config_path.c_str());
            return false;
        }
        nlohmann::json config = nlohmann::json::parse(config_file);
        ALOGI("load config: \n%s\n", config.dump(4).c_str());

        // Missing keys must not crash: nlohmann operator[] returns null for an
        // absent key, and converting null -> bool/number throws type_error.302.
        // Use value(key, default) so an incomplete post_config.json degrades
        // gracefully (same approach as the frequency/presence penalties below).
        enable_temperature = config.value("enable_temperature", false);
        temperature = config.value("temperature", 1.0f);
        if (temperature <= 0.0f) temperature = 1.0f;

        enable_repetition_penalty = config.value("enable_repetition_penalty", false);
        repetition_penalty = config.value("repetition_penalty", 1.0f);
        penalty_window = config.value("penalty_window", 20);
        if (penalty_window < 0) penalty_window = 0;
        if (repetition_penalty < 0.0f) repetition_penalty = 1.0f;

        enable_top_p_sampling = config.value("enable_top_p_sampling", false);
        top_p = config.value("top_p", 1.0f);
        if (top_p <= 0.0f) top_p = 0.9f; // reasonable default
        if (top_p > 1.0f) top_p = 1.0f;

        enable_top_k_sampling = config.value("enable_top_k_sampling", false);
        top_k = config.value("top_k", 1);
        if (top_k < 1) top_k = 1;

        // Optional OpenAI-style penalties. Backward compatible: absent keys ->
        // disabled. A non-zero value implies enabled even without the flag.
        enable_frequency_penalty = config.value("enable_frequency_penalty", false);
        frequency_penalty = config.value("frequency_penalty", 0.0f);
        enable_presence_penalty = config.value("enable_presence_penalty", false);
        presence_penalty = config.value("presence_penalty", 0.0f);
        if (frequency_penalty != 0.0f) enable_frequency_penalty = true;
        if (presence_penalty != 0.0f) enable_presence_penalty = true;
        if (!enable_frequency_penalty) frequency_penalty = 0.0f;
        if (!enable_presence_penalty) presence_penalty = 0.0f;

        // 互斥处理：若同时开启 top_p 与 top_k，则优先 top_p
        if (enable_top_p_sampling && enable_top_k_sampling)
        {
            ALOGW("Both top_p and top_k enabled; prefer top_p and disable top_k");
            enable_top_k_sampling = false;
        }
        return true;
    }

    int apply(std::vector<float> &logits, const std::vector<int> &history)
    {
        return apply_logits(logits, history, resolve_effective_sampling());
    }

    int apply_bf16(const unsigned short *logits, int n, const std::vector<int> &history)
    {
        const auto eff = resolve_effective_sampling();
        if (!eff.enable_repetition_penalty &&
            !eff.enable_frequency_penalty &&
            !eff.enable_presence_penalty &&
            !eff.enable_diversity_penalty &&
            !eff.enable_top_p_sampling &&
            !eff.enable_top_k_sampling &&
            !eff.enable_temperature)
        {
            return argmax_index_bf16(logits, n);
        }

        std::vector<float> fp32_logits((size_t)std::max(0, n));
        for (int i = 0; i < n; ++i)
            fp32_logits[(size_t)i] = bfloat16(logits[i]).fp32();
        return apply_logits(fp32_logits, history, eff);
    }
};
