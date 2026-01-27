#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_set>
#include "utils/json.hpp"
#include "utils/sample_log.h"

class LLMPostprocess
{
private:
    // 	控制随机性
    void apply_temperature(std::vector<float> &logits, float temperature)
    {
        for (float &logit : logits)
        {
            logit /= temperature;
        }
    }

    // 防止重复
    void apply_repetition_penalty(std::vector<float> &logits, const std::vector<int> &history, float penalty)
    {
        for (int token : history)
        {
            if (token < logits.size())
            {
                logits[token] = logits[token] < 0 ? logits[token] * penalty : logits[token] / penalty;
            }
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

    // 	动态裁剪低概率 token
    int faster_top_p_sampling(const std::vector<float> &logits, float top_p)
    {
        // 计算softmax
        std::vector<float> probs = softmax(logits);

        // 构建最大堆（概率和索引的配对）
        std::vector<std::pair<float, size_t>> prob_index;
        prob_index.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
        {
            prob_index.emplace_back(probs[i], i);
        }
        auto cmp = [](const auto &a, const auto &b)
        { return a.first < b.first; };
        std::make_heap(prob_index.begin(), prob_index.end(), cmp);

        // 提取top-p元素
        std::vector<size_t> filtered_indices;
        std::vector<float> filtered_probs;
        float cumulative_prob = 0.0f;

        while (!prob_index.empty() && cumulative_prob < top_p)
        {
            std::pop_heap(prob_index.begin(), prob_index.end(), cmp);
            auto [prob, index] = prob_index.back();
            prob_index.pop_back();

            cumulative_prob += prob;
            filtered_indices.push_back(index);
            filtered_probs.push_back(prob);

            if (cumulative_prob >= top_p)
                break;
        }

        // 处理边缘情况（概率全零时返回第一个元素）
        if (filtered_indices.empty())
            return 0;

        // 使用thread_local随机数生成器（线程安全）
        static thread_local std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        return filtered_indices[dist(gen)];
    }
    int top_p_sampling(const std::vector<float> &logits, float top_p)
    {
        std::vector<float> probs = softmax(logits);

        // Sort indices by probability in descending order
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j)
                  { return probs[i] > probs[j]; });

        // Compute cumulative probabilities
        float cumulative_prob = 0.0f;
        size_t cut_off = 0;
        for (; cut_off < indices.size(); ++cut_off)
        {
            cumulative_prob += probs[indices[cut_off]];
            if (cumulative_prob >= top_p)
                break;
        }

        // Keep only the top-p probabilities
        std::vector<size_t> filtered_indices(indices.begin(), indices.begin() + cut_off + 1);
        std::vector<float> filtered_probs(filtered_indices.size());
        for (size_t i = 0; i < filtered_indices.size(); ++i)
        {
            filtered_probs[i] = probs[filtered_indices[i]];
        }

        // Normalize the probabilities
        float filtered_sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
        for (float &p : filtered_probs)
        {
            p /= filtered_sum;
        }

        // Sample from the filtered distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        return filtered_indices[dist(gen)];
    }

    // 限制候选 token 数
    int top_k_sampling(const std::vector<float> &logits, int k)
    {
        // std::vector<float> probs = softmax(logits);

        // 获取 top-k 索引
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](size_t i, size_t j)
                          { return logits[i] > logits[j]; });

        // 仅保留 top-k 概率
        std::vector<size_t> filtered_indices(indices.begin(), indices.begin() + k);
        std::vector<float> filtered_probs(k);
        for (size_t i = 0; i < k; ++i)
        {
            filtered_probs[i] = logits[filtered_indices[i]];
        }
        filtered_probs = softmax(filtered_probs);

        // 归一化
        float sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
        for (float &p : filtered_probs)
        {
            p /= sum;
        }

        // 采样
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        return filtered_indices[dist(gen)];
    }

    bool enable_temperature = false;
    float temperature = 1.0f;

    bool enable_repetition_penalty = false;
    float repetition_penalty = 1.0f;
    int penalty_window = 20;

    bool enable_diversity_penalty = false;
    std::vector<int> common_phrases;
    float diversity_penalty = 1.0f;

    bool enable_top_p_sampling = false;
    float top_p = 1.0f;

    bool enable_top_k_sampling = false;
    int top_k = 1;

public:
    LLMPostprocess() {}

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

        enable_temperature = config["enable_temperature"];
        temperature = config["temperature"];

        enable_repetition_penalty = config["enable_repetition_penalty"];
        repetition_penalty = config["repetition_penalty"];
        penalty_window = config["penalty_window"];

        enable_top_p_sampling = config["enable_top_p_sampling"];
        top_p = config["top_p"];

        enable_top_k_sampling = config["enable_top_k_sampling"];
        top_k = config["top_k"];
        return true;
    }

    int apply(std::vector<float> &logits, const std::vector<int> &history)
    {
        if (enable_temperature)
            apply_temperature(logits, temperature);
        if (enable_repetition_penalty)
            apply_repetition_penalty(logits, history, repetition_penalty, penalty_window);
        if (enable_diversity_penalty)
            apply_diversity_penalty(logits, common_phrases, diversity_penalty);

        if (enable_top_p_sampling)
            return faster_top_p_sampling(logits, top_p);
        else if (enable_top_k_sampling)
            return top_k_sampling(logits, top_k);
        else
        {
            // 最大值
            float max_logit = *std::max_element(logits.begin(), logits.end());
            int max_index = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
            return max_index;
        }
    }
};
