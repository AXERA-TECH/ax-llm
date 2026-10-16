#pragma once

#include <cstdint>

// Per-request sampling parameters (OpenAI chat API plus the vLLM extensions top_k, min_p and
// repetition_penalty). Each has_* flag says whether the request set that field; unset fields fall
// back to post_config.json. A request that sets none of temperature/top_p/top_k/min_p decodes greedily.
struct SamplingRequest
{
    bool has_temperature = false;
    float temperature = 1.0f;
    bool has_top_p = false;
    float top_p = 1.0f;
    bool has_top_k = false;
    int top_k = 0; // <= 0: no top-k limit
    bool has_min_p = false;
    float min_p = 0.0f;
    bool has_frequency_penalty = false;
    float frequency_penalty = 0.0f;
    bool has_presence_penalty = false;
    float presence_penalty = 0.0f;
    bool has_repetition_penalty = false;
    float repetition_penalty = 1.0f; // HF/vLLM rule over all generated tokens
    bool has_seed = false;
    uint64_t seed = 0;
};
