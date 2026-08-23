// LLM public API wrappers. Impl class -> LLMImpl.hpp; embed bodies -> LLM_embed.cpp.
#include "LLMImpl.hpp"


// Public LLM thin wrappers

LLM::LLM() : impl_(new Impl()) {}
LLM::~LLM() = default;

bool LLM::Init(LLMAttrType attr) { return impl_->Init(std::move(attr)); }
void LLM::Deinit() { impl_->Deinit(); }
void LLM::Stop() { impl_->Stop(); }

LLMAttrType *LLM::getAttr() { return &impl_->_attr; }
LLMPostprocess *LLM::getPostprocess() { return &impl_->postprocess; }
LLaMaEmbedSelector *LLM::getEmbedSelector() { return &impl_->embed_selector; }
void LLM::SetRequestSamplingOverride(bool has_temperature, float temperature, bool has_top_p, float top_p, bool has_frequency_penalty, float frequency_penalty, bool has_presence_penalty, float presence_penalty) { impl_->postprocess.set_request_sampling_override(has_temperature, temperature, has_top_p, top_p, has_frequency_penalty, frequency_penalty, has_presence_penalty, presence_penalty); }
void LLM::ClearRequestSamplingOverride() { impl_->postprocess.clear_request_sampling_override(); }
void LLM::SetRequestThinkingMode(ThinkingMode mode) { if (impl_->tokenizer) impl_->tokenizer->set_generation_thinking_mode(mode); }
void LLM::ClearRequestThinkingMode() { if (impl_->tokenizer) impl_->tokenizer->set_generation_thinking_mode(impl_->_attr.generation_thinking_mode); }
bool LLM::TokenizerSupportsThinkingToggle() const { return impl_->tokenizer ? impl_->tokenizer->supports_thinking_toggle() : false; }
void LLM::MarkRequestStart() { impl_->MarkRequestStart(); }
void LLM::ClearRequestStart() { impl_->ClearRequestStart(); }
std::string LLM::GetLastError() const { return impl_->get_last_error(); }
int LLM::GetLastPromptTokenNum() const { return impl_->get_last_prompt_token_num(); }
int LLM::GetLastCompletionTokenNum() const { return impl_->get_last_completion_token_num(); }
float LLM::GetLastTtftMs() const { return impl_->get_last_ttft_ms(); }
float LLM::GetLastDecodeTps() const { return impl_->get_last_decode_tps(); }

bool LLM::Embed(const std::string &text, std::vector<float> &out_embedding) { return impl_->EmbedText(text, out_embedding); }
bool LLM::Embed(const std::vector<Content> &history, const std::vector<MediaInputs> &media_inputs, std::vector<float> &out_embedding) { return impl_->EmbedHistory(history, media_inputs, out_embedding); }
bool LLM::EmbedBatch(const std::vector<std::string> &inputs, std::vector<std::vector<float>> &out_embeddings) { return impl_->EmbedBatch(inputs, out_embeddings); }

int LLM::GenerateKVCachePrefill(std::vector<int> &ids, std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int &pre_len) { return impl_->GenerateKVCachePrefill(ids, k, v, pre_len); }
int LLM::GetKVCache(std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int &pre_len) { return impl_->GetKVCache(k, v, pre_len); }
uint64_t LLM::HashActiveKV() { return impl_->hash_active_kv(); }
uint64_t LLM::HashLastVisionEmbed() { return impl_->hash_last_vision_embed(); }
int LLM::SetKVCache(std::vector<std::vector<unsigned short>> &k, std::vector<std::vector<unsigned short>> &v, int pre_len, int in_tokens) { return impl_->SetKVCache(k, v, pre_len, in_tokens); }
void LLM::ResetKVCache() { impl_->ResetKVCache(); }

std::vector<Content> LLM::Run(std::vector<Content> history, int output_max_token) { return impl_->Run(std::move(history), output_max_token); }
std::vector<Content> LLM::Run(std::vector<Content> history, const std::vector<MediaInputs> &media_inputs, int output_max_token) { return impl_->Run(std::move(history), media_inputs, output_max_token); }
std::string LLM::Run(std::vector<unsigned short> &embed, int output_max_token) { return impl_->Run(embed, output_max_token); }

