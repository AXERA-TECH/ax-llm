#pragma once

// Per-VLM audio encoder(s), extracted out of VisionModule (audio doesn't belong in a
// vision module). Owns its own encoder runner + audio profiles; the multimodal injection
// (locating audio placeholders in input_ids and injecting embeds) stays in VisionModule.
// Only Gemma4VL (dual 5s/30s profiles) and Qwen3Omni (Whisper 30s) have audio.

#include <memory>
#include <string>
#include <vector>

namespace vision {

class AudioEncoder {
public:
    enum class Kind { None, Gemma4, Whisper };

    AudioEncoder();
    ~AudioEncoder();
    AudioEncoder(const AudioEncoder&) = delete;
    AudioEncoder& operator=(const AudioEncoder&) = delete;

    // Initialize the audio encoder(s). Gemma4 uses enc_5s + enc_30s; Whisper (Qwen3Omni)
    // uses enc_30s only. Missing/empty paths are tolerated (audio then simply rejected).
    bool Init(Kind kind, const std::string& enc_5s, const std::string& enc_30s,
              int devid, int tokens_embed_size, std::string& err);

    // Whether any audio encoder actually initialized.
    bool enabled() const;

    // Encode one audio file -> one bf16 block + tokenizer media count + placeholder token count.
    bool Encode(const std::vector<std::string>& uris,
                std::vector<unsigned short>& out_block,
                int& out_num_media_for_tokenizer,
                int& out_num_media_tokens,
                std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision
