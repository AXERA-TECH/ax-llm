#pragma once

#include <string>
#include <vector>

namespace vision::audio {

struct Gemma4AudioProfile {
    float duration_sec = 0.0f;
    int sampling_rate = 16000;
    int feature_size = 128;
    int frame_length = 320;
    int hop_length = 160;
    int num_mel_frames = 0;
    int num_audio_tokens = 0;
};

struct WhisperAudioProfile {
    float duration_sec = 0.0f;
    int sampling_rate = 16000;
    int feature_size = 128;
    int n_fft = 400;
    int hop_length = 160;
    int num_mel_frames = 0;
    int num_audio_tokens = 0;
};

struct MossAudioChunk {
    std::vector<float> input_features;
    int num_tokens = 0;
};

int NumMelFrames(float duration_sec, int sampling_rate, int frame_length, int hop_length);
int NumAudioTokens(int num_mel_frames);

bool InferGemma4AudioProfileFromPath(const std::string& path, Gemma4AudioProfile& out_profile);

bool ReadAudioDurationSeconds(const std::string& audio_path,
                              float& out_duration_sec,
                              std::string& err);

bool LoadGemma4AudioInputFeatures(const std::string& audio_path,
                                  const Gemma4AudioProfile& profile,
                                  std::vector<float>& input_features,
                                  float* out_duration_sec,
                                  std::string& err);

bool LoadWhisperAudioInputFeatures(const std::string& audio_path,
                                   const WhisperAudioProfile& profile,
                                   std::vector<float>& input_features,
                                   float* out_duration_sec,
                                   std::string& err);

// MOSS-Transcribe-Diarize audio frontend. Decodes/resamples to 16 kHz mono,
// splits into padded 30-second log-Mel chunks, and returns the valid audio
// token count for each chunk using the Whisper 2x stride and VQ 4x merge:
// num_tokens = (valid_samples - 1) / (hop_length * 2 * 4) + 1.
bool LoadMossAudioInputChunks(const std::string& audio_path,
                              const WhisperAudioProfile& profile,
                              std::vector<MossAudioChunk>& chunks,
                              std::string& err);

// Expand one audio placeholder into `audio_seq_len` audio-pad tokens with
// numeric time anchors inserted at the configured interval. The returned
// sequence is longer than `audio_seq_len` by the number of digit tokens.
bool BuildMossAudioSpan(int audio_pad_id,
                        int audio_seq_len,
                        const std::vector<int>& digit_ids,
                        float audio_tokens_per_second,
                        int time_marker_every_seconds,
                        bool enable_time_markers,
                        std::vector<int>& out_span,
                        std::string& err);

} // namespace vision::audio
