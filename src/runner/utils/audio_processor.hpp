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

} // namespace vision::audio
