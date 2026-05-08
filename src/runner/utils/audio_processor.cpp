#include "audio_processor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kMelFloor = 1e-3f;
constexpr float kMinFrequency = 0.0f;
constexpr float kMaxFrequency = 8000.0f;

static inline uint16_t read_u16_le(const unsigned char* p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t read_u32_le(const unsigned char* p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

struct WavFormat {
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 0;
};

class FFTPlan {
public:
    explicit FFTPlan(int n) : n_(n), rev_((size_t)n), twiddles_((size_t)n / 2)
    {
        int lg = 0;
        while ((1 << lg) < n_) ++lg;
        for (int i = 0; i < n_; ++i) {
            int x = i;
            int y = 0;
            for (int b = 0; b < lg; ++b) {
                y = (y << 1) | (x & 1);
                x >>= 1;
            }
            rev_[(size_t)i] = y;
        }
        for (int i = 0; i < n_ / 2; ++i) {
            const float ang = -2.0f * kPi * (float)i / (float)n_;
            twiddles_[(size_t)i] = std::complex<float>(std::cos(ang), std::sin(ang));
        }
    }

    void transform(std::vector<std::complex<float>>& a) const
    {
        for (int i = 0; i < n_; ++i) {
            const int j = rev_[(size_t)i];
            if (i < j) std::swap(a[(size_t)i], a[(size_t)j]);
        }

        for (int len = 2; len <= n_; len <<= 1) {
            const int half = len >> 1;
            const int step = n_ / len;
            for (int i = 0; i < n_; i += len) {
                for (int j = 0; j < half; ++j) {
                    const auto& w = twiddles_[(size_t)j * (size_t)step];
                    const auto u = a[(size_t)i + (size_t)j];
                    const auto v = a[(size_t)i + (size_t)j + (size_t)half] * w;
                    a[(size_t)i + (size_t)j] = u + v;
                    a[(size_t)i + (size_t)j + (size_t)half] = u - v;
                }
            }
        }
    }

private:
    int n_ = 0;
    std::vector<int> rev_;
    std::vector<std::complex<float>> twiddles_;
};

static std::vector<float> resample_linear(const std::vector<float>& waveform, int src_rate, int dst_rate)
{
    if (src_rate <= 0 || dst_rate <= 0 || waveform.empty() || src_rate == dst_rate) {
        return waveform;
    }

    const size_t dst_len = std::max<size_t>(1, (size_t)std::llround((double)waveform.size() * (double)dst_rate / (double)src_rate));
    std::vector<float> out(dst_len, 0.0f);
    for (size_t i = 0; i < dst_len; ++i) {
        const double src_pos = (double)i * (double)src_rate / (double)dst_rate;
        const size_t idx0 = std::min<size_t>((size_t)src_pos, waveform.size() - 1);
        const size_t idx1 = std::min<size_t>(idx0 + 1, waveform.size() - 1);
        const float frac = (float)(src_pos - (double)idx0);
        out[i] = waveform[idx0] * (1.0f - frac) + waveform[idx1] * frac;
    }
    return out;
}

static float hertz_to_mel(float freq)
{
    return 2595.0f * std::log10(1.0f + freq / 700.0f);
}

static float mel_to_hertz(float mel)
{
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

static std::vector<float> make_periodic_hann_window(int frame_length)
{
    std::vector<float> out((size_t)frame_length, 0.0f);
    for (int i = 0; i < frame_length; ++i) {
        out[(size_t)i] = 0.5f - 0.5f * std::cos(2.0f * kPi * (float)i / (float)frame_length);
    }
    return out;
}

static std::vector<float> make_mel_filter_bank(int num_frequency_bins,
                                               int num_mel_filters,
                                               int sampling_rate)
{
    const float mel_min = hertz_to_mel(kMinFrequency);
    const float mel_max = hertz_to_mel(kMaxFrequency);

    std::vector<float> filter_freqs((size_t)num_mel_filters + 2, 0.0f);
    for (int i = 0; i < num_mel_filters + 2; ++i) {
        const float ratio = (float)i / (float)(num_mel_filters + 1);
        const float mel = mel_min + (mel_max - mel_min) * ratio;
        filter_freqs[(size_t)i] = mel_to_hertz(mel);
    }

    std::vector<float> out((size_t)num_frequency_bins * (size_t)num_mel_filters, 0.0f);
    const float fft_bin_width = (float)sampling_rate / ((float)(num_frequency_bins - 1) * 2.0f);

    for (int bin = 0; bin < num_frequency_bins; ++bin) {
        const float freq = fft_bin_width * (float)bin;
        for (int mel_idx = 0; mel_idx < num_mel_filters; ++mel_idx) {
            const float left = filter_freqs[(size_t)mel_idx];
            const float center = filter_freqs[(size_t)mel_idx + 1];
            const float right = filter_freqs[(size_t)mel_idx + 2];
            const float down = (freq - left) / std::max(center - left, std::numeric_limits<float>::min());
            const float up = (right - freq) / std::max(right - center, std::numeric_limits<float>::min());
            out[(size_t)bin * (size_t)num_mel_filters + (size_t)mel_idx] =
                std::max(0.0f, std::min(down, up));
        }
    }

    return out;
}

static bool read_wav_mono_f32(const std::string& path,
                              std::vector<float>& waveform,
                              int& sample_rate,
                              std::string& err)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        err = "failed to open wav: " + path;
        return false;
    }

    unsigned char riff_header[12];
    if (!ifs.read(reinterpret_cast<char*>(riff_header), sizeof(riff_header))) {
        err = "failed to read wav header: " + path;
        return false;
    }
    if (std::memcmp(riff_header, "RIFF", 4) != 0 || std::memcmp(riff_header + 8, "WAVE", 4) != 0) {
        err = "unsupported wav container (expect RIFF/WAVE): " + path;
        return false;
    }

    WavFormat fmt{};
    std::vector<unsigned char> data_bytes;

    while (ifs) {
        unsigned char chunk_header[8];
        if (!ifs.read(reinterpret_cast<char*>(chunk_header), sizeof(chunk_header))) break;
        const uint32_t chunk_size = read_u32_le(chunk_header + 4);
        const std::string chunk_id(reinterpret_cast<const char*>(chunk_header), 4);

        if (chunk_id == "fmt ") {
            std::vector<unsigned char> fmt_bytes(chunk_size, 0);
            if (!ifs.read(reinterpret_cast<char*>(fmt_bytes.data()), chunk_size)) {
                err = "failed to read wav fmt chunk: " + path;
                return false;
            }
            if (chunk_size < 16) {
                err = "wav fmt chunk too small: " + path;
                return false;
            }
            fmt.audio_format = read_u16_le(fmt_bytes.data() + 0);
            fmt.num_channels = read_u16_le(fmt_bytes.data() + 2);
            fmt.sample_rate = read_u32_le(fmt_bytes.data() + 4);
            fmt.block_align = read_u16_le(fmt_bytes.data() + 12);
            fmt.bits_per_sample = read_u16_le(fmt_bytes.data() + 14);
        } else if (chunk_id == "data") {
            data_bytes.resize(chunk_size);
            if (!ifs.read(reinterpret_cast<char*>(data_bytes.data()), chunk_size)) {
                err = "failed to read wav data chunk: " + path;
                return false;
            }
        } else {
            ifs.seekg((std::streamoff)chunk_size, std::ios::cur);
        }

        if ((chunk_size & 1u) != 0u) ifs.seekg(1, std::ios::cur);
    }

    if (fmt.num_channels == 0 || fmt.sample_rate == 0 || fmt.bits_per_sample == 0 || data_bytes.empty()) {
        err = "wav fmt/data chunk missing or invalid: " + path;
        return false;
    }
    if (fmt.audio_format != 1 && fmt.audio_format != 3) {
        err = "unsupported wav format code: " + std::to_string(fmt.audio_format);
        return false;
    }

    const int bytes_per_sample = std::max<int>(1, (int)fmt.bits_per_sample / 8);
    const int frame_bytes = std::max<int>(1, bytes_per_sample * (int)fmt.num_channels);
    const size_t frame_count = data_bytes.size() / (size_t)frame_bytes;
    waveform.assign(frame_count, 0.0f);

    auto decode_sample = [&](const unsigned char* p) -> float {
        if (fmt.audio_format == 3 && fmt.bits_per_sample == 32) {
            float v = 0.0f;
            std::memcpy(&v, p, sizeof(float));
            return v;
        }

        switch (fmt.bits_per_sample) {
        case 8:
            return ((float)(int)p[0] - 128.0f) / 128.0f;
        case 16: {
            int16_t v = (int16_t)read_u16_le(p);
            return (float)v / 32768.0f;
        }
        case 24: {
            int32_t v = (int32_t)p[0] | ((int32_t)p[1] << 8) | ((int32_t)p[2] << 16);
            if (v & 0x00800000) v |= ~0x00FFFFFF;
            return (float)v / 8388608.0f;
        }
        case 32: {
            int32_t v = (int32_t)read_u32_le(p);
            return (float)v / 2147483648.0f;
        }
        default:
            return 0.0f;
        }
    };

    for (size_t frame = 0; frame < frame_count; ++frame) {
        const unsigned char* base = data_bytes.data() + frame * (size_t)frame_bytes;
        double acc = 0.0;
        for (int ch = 0; ch < fmt.num_channels; ++ch) {
            acc += (double)decode_sample(base + (size_t)ch * (size_t)bytes_per_sample);
        }
        waveform[frame] = (float)std::clamp(acc / (double)fmt.num_channels, -1.0, 1.0);
    }

    sample_rate = (int)fmt.sample_rate;
    return true;
}

static bool compute_log_mel_features(const std::vector<float>& waveform,
                                     const vision::audio::Gemma4AudioProfile& profile,
                                     std::vector<float>& out_features,
                                     std::string& err)
{
    const int num_mel_frames = profile.num_mel_frames;
    if (num_mel_frames <= 0) {
        err = "invalid num_mel_frames";
        return false;
    }

    int fft_length = 1;
    while (fft_length < profile.frame_length) fft_length <<= 1;
    if (fft_length <= 0) {
        err = "invalid fft_length";
        return false;
    }

    const std::vector<float> window = make_periodic_hann_window(profile.frame_length);
    const int num_frequency_bins = fft_length / 2 + 1;
    const std::vector<float> mel_filters = make_mel_filter_bank(num_frequency_bins, profile.feature_size, profile.sampling_rate);
    FFTPlan plan(fft_length);

    const int pad_left = profile.frame_length / 2;
    std::vector<float> padded((size_t)pad_left + waveform.size(), 0.0f);
    std::copy(waveform.begin(), waveform.end(), padded.begin() + pad_left);

    out_features.assign((size_t)num_mel_frames * (size_t)profile.feature_size, 0.0f);
    std::vector<std::complex<float>> fft_buf((size_t)fft_length, std::complex<float>(0.0f, 0.0f));
    std::vector<float> magnitude((size_t)num_frequency_bins, 0.0f);

    for (int frame_idx = 0; frame_idx < num_mel_frames; ++frame_idx) {
        const size_t base = (size_t)frame_idx * (size_t)profile.hop_length;
        std::fill(fft_buf.begin(), fft_buf.end(), std::complex<float>(0.0f, 0.0f));
        for (int i = 0; i < profile.frame_length; ++i) {
            fft_buf[(size_t)i] = std::complex<float>(padded[base + (size_t)i] * window[(size_t)i], 0.0f);
        }

        plan.transform(fft_buf);
        for (int bin = 0; bin < num_frequency_bins; ++bin) {
            magnitude[(size_t)bin] = std::abs(fft_buf[(size_t)bin]);
        }

        float* dst = out_features.data() + (size_t)frame_idx * (size_t)profile.feature_size;
        for (int mel_idx = 0; mel_idx < profile.feature_size; ++mel_idx) {
            double acc = 0.0;
            for (int bin = 0; bin < num_frequency_bins; ++bin) {
                acc += (double)magnitude[(size_t)bin] *
                       (double)mel_filters[(size_t)bin * (size_t)profile.feature_size + (size_t)mel_idx];
            }
            dst[(size_t)mel_idx] = std::log((float)acc + kMelFloor);
        }
    }

    return true;
}

} // namespace

namespace vision::audio {

int NumMelFrames(float duration_sec, int sampling_rate, int frame_length, int hop_length)
{
    const int num_samples = (int)std::llround((double)duration_sec * (double)sampling_rate);
    const int frame_size_for_unfold = frame_length + 1;
    const int pad_left = frame_length / 2;
    return (num_samples + pad_left - frame_size_for_unfold) / hop_length + 1;
}

int NumAudioTokens(int num_mel_frames)
{
    int tokens = num_mel_frames;
    for (int i = 0; i < 2; ++i) {
        tokens = (tokens + 2 - 3) / 2 + 1;
    }
    return tokens;
}

bool InferGemma4AudioProfileFromPath(const std::string& path, Gemma4AudioProfile& out_profile)
{
    if (path.find("audio_5s") == std::string::npos && path.find("audio_30s") == std::string::npos) {
        return false;
    }

    out_profile = {};
    out_profile.duration_sec = (path.find("audio_5s") != std::string::npos) ? 5.0f : 30.0f;
    out_profile.num_mel_frames = NumMelFrames(out_profile.duration_sec,
                                              out_profile.sampling_rate,
                                              out_profile.frame_length,
                                              out_profile.hop_length);
    out_profile.num_audio_tokens = NumAudioTokens(out_profile.num_mel_frames);
    return true;
}

bool ReadAudioDurationSeconds(const std::string& audio_path,
                              float& out_duration_sec,
                              std::string& err)
{
    std::vector<float> waveform;
    int source_sample_rate = 0;
    if (!read_wav_mono_f32(audio_path, waveform, source_sample_rate, err)) {
        return false;
    }
    out_duration_sec = source_sample_rate > 0 ? (float)waveform.size() / (float)source_sample_rate : 0.0f;
    return true;
}

bool LoadGemma4AudioInputFeatures(const std::string& audio_path,
                                  const Gemma4AudioProfile& profile,
                                  std::vector<float>& input_features,
                                  float* out_duration_sec,
                                  std::string& err)
{
    std::vector<float> waveform;
    int source_sample_rate = 0;
    if (!read_wav_mono_f32(audio_path, waveform, source_sample_rate, err)) {
        return false;
    }

    if (out_duration_sec) {
        *out_duration_sec = source_sample_rate > 0 ? (float)waveform.size() / (float)source_sample_rate : 0.0f;
    }

    std::vector<float> mono = resample_linear(waveform, source_sample_rate, profile.sampling_rate);
    const size_t target_samples = (size_t)std::llround((double)profile.duration_sec * (double)profile.sampling_rate);
    if (mono.size() < target_samples) {
        mono.resize(target_samples, 0.0f);
    } else if (mono.size() > target_samples) {
        mono.resize(target_samples);
    }

    Gemma4AudioProfile fixed_profile = profile;
    if (fixed_profile.num_mel_frames <= 0) {
        fixed_profile.num_mel_frames = NumMelFrames(fixed_profile.duration_sec,
                                                    fixed_profile.sampling_rate,
                                                    fixed_profile.frame_length,
                                                    fixed_profile.hop_length);
    }
    if (fixed_profile.num_audio_tokens <= 0) {
        fixed_profile.num_audio_tokens = NumAudioTokens(fixed_profile.num_mel_frames);
    }

    return compute_log_mel_features(mono, fixed_profile, input_features, err);
}

} // namespace vision::audio
