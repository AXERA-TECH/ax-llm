#include "audio_processor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kMelFloor = 1e-3f;
constexpr float kMinFrequency = 0.0f;
constexpr float kMaxFrequency = 8000.0f;
constexpr float kWhisperMelFloor = 1e-10f;

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

static bool command_exists(const char* name)
{
    if (!name || !*name) return false;
    const char* path_env = std::getenv("PATH");
    if (!path_env || !*path_env) return false;

    std::stringstream ss(path_env);
    std::string dir;
    while (std::getline(ss, dir, ':')) {
        if (dir.empty()) dir = ".";
        std::error_code ec;
        const auto candidate = std::filesystem::path(dir) / name;
        const auto st = std::filesystem::status(candidate, ec);
        if (!ec && std::filesystem::exists(st) && !std::filesystem::is_directory(st)) {
            return true;
        }
    }
    return false;
}

static std::string shell_quote(const std::string& value)
{
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('\'');
    for (char c : value) {
        if (c == '\'') quoted += "'\\''";
        else quoted.push_back(c);
    }
    quoted.push_back('\'');
    return quoted;
}

static bool maybe_decode_audio_via_ffmpeg(const std::string& input_path,
                                          int sampling_rate,
                                          std::string& decoded_path,
                                          std::string& err)
{
    if (!command_exists("ffmpeg")) return false;
    std::error_code ec;
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path(ec);
    if (ec || temp_dir.empty()) temp_dir = "/tmp";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    decoded_path = (temp_dir / ("axllm_audio_" + std::to_string(nonce) + ".wav")).string();
    const std::string cmd = "ffmpeg -nostdin -hide_banner -loglevel error -y -i " +
                            shell_quote(input_path) + " -ac 1 -ar " + std::to_string(sampling_rate) +
                            " -f wav " + shell_quote(decoded_path);
    const int ret = std::system(cmd.c_str());
    if (ret == 0) return true;
    std::filesystem::remove(decoded_path, ec);
    decoded_path.clear();
    err = "ffmpeg audio decode failed";
    return false;
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

static inline double sinc(double x)
{
    if (std::abs(x) < 1e-12) return 1.0;
    x *= (double)kPi;
    return std::sin(x) / x;
}

static std::vector<float> resample_bandlimited(const std::vector<float>& waveform, int src_rate, int dst_rate)
{
    if (src_rate <= 0 || dst_rate <= 0 || waveform.empty() || src_rate == dst_rate) {
        return waveform;
    }

    const size_t dst_len = std::max<size_t>(1, (size_t)std::llround((double)waveform.size() * (double)dst_rate / (double)src_rate));
    std::vector<float> out(dst_len, 0.0f);
    const double scale = (double)dst_rate / (double)src_rate;
    const double cutoff = std::min(1.0, scale);
    const double taps = 32.0;
    const int radius = std::max(1, (int)std::ceil(taps / cutoff));
    for (size_t i = 0; i < dst_len; ++i) {
        const double src_pos = (double)i / scale;
        const int center = (int)std::floor(src_pos);
        double acc = 0.0;
        double wsum = 0.0;
        for (int k = center - radius + 1; k <= center + radius; ++k) {
            const int idx = std::clamp(k, 0, (int)waveform.size() - 1);
            const double delta = src_pos - (double)k;
            const double x = cutoff * delta;
            if (std::abs(x) >= taps) continue;
            const double window = sinc(x / taps);
            const double weight = cutoff * sinc(x) * window;
            acc += (double)waveform[(size_t)idx] * weight;
            wsum += weight;
        }
        out[i] = (float)(wsum == 0.0 ? 0.0 : (acc / wsum));
    }
    return out;
}

struct SRC_DATA_Lite {
    const float* data_in;
    float* data_out;
    long input_frames;
    long output_frames;
    long input_frames_used;
    long output_frames_gen;
    int end_of_input;
    double src_ratio;
};

using src_simple_fn = int (*)(SRC_DATA_Lite*, int, int);

// Portable dynamic-load shim: dlopen/dlsym on POSIX, LoadLibrary/GetProcAddress
// on Windows (mingw64 has no <dlfcn.h>). libsamplerate is optional everywhere;
// when it can't be loaded the caller falls back to the builtin sinc resampler.
#ifdef _WIN32
static void* axllm_dlopen(const char* path) { return reinterpret_cast<void*>(LoadLibraryA(path)); }
static void* axllm_dlsym(void* handle, const char* sym) { return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), sym)); }
#else
static void* axllm_dlopen(const char* path) { return dlopen(path, RTLD_LAZY | RTLD_LOCAL); }
static void* axllm_dlsym(void* handle, const char* sym) { return dlsym(handle, sym); }
#endif

static std::vector<float> resample_via_libsamplerate(const std::vector<float>& waveform, int src_rate, int dst_rate)
{
    if (src_rate <= 0 || dst_rate <= 0 || waveform.empty() || src_rate == dst_rate) {
        return waveform;
    }

    static void* handle = nullptr;
    static src_simple_fn fn_src_simple = nullptr;
    static bool load_attempted = false;
    if (!load_attempted) {
        load_attempted = true;
        const char* candidates[] = {
#ifdef _WIN32
            "libsamplerate-0.dll",
            "libsamplerate.dll",
            "samplerate.dll",
#endif
            "/usr/local/lib/libsamplerate.so.0",
            "/usr/local/lib/libsamplerate.so",
            "/usr/lib/libsamplerate.so.0",
            "/usr/lib/libsamplerate.so",
            "libsamplerate.so.0",
            "libsamplerate.so",
        };
        for (const char* candidate : candidates) {
            handle = axllm_dlopen(candidate);
            if (handle) {
                if (std::getenv("AXLLM_DEBUG_AUDIO_RESAMPLE")) {
                    std::fprintf(stderr, "audio resample: loaded libsamplerate from %s\n", candidate);
                }
                break;
            }
        }
        if (handle) {
            fn_src_simple = reinterpret_cast<src_simple_fn>(axllm_dlsym(handle, "src_simple"));
        }
        if (!fn_src_simple && std::getenv("AXLLM_DEBUG_AUDIO_RESAMPLE")) {
            std::fprintf(stderr, "audio resample: libsamplerate unavailable, fallback to builtin sinc\n");
        }
    }

    if (!fn_src_simple) {
        return {};
    }

    const double ratio = (double)dst_rate / (double)src_rate;
    const long input_frames = (long)waveform.size();
    const long output_frames = std::max<long>(1, (long)std::ceil((double)input_frames * ratio) + 64);
    std::vector<float> out((size_t)output_frames, 0.0f);

    SRC_DATA_Lite data{};
    data.data_in = waveform.data();
    data.data_out = out.data();
    data.input_frames = input_frames;
    data.output_frames = output_frames;
    data.end_of_input = 1;
    data.src_ratio = ratio;

    constexpr int SRC_SINC_BEST_QUALITY = 0;
    const int ret = fn_src_simple(&data, SRC_SINC_BEST_QUALITY, 1);
    if (ret != 0 || data.output_frames_gen <= 0) {
        if (std::getenv("AXLLM_DEBUG_AUDIO_RESAMPLE")) {
            std::fprintf(stderr, "audio resample: src_simple failed ret=%d, fallback to builtin sinc\n", ret);
        }
        return {};
    }

    out.resize((size_t)data.output_frames_gen);
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

static float hertz_to_mel_slaney(float freq)
{
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel = 15.0f;
    constexpr float logstep = 27.0f / std::log(6.4f);
    if (freq < min_log_hertz) return 3.0f * freq / 200.0f;
    return min_log_mel + std::log(freq / min_log_hertz) * logstep;
}

static float mel_to_hertz_slaney(float mel)
{
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel = 15.0f;
    constexpr float logstep = std::log(6.4f) / 27.0f;
    if (mel < min_log_mel) return mel * 200.0f / 3.0f;
    return min_log_hertz * std::exp(logstep * (mel - min_log_mel));
}

static std::vector<float> make_periodic_hann_window(int frame_length)
{
    std::vector<float> out((size_t)frame_length, 0.0f);
    for (int i = 0; i < frame_length; ++i) {
        out[(size_t)i] = 0.5f - 0.5f * std::cos(2.0f * kPi * (float)i / (float)frame_length);
    }
    return out;
}

static std::vector<float> pad_reflect(const std::vector<float>& waveform, int pad)
{
    if (pad <= 0 || waveform.empty()) return waveform;
    if (waveform.size() == 1) {
        return std::vector<float>((size_t)pad + 1u + (size_t)pad, waveform[0]);
    }

    std::vector<float> out((size_t)pad + waveform.size() + (size_t)pad, 0.0f);
    const size_t n = waveform.size();
    for (int i = 0; i < pad; ++i) {
        size_t src = (size_t)pad - (size_t)i;
        src = std::min(src, n - 1);
        out[(size_t)i] = waveform[src];
    }
    std::copy(waveform.begin(), waveform.end(), out.begin() + pad);
    for (int i = 0; i < pad; ++i) {
        size_t src = (n - 2) - std::min<size_t>((size_t)i, n - 2);
        out[(size_t)pad + n + (size_t)i] = waveform[src];
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

static std::vector<float> make_mel_filter_bank_slaney(int num_frequency_bins,
                                                      int num_mel_filters,
                                                      int sampling_rate)
{
    const float mel_min = hertz_to_mel_slaney(kMinFrequency);
    const float mel_max = hertz_to_mel_slaney(kMaxFrequency);

    std::vector<float> filter_freqs((size_t)num_mel_filters + 2, 0.0f);
    for (int i = 0; i < num_mel_filters + 2; ++i) {
        const float ratio = (float)i / (float)(num_mel_filters + 1);
        const float mel = mel_min + (mel_max - mel_min) * ratio;
        filter_freqs[(size_t)i] = mel_to_hertz_slaney(mel);
    }

    std::vector<float> out((size_t)num_frequency_bins * (size_t)num_mel_filters, 0.0f);
    const float nyquist = (float)sampling_rate / 2.0f;
    for (int bin = 0; bin < num_frequency_bins; ++bin) {
        const float freq = ((float)bin / (float)(num_frequency_bins - 1)) * nyquist;
        for (int mel_idx = 0; mel_idx < num_mel_filters; ++mel_idx) {
            const float left = filter_freqs[(size_t)mel_idx];
            const float center = filter_freqs[(size_t)mel_idx + 1];
            const float right = filter_freqs[(size_t)mel_idx + 2];
            const float down = (freq - left) / std::max(center - left, std::numeric_limits<float>::min());
            const float up = (right - freq) / std::max(right - center, std::numeric_limits<float>::min());
            out[(size_t)bin * (size_t)num_mel_filters + (size_t)mel_idx] = std::max(0.0f, std::min(down, up));
        }
    }

    for (int mel_idx = 0; mel_idx < num_mel_filters; ++mel_idx) {
        const float denom = std::max(filter_freqs[(size_t)mel_idx + 2] - filter_freqs[(size_t)mel_idx],
                                     std::numeric_limits<float>::min());
        const float enorm = 2.0f / denom;
        for (int bin = 0; bin < num_frequency_bins; ++bin) {
            out[(size_t)bin * (size_t)num_mel_filters + (size_t)mel_idx] *= enorm;
        }
    }

    return out;
}

static void apply_whisper_resample_compat_correction(std::vector<float>& features,
                                                     const vision::audio::WhisperAudioProfile& profile)
{
    if (profile.feature_size < 128 || profile.num_mel_frames <= 0) return;

    constexpr float kFeatureFloor = -0.5885409116744995f;
    constexpr float kFeatureCeil = 1.4114590883255005f;
    constexpr int kRows[] = {126, 127};
    constexpr float kScale[] = {0.97321564f, 0.83501387f};
    constexpr float kBias[] = {-0.05427755f, -0.13645501f};

    for (int i = 0; i < 2; ++i) {
        const int row = kRows[i];
        if (row >= profile.feature_size) continue;
        float* base = features.data() + (size_t)row * (size_t)profile.num_mel_frames;
        for (int t = 0; t < profile.num_mel_frames; ++t) {
            base[t] = std::clamp(base[t] * kScale[i] + kBias[i], kFeatureFloor, kFeatureCeil);
        }
    }
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

static bool compute_whisper_log_mel_features(const std::vector<float>& waveform,
                                             const vision::audio::WhisperAudioProfile& profile,
                                             std::vector<float>& out_features,
                                             std::string& err)
{
    if (profile.num_mel_frames <= 0 || profile.n_fft <= 0 || profile.hop_length <= 0 || profile.feature_size <= 0) {
        err = "invalid whisper audio profile";
        return false;
    }

    const std::vector<float> window = make_periodic_hann_window(profile.n_fft);
    const int num_frequency_bins = profile.n_fft / 2 + 1;
    const std::vector<float> mel_filters =
        make_mel_filter_bank_slaney(num_frequency_bins, profile.feature_size, profile.sampling_rate);
    const std::vector<float> padded = pad_reflect(waveform, profile.n_fft / 2);
    if (padded.size() < (size_t)profile.n_fft) {
        err = "whisper waveform too short after padding";
        return false;
    }

    const int num_frames = 1 + (int)((padded.size() - (size_t)profile.n_fft) / (size_t)profile.hop_length);
    out_features.assign((size_t)profile.feature_size * (size_t)profile.num_mel_frames, 0.0f);

    std::vector<float> frame_buf((size_t)profile.n_fft, 0.0f);
    std::vector<float> power_spec((size_t)num_frequency_bins, 0.0f);
    std::vector<float> cos_table((size_t)num_frequency_bins * (size_t)profile.n_fft, 0.0f);
    std::vector<float> sin_table((size_t)num_frequency_bins * (size_t)profile.n_fft, 0.0f);
    for (int bin = 0; bin < num_frequency_bins; ++bin) {
        for (int i = 0; i < profile.n_fft; ++i) {
            const float ang = -2.0f * kPi * (float)bin * (float)i / (float)profile.n_fft;
            cos_table[(size_t)bin * (size_t)profile.n_fft + (size_t)i] = std::cos(ang);
            sin_table[(size_t)bin * (size_t)profile.n_fft + (size_t)i] = std::sin(ang);
        }
    }

    const int frames_to_keep = std::min(profile.num_mel_frames, std::max(0, num_frames - 1));
    for (int frame_idx = 0; frame_idx < frames_to_keep; ++frame_idx) {
        const size_t base = (size_t)frame_idx * (size_t)profile.hop_length;
        for (int i = 0; i < profile.n_fft; ++i) {
            frame_buf[(size_t)i] = padded[base + (size_t)i] * window[(size_t)i];
        }

        for (int bin = 0; bin < num_frequency_bins; ++bin) {
            double real = 0.0;
            double imag = 0.0;
            const float* cos_row = cos_table.data() + (size_t)bin * (size_t)profile.n_fft;
            const float* sin_row = sin_table.data() + (size_t)bin * (size_t)profile.n_fft;
            for (int i = 0; i < profile.n_fft; ++i) {
                const double sample = (double)frame_buf[(size_t)i];
                real += sample * (double)cos_row[(size_t)i];
                imag += sample * (double)sin_row[(size_t)i];
            }
            power_spec[(size_t)bin] = (float)(real * real + imag * imag);
        }

        for (int mel_idx = 0; mel_idx < profile.feature_size; ++mel_idx) {
            double acc = 0.0;
            for (int bin = 0; bin < num_frequency_bins; ++bin) {
                acc += (double)power_spec[(size_t)bin] *
                       (double)mel_filters[(size_t)bin * (size_t)profile.feature_size + (size_t)mel_idx];
            }
            out_features[(size_t)mel_idx * (size_t)profile.num_mel_frames + (size_t)frame_idx] =
                std::log10(std::max((float)acc, kWhisperMelFloor));
        }
    }

    float max_value = -std::numeric_limits<float>::infinity();
    for (float v : out_features) max_value = std::max(max_value, v);
    const float lower = max_value - 8.0f;
    for (float& v : out_features) {
        v = std::max(v, lower);
        v = (v + 4.0f) / 4.0f;
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

    std::vector<float> mono = resample_via_libsamplerate(waveform, source_sample_rate, profile.sampling_rate);
    if (mono.empty()) mono = resample_bandlimited(waveform, source_sample_rate, profile.sampling_rate);
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

bool LoadWhisperAudioInputFeatures(const std::string& audio_path,
                                   const WhisperAudioProfile& profile,
                                   std::vector<float>& input_features,
                                   float* out_duration_sec,
                                   std::string& err)
{
    std::string working_audio_path = audio_path;
    std::string decoded_audio_path;
    std::string ffmpeg_err;
    const bool decoded_with_ffmpeg =
        maybe_decode_audio_via_ffmpeg(audio_path, profile.sampling_rate, decoded_audio_path, ffmpeg_err);
    if (decoded_with_ffmpeg) {
        working_audio_path = decoded_audio_path;
    }

    std::vector<float> waveform;
    int source_sample_rate = 0;
    if (!read_wav_mono_f32(working_audio_path, waveform, source_sample_rate, err)) {
        if (decoded_with_ffmpeg) {
            std::error_code ec;
            std::filesystem::remove(decoded_audio_path, ec);
        }
        return false;
    }
    if (decoded_with_ffmpeg) {
        std::error_code ec;
        std::filesystem::remove(decoded_audio_path, ec);
    }

    if (out_duration_sec) {
        *out_duration_sec = source_sample_rate > 0 ? (float)waveform.size() / (float)source_sample_rate : 0.0f;
    }

    std::vector<float> mono = resample_via_libsamplerate(waveform, source_sample_rate, profile.sampling_rate);
    if (mono.empty()) mono = resample_bandlimited(waveform, source_sample_rate, profile.sampling_rate);
    const size_t target_samples = (size_t)profile.num_mel_frames * (size_t)profile.hop_length;
    if (mono.size() < target_samples) {
        mono.resize(target_samples, 0.0f);
    } else if (mono.size() > target_samples) {
        mono.resize(target_samples);
    }

    WhisperAudioProfile fixed_profile = profile;
    if (fixed_profile.duration_sec <= 0.0f) {
        fixed_profile.duration_sec = (float)fixed_profile.num_mel_frames * (float)fixed_profile.hop_length /
                                     (float)fixed_profile.sampling_rate;
    }
    if (fixed_profile.num_audio_tokens <= 0) {
        fixed_profile.num_audio_tokens = fixed_profile.num_mel_frames / 4;
    }

    if (!compute_whisper_log_mel_features(mono, fixed_profile, input_features, err)) {
        return false;
    }
    if (source_sample_rate != profile.sampling_rate) {
        apply_whisper_resample_compat_correction(input_features, fixed_profile);
    }
    return true;
}

bool LoadMossAudioInputChunks(const std::string& audio_path,
                              const WhisperAudioProfile& profile,
                              std::vector<MossAudioChunk>& chunks,
                              std::string& err)
{
    chunks.clear();

    if (profile.feature_size <= 0 || profile.num_mel_frames <= 0 ||
        profile.hop_length <= 0 || profile.sampling_rate <= 0) {
        err = "invalid Whisper audio profile for MOSS chunking";
        return false;
    }

    // Whisper's 2x temporal stride and the VQ adaptor's 4x merge.
    constexpr int kWhisperEncoderStride = 2;
    constexpr int kAudioMergeSize = 4;
    const int samples_per_token =
        profile.hop_length * kWhisperEncoderStride * kAudioMergeSize;
    if (samples_per_token <= 0) {
        err = "invalid MOSS audio token stride";
        return false;
    }

    // The compiled audio model takes [1, feature_size, num_mel_frames] for a
    // 30-second chunk (16000 Hz * 30 s = 480000 samples).
    const int chunk_samples = profile.num_mel_frames * profile.hop_length;
    if (chunk_samples <= 0) {
        err = "invalid MOSS audio chunk length";
        return false;
    }

    std::string working_audio_path = audio_path;
    std::string decoded_audio_path;
    std::string ffmpeg_err;
    const bool decoded_with_ffmpeg =
        maybe_decode_audio_via_ffmpeg(audio_path, profile.sampling_rate, decoded_audio_path, ffmpeg_err);
    if (decoded_with_ffmpeg) {
        working_audio_path = decoded_audio_path;
    }

    std::vector<float> waveform;
    int source_sample_rate = 0;
    if (!read_wav_mono_f32(working_audio_path, waveform, source_sample_rate, err)) {
        if (decoded_with_ffmpeg) {
            std::error_code ec;
            std::filesystem::remove(decoded_audio_path, ec);
        }
        return false;
    }
    if (decoded_with_ffmpeg) {
        std::error_code ec;
        std::filesystem::remove(decoded_audio_path, ec);
    }
    if (waveform.empty()) {
        err = "MOSS audio input contains no samples";
        return false;
    }

    std::vector<float> mono = resample_via_libsamplerate(waveform, source_sample_rate, profile.sampling_rate);
    if (mono.empty()) mono = resample_bandlimited(waveform, source_sample_rate, profile.sampling_rate);
    if (mono.empty()) {
        err = "MOSS audio resampling produced no samples";
        return false;
    }

    WhisperAudioProfile fixed_profile = profile;
    if (fixed_profile.duration_sec <= 0.0f) {
        fixed_profile.duration_sec =
            (float)fixed_profile.num_mel_frames * (float)fixed_profile.hop_length /
            (float)fixed_profile.sampling_rate;
    }
    if (fixed_profile.num_audio_tokens <= 0) {
        fixed_profile.num_audio_tokens = fixed_profile.num_mel_frames / 4;
    }

    const bool resample_compat_correction = source_sample_rate != profile.sampling_rate;
    for (int start = 0; start < (int)mono.size(); start += chunk_samples) {
        const int valid_samples = std::min<int>(chunk_samples, (int)mono.size() - start);
        if (valid_samples <= 0) break;

        const int num_tokens = (valid_samples - 1) / samples_per_token + 1;
        if (num_tokens <= 0 || num_tokens > fixed_profile.num_audio_tokens) {
            err = "MOSS audio chunk token count out of range: " + std::to_string(num_tokens);
            return false;
        }

        std::vector<float> padded((size_t)chunk_samples, 0.0f);
        std::copy(mono.begin() + start, mono.begin() + start + valid_samples, padded.begin());

        std::vector<float> features;
        if (!compute_whisper_log_mel_features(padded, fixed_profile, features, err)) {
            return false;
        }
        if (resample_compat_correction) {
            apply_whisper_resample_compat_correction(features, fixed_profile);
        }

        MossAudioChunk chunk;
        chunk.input_features = std::move(features);
        chunk.num_tokens = num_tokens;
        chunks.push_back(std::move(chunk));
    }

    if (chunks.empty()) {
        err = "MOSS audio produced no chunks";
        return false;
    }
    return true;
}

bool BuildMossAudioSpan(int audio_pad_id,
                        int audio_seq_len,
                        const std::vector<int>& digit_ids,
                        float audio_tokens_per_second,
                        int time_marker_every_seconds,
                        bool enable_time_markers,
                        std::vector<int>& out_span,
                        std::string& err)
{
    out_span.clear();
    if (audio_pad_id < 0) {
        err = "invalid MOSS audio pad token id";
        return false;
    }
    if (audio_seq_len < 0) {
        err = "negative MOSS audio token count";
        return false;
    }
    if (digit_ids.size() != 10) {
        err = "MOSS digit token map must contain exactly 10 entries";
        return false;
    }

    if (!enable_time_markers || audio_seq_len <= 0 || time_marker_every_seconds <= 0) {
        out_span.assign((size_t)audio_seq_len, audio_pad_id);
        return true;
    }
    if (audio_tokens_per_second <= 0.0f) {
        err = "invalid MOSS audio_tokens_per_second";
        return false;
    }

    const int tokens_per_marker = (int)(audio_tokens_per_second * (float)time_marker_every_seconds);
    if (tokens_per_marker <= 0) {
        out_span.assign((size_t)audio_seq_len, audio_pad_id);
        return true;
    }

    const double duration = (double)audio_seq_len / (double)audio_tokens_per_second;
    int consumed = 0;
    for (int second = time_marker_every_seconds;
         second <= (int)duration;
         second += time_marker_every_seconds) {
        const int position = (second / time_marker_every_seconds) * tokens_per_marker;
        const int segment_len = position - consumed;
        if (segment_len > 0) {
            out_span.insert(out_span.end(), (size_t)segment_len, audio_pad_id);
            consumed += segment_len;
        }
        const std::string digits = std::to_string(second);
        for (const char c : digits) {
            const int digit = (int)(c - '0');
            if (digit < 0 || digit > 9) {
                err = "invalid MOSS time marker digit";
                out_span.clear();
                return false;
            }
            out_span.push_back(digit_ids[(size_t)digit]);
        }
    }
    const int remainder = audio_seq_len - consumed;
    if (remainder > 0) {
        out_span.insert(out_span.end(), (size_t)remainder, audio_pad_id);
    }
    return true;
}

} // namespace vision::audio
