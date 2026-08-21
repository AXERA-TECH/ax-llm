#include "audio_processor.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

static void append_u16le(std::vector<unsigned char>& out, uint16_t v)
{
    out.push_back((unsigned char)(v & 0xFF));
    out.push_back((unsigned char)((v >> 8) & 0xFF));
}

static void append_u32le(std::vector<unsigned char>& out, uint32_t v)
{
    out.push_back((unsigned char)(v & 0xFF));
    out.push_back((unsigned char)((v >> 8) & 0xFF));
    out.push_back((unsigned char)((v >> 16) & 0xFF));
    out.push_back((unsigned char)((v >> 24) & 0xFF));
}

static bool write_sine_wav(const std::string& path, int sample_rate, int num_samples)
{
    std::vector<int16_t> samples((size_t)num_samples);
    for (int i = 0; i < num_samples; ++i) {
        const double t = (double)i / (double)sample_rate;
        samples[(size_t)i] = (int16_t)(0.25 * 32767.0 * std::sin(2.0 * 3.14159265358979323846 * 440.0 * t));
    }

    const uint32_t byte_rate = (uint32_t)(sample_rate * 2);
    std::vector<unsigned char> bytes;
    bytes.reserve(44 + samples.size() * 2);
    bytes.insert(bytes.end(), {'R', 'I', 'F', 'F'});
    append_u32le(bytes, (uint32_t)(36 + samples.size() * 2));
    bytes.insert(bytes.end(), {'W', 'A', 'V', 'E'});
    bytes.insert(bytes.end(), {'f', 'm', 't', ' '});
    append_u32le(bytes, 16);
    append_u16le(bytes, 1); // PCM
    append_u16le(bytes, 1); // mono
    append_u32le(bytes, (uint32_t)sample_rate);
    append_u32le(bytes, byte_rate);
    append_u16le(bytes, 2);
    append_u16le(bytes, 16);
    bytes.insert(bytes.end(), {'d', 'a', 't', 'a'});
    append_u32le(bytes, (uint32_t)(samples.size() * 2));
    for (const int16_t s : samples) {
        const uint16_t u = (uint16_t)s;
        append_u16le(bytes, u);
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
    return ofs.good();
}

static int fail(const std::string& message)
{
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    return 1;
}

} // namespace

int main()
{
    const std::string wav_path =
        (std::filesystem::temp_directory_path() / "moss_audio_test_1s.wav").string();
    if (!write_sine_wav(wav_path, 16000, 16000)) {
        return fail("failed to write test WAV");
    }

    vision::audio::WhisperAudioProfile profile;
    profile.duration_sec = 30.0f;
    profile.sampling_rate = 16000;
    profile.feature_size = 80;
    profile.n_fft = 400;
    profile.hop_length = 160;
    profile.num_mel_frames = 3000;
    profile.num_audio_tokens = 0;

    std::vector<vision::audio::MossAudioChunk> chunks;
    std::string err;
    if (!vision::audio::LoadMossAudioInputChunks(wav_path, profile, chunks, err)) {
        return fail("LoadMossAudioInputChunks failed: " + err);
    }
    if (chunks.size() != 1) {
        return fail("expected 1 chunk, got " + std::to_string(chunks.size()));
    }
    if (chunks[0].num_tokens != 13) {
        return fail("expected 13 audio tokens for 1s, got " + std::to_string(chunks[0].num_tokens));
    }
    if (chunks[0].input_features.size() != (size_t)80 * 3000) {
        return fail("unexpected feature count: " + std::to_string(chunks[0].input_features.size()));
    }
    for (const float v : chunks[0].input_features) {
        if (!std::isfinite(v)) {
            return fail("non-finite log-Mel feature");
        }
    }

    std::vector<int> digit_ids = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    std::vector<int> span;
    if (!vision::audio::BuildMossAudioSpan(7, 63, digit_ids, 12.5f, 5, true, span, err)) {
        return fail("BuildMossAudioSpan failed: " + err);
    }
    int pad_count = 0;
    for (const int id : span) {
        if (id == 7) ++pad_count;
    }
    if (pad_count != 63) {
        return fail("expected 63 audio pads, got " + std::to_string(pad_count));
    }
    if (span.size() != 64 || span[62] != 105) {
        return fail("time anchor for 5s was not inserted at the expected position");
    }

    span.clear();
    if (!vision::audio::BuildMossAudioSpan(7, 125, digit_ids, 12.5f, 5, true, span, err)) {
        return fail("BuildMossAudioSpan(10s) failed: " + err);
    }
    pad_count = 0;
    int digit_count = 0;
    for (const int id : span) {
        if (id == 7) ++pad_count;
        else if (id >= 100 && id <= 109) ++digit_count;
    }
    if (pad_count != 125 || digit_count != 3) {
        return fail("10s span has wrong pad/digit counts: pads=" + std::to_string(pad_count) +
                    " digits=" + std::to_string(digit_count));
    }

    span.clear();
    if (!vision::audio::BuildMossAudioSpan(7, 13, digit_ids, 12.5f, 5, false, span, err)) {
        return fail("BuildMossAudioSpan(disabled) failed: " + err);
    }
    if (span.size() != 13) {
        return fail("disabled markers should keep the raw pad count");
    }
    for (const int id : span) {
        if (id != 7) return fail("disabled markers span contains a non-pad token");
    }

    std::error_code ec;
    std::filesystem::remove(wav_path, ec);
    std::printf("moss_audio_test: all checks passed\n");
    return 0;
}
