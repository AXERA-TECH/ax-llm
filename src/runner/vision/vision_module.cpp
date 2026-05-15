#include "vision_module.hpp"

#include <cctype>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <regex>
#include <numeric>
#include <sstream>

#include "bfloat16.hpp"
#include "sample_log.h"
#include "utils/files.hpp"
#include "utils/audio_processor.hpp"
#include "utils/ax_cv.hpp"
#include "utils/image_processor.hpp"
#include "utils/mrope.hpp"

#ifdef USE_AXCL
#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "utils/axcl_manager.h"
using ax_runner_t = ax_runner_axcl;
static inline void v_h2d(void *phy_dst, const void *src, size_t n, int devid) { axcl_Memcpy(phy_dst, src, n, AXCL_MEMCPY_HOST_TO_DEVICE, devid); }
static inline void v_d2h(void *dst, const void *phy_src, size_t n, int devid) { axcl_Memcpy(dst, phy_src, n, AXCL_MEMCPY_DEVICE_TO_HOST, devid); }
#define V_WADDR(t) ((void *)(t).phyAddr)
#define V_RADDR(t) ((const void *)(t).phyAddr)
#else
#include "ax_model_runner/ax_model_runner_ax650.hpp"
using ax_runner_t = ax_runner_ax650;
static inline void v_h2d(void *vir_dst, const void *src, size_t n, int /*devid*/) { memcpy(vir_dst, src, n); }
static inline void v_d2h(void *dst, const void *vir_src, size_t n, int /*devid*/) { memcpy(dst, vir_src, n); }
#define V_WADDR(t) ((t).pVirAddr)
#define V_RADDR(t) ((const void *)(t).pVirAddr)
#endif

namespace vision {

namespace {

constexpr double kDefaultRawVideoSampleFps = 2.0;

struct AudioEncoderRuntime {
    ax_runner_t encoder;
    bool encoder_inited = false;
    int encoder_output_is_bf16 = -1;
    audio::Gemma4AudioProfile profile;
    std::string axmodel_path;
};

struct ScopedTempDirs {
    std::vector<std::string> dirs;

    ~ScopedTempDirs()
    {
        for (auto it = dirs.rbegin(); it != dirs.rend(); ++it) {
            std::error_code ec;
            std::filesystem::remove_all(*it, ec);
        }
    }

    void add(const std::string& dir)
    {
        if (!dir.empty()) dirs.push_back(dir);
    }
};

struct VideoFrameFitResult {
    int frame_count = 0;
    int tail_tokens = -1;
};

struct Gemma4VideoPlan {
    std::vector<std::string> sampled_uris;
    ScopedTempDirs temp_dirs;
    int frame_count = 0;
    int fitted_tail_tokens = -1;
    int total_frame_count = 0;
    int max_tail_tokens = -1;
    int precompute_len = 0;
};

} // namespace

static std::string lower_ext(const std::string& path)
{
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return ext;
}

static bool is_supported_frame_file(const std::string& path)
{
    const std::string ext = lower_ext(path);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

static std::vector<std::string> filter_supported_frame_files(const std::vector<std::string>& files)
{
    std::vector<std::string> filtered;
    filtered.reserve(files.size());
    for (const auto& file : files) {
        if (is_supported_frame_file(file)) filtered.push_back(file);
    }
    return filtered;
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

static bool parse_positive_finite_double(const std::string& text, double& value)
{
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || !end) return false;
    while (*end != '\0') {
        if (!std::isspace((unsigned char)*end)) return false;
        ++end;
    }
    if (errno == ERANGE || !std::isfinite(parsed) || parsed <= 0.0) return false;
    value = parsed;
    return true;
}

static bool split_raw_video_fps_suffix(const std::string& uri,
                                       std::string& video_path,
                                       double& sample_fps,
                                       bool& has_sample_fps,
                                       std::string& err)
{
    video_path = uri;
    sample_fps = 0.0;
    has_sample_fps = false;

    const size_t pos = uri.rfind(':');
    if (pos == std::string::npos || pos == 0) return true;

    const std::string candidate_path = uri.substr(0, pos);
    if (!is_file(candidate_path) || is_supported_frame_file(candidate_path)) return true;

    const std::string fps_text = uri.substr(pos + 1);
    if (!parse_positive_finite_double(fps_text, sample_fps)) {
        err = "invalid video fps suffix: " + fps_text + " (must be a positive finite number)";
        return false;
    }

    video_path = candidate_path;
    has_sample_fps = true;
    return true;
}

static bool read_video_duration_ffprobe(const std::string& video_path,
                                        double& duration_sec,
                                        std::string& err)
{
    duration_sec = 0.0;
    if (!command_exists("ffprobe")) {
        err = "ffprobe is not available in PATH, so video fps sampling cannot determine duration";
        return false;
    }

    const std::string cmd =
        "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 " +
        shell_quote(video_path) + " 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        err = "failed to run ffprobe for video duration: " + video_path;
        return false;
    }

    std::string output;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) output += buf;
    const int status = pclose(pipe);
    if (status != 0) {
        err = "ffprobe failed to read video duration: " + video_path;
        return false;
    }

    double parsed = 0.0;
    if (!parse_positive_finite_double(output, parsed)) {
        err = "ffprobe returned invalid video duration for: " + video_path;
        return false;
    }

    duration_sec = parsed;
    return true;
}

static int video_fps_target_count(double duration_sec, double sample_fps)
{
    const double target_frames = duration_sec * sample_fps;
    if (target_frames > (double)std::numeric_limits<int>::max()) {
        return std::numeric_limits<int>::max();
    }
    return std::max(1, (int)std::llround(target_frames));
}

static std::string format_ffmpeg_double(double value)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", value);
    return std::string(buf);
}

static bool make_temp_dir(const std::string& prefix, std::string& out_dir, std::string& err)
{
    std::filesystem::path root;
    try {
        root = std::filesystem::temp_directory_path() / "axllm_video_frames";
        std::filesystem::create_directories(root);
    } catch (const std::exception& e) {
        err = "failed to create temporary video root: " + std::string(e.what());
        return false;
    }

    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 64; ++attempt) {
        const auto candidate = root / (prefix + "_" + std::to_string((long long)now) + "_" + std::to_string(attempt));
        std::error_code ec;
        if (std::filesystem::create_directory(candidate, ec)) {
            out_dir = candidate.string();
            return true;
        }
        if (ec && ec.value() != 0 && ec != std::errc::file_exists) {
            err = "failed to create temporary frame directory: " + ec.message();
            return false;
        }
    }

    err = "failed to allocate a unique temporary frame directory";
    return false;
}

static bool extract_video_frames_ffmpeg(const std::string& video_path,
                                        std::vector<std::string>& frame_files,
                                        std::string& temp_dir_out,
                                        double output_fps,
                                        std::string& err)
{
    frame_files.clear();
    temp_dir_out.clear();

    if (!command_exists("ffmpeg")) {
        err = "ffmpeg is not available in PATH, so raw video container decoding is unavailable";
        return false;
    }

    if (!make_temp_dir("video", temp_dir_out, err)) return false;

    const auto pattern = (std::filesystem::path(temp_dir_out) / "frame_%06d.jpg").string();
    std::string filter_arg;
    if (output_fps > 0.0) {
        filter_arg = " -vf " + shell_quote("fps=fps=" + format_ffmpeg_double(output_fps));
    }
    const std::string cmd =
        "ffmpeg -hide_banner -loglevel error -y -i " + shell_quote(video_path) +
        filter_arg + " -vsync 0 " + shell_quote(pattern) + " >/dev/null 2>&1";

    ALOGI("Extracting raw video container to frames: %s -> %s%s",
          video_path.c_str(),
          temp_dir_out.c_str(),
          output_fps > 0.0 ? (" fps=" + format_ffmpeg_double(output_fps)).c_str() : "");
    if (std::system(cmd.c_str()) != 0) {
        std::error_code ec;
        std::filesystem::remove_all(temp_dir_out, ec);
        temp_dir_out.clear();
        err = "ffmpeg failed to decode video container: " + video_path;
        return false;
    }

    frame_files = filter_supported_frame_files(list_files(temp_dir_out));
    if (frame_files.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(temp_dir_out, ec);
        temp_dir_out.clear();
        err = "ffmpeg produced no decodable image frames for video container: " + video_path;
        return false;
    }

    return true;
}

static std::vector<std::string> sample_frame_paths(const std::vector<std::string>& frame_files,
                                                   int target_count,
                                                   bool do_sample_frames);

static bool collect_video_frame_paths(const std::vector<std::string>& uris,
                                      std::vector<std::string>& frame_files,
                                      ScopedTempDirs* temp_dirs,
                                      std::string& err)
{
    frame_files.clear();
    if (uris.empty()) {
        err = "media.uris empty";
        return false;
    }

    if (uris.size() == 1) {
        const auto& uri = uris[0];
        if (is_directory(uri)) {
            frame_files = filter_supported_frame_files(list_files(uri));
            if (frame_files.empty()) {
                err = "no video frames loaded";
                return false;
            }
            return true;
        }

        std::string video_path;
        double sample_fps = 0.0;
        bool has_sample_fps = false;
        if (!split_raw_video_fps_suffix(uri, video_path, sample_fps, has_sample_fps, err)) return false;

        if (!is_file(video_path)) {
            err = "invalid video uri: " + uri;
            return false;
        }
        if (is_supported_frame_file(video_path)) {
            if (has_sample_fps) {
                err = "video fps suffix is only supported for raw video files: " + uri;
                return false;
            }
            frame_files.push_back(video_path);
            return true;
        }

        if (!has_sample_fps) sample_fps = kDefaultRawVideoSampleFps;

        double duration_sec = 0.0;
        if (!read_video_duration_ffprobe(video_path, duration_sec, err)) return false;
        const int target_count = video_fps_target_count(duration_sec, sample_fps);
        const double output_fps = (duration_sec > 0.0) ? ((double)target_count / duration_sec) : sample_fps;

        std::string temp_dir;
        if (!extract_video_frames_ffmpeg(video_path, frame_files, temp_dir, output_fps, err)) return false;
        if ((int)frame_files.size() > target_count) {
            frame_files = sample_frame_paths(frame_files, target_count, true);
        }
        ALOGI("Video fps sampling: path=%s fps=%.6g duration=%.3fs target_frames=%d selected=%zu",
              video_path.c_str(),
              sample_fps,
              duration_sec,
              target_count,
              frame_files.size());
        if (frame_files.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(temp_dir, ec);
            err = "video fps sampling produced no frames: " + video_path;
            return false;
        }
        if (temp_dirs) temp_dirs->add(temp_dir);
        return true;
    }

    frame_files.reserve(uris.size());
    for (const auto& uri : uris) {
        if (is_directory(uri)) {
            err = "VIDEO uri list contains a directory; use a single frames_dir uri instead";
            return false;
        }
        if (!is_file(uri)) {
            err = "invalid video frame uri: " + uri;
            return false;
        }
        if (!is_supported_frame_file(uri)) {
            err = "VIDEO uri list contains a non-image file; pass a single raw video file or an ordered image-frame list";
            return false;
        }
        frame_files.push_back(uri);
    }
    return true;
}

static std::vector<std::string> sample_frame_paths(const std::vector<std::string>& frame_files,
                                                   int target_count,
                                                   bool do_sample_frames)
{
    if (target_count <= 0 || frame_files.empty()) return {};
    if (target_count >= (int)frame_files.size()) return frame_files;
    if (!do_sample_frames) return std::vector<std::string>(frame_files.begin(), frame_files.begin() + target_count);

    std::vector<std::string> sampled;
    sampled.reserve((size_t)target_count);
    const size_t n = frame_files.size();
    for (int i = 0; i < target_count; ++i) {
        const size_t idx = (size_t)(((unsigned long long)(2 * i + 1) * (unsigned long long)n) /
                                    (unsigned long long)(2 * target_count));
        sampled.push_back(frame_files[std::min(idx, n - 1)]);
    }
    return sampled;
}

static int count_appended_tokens(const std::vector<int>& prev_tokens,
                                 const std::vector<int>& next_tokens)
{
    if (prev_tokens.empty()) return (int)next_tokens.size();
    if (next_tokens.size() >= prev_tokens.size() &&
        std::equal(prev_tokens.begin(), prev_tokens.end(), next_tokens.begin())) {
        return (int)(next_tokens.size() - prev_tokens.size());
    }
    return (int)next_tokens.size();
}

static VideoFrameFitResult fit_gemma4_video_frame_count(const std::shared_ptr<BaseTokenizer>& tokenizer,
                                                        std::vector<Content> probe_history,
                                                        size_t content_index,
                                                        int frame_cap,
                                                        int tokens_per_block,
                                                        const std::vector<int>& prev_tokens,
                                                        int max_tail_tokens)
{
    VideoFrameFitResult result;
    if (!tokenizer || content_index >= probe_history.size() || frame_cap <= 0 || max_tail_tokens <= 0)
        return result;

    auto fits_frame_count = [&](int frame_count, int* tail_tokens_out) -> bool {
        probe_history[content_index].num_media = frame_count;
        probe_history[content_index].num_media_tokens = tokens_per_block;
        const auto probe_ids = tokenizer->encode(probe_history);
        const int tail_tokens = count_appended_tokens(prev_tokens, probe_ids);
        if (tail_tokens_out) *tail_tokens_out = tail_tokens;
        return tail_tokens <= max_tail_tokens;
    };

    int lo = 1;
    int hi = frame_cap;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        int tail_tokens = 0;
        if (fits_frame_count(mid, &tail_tokens)) {
            result.frame_count = mid;
            result.tail_tokens = tail_tokens;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    if (result.frame_count <= 0) {
        int min_tail_tokens = 0;
        (void)fits_frame_count(1, &min_tail_tokens);
        result.tail_tokens = min_tail_tokens;
    }

    return result;
}

static bool has_non_system_history_before(const std::vector<Content>& history, size_t end_index)
{
    for (size_t i = 0; i < end_index; ++i) {
        if (history[i].role != SYSTEM) return true;
    }
    return false;
}

static void build_video_fresh_history(const std::vector<Content>& history_in,
                                      size_t keep_index,
                                      std::vector<Content>& history_out,
                                      size_t& keep_index_out)
{
    history_out.clear();
    history_out.reserve(history_in.size());
    for (const auto& c : history_in) {
        if (c.role == SYSTEM) history_out.push_back(c);
    }
    keep_index_out = history_out.size();
    history_out.push_back(history_in[keep_index]);
}

struct VisionModule::Impl {
    ax_runner_t encoder;
    bool encoder_inited = false;
    int encoder_output_is_bf16 = -1;

    AudioEncoderRuntime audio_5s;
    AudioEncoderRuntime audio_30s;

    // For "classic image encoder" layout detection.
    int input_is_nchw = -1; // 1=NCHW float32, 0=NHWC u8, -1=unknown
};

static bool env_flag_false(const char* v)
{
    if (!v) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return (s == "0" || s == "false" || s == "off" || s == "no");
}

static size_t env_size_t(const char* v, size_t fallback)
{
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || (end && *end != '\0') || n < 0) return fallback;
    return (size_t)n;
}

static bool try_infer_qwen_hw_from_input_bytes(size_t input_nbytes,
                                               int temporal_patch_size,
                                               int channels,
                                               int patch_size,
                                               int& io_h,
                                               int& io_w,
                                               std::string& note)
{
    if (temporal_patch_size <= 0 || channels <= 0) return false;
    const size_t denom = (size_t)temporal_patch_size * (size_t)channels;
    if (denom == 0 || (input_nbytes % denom) != 0) return false;

    const size_t hw = input_nbytes / denom; // H*W
    if (hw == 0) return false;

    // Prefer square if possible.
    const int side = (int)(std::sqrt((double)hw) + 0.5);
    if ((size_t)side * (size_t)side == hw) {
        if (patch_size > 0 && (side % patch_size) != 0) {
            note = "perfect square but not divisible by patch_size";
            return false;
        }
        io_h = side;
        io_w = side;
        note = "square";
        return true;
    }

    // Try keep configured height and solve width.
    if (io_h > 0 && (hw % (size_t)io_h) == 0) {
        int w = (int)(hw / (size_t)io_h);
        if (patch_size <= 0 || ((io_h % patch_size) == 0 && (w % patch_size) == 0)) {
            io_w = w;
            note = "matched config height";
            return true;
        }
    }

    // Fallback: search a reasonable factor pair close to sqrt, honoring patch_size divisibility.
    size_t best_diff = (size_t)-1;
    int best_h = -1, best_w = -1;
    for (size_t h = 1; h * h <= hw; ++h) {
        if (hw % h) continue;
        size_t w = hw / h;
        if (patch_size > 0) {
            if (((int)h % patch_size) != 0) continue;
            if (((int)w % patch_size) != 0) continue;
        }
        size_t diff = (w > h) ? (w - h) : (h - w);
        if (diff < best_diff) {
            best_diff = diff;
            best_h = (int)h;
            best_w = (int)w;
        }
    }
    if (best_h > 0 && best_w > 0) {
        io_h = best_h;
        io_w = best_w;
        note = "factor-search";
        return true;
    }

    return false;
}

template <typename ShapeVec>
static bool try_infer_hw_from_4d_shape_with_c3(const ShapeVec& shape, int& out_h, int& out_w, int* out_is_nchw = nullptr)
{
    if (shape.size() != 4) return false;
    // NCHW
    if ((int)shape[1] == 3) {
        out_h = (int)shape[2];
        out_w = (int)shape[3];
        if (out_is_nchw) *out_is_nchw = 1;
        return (out_h > 0 && out_w > 0);
    }
    // NHWC
    if ((int)shape[3] == 3) {
        out_h = (int)shape[1];
        out_w = (int)shape[2];
        if (out_is_nchw) *out_is_nchw = 0;
        return (out_h > 0 && out_w > 0);
    }
    return false;
}

static bool get_single_token_id(const std::shared_ptr<BaseTokenizer>& tok, const std::string& s, int& out_id, std::string& err)
{
    auto ids = tok->encode(s);
    if (ids.size() != 1) {
        err = "special token is not a single id: '" + s + "' size=" + std::to_string(ids.size());
        return false;
    }
    out_id = ids[0];
    return true;
}

static bool parse_gemma4_profile_from_path(const std::string& path, int& out_h, int& out_w, int& out_tokens)
{
    std::smatch m;
    if (!std::regex_search(path, m, std::regex("_h(\\d+)_w(\\d+)_t(\\d+)"))) return false;
    out_h = std::stoi(m[1].str());
    out_w = std::stoi(m[2].str());
    out_tokens = std::stoi(m[3].str());
    return true;
}

static bool try_pick_tokens_by_output_bytes(const ax_runner_tensor_t& out0,
                                            int tokens_embed_size,
                                            int& out_is_bf16,
                                            int& out_tokens_per_block)
{
    for (int bytes_per_elem : {4, 2}) {
        if ((out0.nSize % bytes_per_elem) != 0) continue;
        const size_t elem = (size_t)out0.nSize / (size_t)bytes_per_elem;
        if (elem == 0 || (elem % (size_t)tokens_embed_size) != 0) continue;
        out_is_bf16 = (bytes_per_elem == 2) ? 1 : 0;
        out_tokens_per_block = (int)(elem / (size_t)tokens_embed_size);
        return true;
    }
    return false;
}

static bool init_audio_profile(AudioEncoderRuntime& runtime,
                               const std::string& axmodel_path,
                               int devid,
                               int tokens_embed_size,
                               std::string& err)
{
    audio::Gemma4AudioProfile profile;
    if (!audio::InferGemma4AudioProfileFromPath(axmodel_path, profile)) {
        err = "failed to infer Gemma4 audio profile from path: " + axmodel_path;
        return false;
    }

    if (runtime.encoder.init(axmodel_path.c_str(), devid) != 0) {
        err = "init audio encoder axmodel failed: " + axmodel_path;
        return false;
    }
    runtime.encoder_inited = true;
    runtime.profile = profile;
    runtime.axmodel_path = axmodel_path;

#ifdef USE_AXCL
    runtime.encoder.set_auto_sync_before_inference(true);
    runtime.encoder.set_auto_sync_after_inference(true);
#endif

    const auto& out0 = runtime.encoder.get_output(0);
    int out_is_bf16 = -1;
    int tokens_per_block = 0;
    auto try_pick_expected = [&](int bytes_per_elem) -> bool {
        if ((out0.nSize % bytes_per_elem) != 0) return false;
        const size_t elem = (size_t)out0.nSize / (size_t)bytes_per_elem;
        if (elem == 0 || (elem % (size_t)tokens_embed_size) != 0) return false;
        const int tokens = (int)(elem / (size_t)tokens_embed_size);
        if (tokens != runtime.profile.num_audio_tokens) return false;
        out_is_bf16 = (bytes_per_elem == 2) ? 1 : 0;
        tokens_per_block = tokens;
        return true;
    };

    if (!try_pick_expected(4) && !try_pick_expected(2)) {
        if (!try_pick_tokens_by_output_bytes(out0, tokens_embed_size, out_is_bf16, tokens_per_block)) {
            err = "failed to infer Gemma4 audio output layout: " + axmodel_path;
            return false;
        }
        ALOGW("Gemma4 audio profile token count mismatch: expected=%d inferred=%d from %s",
              runtime.profile.num_audio_tokens, tokens_per_block, axmodel_path.c_str());
    }

    runtime.encoder_output_is_bf16 = out_is_bf16;
    runtime.profile.num_audio_tokens = tokens_per_block;
    ALOGI("Gemma4 audio profile init ok: path=%s duration=%.1fs mel_frames=%d tokens=%d out_dtype=%s",
          axmodel_path.c_str(),
          runtime.profile.duration_sec,
          runtime.profile.num_mel_frames,
          runtime.profile.num_audio_tokens,
          (runtime.encoder_output_is_bf16 ? "bf16" : "fp32"));
    return true;
}

static AudioEncoderRuntime* select_audio_profile(AudioEncoderRuntime* p5,
                                                 AudioEncoderRuntime* p30,
                                                 float duration_sec)
{
    if (p5 && !p5->encoder_inited) p5 = nullptr;
    if (p30 && !p30->encoder_inited) p30 = nullptr;

    if (p5 && duration_sec <= p5->profile.duration_sec + 0.25f) return p5;
    if (p30) return p30;
    if (p5) return p5;
    return nullptr;
}

static bool file_sig(const std::string& path, uint64_t& size_out, uint64_t& mtime_ns_out)
{
    std::error_code ec;
    const auto file_size = std::filesystem::file_size(path, ec);
    if (ec) return false;

    const auto write_time = std::filesystem::last_write_time(path, ec);
    if (ec) return false;

    size_out = static_cast<uint64_t>(file_size);
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(write_time.time_since_epoch()).count();
    mtime_ns_out = static_cast<uint64_t>(ns);
    return true;
}

static std::string normalize_path_for_key(const std::string& path)
{
    try {
        return std::filesystem::absolute(path).string();
    } catch (...) {
        return path;
    }
}

static std::string to_hex_u64(uint64_t v)
{
    static const char* kHex = "0123456789abcdef";
    char buf[16];
    for (int i = 15; i >= 0; --i) { buf[i] = kHex[v & 0xF]; v >>= 4; }
    return std::string(buf, 16);
}

static uint64_t fnv1a64(const void* data, size_t n, uint64_t seed = 14695981039346656037ull)
{
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = seed;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t fnv1a64_str(const std::string& s, uint64_t seed = 14695981039346656037ull)
{
    return fnv1a64(s.data(), s.size(), seed);
}

static std::string make_image_cache_key(const std::string& prefix, const std::string& path)
{
    uint64_t sz = 0, mt = 0;
    if (!file_sig(path, sz, mt)) return prefix + "|missing:" + normalize_path_for_key(path);
    return prefix + "|file:" + normalize_path_for_key(path) + "|sz:" + std::to_string(sz) + "|mt_ns:" + std::to_string(mt);
}

struct DiskHeaderV1 {
    uint32_t magic = 0x41585631; // "AXV1"
    uint32_t version = 1;
    uint32_t tokens_embed_size = 0;
    uint32_t tokens_per_block = 0;
    uint32_t num_blocks = 0;
    uint32_t key_len = 0;
};

struct DiskHeaderV2 {
    uint32_t magic = 0x41585632; // "AXV2"
    uint32_t version = 2;
    uint32_t tokens_embed_size = 0;
    uint32_t tokens_per_block = 0;
    uint32_t num_blocks = 0;
    uint32_t key_len = 0;
    uint32_t deepstack_layers = 0;
    uint32_t deepstack_elem_count = 0; // per layer, float count
};

static bool disk_cache_load(const std::string& cache_dir,
                            const std::string& key,
                            int tokens_embed_size,
                            int tokens_per_block,
                            int deepstack_layers,
                            std::vector<std::vector<unsigned short>>& blocks_out,
                            std::vector<std::vector<float>>& deepstack_out)
{
    if (cache_dir.empty()) return false;
    std::filesystem::path dir(cache_dir);
    std::filesystem::path file = dir / (to_hex_u64(fnv1a64_str(key)) + ".bin");
    std::FILE* fp = std::fopen(file.string().c_str(), "rb");
    if (!fp) return false;

    // Guard against corrupted/truncated cache files (can happen if previous run crashes mid-write).
    // Without these bounds, we might try to allocate huge vectors and crash on next run.
    constexpr uint32_t kMaxKeyLen = 16 * 1024;
    constexpr uint32_t kMaxBlocks = 32 * 1024;
    // Keep this generous: allow large image directories, but prevent pathological allocations.
    constexpr uint64_t kMaxTotalElems = 512ull * 1024ull * 1024ull; // bf16 element count

    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(uint32_t), 1, fp) != 1) { std::fclose(fp); return false; }
    std::fseek(fp, 0, SEEK_SET);

    bool is_v2 = (magic == 0x41585632);

    uint32_t num_blocks = 0;
    uint32_t key_len = 0;
    uint32_t ds_layers = 0;
    uint32_t ds_elem_count = 0;

    if (is_v2) {
        DiskHeaderV2 hdr{};
        if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1) { std::fclose(fp); return false; }
        if (hdr.magic != 0x41585632 || hdr.version != 2) { std::fclose(fp); return false; }
        if ((int)hdr.tokens_embed_size != tokens_embed_size) { std::fclose(fp); return false; }
        if ((int)hdr.tokens_per_block != tokens_per_block) { std::fclose(fp); return false; }
        if ((int)hdr.deepstack_layers != deepstack_layers) { std::fclose(fp); return false; }
        if (hdr.key_len > kMaxKeyLen) { std::fclose(fp); return false; }
        if (hdr.num_blocks > kMaxBlocks) { std::fclose(fp); return false; }
        num_blocks = hdr.num_blocks;
        key_len = hdr.key_len;
        ds_layers = hdr.deepstack_layers;
        ds_elem_count = hdr.deepstack_elem_count;
    } else {
        DiskHeaderV1 hdr{};
        if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1) { std::fclose(fp); return false; }
        if (hdr.magic != 0x41585631 || hdr.version != 1) { std::fclose(fp); return false; }
        if (deepstack_layers != 0) { std::fclose(fp); return false; } // old cache has no deepstack
        if ((int)hdr.tokens_embed_size != tokens_embed_size) { std::fclose(fp); return false; }
        if ((int)hdr.tokens_per_block != tokens_per_block) { std::fclose(fp); return false; }
        if (hdr.key_len > kMaxKeyLen) { std::fclose(fp); return false; }
        if (hdr.num_blocks > kMaxBlocks) { std::fclose(fp); return false; }
        num_blocks = hdr.num_blocks;
        key_len = hdr.key_len;
        ds_layers = 0;
        ds_elem_count = 0;
    }

    std::string key_read;
    try {
        key_read.resize(key_len);
        if (key_len > 0) {
            if (std::fread(key_read.data(), 1, key_len, fp) != key_len) { std::fclose(fp); return false; }
            if (key_read != key) { std::fclose(fp); return false; }
        }
    } catch (...) {
        std::fclose(fp);
        return false;
    }

    std::vector<uint32_t> elem_counts(num_blocks);
    if (num_blocks > 0) {
        if (std::fread(elem_counts.data(), sizeof(uint32_t), num_blocks, fp) != num_blocks) { std::fclose(fp); return false; }
    }

    uint64_t total_elems = 0;
    for (uint32_t n : elem_counts) total_elems += (uint64_t)n;
    if (total_elems > kMaxTotalElems) { std::fclose(fp); return false; }
    if (tokens_embed_size > 0 && (total_elems % (uint64_t)tokens_embed_size) != 0) { std::fclose(fp); return false; }
    if (ds_layers > 0 && ds_elem_count != (uint32_t)total_elems) { std::fclose(fp); return false; }

    blocks_out.clear();
    blocks_out.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t n = elem_counts[i];
        std::vector<unsigned short> b;
        try {
            b.resize(n);
            if (n > 0) {
                if (std::fread(b.data(), sizeof(unsigned short), n, fp) != n) { std::fclose(fp); return false; }
            }
        } catch (...) {
            std::fclose(fp);
            return false;
        }
        blocks_out.push_back(std::move(b));
    }

    deepstack_out.clear();
    if (ds_layers > 0) {
        if (ds_elem_count == 0) { std::fclose(fp); return false; }
        try {
            deepstack_out.resize(ds_layers);
            for (uint32_t li = 0; li < ds_layers; ++li) {
                auto& v = deepstack_out[li];
                v.resize(ds_elem_count);
                if (std::fread(v.data(), sizeof(float), ds_elem_count, fp) != ds_elem_count) { std::fclose(fp); return false; }
            }
        } catch (...) {
            std::fclose(fp);
            return false;
        }
    }

    std::fclose(fp);
    return true;
}

static void disk_cache_save(const std::string& cache_dir,
                            const std::string& key,
                            int tokens_embed_size,
                            int tokens_per_block,
                            const std::vector<std::vector<unsigned short>>& blocks,
                            const std::vector<std::vector<float>>& deepstack)
{
    if (cache_dir.empty()) return;
    try { std::filesystem::create_directories(cache_dir); } catch (...) { return; }

    std::filesystem::path dir(cache_dir);
    std::filesystem::path file = dir / (to_hex_u64(fnv1a64_str(key)) + ".bin");
    // Write to a temp file then atomically replace, so a crash can't leave a corrupted cache file.
    std::filesystem::path tmp = file;
    tmp += ".tmp";

    std::FILE* fp = std::fopen(tmp.string().c_str(), "wb");
    if (!fp) return;

    DiskHeaderV2 hdr{};
    hdr.tokens_embed_size = (uint32_t)tokens_embed_size;
    hdr.tokens_per_block = (uint32_t)tokens_per_block;
    hdr.num_blocks = (uint32_t)blocks.size();
    hdr.key_len = (uint32_t)key.size();
    hdr.deepstack_layers = (uint32_t)deepstack.size();
    uint32_t ds_elem_count = 0;
    if (!deepstack.empty()) ds_elem_count = (uint32_t)deepstack[0].size();
    hdr.deepstack_elem_count = ds_elem_count;
    (void)std::fwrite(&hdr, sizeof(hdr), 1, fp);
    if (!key.empty()) (void)std::fwrite(key.data(), 1, key.size(), fp);

    std::vector<uint32_t> elem_counts;
    elem_counts.reserve(blocks.size());
    for (const auto& b : blocks) elem_counts.push_back((uint32_t)b.size());
    if (!elem_counts.empty()) (void)std::fwrite(elem_counts.data(), sizeof(uint32_t), elem_counts.size(), fp);

    for (const auto& b : blocks) {
        if (!b.empty()) (void)std::fwrite(b.data(), sizeof(unsigned short), b.size(), fp);
    }

    if (!deepstack.empty()) {
        // Require consistent sizes.
        for (const auto& v : deepstack) {
            if (v.size() != ds_elem_count) { std::fclose(fp); std::remove(tmp.string().c_str()); return; }
        }
        for (const auto& v : deepstack) {
            (void)std::fwrite(v.data(), sizeof(float), v.size(), fp);
        }
    }

    std::fclose(fp);

    std::error_code ec;
    std::filesystem::rename(tmp, file, ec);
    if (ec) {
        // Best-effort: remove temp file; keep existing cache file unchanged.
        std::remove(tmp.string().c_str());
    }
}

VisionModule::VisionModule() = default;
VisionModule::~VisionModule() { Deinit(); }
VisionModule::VisionModule(VisionModule&&) noexcept = default;
VisionModule& VisionModule::operator=(VisionModule&&) noexcept = default;

void VisionModule::Deinit()
{
    if (impl_) {
        if (impl_->encoder_inited) impl_->encoder.deinit();
        if (impl_->audio_5s.encoder_inited) impl_->audio_5s.encoder.deinit();
        if (impl_->audio_30s.encoder_inited) impl_->audio_30s.encoder.deinit();
        impl_.reset();
    }
    enabled_ = false;
    cache_enabled_ = true;
    type_ = VLMType::None;
    tokens_per_block_ = 0;
    deepstack_layers_ = 0;
    image_pad_id_ = -1;
    video_pad_id_ = -1;
    audio_pad_id_ = -1;
    vision_start_id_ = -1;
    image_cache_.clear();
    image_cache_lru_.clear();
    image_cache_lru_pos_.clear();
    image_cache_max_entries_ = 8;
}

bool VisionModule::Init(VLMType type,
                        const std::string& encoder_axmodel,
                        const std::string& cache_dir,
                        int tokens_embed_size,
                        int devid,
                        const std::shared_ptr<BaseTokenizer>& tokenizer,
                        int vision_width,
                        int vision_height,
                        int temporal_patch_size,
                        int spatial_merge_size,
                        int patch_size,
                        int fps,
                        int tokens_per_second,
                        int video_num_frames,
                        bool video_do_sample_frames,
                        const std::string& audio_encoder_axmodel_5s,
                        const std::string& audio_encoder_axmodel_30s,
                        std::string& err)
{
    Deinit();

    tokens_embed_size_ = tokens_embed_size;
    tokenizer_ = tokenizer;
    cache_dir_ = cache_dir;
    type_ = type;

    // Debug switch: disable both disk+memory vision cache.
    cache_enabled_ = !env_flag_false(std::getenv("AXLLM_VISION_CACHE"));
    if (!cache_enabled_) {
        ALOGW("Vision cache disabled by env AXLLM_VISION_CACHE=0 (disk+mem cache bypassed).");
        cache_dir_.clear();
    }
    // Memory cache size limit: protects long-running `serve` mode from unbounded growth
    // when users send lots of distinct images (especially base64 uploads that become temp files).
    image_cache_max_entries_ = env_size_t(std::getenv("AXLLM_VISION_MEM_CACHE_SIZE"), image_cache_max_entries_);

    if (type_ == VLMType::None) {
        enabled_ = false;
        return true;
    }

    impl_.reset(new Impl());

    vision_width_ = vision_width;
    vision_height_ = vision_height;
    temporal_patch_size_ = temporal_patch_size;
    spatial_merge_size_ = spatial_merge_size;
    patch_size_ = patch_size;
    fps_ = fps;
    tokens_per_second_ = tokens_per_second;
    video_num_frames_ = video_num_frames;
    video_do_sample_frames_ = video_do_sample_frames;

    // Load encoder axmodel (all supported VLM types in this repo need it).
    if (encoder_axmodel.empty()) {
        err = "filename_image_encoder_axmodel is empty";
        return false;
    }
    if (impl_->encoder.init(encoder_axmodel.c_str(), devid) != 0) {
        err = "init vision encoder axmodel failed: " + encoder_axmodel;
        return false;
    }
    impl_->encoder_inited = true;

#ifdef USE_AXCL
    impl_->encoder.set_auto_sync_before_inference(true);
    impl_->encoder.set_auto_sync_after_inference(true);
#endif

    const auto& in0 = impl_->encoder.get_input(0);

    // Auto-resolve vision width/height from encoder input shape/size, so users don't need to manually set them.
    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) {
        const int old_w = vision_width_;
        const int old_h = vision_height_;

        // PaddleOCRVL VIT takes float32 input (not uint8 like Qwen-VL);
        // divide nSize by sizeof(float) to get the effective element count.
        size_t eff_nSize = (size_t)in0.nSize;
        if (type_ == VLMType::PaddleOCRVL && (eff_nSize % sizeof(float)) == 0) {
            eff_nSize /= sizeof(float);
            ALOGI("PaddleOCRVL: encoder input nSize=%zu -> eff_nSize=%zu (float32 input)",
                  (size_t)in0.nSize, eff_nSize);
        }

        const size_t cfg_bytes = (size_t)std::max(0, old_h) * (size_t)std::max(0, old_w) *
                                 (size_t)std::max(1, temporal_patch_size_) * (size_t)3;

        int h = old_h;
        int w = old_w;
        std::string note;
        if (try_infer_qwen_hw_from_input_bytes(eff_nSize,
                                               std::max(1, temporal_patch_size_),
                                               3,
                                               std::max(1, patch_size_),
                                               h, w, note)) {
            if (w != old_w || h != old_h) {
                ALOGW("Qwen-VL vision size override: cfg=%dx%d bytes=%zu, model_input_bytes=%zu -> %dx%d (%s).",
                      old_w, old_h, cfg_bytes, (size_t)in0.nSize, w, h, note.c_str());
            }
            vision_width_ = w;
            vision_height_ = h;
        } else {
            if (vision_width_ <= 0 || vision_height_ <= 0) {
                err = "failed to infer Qwen-VL vision_width/vision_height from encoder input";
                return false;
            }
            if (cfg_bytes != (size_t)in0.nSize) {
                ALOGW("Qwen-VL vision size mismatch (cfg=%dx%d bytes=%zu, model_input_bytes=%zu). Will pad/zero input tail.",
                      vision_width_, vision_height_, cfg_bytes, (size_t)in0.nSize);
            }
        }
    } else if (type_ == VLMType::Gemma4VL) {
        const int old_w = vision_width_;
        const int old_h = vision_height_;
        int parsed_h = 0, parsed_w = 0, parsed_tokens = 0;
        const bool parsed_profile = parse_gemma4_profile_from_path(encoder_axmodel, parsed_h, parsed_w, parsed_tokens);

        const size_t eff_nsize = ((size_t)in0.nSize % sizeof(float) == 0) ? ((size_t)in0.nSize / sizeof(float)) : (size_t)in0.nSize;
        const int pixel_dim = std::max(1, patch_size_) * std::max(1, patch_size_) * 3;
        if (pixel_dim <= 0 || eff_nsize % (size_t)pixel_dim != 0) {
            err = "failed to infer Gemma4 vision patch layout from encoder input";
            return false;
        }

        const int patch_count = (int)(eff_nsize / (size_t)pixel_dim);
        if (parsed_profile) {
            vision_height_ = parsed_h;
            vision_width_ = parsed_w;
        } else if (vision_width_ > 0 && vision_height_ > 0) {
            const int expected_patch_count = (vision_height_ / std::max(1, patch_size_)) * (vision_width_ / std::max(1, patch_size_));
            if (expected_patch_count != patch_count) {
                ALOGW("Gemma4 input patch count mismatch: cfg=%dx%d -> %d patches, model=%d patches",
                      vision_width_, vision_height_, expected_patch_count, patch_count);
            }
        } else {
            err = "failed to infer Gemma4 vision_width/vision_height from encoder filename; please set config";
            return false;
        }

        if (old_w != vision_width_ || old_h != vision_height_) {
            ALOGW("Gemma4 vision size override: cfg=%dx%d -> model=%dx%d",
                  old_w, old_h, vision_width_, vision_height_);
        }
    } else if (type_ == VLMType::InternVL3 || type_ == VLMType::FastVLM) {
        // Classic image encoder: detect NCHW/NHWC layout and image size from model input shape.
        impl_->input_is_nchw = -1;
        int h = 0, w = 0, is_nchw = -1;
        const bool got_shape = try_infer_hw_from_4d_shape_with_c3(in0.vShape, h, w, &is_nchw);
        if (got_shape) {
            impl_->input_is_nchw = is_nchw;
            if (vision_width_ > 0 && vision_height_ > 0 && (vision_width_ != w || vision_height_ != h)) {
                ALOGW("classic vision size override: cfg=%dx%d -> model=%dx%d (from input shape)",
                      vision_width_, vision_height_, w, h);
            }
            vision_width_ = w;
            vision_height_ = h;
        } else {
            // Fallback: if shape cannot be parsed, try layout by config+nSize.
            if (vision_width_ <= 0 || vision_height_ <= 0) {
                err = "classic vision encoder input shape missing; please provide valid vision_width/vision_height";
                return false;
            }
            const size_t need_nhwc_u8 = (size_t)vision_width_ * (size_t)vision_height_ * (size_t)3;
            const size_t need_nchw_f32 = need_nhwc_u8 * sizeof(float);
            if ((size_t)in0.nSize == need_nchw_f32) impl_->input_is_nchw = 1;
            else if ((size_t)in0.nSize == need_nhwc_u8) impl_->input_is_nchw = 0;
            else {
                err = "classic vision encoder layout not detected from input shape/nSize";
                return false;
            }
            ALOGW("classic vision input shape unavailable; fallback to cfg size %dx%d by nSize=%zu (layout=%s)",
                  vision_width_, vision_height_, (size_t)in0.nSize,
                  (impl_->input_is_nchw == 1 ? "NCHW-fp32" : "NHWC-u8"));
        }
    } else if (type_ == VLMType::SmolVLM2) {
        // SmolVLM2 encoder is usually NHWC u8; try shape first, then infer from nSize.
        const int old_w = vision_width_;
        const int old_h = vision_height_;
        int h = 0, w = 0, tmp_layout = -1;
        bool inferred = try_infer_hw_from_4d_shape_with_c3(in0.vShape, h, w, &tmp_layout);
        if (!inferred && ((size_t)in0.nSize % 3u) == 0u) {
            const size_t hw = (size_t)in0.nSize / 3u;
            const int side = (int)(std::sqrt((double)hw) + 0.5);
            if ((size_t)side * (size_t)side == hw) {
                h = side;
                w = side;
                inferred = true;
            }
        }
        if (!inferred) {
            if (vision_width_ <= 0 || vision_height_ <= 0) {
                err = "failed to infer SmolVLM2 vision_width/vision_height from encoder input";
                return false;
            }
            ALOGW("SmolVLM2 vision size inference failed; keep cfg=%dx%d", vision_width_, vision_height_);
        } else {
            if (old_w != w || old_h != h) {
                ALOGW("SmolVLM2 vision size override: cfg=%dx%d -> model=%dx%d", old_w, old_h, w, h);
            }
            vision_width_ = w;
            vision_height_ = h;
        }
    }

    // Detect encoder output dtype + tokens_per_block.
    // Some AX* runners/models report unreliable vShape for Qwen-VL vision encoders; prefer nSize + config sanity.
    {
        const auto& out0 = impl_->encoder.get_output(0);

        auto try_pick_by_bytes = [&](int bytes_per_elem, int& out_is_bf16, int& out_tokens_per_block) -> bool {
            if (bytes_per_elem <= 0) return false;
            if ((out0.nSize % (size_t)bytes_per_elem) != 0) return false;
            const size_t elem = (size_t)out0.nSize / (size_t)bytes_per_elem;
            if (elem % (size_t)tokens_embed_size_ != 0) return false;
            out_tokens_per_block = (int)(elem / (size_t)tokens_embed_size_);
            out_is_bf16 = (bytes_per_elem == 2) ? 1 : 0;
            return true;
        };

        bool picked = false;

        if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) {
            const int grid_h = vision_height_ / std::max(1, patch_size_);
            const int grid_w = vision_width_ / std::max(1, patch_size_);
            const int llm_grid_h = grid_h / std::max(1, spatial_merge_size_);
            const int llm_grid_w = grid_w / std::max(1, spatial_merge_size_);
            const int expected_tokens = std::max(1, llm_grid_h) * std::max(1, llm_grid_w);

            // Prefer fp32 if it matches the expected token count (Qwen3-VL reference branches use fp32 outputs).
            int out_is_bf16 = -1, tpb = 0;
            if (try_pick_by_bytes(4, out_is_bf16, tpb) && tpb == expected_tokens) {
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            } else if (try_pick_by_bytes(2, out_is_bf16, tpb) && tpb == expected_tokens) {
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            } else if (try_pick_by_bytes(4, out_is_bf16, tpb)) {
                // Fallback: still prefer fp32 if valid, but warn about unexpected token count.
                ALOGW("vision encoder tokens_per_block=%d (expected=%d). Using fp32 by nSize inference (out0.nSize=%zu).",
                      tpb, expected_tokens, (size_t)out0.nSize);
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            } else if (try_pick_by_bytes(2, out_is_bf16, tpb)) {
                ALOGW("vision encoder tokens_per_block=%d (expected=%d). Using bf16 by nSize inference (out0.nSize=%zu).",
                      tpb, expected_tokens, (size_t)out0.nSize);
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            }
        } else if (type_ == VLMType::Gemma4VL) {
            int expected_h = 0, expected_w = 0, expected_tokens = 0;
            const bool parsed_profile = parse_gemma4_profile_from_path(encoder_axmodel, expected_h, expected_w, expected_tokens);
            int out_is_bf16 = -1, tpb = 0;
            if (try_pick_by_bytes(4, out_is_bf16, tpb) && (!parsed_profile || tpb == expected_tokens)) {
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            } else if (try_pick_by_bytes(2, out_is_bf16, tpb) && (!parsed_profile || tpb == expected_tokens)) {
                impl_->encoder_output_is_bf16 = out_is_bf16;
                tokens_per_block_ = tpb;
                picked = true;
            }
        }

        if (!picked) {
            // Generic path: rely on vShape + nSize.
            int elem_count = 1;
            for (auto d : out0.vShape) elem_count *= (int)d;
            if (elem_count * 2 == out0.nSize) impl_->encoder_output_is_bf16 = 1;
            else if (elem_count * 4 == out0.nSize) impl_->encoder_output_is_bf16 = 0;
            else {
                err = "vision encoder output dtype not supported (nSize mismatch)";
                return false;
            }
            if (elem_count % tokens_embed_size_ != 0) {
                err = "vision encoder output element count not divisible by tokens_embed_size";
                return false;
            }
            tokens_per_block_ = elem_count / tokens_embed_size_;
        }
    }

    // Optional deepstack (Qwen3VL family from older branches): encoder provides extra float outputs.
    deepstack_layers_ = 0;
    if (type_ == VLMType::Qwen3VL) {
        const int nout = impl_->encoder.get_num_outputs();
        if (nout > 1) {
            deepstack_layers_ = std::min(3, nout - 1);
        }
    }

    if (type_ == VLMType::Gemma4VL) {
        if (video_num_frames_ <= 0) video_num_frames_ = 32;
        if (!audio_encoder_axmodel_5s.empty() && is_file(audio_encoder_axmodel_5s)) {
            if (!init_audio_profile(impl_->audio_5s, audio_encoder_axmodel_5s, devid, tokens_embed_size_, err)) {
                return false;
            }
        }
        if (!audio_encoder_axmodel_30s.empty() && is_file(audio_encoder_axmodel_30s)) {
            if (!init_audio_profile(impl_->audio_30s, audio_encoder_axmodel_30s, devid, tokens_embed_size_, err)) {
                return false;
            }
        }
        if (!impl_->audio_5s.encoder_inited && !impl_->audio_30s.encoder_inited) {
            ALOGW("Gemma4 audio encoders are not configured or missing; AUDIO inputs will be rejected.");
        }
        ALOGI("Gemma4 video config: num_frames=%d do_sample_frames=%d",
              video_num_frames_,
              video_do_sample_frames_ ? 1 : 0);
    }

    cache_key_prefix_ = "vlm=" + std::string(VLMTypeName(type_)) + "|enc=" + normalize_path_for_key(encoder_axmodel) +
                        "|e=" + std::to_string(tokens_embed_size_) +
                        "|t=" + std::to_string(tokens_per_block_) +
                        "|ds=" + std::to_string(deepstack_layers_) +
                        "|vw=" + std::to_string(vision_width_) +
                        "|vh=" + std::to_string(vision_height_) +
                        "|tp=" + std::to_string(temporal_patch_size_) +
                        "|sm=" + std::to_string(spatial_merge_size_) +
                        "|ps=" + std::to_string(patch_size_) +
                        "|fps=" + std::to_string(fps_) +
                        "|tps=" + std::to_string(tokens_per_second_);
    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) {
        cache_key_prefix_ += "|resize=pillow_bicubic";
    }
    cache_key_prefix_ += "|bf16=rn_even";

    // Sanity checks after auto inference.
    if (vision_width_ <= 0 || vision_height_ <= 0) {
        err = "invalid vision size after inference: " + std::to_string(vision_width_) + "x" + std::to_string(vision_height_);
        return false;
    }
    if (type_ == VLMType::InternVL3 || type_ == VLMType::FastVLM) {
        if (impl_->input_is_nchw != 0 && impl_->input_is_nchw != 1) {
            err = "classic vision encoder input layout (NCHW/NHWC) not detected";
            return false;
        }
    }

    // Token ids for placeholder locating.
    switch (type_) {
    case VLMType::Qwen2_5VL:
    case VLMType::Qwen3VL:
        if (!get_single_token_id(tokenizer_, "<|image_pad|>", image_pad_id_, err)) return false;
        if (!get_single_token_id(tokenizer_, "<|video_pad|>", video_pad_id_, err)) return false;
        if (!get_single_token_id(tokenizer_, "<|vision_start|>", vision_start_id_, err)) return false;
        ALOGI("Qwen-VL token ids: vision_start=%d image_pad=%d video_pad=%d", vision_start_id_, image_pad_id_, video_pad_id_);
        break;
    case VLMType::PaddleOCRVL:
        if (!get_single_token_id(tokenizer_, "<|IMAGE_PLACEHOLDER|>", image_pad_id_, err)) return false;
        // PaddleOCRVL uses the same placeholder token for both image and video blocks.
        // Keep `video_pad_id_` aligned with the tokenizer chat template.
        video_pad_id_ = image_pad_id_;
        if (!get_single_token_id(tokenizer_, "<|IMAGE_START|>", vision_start_id_, err)) return false;
        ALOGI("PaddleOCR-VL token ids: vision_start=%d image_pad=%d video_pad=%d", vision_start_id_, image_pad_id_, video_pad_id_);
        break;
    case VLMType::InternVL3:
        if (!get_single_token_id(tokenizer_, "<IMG_CONTEXT>", image_pad_id_, err)) return false;
        video_pad_id_ = image_pad_id_;
        if (!get_single_token_id(tokenizer_, "<img>", vision_start_id_, err)) {
            // Not required for injection; mRoPE not used.
            vision_start_id_ = -1;
        }
        break;
    case VLMType::FastVLM:
        if (!get_single_token_id(tokenizer_, "<image>", image_pad_id_, err)) return false;
        video_pad_id_ = image_pad_id_;
        vision_start_id_ = -1;
        break;
    case VLMType::SmolVLM2:
        if (!get_single_token_id(tokenizer_, "<image>", image_pad_id_, err)) return false;
        video_pad_id_ = image_pad_id_;
        vision_start_id_ = -1;
        break;
    case VLMType::Gemma4VL:
        if (!get_single_token_id(tokenizer_, "<|image|>", image_pad_id_, err)) return false;
        if (!get_single_token_id(tokenizer_, "<|video|>", video_pad_id_, err)) {
            video_pad_id_ = -1;
        }
        if (!get_single_token_id(tokenizer_, "<|audio|>", audio_pad_id_, err)) return false;
        vision_start_id_ = -1;
        ALOGI("Gemma4-VL token ids: image_pad=%d video_pad=%d audio_pad=%d", image_pad_id_, video_pad_id_, audio_pad_id_);
        break;
    default:
        break;
    }

    enabled_ = true;
    ALOGI("VisionModule init ok: type=%s, tokens_per_block=%d, embed_size=%d, out_dtype=%s",
          std::string(VLMTypeName(type_)).c_str(),
          tokens_per_block_,
          tokens_embed_size_,
          (impl_->encoder_output_is_bf16 ? "bf16" : "fp32"));
    if (deepstack_layers_ > 0) {
        ALOGI("VisionModule deepstack enabled: layers=%d", deepstack_layers_);
    }
#if !defined(AXLLM_USE_OPENCV)
    ALOGW("Vision preprocess backend: SimpleCV (OpenCV not found at build time; minor differences vs OpenCV are possible)");
#endif
    return true;
}

// PaddleOCRVL: convert uint8 patches to normalized float32 before feeding to the encoder.
// Normalization: (pixel / 255.0 - mean) / std, with mean=0.5, std=0.5 => pixel / 127.5 - 1.0
static bool encode_block_normalized_float(ax_runner_t& enc, int devid, int out_is_bf16,
                                          const std::vector<unsigned char>& bytes,
                                          std::vector<unsigned short>& out_bf16,
                                          float img_mean, float img_std,
                                          int deepstack_layers,
                                          std::vector<std::vector<float>>* deepstack_out,
                                          std::string& err)
{
    const auto& in0 = enc.get_input(0);
    const size_t expected_float_elems = bytes.size();
    const size_t expected_float_bytes = expected_float_elems * sizeof(float);
    if ((size_t)in0.nSize < expected_float_bytes) {
        err = "encoder input tensor too small for float32 conversion";
        return false;
    }

    // Convert uint8 -> normalized float32
    const float scale = 1.0f / (255.0f * img_std);
    const float shift = -img_mean / img_std;
    std::vector<float> fp32(expected_float_elems);
    for (size_t i = 0; i < expected_float_elems; ++i) {
        fp32[i] = (float)bytes[i] * scale + shift;
    }

    // Copy float32 data to encoder input; zero-pad tail if needed.
    if (expected_float_bytes == (size_t)in0.nSize) {
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, fp32.data(), expected_float_bytes);
        } else {
            v_h2d(V_WADDR(in0), fp32.data(), expected_float_bytes, devid);
        }
    } else {
        std::vector<unsigned char> tmp((size_t)in0.nSize, 0);
        std::memcpy(tmp.data(), fp32.data(), expected_float_bytes);
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, tmp.data(), tmp.size());
        } else {
            v_h2d(V_WADDR(in0), tmp.data(), tmp.size(), devid);
        }
    }
    fp32.clear();
    fp32.shrink_to_fit();

    enc.inference();

    // Read output - reuse the same logic as encode_block_u8.
    const auto& out0 = enc.get_output(0);
    int elem_count = 0;
    if (out_is_bf16) {
        elem_count = (int)((size_t)out0.nSize / sizeof(unsigned short));
    } else {
        elem_count = (int)((size_t)out0.nSize / sizeof(float));
    }
    if (elem_count <= 0) { err = "vision encoder output elem_count invalid"; return false; }
    out_bf16.resize(elem_count);
    if (out_is_bf16) {
        if (out0.pVirAddr) std::memcpy(out_bf16.data(), out0.pVirAddr, (size_t)elem_count * sizeof(unsigned short));
        else v_d2h(out_bf16.data(), V_RADDR(out0), (size_t)elem_count * sizeof(unsigned short), devid);
    } else {
        std::vector<float> tmp(elem_count);
        if (out0.pVirAddr) std::memcpy(tmp.data(), out0.pVirAddr, (size_t)elem_count * sizeof(float));
        else v_d2h(tmp.data(), V_RADDR(out0), (size_t)elem_count * sizeof(float), devid);
        for (int i = 0; i < elem_count; ++i) out_bf16[i] = fp32_to_bfloat16_rne(tmp[i]);
    }
    return true;
}

static bool encode_block_fp32(ax_runner_t& enc, int devid, int out_is_bf16,
                              const std::vector<float>& values,
                              std::vector<unsigned short>& out_bf16,
                              std::string& err)
{
    const auto& in0 = enc.get_input(0);
    const size_t input_bytes = values.size() * sizeof(float);
    if ((size_t)in0.nSize < input_bytes) {
        err = "encoder input tensor too small for float32 input";
        return false;
    }

    if (input_bytes == (size_t)in0.nSize) {
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, values.data(), input_bytes);
        } else {
            v_h2d(V_WADDR(in0), values.data(), input_bytes, devid);
        }
    } else {
        std::vector<unsigned char> tmp((size_t)in0.nSize, 0);
        std::memcpy(tmp.data(), values.data(), input_bytes);
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, tmp.data(), tmp.size());
        } else {
            v_h2d(V_WADDR(in0), tmp.data(), tmp.size(), devid);
        }
    }

    enc.inference();

    const auto& out0 = enc.get_output(0);
    int elem_count = 0;
    if (out_is_bf16) {
        elem_count = (int)((size_t)out0.nSize / sizeof(unsigned short));
    } else {
        elem_count = (int)((size_t)out0.nSize / sizeof(float));
    }
    if (elem_count <= 0) {
        err = "audio encoder output elem_count invalid";
        return false;
    }
    out_bf16.resize(elem_count);

    if (out_is_bf16) {
        if (out0.pVirAddr) {
            std::memcpy(out_bf16.data(), out0.pVirAddr, (size_t)elem_count * sizeof(unsigned short));
        } else {
            v_d2h(out_bf16.data(), V_RADDR(out0), (size_t)elem_count * sizeof(unsigned short), devid);
        }
        return true;
    }

    std::vector<float> tmp(elem_count);
    if (out0.pVirAddr) {
        std::memcpy(tmp.data(), out0.pVirAddr, (size_t)elem_count * sizeof(float));
    } else {
        v_d2h(tmp.data(), V_RADDR(out0), (size_t)elem_count * sizeof(float), devid);
    }
    for (int i = 0; i < elem_count; ++i) out_bf16[i] = bfloat16(tmp[i]).data;
    return true;
}

static bool encode_block_u8(ax_runner_t& enc, int devid, int out_is_bf16,
                            const std::vector<unsigned char>& bytes,
                            std::vector<unsigned short>& out_bf16,
                            int deepstack_layers,
                            std::vector<std::vector<float>>* deepstack_out,
                            std::string& err)
{
    const auto& in0 = enc.get_input(0);
    if ((size_t)in0.nSize < bytes.size()) {
        err = "encoder input tensor too small";
        return false;
    }
    // If the model expects more bytes than the patchifier produced (config mismatch),
    // we must zero-pad the tail; otherwise leftover bytes from previous runs can dominate.
    if (bytes.size() == (size_t)in0.nSize) {
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, bytes.data(), bytes.size());
        } else {
            v_h2d(V_WADDR(in0), bytes.data(), bytes.size(), devid);
        }
    } else {
        static bool warned = false;
        if (!warned) {
            ALOGW("vision encoder input size mismatch: write=%zu < tensor=%zu (zero-pad tail).", bytes.size(), (size_t)in0.nSize);
            warned = true;
        }
        std::vector<unsigned char> tmp;
        tmp.assign((size_t)in0.nSize, 0);
        if (!bytes.empty()) std::memcpy(tmp.data(), bytes.data(), bytes.size());
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, tmp.data(), tmp.size());
        } else {
            v_h2d(V_WADDR(in0), tmp.data(), tmp.size(), devid);
        }
    }
    enc.inference();

    const auto& out0 = enc.get_output(0);
    int elem_count = 0;
    if (out_is_bf16) {
        elem_count = (int)((size_t)out0.nSize / sizeof(unsigned short));
    } else {
        elem_count = (int)((size_t)out0.nSize / sizeof(float));
    }
    if (elem_count <= 0) {
        err = "vision encoder output elem_count invalid";
        return false;
    }
    out_bf16.resize(elem_count);

    if (out_is_bf16) {
        if (out0.pVirAddr) {
            std::memcpy(out_bf16.data(), out0.pVirAddr, (size_t)elem_count * sizeof(unsigned short));
        } else {
            v_d2h(out_bf16.data(), V_RADDR(out0), (size_t)elem_count * sizeof(unsigned short), devid);
        }
    } else {
        std::vector<float> tmp(elem_count);
        if (out0.pVirAddr) {
            std::memcpy(tmp.data(), out0.pVirAddr, (size_t)elem_count * sizeof(float));
        } else {
            v_d2h(tmp.data(), V_RADDR(out0), (size_t)elem_count * sizeof(float), devid);
        }
        for (int i = 0; i < elem_count; ++i) out_bf16[i] = fp32_to_bfloat16_rne(tmp[i]);
    }

    if (deepstack_out && deepstack_layers > 0) {
        if ((int)deepstack_out->size() != deepstack_layers) deepstack_out->assign((size_t)deepstack_layers, {});
        for (int li = 0; li < deepstack_layers; ++li) {
            const auto& o = enc.get_output(li + 1);
            int o_elems = (int)((size_t)o.nSize / sizeof(float));
            int o_is_fp32 = 1;
            if ((size_t)o_elems * sizeof(float) != (size_t)o.nSize) {
                // fallback: bf16 -> fp32
                o_elems = (int)((size_t)o.nSize / sizeof(unsigned short));
                o_is_fp32 = 0;
                if ((size_t)o_elems * sizeof(unsigned short) != (size_t)o.nSize) {
                    err = "deepstack output dtype not supported (nSize mismatch)";
                    return false;
                }
            }
            std::vector<float> feat;
            feat.resize(o_elems);

            if (o_is_fp32) {
                if (o.pVirAddr) {
                    std::memcpy(feat.data(), o.pVirAddr, (size_t)o_elems * sizeof(float));
                } else {
                    v_d2h(feat.data(), V_RADDR(o), (size_t)o_elems * sizeof(float), devid);
                }
            } else {
                std::vector<unsigned short> tmp_bf16(o_elems);
                if (o.pVirAddr) {
                    std::memcpy(tmp_bf16.data(), o.pVirAddr, (size_t)o_elems * sizeof(unsigned short));
                } else {
                    v_d2h(tmp_bf16.data(), V_RADDR(o), (size_t)o_elems * sizeof(unsigned short), devid);
                }
                for (int i = 0; i < o_elems; ++i) feat[i] = bfloat16(tmp_bf16[i]).fp32();
            }

            (*deepstack_out)[li].insert((*deepstack_out)[li].end(), feat.begin(), feat.end());
        }
    }

    return true;
}

static bool encode_classic_image(ax_runner_t& enc, int devid, int out_is_bf16, int input_is_nchw,
                                 int tgt_w, int tgt_h,
                                 const axcv::Mat& img_bgr,
                                 std::vector<unsigned short>& out_bf16,
                                 std::string& err)
{
    if (axcv::empty(img_bgr)) { err = "empty image"; return false; }

    axcv::Mat dst_rs;
    axcv::resize(img_bgr, dst_rs, tgt_w, tgt_h);
    axcv::Mat dst;
    axcv::cvtColorBGR2RGB(dst_rs, dst);

    const auto& in0 = enc.get_input(0);

    if (input_is_nchw) {
        // float32 NCHW with imagenet mean/std
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float stdv[3] = {0.229f, 0.224f, 0.225f};

        std::vector<float> tmp((size_t)3 * tgt_h * tgt_w);
        for (int h = 0; h < tgt_h; h++) {
            const uint8_t* row = axcv::row_ptr(dst, h);
            for (int w = 0; w < tgt_w; w++) {
                for (int c = 0; c < 3; c++) {
                    int in_index = w * 3 + c;
                    int out_index = c * tgt_h * tgt_w + h * tgt_w + w;
                    tmp[out_index] = (float(row[in_index]) / 255.0f - mean[c]) / stdv[c];
                }
            }
        }
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, tmp.data(), tmp.size() * sizeof(float));
        } else {
            v_h2d(V_WADDR(in0), tmp.data(), tmp.size() * sizeof(float), devid);
        }
    } else {
        // u8 NHWC
        const size_t need = (size_t)tgt_h * tgt_w * 3;
        if ((size_t)in0.nSize < need) { err = "encoder input tensor too small"; return false; }
        // Pack tightly row-major.
        std::vector<uint8_t> packed;
        packed.resize(need);
        for (int r = 0; r < tgt_h; ++r) {
            const uint8_t* sp = axcv::row_ptr(dst, r);
            std::memcpy(packed.data() + (size_t)r * (size_t)tgt_w * 3, sp, (size_t)tgt_w * 3);
        }
        if (in0.pVirAddr) {
            std::memcpy(in0.pVirAddr, packed.data(), packed.size());
        } else {
            v_h2d(V_WADDR(in0), packed.data(), packed.size(), devid);
        }
    }

    enc.inference();

    const auto& out0 = enc.get_output(0);
    int elem_count = 0;
    if (out_is_bf16) {
        elem_count = (int)((size_t)out0.nSize / sizeof(unsigned short));
    } else {
        elem_count = (int)((size_t)out0.nSize / sizeof(float));
    }
    if (elem_count <= 0) {
        err = "vision encoder output elem_count invalid";
        return false;
    }
    out_bf16.resize(elem_count);

    if (out_is_bf16) {
        v_d2h(out_bf16.data(), V_RADDR(out0), (size_t)elem_count * sizeof(unsigned short), devid);
        return true;
    }

    std::vector<float> tmp(elem_count);
    v_d2h(tmp.data(), V_RADDR(out0), (size_t)elem_count * sizeof(float), devid);
    for (int i = 0; i < elem_count; ++i) out_bf16[i] = fp32_to_bfloat16_rne(tmp[i]);
    return true;
}

bool VisionModule::EncodeForContent(const Content& content,
                                    const MediaInputs& media,
                                    int& out_num_media_for_tokenizer,
                                    int& out_num_media_tokens,
                                    std::vector<std::vector<unsigned short>>& out_blocks,
                                    std::vector<std::vector<float>>* out_deepstack_append,
                                    std::vector<std::vector<int>>& out_image_grid_thw,
                                    std::vector<std::vector<int>>& out_video_grid_thw,
                                    std::string& err)
{
    if (!enabled_ || !impl_ || !impl_->encoder_inited) { err = "vision module not initialized"; return false; }
    if (content.type != IMAGE && content.type != VIDEO && content.type != AUDIO) { err = "content is not image/video/audio"; return false; }
    if (media.uris.empty()) { err = "media.uris empty"; return false; }

    out_blocks.clear();
    out_image_grid_thw.clear();
    out_video_grid_thw.clear();
    out_num_media_for_tokenizer = 0;
    out_num_media_tokens = 0;

    const int devid = impl_->encoder.get_devid();

    if (content.type == AUDIO) {
        if (type_ == VLMType::Gemma4VL) {
            if (media.uris.size() != 1) {
                err = "Gemma4 audio expects exactly 1 audio file per message";
                return false;
            }

            float duration_sec = 0.0f;
            if (!audio::ReadAudioDurationSeconds(media.uris[0], duration_sec, err)) {
                return false;
            }

            auto* runtime = select_audio_profile(&impl_->audio_5s, &impl_->audio_30s, duration_sec);
            if (!runtime) {
                err = "Gemma4 audio encoder profile is not initialized";
                return false;
            }

            std::vector<float> input_features;
            if (!audio::LoadGemma4AudioInputFeatures(media.uris[0], runtime->profile, input_features, nullptr, err)) {
                return false;
            }

            std::vector<unsigned short> emb;
            if (!encode_block_fp32(runtime->encoder,
                                   runtime->encoder.get_devid(),
                                   runtime->encoder_output_is_bf16,
                                   input_features,
                                   emb,
                                   err)) {
                return false;
            }

            out_num_media_for_tokenizer = 1;
            out_num_media_tokens = runtime->profile.num_audio_tokens;
            out_blocks.push_back(std::move(emb));
            if (duration_sec > runtime->profile.duration_sec) {
                ALOGW("Gemma4 audio input %.3fs exceeds selected %.1fs profile; trailing audio will be truncated",
                      duration_sec, runtime->profile.duration_sec);
            } else {
                ALOGI("Gemma4 audio profile selected: %.1fs -> %d tokens (input=%.3fs)",
                      runtime->profile.duration_sec,
                      runtime->profile.num_audio_tokens,
                      duration_sec);
            }
            return true;
        }
        err = "AUDIO not supported for this vlm_type";
        return false;
    }

    if (content.type == IMAGE) {
        // Expand all uris into a flat image file list (directory is treated as multiple images).
        std::vector<std::string> image_files;
        for (const auto& uri : media.uris) {
            if (is_file(uri)) {
                image_files.push_back(uri);
            } else if (is_directory(uri)) {
                auto files = list_files(uri);
                image_files.insert(image_files.end(), files.begin(), files.end());
            } else {
                err = "invalid image uri: " + uri;
                return false;
            }
        }
        if (image_files.empty()) { err = "no images found"; return false; }

        const int grid_h = vision_height_ / patch_size_;
        const int grid_w = vision_width_ / patch_size_;

        out_num_media_for_tokenizer = (int)image_files.size();
        out_num_media_tokens = tokens_per_block_;
        out_image_grid_thw.reserve(image_files.size());

        const bool mem_cache_enabled = cache_enabled_ && image_cache_max_entries_ > 0;
        auto touch_cache_key = [&](const std::string& key) {
            if (!mem_cache_enabled) return;
            auto it = image_cache_lru_pos_.find(key);
            if (it != image_cache_lru_pos_.end()) {
                image_cache_lru_.splice(image_cache_lru_.end(), image_cache_lru_, it->second);
                it->second = std::prev(image_cache_lru_.end());
                return;
            }
            image_cache_lru_.push_back(key);
            image_cache_lru_pos_[key] = std::prev(image_cache_lru_.end());
        };
        auto evict_cache_if_needed = [&]() {
            if (!mem_cache_enabled) return;
            while (image_cache_max_entries_ > 0 && image_cache_.size() > image_cache_max_entries_) {
                const std::string old = image_cache_lru_.front();
                image_cache_lru_.pop_front();
                image_cache_lru_pos_.erase(old);
                image_cache_.erase(old);
            }
        };

        for (const auto& file : image_files) {
            const std::string key = make_image_cache_key(cache_key_prefix_, file);

            if (cache_enabled_) {
                auto it = mem_cache_enabled ? image_cache_.find(key) : image_cache_.end();
                if (it == image_cache_.end()) {
                    std::vector<std::vector<unsigned short>> cached_blocks;
                    std::vector<std::vector<float>> cached_deepstack;
                    if (disk_cache_load(cache_dir_, key, tokens_embed_size_, tokens_per_block_, deepstack_layers_, cached_blocks, cached_deepstack)) {
                        ALOGI("vision cache hit (disk): %s", file.c_str());
                        CachedImage ci;
                        ci.blocks_bf16 = std::move(cached_blocks);
                        ci.deepstack_features = std::move(cached_deepstack);
                        if (mem_cache_enabled) {
                            it = image_cache_.emplace(key, std::move(ci)).first;
                            touch_cache_key(key);
                            evict_cache_if_needed();
                        } else {
                            // Mem cache disabled: directly use disk cached data without keeping it.
                            for (const auto& b : ci.blocks_bf16) out_blocks.push_back(b);
                            if (out_deepstack_append && !ci.deepstack_features.empty()) {
                                if (out_deepstack_append->size() != ci.deepstack_features.size()) {
                                    err = "deepstack cache layer count mismatch";
                                    return false;
                                }
                                for (size_t li = 0; li < ci.deepstack_features.size(); ++li) {
                                    const auto& v = ci.deepstack_features[li];
                                    (*out_deepstack_append)[li].insert((*out_deepstack_append)[li].end(), v.begin(), v.end());
                                }
                            }
                            if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) out_image_grid_thw.push_back({1, grid_h, grid_w});
                            continue;
                        }
                    }
                }

                if (it != image_cache_.end()) {
                    ALOGI("vision cache hit (mem): %s", file.c_str());
                    touch_cache_key(key);
                    for (const auto& b : it->second.blocks_bf16) out_blocks.push_back(b);
                    if (out_deepstack_append && !it->second.deepstack_features.empty()) {
                        if (out_deepstack_append->size() != it->second.deepstack_features.size()) {
                            err = "deepstack cache layer count mismatch";
                            return false;
                        }
                        for (size_t li = 0; li < it->second.deepstack_features.size(); ++li) {
                            const auto& v = it->second.deepstack_features[li];
                            (*out_deepstack_append)[li].insert((*out_deepstack_append)[li].end(), v.begin(), v.end());
                        }
                    }
                    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) out_image_grid_thw.push_back({1, grid_h, grid_w});
                    continue;
                }
            }

            axcv::Mat img = axcv::imread(file, axcv::IMREAD_COLOR);
            if (axcv::empty(img)) { err = "failed to read image: " + file; return false; }

            std::vector<std::vector<unsigned short>> blocks_for_one;
            std::vector<std::vector<float>> deepstack_for_one;
            if (out_deepstack_append && deepstack_layers_ > 0) deepstack_for_one.resize((size_t)deepstack_layers_);

            if (type_ == VLMType::SmolVLM2) {
                std::vector<axcv::Mat> one{img};
                std::vector<std::vector<unsigned char>> pixel_values;
                Smolvlm2ImageProcessor(one, pixel_values, vision_width_, vision_height_);
                blocks_for_one.reserve(pixel_values.size()); // expected 5
                for (auto& pv : pixel_values) {
                    std::vector<unsigned short> emb;
                    if (!encode_block_u8(impl_->encoder, devid, impl_->encoder_output_is_bf16, pv, emb, 0, nullptr, err)) return false;
                    blocks_for_one.push_back(std::move(emb));
                }
            }
            else if (type_ == VLMType::PaddleOCRVL) {
                // PaddleOCR-VL VIT expects patches in [N, C, pH, pW] format (channel-first per patch,
                // no spatial merge in preprocessing — merge happens inside the VIT model).
                std::vector<unsigned char> pv;
                PaddleOCRVLImageProcessor(img, pv, vision_height_, vision_width_, patch_size_);
                {
                    unsigned char mn = 255, mx = 0;
                    for (unsigned char b : pv) { if (b < mn) mn = b; if (b > mx) mx = b; }
                    ALOGI("PaddleOCRVL pixel_values bytes=%zu min=%u max=%u (w=%d h=%d ps=%d)",
                          pv.size(), (unsigned)mn, (unsigned)mx,
                          vision_width_, vision_height_, patch_size_);
                }
                std::vector<unsigned short> emb;
                if (!encode_block_normalized_float(impl_->encoder, devid, impl_->encoder_output_is_bf16,
                                                  pv, emb, 0.5f, 0.5f,
                                                  0, nullptr, err))
                    return false;
                blocks_for_one.push_back(std::move(emb));
            }
            else if (type_ == VLMType::Gemma4VL) {
                std::vector<unsigned char> pv;
                Gemma4ImageProcessor(img, pv, vision_height_, vision_width_, patch_size_);
                std::vector<unsigned short> emb;
                if (!encode_block_normalized_float(impl_->encoder, devid, impl_->encoder_output_is_bf16,
                                                  pv, emb, 0.0f, 1.0f,
                                                  0, nullptr, err))
                    return false;
                blocks_for_one.push_back(std::move(emb));
            }
            else if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL) {
                std::vector<axcv::Mat> one{img};
                std::vector<std::vector<unsigned char>> pixel_values;
                Qwen2VideoProcessor(one, pixel_values, vision_height_, vision_width_, temporal_patch_size_, spatial_merge_size_, patch_size_);
                if (pixel_values.size() != 1) { err = "Qwen2VideoProcessor(image) returned != 1 block"; return false; }
                {
                    // Quick sanity: if preprocessing is broken, pixel_values often becomes all zeros.
                    const auto& pv = pixel_values[0];
                    unsigned char mn = 255, mx = 0;
                    for (unsigned char b : pv) { if (b < mn) mn = b; if (b > mx) mx = b; }
                    ALOGI("Qwen-VL pixel_values[0] bytes=%zu min=%u max=%u (w=%d h=%d tp=%d ps=%d sm=%d)",
                          pv.size(), (unsigned)mn, (unsigned)mx,
                          vision_width_, vision_height_, temporal_patch_size_, patch_size_, spatial_merge_size_);
                }
                std::vector<unsigned short> emb;
                if (!encode_block_u8(impl_->encoder, devid, impl_->encoder_output_is_bf16, pixel_values[0], emb,
                                     deepstack_layers_, (out_deepstack_append ? &deepstack_for_one : nullptr), err))
                    return false;
                blocks_for_one.push_back(std::move(emb));
            }
            else if (type_ == VLMType::InternVL3 || type_ == VLMType::FastVLM) {
                std::vector<unsigned short> emb;
                if (!encode_classic_image(impl_->encoder, devid, impl_->encoder_output_is_bf16,
                                          impl_->input_is_nchw, vision_width_, vision_height_, img, emb, err))
                    return false;
                blocks_for_one.push_back(std::move(emb));
            }
            else {
                err = "IMAGE not supported for this vlm_type";
                return false;
            }

            if (cache_enabled_) {
                ALOGI("vision cache store: %s", file.c_str());
                disk_cache_save(cache_dir_, key, tokens_embed_size_, tokens_per_block_, blocks_for_one, deepstack_for_one);
                if (mem_cache_enabled) {
                    CachedImage ci;
                    ci.blocks_bf16 = blocks_for_one;
                    ci.deepstack_features = deepstack_for_one;
                    image_cache_[key] = std::move(ci);
                    touch_cache_key(key);
                    evict_cache_if_needed();
                }
            }

            for (auto& b : blocks_for_one) out_blocks.push_back(std::move(b));
            if (out_deepstack_append && !deepstack_for_one.empty()) {
                if (out_deepstack_append->size() != deepstack_for_one.size()) {
                    err = "deepstack layer count mismatch";
                    return false;
                }
                for (size_t li = 0; li < deepstack_for_one.size(); ++li) {
                    const auto& v = deepstack_for_one[li];
                    (*out_deepstack_append)[li].insert((*out_deepstack_append)[li].end(), v.begin(), v.end());
                }
            }
            if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) out_image_grid_thw.push_back({1, grid_h, grid_w});
        }

        return true;

        err = "IMAGE not supported for this vlm_type";
        return false;
    }

    // VIDEO
    // Supports:
    // - 1 uri: a directory of frames (recommended), or a single image frame file
    // - N uris: an ordered list of image frame files (useful for base64 uploads)
    ScopedTempDirs temp_dirs;
    std::vector<std::string> frame_files;
    if (!collect_video_frame_paths(media.uris, frame_files, &temp_dirs, err)) return false;

    std::vector<axcv::Mat> frames;
    frames.reserve(frame_files.size());
    for (const auto& file : frame_files) {
        axcv::Mat img = axcv::imread(file, axcv::IMREAD_COLOR);
        if (axcv::empty(img)) {
            err = "failed to read video frame: " + file;
            return false;
        }
        frames.push_back(img);
    }
    if (frames.empty()) { err = "no video frames loaded"; return false; }

    if (type_ == VLMType::Gemma4VL) {
        out_num_media_for_tokenizer = (int)frames.size();
        out_num_media_tokens = tokens_per_block_;
        out_blocks.reserve(frames.size());
        for (auto& frame : frames) {
            std::vector<unsigned char> pv;
            Gemma4ImageProcessor(frame, pv, vision_height_, vision_width_, patch_size_);
            std::vector<unsigned short> emb;
            if (!encode_block_normalized_float(impl_->encoder, devid, impl_->encoder_output_is_bf16,
                                               pv, emb, 0.0f, 1.0f, 0, nullptr, err))
                return false;
            out_blocks.push_back(std::move(emb));
        }
        return true;
    }

    if (type_ == VLMType::SmolVLM2) {
        std::vector<std::vector<unsigned char>> pixel_values;
        Smolvlm2VideoProcessor(frames, pixel_values, vision_width_, vision_height_);
        out_num_media_for_tokenizer = (int)pixel_values.size();
        out_num_media_tokens = tokens_per_block_;
        out_blocks.reserve(pixel_values.size());
        for (auto& pv : pixel_values) {
            std::vector<unsigned short> emb;
            if (!encode_block_u8(impl_->encoder, devid, impl_->encoder_output_is_bf16, pv, emb, 0, nullptr, err)) return false;
            out_blocks.push_back(std::move(emb));
        }
        return true;
    }

    if (type_ == VLMType::PaddleOCRVL) {
        // PaddleOCR-VL video: process each frame independently with the same VIT as images.
        const int grid_h = vision_height_ / patch_size_;
        const int grid_w = vision_width_ / patch_size_;

        out_num_media_for_tokenizer = (int)frames.size();
        out_num_media_tokens = tokens_per_block_;
        out_video_grid_thw.push_back({(int)frames.size(), grid_h, grid_w});
        out_blocks.reserve(frames.size());
        for (auto& frame : frames) {
            std::vector<unsigned char> pv;
            PaddleOCRVLImageProcessor(frame, pv, vision_height_, vision_width_, patch_size_);
            std::vector<unsigned short> emb;
            if (!encode_block_normalized_float(impl_->encoder, devid, impl_->encoder_output_is_bf16,
                                               pv, emb, 0.5f, 0.5f, 0, nullptr, err))
                return false;
            out_blocks.push_back(std::move(emb));
        }
        return true;
    }

    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL) {
        const int grid_h = vision_height_ / patch_size_;
        const int grid_w = vision_width_ / patch_size_;

        std::vector<std::vector<unsigned char>> pixel_values;
        Qwen2VideoProcessor(frames, pixel_values, vision_height_, vision_width_, temporal_patch_size_, spatial_merge_size_, patch_size_);
        out_num_media_for_tokenizer = (int)pixel_values.size();
        out_num_media_tokens = tokens_per_block_;
        out_video_grid_thw.push_back({(int)pixel_values.size(), grid_h, grid_w});
        out_blocks.reserve(pixel_values.size());
        for (auto& pv : pixel_values) {
            std::vector<unsigned short> emb;
            if (!encode_block_u8(impl_->encoder, devid, impl_->encoder_output_is_bf16, pv, emb,
                                 deepstack_layers_, out_deepstack_append, err))
                return false;
            out_blocks.push_back(std::move(emb));
        }
        return true;
    }

    err = "VIDEO not supported for this vlm_type";
    return false;
}

bool VisionModule::BuildInjectionState(const std::vector<int>& input_ids,
                                       const std::vector<std::vector<unsigned short>>& blocks,
                                       const std::vector<std::vector<float>>& deepstack,
                                       RunState& state_out,
                                       std::string& err)
{
    state_out = {};
    state_out.pos2vision.assign(input_ids.size(), -1);

    // Collect placeholder positions in order.
    std::vector<int> placeholder_pos;
    placeholder_pos.reserve(input_ids.size());
    for (size_t i = 0; i < input_ids.size(); ++i) {
        const int id = input_ids[i];
        if (id == image_pad_id_ || id == video_pad_id_ || id == audio_pad_id_) placeholder_pos.push_back((int)i);
    }

    // Flatten blocks to vision tokens (bf16).
    size_t total_elems = 0;
    for (const auto& b : blocks) total_elems += b.size();
    if (total_elems % (size_t)tokens_embed_size_ != 0) {
        err = "vision blocks total size not divisible by tokens_embed_size";
        return false;
    }
    const size_t vision_token_count = total_elems / (size_t)tokens_embed_size_;

    if (placeholder_pos.size() != vision_token_count) {
        err = "placeholder token count mismatch: placeholder=" + std::to_string(placeholder_pos.size()) +
              " vision_tokens=" + std::to_string(vision_token_count);
        return false;
    }

    state_out.vision_embed.resize(total_elems);
    size_t off = 0;
    for (const auto& b : blocks) {
        memcpy(state_out.vision_embed.data() + off, b.data(), b.size() * sizeof(unsigned short));
        off += b.size();
    }

    state_out.deepstack_features.clear();
    if (!deepstack.empty()) {
        // Expect one entry per layer, each flattened as [total_elems].
        state_out.deepstack_features = deepstack;
        for (const auto& v : state_out.deepstack_features) {
            if (v.size() != total_elems) {
                err = "deepstack feature size mismatch";
                return false;
            }
        }

        // Match python reference: deepstack visual embeds are cast to bf16 before injection.
        for (auto& v : state_out.deepstack_features) {
            for (size_t i = 0; i < v.size(); ++i) {
                const unsigned short bf = fp32_to_bfloat16_rne(v[i]);
                std::uint32_t proc = (std::uint32_t)bf << 16;
                float fp32;
                std::memcpy(&fp32, &proc, sizeof(fp32));
                v[i] = fp32;
            }
        }
    }

    for (size_t i = 0; i < placeholder_pos.size(); ++i) state_out.pos2vision[placeholder_pos[i]] = (int)i;
    return true;
}

bool VisionModule::Prepare(const std::vector<Content>& history_in,
                           const std::vector<MediaInputs>& media_inputs,
                           const PromptBudget* budget,
                           std::vector<Content>& history_out,
                           std::vector<int>& input_ids_out,
                           RunState& state_out,
                           std::string& err,
                           PrepareMetadata* meta)
{
    if (!enabled_) { err = "vision module disabled"; return false; }
    if (!tokenizer_) { err = "tokenizer not set"; return false; }

    if (meta) *meta = {};

    history_out = history_in;
    std::vector<MediaInputs> media_inputs_work = media_inputs;
    std::unordered_map<size_t, Gemma4VideoPlan> gemma4_video_plans;
    ScopedTempDirs planned_temp_dirs;

    if (type_ == VLMType::Gemma4VL && budget) {
        size_t video_count = 0;
        size_t video_index = (size_t)-1;
        for (size_t i = 0; i < history_in.size(); ++i) {
            if (history_in[i].type == VIDEO) {
                ++video_count;
                video_index = i;
            }
        }

        if (video_count == 1 && video_index == history_in.size() - 1 && history_in[video_index].role == USER) {
            auto media_it = std::find_if(media_inputs.begin(), media_inputs.end(),
                                         [video_index](const MediaInputs& m) { return m.content_index == video_index; });
            if (media_it != media_inputs.end()) {
                Gemma4VideoPlan plan;
                std::vector<std::string> all_frame_files;
                if (!collect_video_frame_paths(media_it->uris, all_frame_files, &planned_temp_dirs, err)) return false;

                const int frame_cap = (video_num_frames_ > 0)
                                          ? std::min((int)all_frame_files.size(), video_num_frames_)
                                          : (int)all_frame_files.size();
                if (frame_cap <= 0) {
                    err = "Gemma4 video has no usable frames";
                    return false;
                }

                const bool has_prior_non_system = has_non_system_history_before(history_in, video_index);
                const bool can_compare_fresh = has_prior_non_system && budget->precompute_len > 0 && budget->max_total_tokens > 0;

                VideoFrameFitResult current_fit;
                if (budget->max_history_tokens > 0 && budget->precompute_len > budget->max_history_tokens) {
                    current_fit.frame_count = 0;
                    current_fit.tail_tokens = budget->max_tail_tokens + 1;
                } else {
                    current_fit = fit_gemma4_video_frame_count(tokenizer_,
                                                               history_in,
                                                               video_index,
                                                               frame_cap,
                                                               tokens_per_block_,
                                                               budget->last_tokens,
                                                               budget->max_tail_tokens);
                }

                VideoFrameFitResult fresh_fit;
                std::vector<Content> fresh_history;
                size_t fresh_video_index = (size_t)-1;
                if (can_compare_fresh) {
                    build_video_fresh_history(history_in, video_index, fresh_history, fresh_video_index);
                    fresh_fit = fit_gemma4_video_frame_count(tokenizer_,
                                                             fresh_history,
                                                             fresh_video_index,
                                                             frame_cap,
                                                             tokens_per_block_,
                                                             {},
                                                             budget->max_total_tokens);
                }

                const bool auto_reset_for_video = can_compare_fresh && fresh_fit.frame_count > current_fit.frame_count;
                if (meta) {
                    meta->auto_reset_for_video = auto_reset_for_video;
                    meta->current_video_frames = current_fit.frame_count;
                    meta->fresh_video_frames = can_compare_fresh ? fresh_fit.frame_count : -1;
                }

                if (auto_reset_for_video) {
                    history_out = std::move(fresh_history);
                    media_inputs_work.clear();
                    media_inputs_work.push_back({fresh_video_index, media_it->uris});
                    video_index = fresh_video_index;
                    plan.frame_count = fresh_fit.frame_count;
                    plan.fitted_tail_tokens = fresh_fit.tail_tokens;
                    plan.max_tail_tokens = budget->max_total_tokens;
                    plan.precompute_len = 0;
                    ALOGI("Gemma4 video auto reset selected: current_frames=%d fresh_frames=%d precompute_len=%d",
                          current_fit.frame_count,
                          fresh_fit.frame_count,
                          budget->precompute_len);
                } else {
                    if (current_fit.frame_count <= 0) {
                        err = "Gemma4 video prompt exceeds current prefill budget: 1 frame requires " +
                              std::to_string(current_fit.tail_tokens) + " tail tokens, budget allows " +
                              std::to_string(budget->max_tail_tokens);
                        return false;
                    }
                    plan.frame_count = current_fit.frame_count;
                    plan.fitted_tail_tokens = current_fit.tail_tokens;
                    plan.max_tail_tokens = budget->max_tail_tokens;
                    plan.precompute_len = budget->precompute_len;
                }

                plan.total_frame_count = (int)all_frame_files.size();
                plan.sampled_uris = sample_frame_paths(all_frame_files, plan.frame_count, video_do_sample_frames_);
                if ((int)plan.sampled_uris.size() != plan.frame_count) {
                    err = "Gemma4 video frame sampler produced an unexpected frame count";
                    return false;
                }
                gemma4_video_plans.emplace(video_index, std::move(plan));
            }
        }
    }

    // Map content_index -> MediaInputs (at most one entry per content index).
    std::unordered_map<size_t, MediaInputs> media_map;
    media_map.reserve(media_inputs_work.size());
    for (const auto& m : media_inputs_work) media_map[m.content_index] = m;

    std::vector<std::vector<unsigned short>> all_blocks;
    std::vector<std::vector<float>> all_deepstack;
    if (deepstack_layers_ > 0) all_deepstack.resize((size_t)deepstack_layers_);
    std::vector<std::vector<int>> image_grid_thw;
    std::vector<std::vector<int>> video_grid_thw;

    for (size_t i = 0; i < history_out.size(); ++i) {
        auto& c = history_out[i];
        if (c.type != IMAGE && c.type != VIDEO && c.type != AUDIO) continue;

        auto it = media_map.find(i);
        if (it == media_map.end()) {
            err = "missing media_inputs for history index " + std::to_string(i);
            return false;
        }

        MediaInputs effective_media = it->second;
        ScopedTempDirs temp_dirs;
        if (type_ == VLMType::Gemma4VL && c.type == VIDEO) {
            auto plan_it = gemma4_video_plans.find(i);
            if (plan_it != gemma4_video_plans.end()) {
                effective_media.uris = plan_it->second.sampled_uris;
                ALOGI("Gemma4 video frames selected: %d/%d (configured_cap=%d, tail_tokens=%d, max_tail=%d, precompute_len=%d)",
                      plan_it->second.frame_count,
                      plan_it->second.total_frame_count,
                      video_num_frames_,
                      plan_it->second.fitted_tail_tokens,
                      plan_it->second.max_tail_tokens,
                      plan_it->second.precompute_len);
            } else {
                std::vector<std::string> all_frame_files;
                if (!collect_video_frame_paths(it->second.uris, all_frame_files, &temp_dirs, err)) return false;

                int frame_cap = (video_num_frames_ > 0)
                                    ? std::min((int)all_frame_files.size(), video_num_frames_)
                                    : (int)all_frame_files.size();
                if (frame_cap <= 0) {
                    err = "Gemma4 video has no usable frames";
                    return false;
                }

                int fitted_tail_tokens = -1;
                if (budget) {
                    if (budget->max_history_tokens > 0 && budget->precompute_len > budget->max_history_tokens) {
                        err = "Gemma4 video prompt exceeds current history budget before frame injection";
                        return false;
                    }

                    const auto fit = fit_gemma4_video_frame_count(tokenizer_,
                                                                  history_out,
                                                                  i,
                                                                  frame_cap,
                                                                  tokens_per_block_,
                                                                  budget->last_tokens,
                                                                  budget->max_tail_tokens);
                    if (fit.frame_count <= 0) {
                        err = "Gemma4 video prompt exceeds current prefill budget: 1 frame requires " +
                              std::to_string(fit.tail_tokens) + " tail tokens, budget allows " +
                              std::to_string(budget->max_tail_tokens);
                        return false;
                    }
                    frame_cap = fit.frame_count;
                    fitted_tail_tokens = fit.tail_tokens;
                }

                effective_media.uris = sample_frame_paths(all_frame_files, frame_cap, video_do_sample_frames_);
                if ((int)effective_media.uris.size() != frame_cap) {
                    err = "Gemma4 video frame sampler produced an unexpected frame count";
                    return false;
                }

                ALOGI("Gemma4 video frames selected: %d/%zu (configured_cap=%d, tail_tokens=%d, max_tail=%d, precompute_len=%d)",
                      frame_cap,
                      all_frame_files.size(),
                      video_num_frames_,
                      fitted_tail_tokens,
                      budget ? budget->max_tail_tokens : -1,
                      budget ? budget->precompute_len : 0);
            }
        }

        int num_media_for_tokenizer = 0;
        int num_media_tokens = 0;
        std::vector<std::vector<unsigned short>> blocks;
        std::vector<std::vector<int>> img_grid, vid_grid;
        if (!EncodeForContent(c, effective_media, num_media_for_tokenizer, num_media_tokens, blocks,
                              (deepstack_layers_ > 0 ? &all_deepstack : nullptr),
                              img_grid, vid_grid, err))
            return false;

        c.num_media = num_media_for_tokenizer;
        c.num_media_tokens = num_media_tokens;

        for (auto& b : blocks) all_blocks.push_back(std::move(b));
        image_grid_thw.insert(image_grid_thw.end(), img_grid.begin(), img_grid.end());
        video_grid_thw.insert(video_grid_thw.end(), vid_grid.begin(), vid_grid.end());
    }

    input_ids_out = tokenizer_->encode(history_out);
    if (!BuildInjectionState(input_ids_out, all_blocks, all_deepstack, state_out, err)) return false;

    // Optional: mRoPE (Qwen-VL)
    if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::Qwen3VL || type_ == VLMType::PaddleOCRVL) {
        mrope::Config cfg;
        cfg.vision_config.temporal_patch_size = temporal_patch_size_;
        cfg.vision_config.tokens_per_second = tokens_per_second_;
        cfg.vision_config.spatial_merge_size = spatial_merge_size_;
        cfg.vision_config.patch_size = patch_size_;
        cfg.vision_config.width = vision_width_;
        cfg.vision_config.height = vision_height_;
        cfg.vision_config.fps = fps_;
        cfg.image_token_id = image_pad_id_;
        cfg.video_token_id = video_pad_id_;
        cfg.vision_start_token_id = vision_start_id_;

        if (type_ == VLMType::Qwen2_5VL || type_ == VLMType::PaddleOCRVL) {
            std::vector<double> second_per_grid_ts;
            second_per_grid_ts.reserve(video_grid_thw.size());
            for (size_t i = 0; i < video_grid_thw.size(); ++i) {
                second_per_grid_ts.push_back(double(temporal_patch_size_) / double(std::max(1, fps_)));
            }
            state_out.position_ids = mrope::get_rope_index_qwen2_5(cfg, input_ids_out, image_grid_thw, video_grid_thw, second_per_grid_ts);
        } else {
            state_out.position_ids = mrope::get_rope_index_qwen3(cfg, input_ids_out, image_grid_thw, video_grid_thw);
        }

        int max_pos = -1;
        for (const auto& row : state_out.position_ids) {
            for (int v : row) max_pos = std::max(max_pos, v);
        }
        if (max_pos >= 0) state_out.decode_start = max_pos + 1;
    }

    return true;
}

} // namespace vision
