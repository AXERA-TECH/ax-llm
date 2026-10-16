#include "media_fetch.hpp"

#include "utils/httplib.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>

namespace axllm::media {

namespace {

FetchOptions g_options;
std::mutex g_options_mutex;

constexpr int kMaxRedirects = 5;

std::string lower(std::string s)
{
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

bool starts_with_ci(const std::string& s, const char* prefix)
{
    const size_t n = std::char_traits<char>::length(prefix);
    return s.size() >= n && lower(s.substr(0, n)) == prefix;
}

std::string ext_from_content_type_impl(const std::string& content_type)
{
    std::string ct = lower(content_type);
    const size_t semi = ct.find(';');
    if (semi != std::string::npos) ct = ct.substr(0, semi);
    while (!ct.empty() && std::isspace((unsigned char)ct.back())) ct.pop_back();
    static const std::pair<const char*, const char*> kMap[] = {
        {"image/jpeg", "jpg"},      {"image/jpg", "jpg"},        {"image/png", "png"},
        {"image/webp", "webp"},     {"image/bmp", "bmp"},        {"image/gif", "gif"},
        {"video/mp4", "mp4"},       {"video/quicktime", "mov"},  {"video/webm", "webm"},
        {"video/x-matroska", "mkv"}, {"video/x-msvideo", "avi"}, {"video/mpeg", "mpg"},
        {"audio/wav", "wav"},       {"audio/x-wav", "wav"},      {"audio/wave", "wav"},
        {"audio/mpeg", "mp3"},      {"audio/mp3", "mp3"},        {"audio/flac", "flac"},
        {"audio/x-flac", "flac"},   {"audio/ogg", "ogg"},        {"audio/mp4", "m4a"},
        {"audio/aac", "aac"},       {"audio/webm", "webm"},
    };
    for (const auto& kv : kMap)
        if (ct == kv.first) return kv.second;
    return {};
}

std::string ext_from_url_path(const std::string& url)
{
    std::string path = url;
    const size_t cut = path.find_first_of("?#");
    if (cut != std::string::npos) path = path.substr(0, cut);
    const size_t slash = path.rfind('/');
    const size_t dot = path.rfind('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return {};
    std::string ext = lower(path.substr(dot + 1));
    if (ext.empty() || ext.size() > 5) return {};
    for (char c : ext)
        if (!std::isalnum((unsigned char)c)) return {};
    return ext;
}

// Media URLs are handed to the curl command line, so refuse anything that is not
// a plain printable URL (clients must percent-encode spaces and control characters).
bool url_is_safe(const std::string& url, std::string& err)
{
    for (unsigned char c : url) {
        if (c <= 0x20 || c == 0x7f || c == '"' || c == '\'' || c == '\\' || c == '`') {
            err = "media URL contains characters that must be percent-encoded";
            return false;
        }
    }
    return true;
}

std::string shell_quote(const std::string& s)
{
#ifdef _WIN32
    return "\"" + s + "\"";
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out += "'";
    return out;
#endif
}

bool curl_available()
{
    static const bool available = [] {
#ifdef _WIN32
        return std::system("curl --version >NUL 2>&1") == 0;
#else
        return std::system("curl --version >/dev/null 2>&1") == 0;
#endif
    }();
    return available;
}

std::filesystem::path make_temp_path(const char* suffix)
{
    std::filesystem::path dir;
    try {
        dir = std::filesystem::temp_directory_path() / "axllm_media";
    } catch (...) {
        dir = std::filesystem::current_path() / "tmp" / "axllm_media";
    }
    std::filesystem::create_directories(dir);
    static std::atomic<uint64_t> seq{0};
    const auto now = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    return dir / ("fetch_" + std::to_string(now) + "_" + std::to_string(seq.fetch_add(1)) + suffix);
}

std::string read_small_file(const std::filesystem::path& p)
{
    std::ifstream ifs(p, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

bool fetch_with_curl(const std::string& url, const FetchOptions& opt,
                     std::vector<uint8_t>& out, std::string& content_type, std::string& err)
{
    if (!curl_available()) {
        err = "https media URLs need the `curl` command on the server, which was not found in PATH";
        return false;
    }
    const auto body = make_temp_path(".body");
    const auto meta = make_temp_path(".meta");
    const auto errf = make_temp_path(".err");
    struct Cleanup {
        std::vector<std::filesystem::path> files;
        ~Cleanup() { for (auto& f : files) { std::error_code ec; std::filesystem::remove(f, ec); } }
    } cleanup{{body, meta, errf}};

    const std::string cmd =
        "curl -sS -L --fail --proto =http,https --max-redirs " + std::to_string(kMaxRedirects) +
        " --max-time " + std::to_string(std::max(1, opt.timeout_s)) +
        " --max-filesize " + std::to_string(opt.max_bytes) +
        " -o " + shell_quote(body.string()) + " -w " + shell_quote("%{content_type}") + " " +
        shell_quote(url) + " >" + shell_quote(meta.string()) + " 2>" + shell_quote(errf.string());
    if (std::system(cmd.c_str()) != 0) {
        std::string detail = read_small_file(errf);
        err = "failed to download media URL" + (detail.empty() ? std::string() : (": " + detail));
        return false;
    }

    std::error_code ec;
    const auto size = std::filesystem::file_size(body, ec);
    if (ec) { err = "failed to download media URL: no data received"; return false; }
    if (size > opt.max_bytes) {
        err = "media URL exceeds the size limit (" + std::to_string(opt.max_bytes) + " bytes)";
        return false;
    }
    std::ifstream ifs(body, std::ios::binary);
    out.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    content_type = read_small_file(meta);
    return true;
}

// Plain-http fetch with the bundled client; redirects are followed by hand so an
// http -> https hop can be handed to curl.
bool fetch_http(const std::string& url, const FetchOptions& opt, int redirects_left,
                std::vector<uint8_t>& out, std::string& content_type, std::string& final_url, std::string& err)
{
    const size_t host_start = std::char_traits<char>::length("http://");
    const size_t path_start = url.find('/', host_start);
    const std::string scheme_host_port = url.substr(0, path_start);
    const std::string path = path_start == std::string::npos ? "/" : url.substr(path_start);

    httplib::Client cli(scheme_host_port);
    if (!cli.is_valid()) { err = "invalid media URL: " + url; return false; }
    cli.set_follow_location(false);
    cli.set_connection_timeout(std::max(1, opt.timeout_s), 0);
    cli.set_read_timeout(std::max(1, opt.timeout_s), 0);

    out.clear();
    int status = 0;
    std::string location;
    bool too_large = false;
    auto result = cli.Get(
        path,
        [&](const httplib::Response& res) {
            status = res.status;
            content_type = res.get_header_value("Content-Type");
            location = res.get_header_value("Location");
            if (status >= 200 && status < 300 && res.has_header("Content-Length")) {
                const unsigned long long len = std::strtoull(res.get_header_value("Content-Length").c_str(), nullptr, 10);
                if (len > opt.max_bytes) { too_large = true; return false; }
                out.reserve((size_t)len);
            }
            return true;
        },
        [&](const char* data, size_t n) {
            if (status < 200 || status >= 300) return true; // drain error/redirect bodies
            if (out.size() + n > opt.max_bytes) { too_large = true; return false; }
            out.insert(out.end(), data, data + n);
            return true;
        });

    if (too_large) {
        err = "media URL exceeds the size limit (" + std::to_string(opt.max_bytes) + " bytes)";
        return false;
    }
    if (!result) {
        err = "failed to download media URL: " + httplib::to_string(result.error());
        return false;
    }
    if (status >= 300 && status < 400 && !location.empty()) {
        if (redirects_left <= 0) { err = "failed to download media URL: too many redirects"; return false; }
        std::string next = location;
        if (next.rfind("/", 0) == 0) next = scheme_host_port + next;
        if (!url_is_safe(next, err)) return false;
        final_url = next;
        if (starts_with_ci(next, "https://")) return fetch_with_curl(next, opt, out, content_type, err);
        if (!starts_with_ci(next, "http://")) { err = "media URL redirected to an unsupported scheme"; return false; }
        return fetch_http(next, opt, redirects_left - 1, out, content_type, final_url, err);
    }
    if (status < 200 || status >= 300) {
        err = "failed to download media URL: HTTP " + std::to_string(status);
        return false;
    }
    return true;
}

} // namespace

void set_fetch_options(const FetchOptions& opts)
{
    std::lock_guard<std::mutex> lock(g_options_mutex);
    g_options = opts;
}

std::string ext_for_mime(const std::string& mime)
{
    return ext_from_content_type_impl(mime);
}

bool is_remote_url(const std::string& uri)
{
    return starts_with_ci(uri, "http://") || starts_with_ci(uri, "https://");
}

bool fetch_url(const std::string& url, std::vector<uint8_t>& out_bytes, std::string& out_ext, std::string& err)
{
    FetchOptions opt;
    {
        std::lock_guard<std::mutex> lock(g_options_mutex);
        opt = g_options;
    }
    out_bytes.clear();
    out_ext.clear();
    if (!opt.allow_remote) {
        err = "remote media URLs are disabled on this server (media_allow_remote=false)";
        return false;
    }
    if (!url_is_safe(url, err)) return false;

    std::string content_type;
    std::string final_url = url;
    const bool ok = starts_with_ci(url, "https://")
                        ? fetch_with_curl(url, opt, out_bytes, content_type, err)
                        : fetch_http(url, opt, kMaxRedirects, out_bytes, content_type, final_url, err);
    if (!ok) return false;
    if (out_bytes.empty()) {
        err = "failed to download media URL: empty response";
        return false;
    }
    out_ext = ext_for_mime(content_type);
    if (out_ext.empty()) out_ext = ext_from_url_path(final_url);
    return true;
}

} // namespace axllm::media
