#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace axllm::media {

// Limits applied when the server downloads media referenced by an http(s) URL
// (OpenAI image_url / vLLM video_url / audio_url).
struct FetchOptions {
    bool allow_remote = true;                 // false: reject http(s) media URLs
    int timeout_s = 60;                       // whole-transfer timeout
    size_t max_bytes = 256u * 1024u * 1024u;  // per-file size cap
};

void set_fetch_options(const FetchOptions& opts);

bool is_remote_url(const std::string& uri);

// File extension for a MIME type such as "image/jpeg" or "audio/x-wav" ("" when unknown).
std::string ext_for_mime(const std::string& mime);

// Downloads `url` into memory. `out_ext` is a file extension guessed from the
// Content-Type header or the URL path ("" when unknown). http:// is fetched
// in-process; https:// (and http redirects to https) go through the `curl`
// command, since the bundled HTTP client is built without TLS.
bool fetch_url(const std::string& url,
               std::vector<uint8_t>& out_bytes,
               std::string& out_ext,
               std::string& err);

} // namespace axllm::media
