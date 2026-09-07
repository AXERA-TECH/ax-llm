#pragma once

#include <cstddef>
#include <vector>

#include "BaseTokenizer.hpp"

struct MediaInputs;

namespace axllm::media_history {

// A video mapping is request media only when it points at the latest user turn.
// Historical mappings remain available so text follow-ups can reuse context.
bool current_request_has_video(const std::vector<Content> &history,
                               const std::vector<MediaInputs> &media_inputs);

// Keep only system messages and the current video turn when a new video is
// submitted after an existing conversation. Invalid/first-turn requests are
// left untouched and return false.
bool isolate_current_video(std::vector<Content> &history,
                           std::vector<MediaInputs> &media_inputs);

} // namespace axllm::media_history
