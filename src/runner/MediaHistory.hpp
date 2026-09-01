#pragma once

#include <cstddef>
#include <vector>

#include "BaseTokenizer.hpp"

struct MediaInputs;

namespace axllm::media_history {

// VIDEO_HISTORY_FIX: distinguish current-turn media from retained session media.
// A media mapping is request media only when it points at the latest user
// turn. Historical mappings remain available so a text follow-up can reuse
// the earlier image/video context.
bool current_request_has_video(const std::vector<Content>     &history,
                               const std::vector<MediaInputs> &media_inputs);

// Keep only system messages and the current video turn when a new video is
// submitted after an existing conversation. Returns false without modifying
// either vector for a first-turn video or an invalid media mapping.
bool isolate_current_video(std::vector<Content> &history, std::vector<MediaInputs> &media_inputs);

}  // namespace axllm::media_history
