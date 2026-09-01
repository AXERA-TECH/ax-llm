#include "MediaHistory.hpp"

#include <string>
#include <utility>

#include "LLM.hpp"

namespace axllm::media_history {
namespace {

size_t latest_user_index(const std::vector<Content> &history) {
    for (size_t i = history.size(); i > 0; --i) {
        if (history[i - 1].role == USER) return i - 1;
    }
    return history.size();
}

const MediaInputs *find_video_media(const std::vector<Content>     &history,
                                    const std::vector<MediaInputs> &media_inputs,
                                    size_t                          user_index) {
    if (user_index >= history.size() || history[user_index].type != VIDEO) return nullptr;

    for (const auto &media : media_inputs) {
        if (media.content_index == user_index && !media.uris.empty()) return &media;
    }
    return nullptr;
}

bool has_prior_non_system_history(const std::vector<Content> &history, size_t user_index) {
    for (size_t i = 0; i < user_index; ++i) {
        if (history[i].role != SYSTEM) return true;
    }
    return false;
}

}  // namespace

bool current_request_has_video(const std::vector<Content>     &history,
                               const std::vector<MediaInputs> &media_inputs) {
    if (history.empty() || history.back().role != USER) return false;
    const size_t user_index = latest_user_index(history);
    return find_video_media(history, media_inputs, user_index) != nullptr;
}

bool isolate_current_video(std::vector<Content> &history, std::vector<MediaInputs> &media_inputs) {
    if (history.empty() || history.back().role != USER) return false;
    const size_t       user_index  = latest_user_index(history);
    const MediaInputs *video_media = find_video_media(history, media_inputs, user_index);
    if (video_media == nullptr || !has_prior_non_system_history(history, user_index)) return false;

    std::vector<Content> normalized;
    normalized.reserve(history.size());
    for (size_t i = 0; i < user_index; ++i) {
        if (history[i].role == SYSTEM) normalized.push_back(history[i]);
    }

    const size_t                   normalized_user_index = normalized.size();
    const std::vector<std::string> current_video_uris    = video_media->uris;
    normalized.push_back(history[user_index]);
    media_inputs = {{normalized_user_index, current_video_uris}};
    history      = std::move(normalized);
    return true;
}

}  // namespace axllm::media_history
