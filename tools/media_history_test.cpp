#include <cassert>

#include "LLM.hpp"
#include "MediaHistory.hpp"

namespace {

Content system_message() { return {SYSTEM, TEXT, "system"}; }

Content user_text(const char *text) { return {USER, TEXT, text}; }

Content user_video(const char *text) { return {USER, VIDEO, text}; }

Content assistant_text(const char *text) { return {ASSISTANT, TEXT, text}; }

}  // namespace

int main() {
    // A first-turn video is not isolated, so its KV can serve a follow-up.
    std::vector<Content>     first_history = {system_message(), user_video("video")};
    std::vector<MediaInputs> first_media   = {{1, {"football.mp4"}}};
    assert(axllm::media_history::current_request_has_video(first_history, first_media));
    assert(!axllm::media_history::isolate_current_video(first_history, first_media));
    assert(first_history.size() == 2);
    assert(first_media.size() == 1 && first_media[0].content_index == 1);

    // A text follow-up must not be classified as a new video request just
    // because the caller keeps historical media mappings in the session.
    first_history.push_back(assistant_text("answer"));
    first_history.push_back(user_text("which player scored?"));
    assert(!axllm::media_history::current_request_has_video(first_history, first_media));
    assert(!axllm::media_history::isolate_current_video(first_history, first_media));
    assert(first_history.size() == 4);
    assert(first_media.size() == 1 && first_media[0].content_index == 1);

    // A genuinely new video after prior dialogue still starts an isolated
    // video context and remaps the current media index.
    first_history.push_back(assistant_text("follow-up answer"));
    first_history.push_back(user_video("new video"));
    first_media.push_back({5, {"new.mp4"}});
    assert(axllm::media_history::current_request_has_video(first_history, first_media));
    assert(axllm::media_history::isolate_current_video(first_history, first_media));
    assert(first_history.size() == 2);
    assert(first_history[0].role == SYSTEM && first_history[1].type == VIDEO);
    assert(first_media.size() == 1 && first_media[0].content_index == 1);
    assert(first_media[0].uris[0] == "new.mp4");

    first_history.push_back(assistant_text("new answer"));
    first_history.push_back(user_text("what happened in it?"));
    assert(!axllm::media_history::current_request_has_video(first_history, first_media));
    assert(!axllm::media_history::isolate_current_video(first_history, first_media));
    assert(first_media.size() == 1 && first_media[0].content_index == 1);

    // Re-running a completed history is not a new video request.
    first_history.push_back(assistant_text("answer"));
    assert(!axllm::media_history::current_request_has_video(first_history, first_media));
    assert(!axllm::media_history::isolate_current_video(first_history, first_media));

    // A stale video mapping whose content is no longer present is ignored.
    std::vector<Content>     text_only   = {system_message(), user_text("text")};
    std::vector<MediaInputs> stale_media = {{0, {"old.mp4"}}};
    assert(!axllm::media_history::current_request_has_video(text_only, stale_media));
    assert(!axllm::media_history::isolate_current_video(text_only, stale_media));
    return 0;
}
