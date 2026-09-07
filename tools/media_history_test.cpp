#include <cassert>

#include "LLM.hpp"
#include "MediaHistory.hpp"

namespace {
Content system_message() { return {SYSTEM, TEXT, "system"}; }
Content user_text(const char *text) { return {USER, TEXT, text}; }
Content user_video(const char *text) { return {USER, VIDEO, text}; }
Content assistant_text(const char *text) { return {ASSISTANT, TEXT, text}; }
}

int main() {
    std::vector<Content> history = {system_message(), user_video("video")};
    std::vector<MediaInputs> media = {{1, {"football.mp4"}}};
    assert(axllm::media_history::current_request_has_video(history, media));
    assert(!axllm::media_history::isolate_current_video(history, media));

    history.push_back(assistant_text("answer"));
    history.push_back(user_text("which player scored?"));
    assert(!axllm::media_history::current_request_has_video(history, media));
    assert(!axllm::media_history::isolate_current_video(history, media));

    history.push_back(assistant_text("follow-up answer"));
    history.push_back(user_video("new video"));
    media.push_back({5, {"new.mp4"}});
    assert(axllm::media_history::isolate_current_video(history, media));
    assert(history.size() == 2 && history[1].type == VIDEO);
    assert(media.size() == 1 && media[0].content_index == 1 && media[0].uris[0] == "new.mp4");

    history.push_back(assistant_text("new answer"));
    history.push_back(user_text("what happened in it?"));
    assert(!axllm::media_history::current_request_has_video(history, media));
    std::vector<Content> text_only = {system_message(), user_text("text")};
    std::vector<MediaInputs> stale = {{0, {"old.mp4"}}};
    assert(!axllm::media_history::current_request_has_video(text_only, stale));
    return 0;
}
