#pragma once

#include <cstddef>
#include <string>

// Streaming filter that strips a model's hidden "channel" thought sections
// ("<|channel>...<channel|>") from generated text, chunk by chunk. Extracted
// verbatim from LLM.cpp -- self-contained, depends only on std::string.
class ChannelSectionFilter
{
public:
    void reset()
    {
        pending_.clear();
        skipping_ = false;
    }

    std::string filter(const std::string &chunk)
    {
        // Gemma4 hidden-thought section is "<|channel>thought\n...<channel|>" per tokenizer x-regex
        // and chat_template.jinja::strip_thinking. <|think|> is a prompt-only marker (emitted when
        // enable_thinking=true in the template); the model does not generate it. Matching it here
        // as a section-opener would make any model output containing the 9-byte literal "<|think|>"
        // be silently swallowed because no paired "<channel|>" would follow.
        static const std::string kChannelStart = "<|channel>";
        static const std::string kChannelEnd = "<channel|>";
        static const size_t kStartGuard = kChannelStart.size() - 1;
        static const size_t kEndGuard = kChannelEnd.size() - 1;

        pending_ += chunk;
        std::string out;

        while (true)
        {
            if (!skipping_)
            {
                size_t start = pending_.find(kChannelStart);

                if (start == std::string::npos)
                {
                    if (pending_.size() > kStartGuard)
                    {
                        const size_t emit_len = pending_.size() - kStartGuard;
                        out.append(pending_, 0, emit_len);
                        pending_.erase(0, emit_len);
                    }
                    break;
                }

                out.append(pending_, 0, start);
                pending_.erase(0, start);
                skipping_ = true;
            }

            const size_t end = pending_.find(kChannelEnd);
            if (end == std::string::npos)
            {
                if (pending_.size() > kEndGuard)
                    pending_.erase(0, pending_.size() - kEndGuard);
                break;
            }

            pending_.erase(0, end + kChannelEnd.size());
            skipping_ = false;
        }

        return out;
    }

    std::string flush()
    {
        if (skipping_)
        {
            pending_.clear();
            return {};
        }

        std::string out;
        out.swap(pending_);
        return out;
    }

private:
    std::string pending_;
    bool skipping_ = false;
};
