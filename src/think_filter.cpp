#include "think_filter.h"

#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// tag tables
// ---------------------------------------------------------------------------

namespace {

struct tag {
    const char * text;
    size_t len;
};

#define TAG(s) { s, sizeof(s) - 1 }

const tag kStartTags[] = {
    TAG("<thi" "nk>"),           // qwen3 / nemotron_h (literal text tokens)
    TAG("<|thinking|>"),
    TAG("<|im_start|>thinking"),
    TAG("<thinking>"),
    TAG("[Thinking]"),
};

const tag kEndTags[] = {
    TAG("</thi" "nk>"),
    TAG("<|/thinking|>"),
    TAG("</thinking>"),
    TAG("[/Thinking]"),
};

#undef TAG

const size_t kStartTagsN = sizeof(kStartTags) / sizeof(kStartTags[0]);
const size_t kEndTagsN   = sizeof(kEndTags)   / sizeof(kEndTags[0]);

// earliest occurrence of any tag in `buf`, or npos
size_t find_tag(const tag * tags, size_t n_tags, const std::string & buf) {
    size_t best = std::string::npos;
    for (size_t i = 0; i < n_tags; i++) {
        size_t p = buf.find(tags[i].text);
        if (p != std::string::npos && (best == std::string::npos || p < best)) best = p;
    }
    return best;
}

// length of the longest tag matching `buf` at `at` (0 if none -- guarded by the caller having
// just found one, but kept defensive). Longest wins so that, e.g., "<|im_start|>thinking" is
// consumed whole rather than an inner "<thinking>"-style prefix.
size_t match_tag_len(const tag * tags, size_t n_tags, const std::string & buf, size_t at) {
    size_t best = 0;
    for (size_t i = 0; i < n_tags; i++) {
        if (tags[i].len > best && buf.compare(at, tags[i].len, tags[i].text) == 0) {
            best = tags[i].len;
        }
    }
    return best;
}

// Length of the longest suffix of `buf` that is a *proper prefix* of some tag. Text before this
// suffix can be flushed safely; the suffix itself must be withheld until more pieces arrive (it
// may complete a tag on a later token boundary).
size_t held_suffix_len(const tag * tags, size_t n_tags, const std::string & buf) {
    size_t held = 0;
    for (size_t i = 0; i < n_tags; i++) {
        size_t hi = tags[i].len - 1;             // proper prefix: at most len-1 chars
        if (buf.size() < hi) hi = buf.size();
        for (size_t l = hi; l > held; l--) {
            if (buf.compare(buf.size() - l, l, tags[i].text, l) == 0) {
                held = l;
                break;
            }
        }
    }
    return held;
}

// earliest occurrence at or after `from` of any tag in `buf`, with the matched tag's length
// written to `len` (longest tag wins at the same position, mirroring match_tag_len above)
size_t find_tag_from(const tag * tags, size_t n_tags, const std::string & buf, size_t from,
                     size_t & len) {
    size_t best = std::string::npos;
    len = 0;
    for (size_t i = 0; i < n_tags; i++) {
        size_t p = buf.find(tags[i].text, from);
        if (p != std::string::npos &&
            (best == std::string::npos || p < best || (p == best && tags[i].len > len))) {
            best = p;
            len = tags[i].len;
        }
    }
    return best;
}

// prints `text` (or hides it, per mode), opening/closing the dim sequence as needed
void emit(thinking_filter & tf, const char * data, size_t n, bool thinking) {
    if (n == 0) return;
    if (thinking && tf.mode == thinking_filter::MODE_HIDDEN) {
        tf.hidden_chars += (long long) n;
        return;
    }
    if (thinking) {
        if (!tf.dim_open) {
            fputs("\033[90m", stdout);
            tf.dim_open = true;
        }
    } else if (tf.dim_open) {
        fputs("\033[0m", stdout);
        tf.dim_open = false;
    }
    fwrite(data, 1, n, stdout);
}

// runs the state machine over tf.pending, printing everything that is currently printable and
// leaving only the (possibly partial) trailing tag candidate behind
void filter_process(thinking_filter & tf) {
    while (true) {
        if (!tf.is_thinking) {
            size_t at = find_tag(kStartTags, kStartTagsN, tf.pending);
            if (at == std::string::npos) {
                size_t held = held_suffix_len(kStartTags, kStartTagsN, tf.pending);
                emit(tf, tf.pending.data(), tf.pending.size() - held, /*thinking=*/false);
                tf.pending.erase(0, tf.pending.size() - held);
                return;
            }
            emit(tf, tf.pending.data(), at, /*thinking=*/false);
            size_t len = match_tag_len(kStartTags, kStartTagsN, tf.pending, at);
            tf.pending.erase(0, at + (len > 0 ? len : 1));
            tf.is_thinking = true;
        } else {
            size_t at = find_tag(kEndTags, kEndTagsN, tf.pending);
            if (at == std::string::npos) {
                size_t held = held_suffix_len(kEndTags, kEndTagsN, tf.pending);
                emit(tf, tf.pending.data(), tf.pending.size() - held, /*thinking=*/true);
                tf.pending.erase(0, tf.pending.size() - held);
                return;
            }
            emit(tf, tf.pending.data(), at, /*thinking=*/true);
            size_t len = match_tag_len(kEndTags, kEndTagsN, tf.pending, at);
            tf.pending.erase(0, at + (len > 0 ? len : 1));
            tf.is_thinking = false;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

void render_filter_reset(thinking_filter & tf) {
    tf.is_thinking = false;
    tf.hidden_chars = 0;
    tf.pending.clear();
    // finish() always closes the dim sequence, so this should already be false -- reset anyway so
    // a missed finish() can never leak colour into the next turn
    if (tf.dim_open) {
        fputs("\033[0m", stdout);
        tf.dim_open = false;
    }
}

void render_filter_begin_thinking(thinking_filter & tf) {
    // no state to synthesise beyond flipping the flag: the opening tag itself was consumed as
    // *prompt* tokens (never rendered), so the generated stream starts mid-block with nothing
    // pending
    tf.is_thinking = true;
    tf.hidden_chars = 0;
    tf.pending.clear();
}

void render_filter_thinking(thinking_filter & tf, const std::string & piece) {
    tf.pending += piece;
    filter_process(tf);
    fflush(stdout); // match the caller's previous per-piece streaming behaviour
}

void render_filter_finish(thinking_filter & tf) {
    if (!tf.pending.empty()) {
        // whatever is left is real output that never completed a tag -- print it as normal text
        // (or as the tail of a thinking block, if generation ended mid-think)
        emit(tf, tf.pending.data(), tf.pending.size(), tf.is_thinking);
        tf.pending.clear();
    }
    if (tf.dim_open) {
        fputs("\033[0m", stdout);
        tf.dim_open = false;
    }
    fflush(stdout);
}

std::string render_filter_strip_thinking(const std::string & s, bool starts_inside) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    bool thinking = starts_inside;
    while (i < s.size()) {
        size_t len = 0;
        if (thinking) {
            size_t at = find_tag_from(kEndTags, kEndTagsN, s, i, len);
            if (at == std::string::npos) break; // unterminated trace: nothing after it to keep
            i = at + len;
            thinking = false;
        } else {
            size_t at = find_tag_from(kStartTags, kStartTagsN, s, i, len);
            if (at == std::string::npos) {
                out.append(s, i, std::string::npos); // rest of the text is answer text
                break;
            }
            out.append(s, i, at - i); // answer text before the next think block
            i = at + len;
            thinking = true;
        }
    }
    return out;
}
