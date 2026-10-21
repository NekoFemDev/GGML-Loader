// Streaming console filter for reasoning-model output (chat): detects thinking-block tags in the
// decodable text stream and either hides or dims the reasoning trace, so the actual answer isn't
// buried under a wall of think tokens.
//
// Token-boundary safety: one "piece" fed to render_filter_thinking() is whatever the model just
// generated -- usually one token, but a tag can be split across tokens (observed with these
// models: "<|thin" + "king|>"), and byte-level BPE can also split a tag mid-character on
// multi-byte text. The filter therefore keeps a rolling `pending` tail: text that could still turn
// out to be the beginning of a tag is withheld until the next piece arrives, and is only printed
// once it provably isn't part of a tag. Nothing is ever emitted twice or dropped mid-way.
//
// Tags recognised (start -> set thinking, end -> clear it). The exact literals live in
// think_filter.cpp's kStartTags/kEndTags tables -- these are the families covered:
//   the qwen3/nemotron_h think open/close text tokens (bare think word, angle-wrapped)
//   "<|thinking|>"            / "<|/thinking|>"     -- tag-style templates
//   "<|im_start|>thinking"    / "</thinking>"       -- ChatML-ish templates
//   "<thinking>"              / "[/Thinking]"
//   "[Thinking]"
// The start/end tables are matched case-sensitively and checked for the *earliest* occurrence, so
// a tag found earlier in the buffer always wins over one found later.
#pragma once

#include <string>

struct thinking_filter {
    enum mode_t {
        MODE_HIDDEN, // thinking blocks are not printed at all (default -- the "declutter" mode)
        MODE_DIMMED, // thinking blocks are printed in ANSI dim gray (\033[90m...\033[0m)
    };

    mode_t mode = MODE_HIDDEN;
    bool is_thinking = false;

    // Number of thinking characters dropped while MODE_HIDDEN since the last reset -- callers can
    // use this to print a short "[N chars of thinking hidden]" note without touching stdout while
    // the model is mid-think.
    long long hidden_chars = 0;

    // internal state
    bool dim_open = false;   // a \033[90m is currently in effect on stdout
    std::string pending;     // withheld tail that might still become (part of) a tag
};

// Re-arms the filter for a new assistant turn (mode is preserved).
void render_filter_reset(thinking_filter & tf);

// Marks the start of a turn that is already inside an open think block because the model's own
// template put the opening tag into the prompt (see generation_prompt_opens_think in model.h):
// generated text is hidden/dimmed from the very first token until the closing tag arrives.
// Call after render_filter_reset().
void render_filter_begin_thinking(thinking_filter & tf);

// Feeds one decoded piece (see the file comment) to the filter. Prints the printable parts to
// stdout immediately (and flushes), honouring the current mode.
void render_filter_thinking(thinking_filter & tf, const std::string & piece);

// Call once generation of a turn has ended: emits anything still withheld (a trailing partial tag
// is real model output, not an artefact) and closes an open dim sequence, so the console isn't
// left in a coloured state.
void render_filter_finish(thinking_filter & tf);

// Non-streaming companion to render_filter_thinking(): returns `s` with every complete thinking
// block removed, using the same tag tables. `starts_inside` must mirror
// render_filter_begin_thinking() (the model's template put the opening tag into the prompt, so the
// text begins mid-think). An unterminated trailing think block is dropped; a trailing *partial*
// tag is left in place -- the caller re-runs this on the next token, where it resolves either way.
// Used by `run --edit-file` so a reasoning trace can never be extracted as the file rewrite, nor
// have the fenced drafts inside it trip the closing-fence early stop (see cmd_run.cpp).
std::string render_filter_strip_thinking(const std::string & s, bool starts_inside);
