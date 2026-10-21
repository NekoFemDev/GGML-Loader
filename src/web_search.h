// Shared web-search tool support for the chat and run modes (--search / --max-searches).
//
// Tool contract: the prompt tells the model to emit "[SEARCH: query]" when it needs live
// information and to wait for the results. The decode loop scans the generated stream for that
// marker (token-boundary safe -- the marker can be split across pieces), pauses generation, runs
// perform_search() (curl | python3 through popen -- both present on Termux), feeds a
// "[SEARCH RESULTS for ...]: ..." block (or "[SEARCH FAILED ...]" when the search produced
// nothing) back into the live context and resumes. Each mode hard-caps the rounds per user
// prompt via --max-searches (default SEARCH_DEFAULT_MAX_ROUNDS), so a runaway model cannot loop.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <cstdint>
#include <string>

struct vocab; // model.h -- only needed by search_fit_to_tokens()

inline constexpr char   SEARCH_OPEN[]             = "[SEARCH: ";
inline constexpr size_t SEARCH_OPEN_LEN           = sizeof(SEARCH_OPEN) - 1;
inline constexpr size_t SEARCH_QUERY_MAX          = 256;  // query length cap (bytes)
inline constexpr size_t SEARCH_RESULT_MAX         = 2000; // results fed back to the model (KV-cache hygiene)
inline constexpr int    SEARCH_DEFAULT_MAX_ROUNDS = 3;    // searches per user prompt unless --max-searches

// The tool contract, appended to the system prompt (chat) / prompt (run) when --search is on.
inline constexpr char SEARCH_TOOL_NOTE[] =
    "If you need real-time information, output [SEARCH: your query]. "
    "Do not answer until you receive the search results.";

enum search_scan_t {
    SEARCH_SCAN_NONE,    // no marker (and no possible marker prefix): all text is ordinary output
    SEARCH_SCAN_PENDING, // text ends in a possible marker prefix -> hold it until it resolves
    SEARCH_SCAN_MARKER,  // a complete "[SEARCH: query]" is present; `query` is filled
};

// Scans `text` for a "[SEARCH: query]" marker. MARKER comes back with `query` and `marker_start`
// (offset of the '['); PENDING while `text` merely ends in a possible marker prefix (the caller
// must withhold that tail from the console until the next piece resolves it); NONE otherwise.
// With at_end=true a marker missing its ']' counts too (the model ran into EOS mid-marker).
search_scan_t search_scan(const std::string & text, std::string & query,
                          size_t & marker_start, bool at_end);

// One search: curl (DuckDuckGo's html endpoint; curl URL-encodes the query itself via -G
// --data-urlencode, so spaces/&/# can't break the URL) piped through a python3 one-liner that
// strips tags, collapses whitespace and caps the result at SEARCH_RESULT_MAX chars. popen() with
// the query single-quoted for the shell (every byte safe -- spaces, quotes, backticks, $(...));
// output is read in bounded chunks into a capped buffer, and the pipe keeps being drained to EOF
// after the cap (a full pipe would block the child and deadlock pclose()); pclose() is always
// called. curl's own --connect-timeout/--max-time bound the whole call (~15s worst case), so the
// caller can stall at most that long -- no extra watchdog needed. Returns false when no usable
// text came back (the caller should inject search_failed_block() instead).
bool perform_search(const std::string & query, std::string & out);

// The block injected into the live context when a search returned text. Deliberately plain text
// with no template tokens fabricated mid-turn, so it works for every architecture's chat format.
std::string search_results_block(const std::string & query, const std::string & results);

// The block injected when the search failed (popen/pipe error, curl nonzero exit, or empty
// output): tells the model explicitly that no results arrived, so it can apologize or try a
// different query instead of silently hallucinating one.
std::string search_failed_block(const std::string & query);

// Fits `text` into at most `max_tokens` tokens (as measured by `vc`), cutting on a word boundary
// and appending " [...]" when something was dropped. Returns an empty string when even a minimal
// prefix doesn't fit. Used to keep injected search blocks from overflowing n_ctx.
std::string search_fit_to_tokens(const vocab & vc, const std::string & text, int64_t max_tokens);
