// See web_search.h for the tool contract. This file hosts the marker scanner, the popen()-based
// search runner and the context-injection block builders shared by cmd_chat.cpp and cmd_run.cpp.
#include "web_search.h"

#include "model.h" // vocab (search_fit_to_tokens)

#include <cerrno>
#include <cstdio>
#include <cstring>

search_scan_t search_scan(const std::string & text, std::string & query,
                          size_t & marker_start, bool at_end) {
    query.clear();
    marker_start = 0;

    const size_t p = text.find(SEARCH_OPEN);
    if (p != std::string::npos) {
        const size_t q = p + SEARCH_OPEN_LEN;
        size_t end = text.find(']', q);
        if (end == std::string::npos && at_end && q < text.size()) end = text.size();
        if (end != std::string::npos) {
            query = text.substr(q, std::min(end - q, SEARCH_QUERY_MAX - 1));
            for (char & c : query) { // control bytes (multi-line output) out of the query
                if ((unsigned char) c < 0x20) c = ' ';
            }
            const size_t b = query.find_first_not_of(' ');
            const size_t e = query.find_last_not_of(' ');
            query = (b == std::string::npos) ? std::string() : query.substr(b, e - b + 1);
            if (query.empty()) return SEARCH_SCAN_NONE; // "[SEARCH: ]" is ordinary text, not a call
            marker_start = p;
            return SEARCH_SCAN_MARKER;
        }
        if (text.size() - q >= SEARCH_QUERY_MAX) return SEARCH_SCAN_NONE; // runaway: not a call
        marker_start = p;
        return SEARCH_SCAN_PENDING; // marker opened, no ']' yet
    }

    if (at_end) return SEARCH_SCAN_NONE;
    // the marker can be split across pieces: hold the longest suffix that is a prefix of it
    const size_t max_k = std::min(SEARCH_OPEN_LEN - 1, text.size());
    for (size_t k = max_k; k > 0; k--) {
        if (strncmp(text.data() + text.size() - k, SEARCH_OPEN, k) == 0) {
            marker_start = text.size() - k;
            return SEARCH_SCAN_PENDING;
        }
    }
    return SEARCH_SCAN_NONE;
}

bool perform_search(const std::string & query, std::string & out) {
    out.clear();

    std::string quoted = "'";
    for (char c : query) {
        if (c == '\'') quoted += "'\\''"; // close quote, escaped literal quote, reopen
        else           quoted += c;
    }
    quoted += "'";

    const std::string py =
        "python3 -c \"import sys, re; html = sys.stdin.read(); "
        "text = re.sub('<[^>]+>', ' ', html); "
        "print(' '.join(text.split())[:" + std::to_string(SEARCH_RESULT_MAX) + "])\"";

    const std::string cmd =
        "curl -sS -L --connect-timeout 5 --max-time 15 -A 'Mozilla/5.0 (Termux; ggml-loader)' "
        "-G 'https://html.duckduckgo.com/html/' --data-urlencode q=" + quoted + " | " + py;

    FILE * fp = popen(cmd.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "[search: popen failed: %s]\n", strerror(errno));
        return false;
    }

    std::string raw;
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, fp)) > 0) {
        if (raw.size() < SEARCH_RESULT_MAX * 2) { // keep reading after the cap: drain, don't stop
            raw.append(buf, std::min(n, SEARCH_RESULT_MAX * 2 - raw.size()));
        }
    }
    const int status = pclose(fp); // always called, on every path past a successful popen()
    if (status == -1) {
        fprintf(stderr, "[search: pclose failed: %s]\n", strerror(errno));
        return false;
    }
    if (status != 0) fprintf(stderr, "[search: command exited with status %d]\n", status);

    const size_t b = raw.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return false; // no output (or only whitespace)
    const size_t e = raw.find_last_not_of(" \t\r\n");
    raw = raw.substr(b, e - b + 1);
    if (raw.size() > SEARCH_RESULT_MAX) { // cut on a word boundary so the tail stays readable
        raw.resize(SEARCH_RESULT_MAX);
        const size_t cut = raw.find_last_of(" \t\n");
        if (cut != std::string::npos && cut > SEARCH_RESULT_MAX / 2) raw.resize(cut);
        raw += " [...]";
    }
    out = std::move(raw);
    return true;
}

std::string search_results_block(const std::string & query, const std::string & results) {
    return "[SEARCH RESULTS for \"" + query + "\"]:\n" + results + "\n";
}

std::string search_failed_block(const std::string & query) {
    return "[SEARCH FAILED for \"" + query + "\"]: no results were returned (network error, empty "
           "response, or timeout). Answer without them or try a different query.\n";
}

std::string search_fit_to_tokens(const vocab & vc, const std::string & text, int64_t max_tokens) {
    if (max_tokens <= 0 || text.empty()) return std::string();
    if ((int64_t) vc.encode(text).size() <= max_tokens) return text;

    // Binary search the longest prefix that still fits. Tokenization isn't strictly monotonic in
    // prefix length (a longer prefix can merge tokens), so this is a heuristic -- the final trim
    // below is what makes the "fits" guarantee hold for the returned string.
    size_t lo = 0, hi = text.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo + 1) / 2;
        if ((int64_t) vc.encode(text.substr(0, mid)).size() <= max_tokens) lo = mid;
        else hi = mid - 1;
    }

    // the byte at [lo) is the first dropped one; if it is a UTF-8 continuation byte, the prefix
    // would end inside a sequence -- step back to that sequence's lead byte and drop it whole
    while (lo > 0 && lo < text.size() && ((unsigned char) text[lo] & 0xC0) == 0x80) lo--;

    std::string out = text.substr(0, lo);
    if (out.empty()) return out;
    const size_t cut = out.find_last_of(" \t\n"); // cut on a word boundary so the tail stays readable
    if (cut != std::string::npos && cut > out.size() / 2) out.resize(cut);

    // keep the truncation marker intact: drop whole trailing code points (starting at a code-point
    // boundary, so no partial UTF-8 sequence can survive) until base + " [...]" fits
    static const char ellipsis[] = " [...]";
    while (!out.empty() && (int64_t) vc.encode(out + ellipsis).size() > max_tokens) {
        while (!out.empty() && ((unsigned char) out.back() & 0xC0) == 0x80) out.pop_back();
        if (!out.empty()) out.pop_back();
    }
    if (out.empty()) return out;
    return out + ellipsis;
}
