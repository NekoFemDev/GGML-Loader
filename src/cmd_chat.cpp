// `ggml-loader chat <model.gguf> [--system "text"] [-n/--n-predict N]`: interactive multi-turn chat.
// Chat formatting (append_chat_turn/append_generation_prompt/turn_end_token) is
// architecture-specific -- see model.cpp -- so this file doesn't hardcode any model family's
// template. Multi-turn history lives entirely in the persistent KV cache -- each turn only
// encodes/prefills the newest user message, not the whole conversation.
// Each reply's budget defaults to the room left in the context (n_ctx - prompt - 64, a 64-token
// EOS margin) unless -n is given; an explicit -n that no longer fits is clamped to the room with
// a warning, not an error. Decoding stops at EOS / the model's end-of-turn token (<|im_end|>,
// </s>, [eos], ...).
// Interactive input: a multi-line block is sent as ONE message -- via bracketed paste, the
// rapid-line burst of a plain paste (tty only), or an explicit '/paste' .. '/end' typed by
// hand (see read_user_message below).
// Local slash commands (intercepted in the REPL, never sent to the model; '/help' lists them):
//   /load-file <path>  attach a file's contents to the next message
//   /save [file]       write the conversation (roles + raw text) to JSON (default chat_history.json)
//   /load <file>       replace the conversation with a saved one: the KV cache is rewound and the
//                      loaded history re-prefilled from scratch
// Anything else that starts with '/' (and every multi-line block) is a normal message.
// --search adds a web-search tool call: the system prompt instructs the model to emit
// "[SEARCH: query]" when it needs live information and to wait for the results; the generation
// loop pauses on that marker, runs the search through popen (curl | python3 -- DuckDuckGo's html
// endpoint, HTML stripped, ~2000 chars kept), feeds a "[SEARCH RESULTS for ...]: ..." block (or a
// "[SEARCH FAILED ...]" note when nothing came back) into the live context and resumes. Up to
// --max-searches rounds per reply (default 3). Off by default.
#include "commands.h"
#include "model.h"
#include "ollama_model.h"
#include "think_filter.h"
#include "web_search.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [model.gguf] [--system \"text\"] [-n/--n-predict N] [--backend cpu|vulkan] [--flash-attn] [--search]\n"
        "           [--temp F] [--top-k N] [--top-p F] [--repeat-penalty F] [--repeat-last-n N] [--seed N]\n"
        "           [--thinking | --no-thinking] [--show-thinking | --hide-thinking] [--ctx N]\n"
        "  -n/--n-predict N: max tokens per reply, counting thinking tokens (optional;\n"
        "  default: n_ctx - prompt - 64, i.e. the room left in the KV cache minus a 64-token\n"
        "  EOS margin). An explicit N that exceeds the room is clamped to it with a warning\n"
        "  local commands (handled by the CLI, never sent to the model -- '/help' in the REPL):\n"
        "    /load-file <path>  attach a file's contents to the next message\n"
        "    /save [file]       write the conversation to JSON (default: chat_history.json)\n"
        "    /load <file>       replace the conversation with a saved one and rebuild the context\n"
        "  multi-line input: a pasted block is sent as one message (bracketed paste, with a\n"
        "  rapid-lines fallback); '/paste' + '/end' types one by hand\n"
        "  --search: web-search tool call. The system prompt tells the model to emit\n"
        "  '[SEARCH: <query>]' when it needs current information; the CLI then pauses generation,\n"
        "  runs the search (curl | python3, ~15s cap) and feeds a '[SEARCH RESULTS for <query>]'\n"
        "  block back into the live context (default 3 searches per reply -- see --max-searches;\n"
        "  each result truncated to ~2000 chars, further if n_ctx demands it)\n"
        "  --max-searches N: hard cap on searches per reply (default 3; implies --search)\n"
        "  temp<=0 (default): greedy/deterministic, same as before these flags existed\n"
        "  --show-thinking/--hide-thinking: a reasoning model's think blocks are shown in dim gray\n"
        "  by default; --hide-thinking collapses them to a short '[thinking hidden]' note instead\n"
        "  (tags are matched across token boundaries, so a split tag is never leaked either way)\n"
        "  --thinking/--no-thinking only matter for reasoning models (auto-detected from the\n"
        "  model's own chat_template); default is --thinking (let the model decide whether to\n"
        "  reason). --no-thinking forces an empty  thinking block to suppress it.\n"
        "  --ctx N: KV cache context size. Default: auto-sized at load time from the RAM free\n"
        "  after the weights are loaded, clamped to the model's trained context length\n", argv0);
}

// Returns true when stdin already has more complete lines buffered (or they arrive within
// `ms`). Used to recognize the rapid line burst a multi-line paste turns into on terminals
// that don't (reliably) deliver bracketed-paste markers: in canonical mode each read hands
// out a whole line, a human cannot produce the next line within the grace window, a paste
// always can. Only consulted for tty stdin -- pipes deliver everything back-to-back.
static bool input_ready_within(int ms) {
    struct pollfd pfd;
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;
    pfd.revents = 0;
    return poll(&pfd, 1, ms) > 0 && (pfd.revents & POLLIN) != 0;
}

// Reads one user message, possibly spanning multiple lines. Three ways in:
//   1. bracketed paste: a real terminal wraps the paste in \x1b[200~ .. \x1b[201~ once the
//      app enables DECSET 2004 (cmd_chat does, on tty stdin); the whole block is reassembled
//      here into ONE message.
//   2. burst fallback: terminals that don't send the markers still deliver the whole paste in
//      one write, so consecutive lines arriving back-to-back are joined into one message.
//   3. "/paste" .. "/end": explicit multi-line accumulation, for typing a block by hand.
// Without this, a pasted code block becomes one user message per line and each line gets its
// own full model reply ("no code provided", "the block is empty", ...).
// Returns false when the session should end (EOF at the prompt, or exit/quit).
static bool read_user_message(std::string & msg) {
    const std::string paste_open  = "\x1b[200~";
    const std::string paste_close = "\x1b[201~";
    msg.clear();

    std::string line;
    while (true) {
        printf("> ");
        fflush(stdout);
        if (!std::getline(std::cin, line)) return false; // EOF (Ctrl-D)
        if (line == "exit" || line == "quit") return false;
        if (line.empty()) continue;
        break;
    }

    // bracketed paste: keep consuming lines until the one carrying the closing marker arrives
    size_t open_pos = line.find(paste_open);
    if (open_pos != std::string::npos) {
        msg = line.substr(0, open_pos);
        std::string rest = line.substr(open_pos + paste_open.size());
        size_t close_pos = rest.find(paste_close);
        if (close_pos != std::string::npos) {
            // single-line paste: opener and closer on the same line (text typed around it stays)
            msg += rest.substr(0, close_pos);
            msg += rest.substr(close_pos + paste_close.size());
            return true;
        }
        msg += rest;
        while (std::getline(std::cin, line)) {
            size_t cp = line.find(paste_close);
            if (cp != std::string::npos) {
                msg += '\n';
                msg += line.substr(0, cp);
                msg += line.substr(cp + paste_close.size()); // anything typed after the paste
                break;
            }
            msg += '\n';
            msg += line;
        }
        return true; // EOF mid-paste: deliver what accumulated
    }

    // manual multi-line mode
    if (line == "/paste") {
        bool first = true;
        while (true) {
            printf(".. ");
            fflush(stdout);
            if (!std::getline(std::cin, line) || line == "/end") break;
            if (!first) msg += '\n';
            msg += line;
            first = false;
        }
        return true;
    }

    // burst fallback (tty only -- pipes deliver everything back-to-back by definition)
    if (isatty(STDIN_FILENO) && input_ready_within(30)) {
        msg = line;
        while (input_ready_within(30) && std::getline(std::cin, line)) {
            msg += '\n';
            msg += line;
        }
        return true;
    }

    msg = line;
    return true;
}

// ---- local slash-command support (/load-file, /save, /load) ---------------------------------
// The REPL mirrors the conversation in `messages` (role + raw text) purely for /save and /load.
// Generation itself is unchanged: it streams straight into the persistent KV cache.

struct chat_message {
    std::string role;    // "system" | "user" | "assistant"
    std::string content;
};

static std::string trim_ws(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) b++;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) e--;
    return s.substr(b, e - b);
}

// True when `line` is exactly `cmd`, or starts with `cmd` followed by whitespace. The trailing
// argument (trimmed) comes back in `arg` ("" when absent). The whitespace delimiter requirement
// is what keeps "/load" from matching "/load-file".
static bool match_command(const std::string & line, const char * cmd, std::string & arg) {
    const size_t n = strlen(cmd);
    if (line.size() < n || line.compare(0, n, cmd) != 0) return false;
    if (line.size() > n && line[n] != ' ' && line[n] != '\t') return false;
    arg = trim_ws(line.substr(n));
    return true;
}

// Reads a whole file, binary-safe, into `out`. `out` is a std::string, so the buffer is released
// by its owner as soon as the caller drops it -- no manual free needed.
static bool read_whole_file(const std::string & path, std::string & out, std::string & err) {
    out.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open '" + path + "'"; return false; }
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size < 0) { err = "cannot determine size of '" + path + "'"; return false; }
    f.seekg(0, std::ios::beg);
    out.resize((size_t) size);
    if (size > 0 && !f.read(out.data(), (std::streamsize) size)) {
        err = "read error on '" + path + "'";
        out.clear();
        return false;
    }
    return true;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    return out;
}

// Writes the message array as `[{"role": ..., "content": ...}, ...]` -- the exact schema the
// reader below accepts.
static bool save_chat_history(const std::string & path, const std::vector<chat_message> & msgs,
                              std::string & err) {
    std::string out;
    out += "[\n";
    for (size_t i = 0; i < msgs.size(); i++) {
        out += "  {\"role\": \"";
        out += json_escape(msgs[i].role);
        out += "\", \"content\": \"";
        out += json_escape(msgs[i].content);
        out += "\"}";
        if (i + 1 < msgs.size()) out += ",";
        out += "\n";
    }
    out += "]\n";

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { err = "cannot open '" + path + "' for writing"; return false; }
    f.write(out.data(), (std::streamsize) out.size());
    if (!f.good()) { err = "write error on '" + path + "'"; return false; }
    return true;
}

// Minimal dependency-free JSON reader for exactly the schema save_chat_history() writes: a
// top-level array of objects with string "role"/"content" members. Unknown members (any JSON
// value, nested objects and arrays included) are skipped structurally, so extra fields don't
// break reading. Escapes, \uXXXX with surrogate pairs included, are decoded.
struct json_reader {
    const std::string & s;
    size_t pos = 0;
    std::string err;

    explicit json_reader(const std::string & src) : s(src) {}

    bool fail(const std::string & what) {
        if (err.empty()) err = what + " (at byte " + std::to_string(pos) + ")";
        return false;
    }

    void skip_ws() {
        while (pos < s.size() &&
               (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) pos++;
    }

    int peek() {
        skip_ws();
        return pos < s.size() ? (unsigned char) s[pos] : -1;
    }

    bool expect(char c) {
        if (peek() != (unsigned char) c) return fail(std::string("expected '") + c + "'");
        pos++;
        return true;
    }

    static void utf8_append(std::string & out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back((char) cp);
        } else if (cp < 0x800) {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char) (0xE0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char) (0xF0 | (cp >> 18)));
            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }

    bool read_hex4(uint32_t & out) {
        if (pos + 4 > s.size()) return fail("truncated \\u escape");
        out = 0;
        for (int i = 0; i < 4; i++) {
            const char c = s[pos++];
            out <<= 4;
            if (c >= '0' && c <= '9')      out |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') out |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') out |= (uint32_t) (c - 'A' + 10);
            else return fail("bad hex digit in \\u escape");
        }
        return true;
    }

    bool parse_string(std::string & out) {
        if (!expect('"')) return false;
        out.clear();
        while (pos < s.size()) {
            const unsigned char c = (unsigned char) s[pos++];
            if (c == '"') return true;
            if (c < 0x20) return fail("raw control character in string");
            if (c != '\\') { out.push_back((char) c); continue; }
            if (pos >= s.size()) return fail("unterminated escape");
            const char e = s[pos++];
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!read_hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        // high surrogate: combine with a following \uXXXX low surrogate when one
                        // is actually there, otherwise emit U+FFFD instead of invalid UTF-8
                        if (pos + 1 < s.size() && s[pos] == '\\' && s[pos + 1] == 'u') {
                            const size_t save = pos;
                            pos += 2;
                            uint32_t lo = 0;
                            if (!read_hex4(lo)) return false;
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                pos = save;
                                cp = 0xFFFD;
                            }
                        } else {
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        cp = 0xFFFD; // lone low surrogate
                    }
                    utf8_append(out, cp);
                    break;
                }
                default: return fail("unknown escape sequence");
            }
        }
        return fail("unterminated string");
    }

    // Skips one complete JSON value of any type. Depth-limited so a pathological file can't
    // recurse the stack away.
    bool skip_value(int depth = 0) {
        if (depth > 64) return fail("nesting too deep");
        const int c = peek();
        if (c < 0) return fail("unexpected end of input");
        if (c == '"') { std::string tmp; return parse_string(tmp); }
        if (c == '{') {
            pos++;
            if (peek() == '}') { pos++; return true; }
            while (true) {
                std::string key;
                if (!parse_string(key)) return false;
                if (!expect(':')) return false;
                if (!skip_value(depth + 1)) return false;
                const int d = peek();
                if (d == ',') { pos++; continue; }
                if (d == '}') { pos++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            pos++;
            if (peek() == ']') { pos++; return true; }
            while (true) {
                if (!skip_value(depth + 1)) return false;
                const int d = peek();
                if (d == ',') { pos++; continue; }
                if (d == ']') { pos++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (s.compare(pos, 4, "true") == 0)  { pos += 4; return true; }
        if (s.compare(pos, 5, "false") == 0) { pos += 5; return true; }
        if (s.compare(pos, 4, "null") == 0)  { pos += 4; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            pos++;
            while (pos < s.size() && ((s[pos] >= '0' && s[pos] <= '9') || s[pos] == '.' ||
                                      s[pos] == '+' || s[pos] == '-' ||
                                      s[pos] == 'e' || s[pos] == 'E')) pos++;
            return true;
        }
        return fail("unexpected character");
    }

    bool parse_history(std::vector<chat_message> & out) {
        if (!expect('[')) return false;
        if (peek() == ']') {
            pos++;
        } else {
            while (true) {
                if (!expect('{')) return false;
                chat_message msg;
                bool have_role = false, have_content = false;
                if (peek() != '}') {
                    while (true) {
                        std::string key;
                        if (!parse_string(key)) return false;
                        if (!expect(':')) return false;
                        if (key == "role") {
                            if (!parse_string(msg.role)) return false;
                            have_role = true;
                        } else if (key == "content") {
                            if (!parse_string(msg.content)) return false;
                            have_content = true;
                        } else if (!skip_value()) {
                            return false;
                        }
                        const int d = peek();
                        if (d == ',') { pos++; continue; }
                        if (d == '}') break;
                        return fail("expected ',' or '}' in message object");
                    }
                }
                pos++; // consume '}'
                if (!have_role || !have_content) return fail("message object without role/content");
                out.push_back(std::move(msg));
                const int d = peek();
                if (d == ',') { pos++; continue; }
                if (d == ']') { pos++; break; }
                return fail("expected ',' or ']'");
            }
        }
        skip_ws();
        if (pos != s.size()) return fail("trailing data after history array");
        return true;
    }
};

// Loads `path` and returns the parsed message array, with role validation: append_chat_turn()
// only has well-defined templates for system/user/assistant, so anything else is rejected up
// front instead of being silently templated as something the model never saw.
static bool load_chat_history(const std::string & path, std::vector<chat_message> & out,
                              std::string & err) {
    out.clear();
    std::string text;
    if (!read_whole_file(path, text, err)) return false; // `text` frees itself on return
    json_reader rd(text);
    if (!rd.parse_history(out)) {
        err = "'" + path + "': " + rd.err;
        out.clear();
        return false;
    }
    for (size_t i = 0; i < out.size(); i++) {
        const std::string & role = out[i].role;
        if (role != "system" && role != "user" && role != "assistant") {
            err = "'" + path + "': message " + std::to_string(i) + " has unsupported role '" +
                  role + "' (expected system/user/assistant)";
            out.clear();
            return false;
        }
    }
    return true;
}

static void print_chat_commands() {
    fprintf(stderr,
        "local commands (handled by the CLI, never sent to the model):\n"
        "  /load-file <path>   attach a file's contents to the next message\n"
        "  /save [file]        write the conversation to JSON (default: chat_history.json)\n"
        "  /load <file>        replace the conversation with a saved one and rebuild the context\n"
        "  /paste .. /end      type a multi-line message by hand\n"
        "  /help               this list\n");
}

// ---- web-search tool support (--search) ---------------------------------------------------------
// The marker scanner, the popen()-based search runner and the result/failure block builders live
// in web_search.h/.cpp (shared with `run`). The loop below is opt-in via --search: without it the
// behavior is exactly as before. When the model emits "[SEARCH: query]", generation pauses,
// perform_search() runs, a "[SEARCH RESULTS for ...]" block is injected into the live context and
// generation resumes -- up to max_searches rounds per reply, then the reply is cut short.

int cmd_chat(int argc, char ** argv) {
    std::string model_path = "Phi-4-mini-instruct-Q4_K_M.gguf";
    bool model_path_set = false;
    std::string system_prompt;
    std::string backend_name = "cpu";
    bool flash_attn = false;
    bool enable_thinking = true;
    bool show_thinking = true; // reasoning visible (dimmed) by default -- --hide-thinking opts out
    bool ctx_set = false;
    int64_t ctx_override = 0;
    int max_reply_tokens = 512; // only used with an explicit -n; see the per-turn sizing below
    bool n_set = false;
    bool search_enabled = false; // --search: advertise and handle the [SEARCH: ...] tool call
    int  max_searches = SEARCH_DEFAULT_MAX_ROUNDS; // --max-searches: hard cap per user prompt
    sampler_params sp;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if ((a == "-n" || a == "--n-predict") && i + 1 < argc) {
            max_reply_tokens = atoi(argv[++i]);
            n_set = true;
        } else if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "--flash-attn") {
            flash_attn = true;
        } else if (a == "--thinking") {
            enable_thinking = true;
        } else if (a == "--no-thinking") {
            enable_thinking = false;
        } else if (a == "--show-thinking") {
            show_thinking = true;
        } else if (a == "--hide-thinking") {
            show_thinking = false;
        } else if (a == "--ctx" && i + 1 < argc) {
            ctx_override = atoll(argv[++i]);
            ctx_set = true;
        } else if (a == "--temp" && i + 1 < argc) {
            sp.temp = (float) atof(argv[++i]);
        } else if (a == "--top-k" && i + 1 < argc) {
            sp.top_k = atoi(argv[++i]);
        } else if (a == "--top-p" && i + 1 < argc) {
            sp.top_p = (float) atof(argv[++i]);
        } else if (a == "--repeat-penalty" && i + 1 < argc) {
            sp.repeat_penalty = (float) atof(argv[++i]);
        } else if (a == "--repeat-last-n" && i + 1 < argc) {
            sp.repeat_last_n = atoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            sp.seed = (uint32_t) strtoul(argv[++i], nullptr, 10);
        } else if (a == "--search") {
            search_enabled = true;
        } else if (a == "--max-searches" && i + 1 < argc) {
            max_searches = atoi(argv[++i]);
            search_enabled = true; // the cap only means something with the tool call enabled
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            // silently treating unknown flags as a model path made typos / unsupported options
            // (e.g. --verbose) look like they had been applied
            fprintf(stderr, "warning: ignoring unknown option '%s'\n", a.c_str());
        } else if (!model_path_set) {
            model_path = a;
            model_path_set = true;
        }
    }
    // reject a nonsense explicit reply budget up front (0/negative would silently produce empty
    // replies); oversized values are fine -- each turn clamps them to the room left in the cache
    if (n_set && max_reply_tokens < 1) {
        fprintf(stderr, "error: -n/--n-predict must be >= 1\n");
        return 1;
    }
    if (ctx_set && ctx_override < 64) {
        fprintf(stderr, "error: --ctx must be >= 64 (got %lld)\n", (long long) ctx_override);
        return 1;
    }
    if (search_enabled && max_searches < 1) {
        fprintf(stderr, "error: --max-searches must be >= 1\n");
        return 1;
    }

    // with --search, the tool contract rides in the system prompt (appended after any user
    // --system text so it reads as the later, specific instruction)
    if (search_enabled) {
        if (!system_prompt.empty()) system_prompt += "\n\n";
        system_prompt += SEARCH_TOOL_NOTE;
    }

    sampler smp = init_sampler(sp);

    model_path = resolve_model_ref(model_path);
    ggml_backend_t backend = init_backend(backend_name, 8);

    struct gguf_context * gctx = nullptr;
    weights_store ws = load_weights(model_path.c_str(), backend, &gctx);

    hparams hp = load_hparams(gctx);
    vocab   vc = load_vocab(gctx);
    model   m  = load_model(ws.ctx, hp);
    const int64_t n_vocab = m.token_embd->ne[1];
    const int32_t tok_end = turn_end_token(hp, vc);
    // some models (observed on qwen3) occasionally emit the literal "<|endoftext|>" token mid-turn
    // instead of the chat template's own end-of-turn token; treat a literal "</s>" / "[eos]"
    // (plus "<|im_end|>" on templates that terminate with something else) the same way, rather
    // than letting generation run past them into repetition. Guarded lookups: absent -> -1.
    auto vocab_id = [&vc](const char * tag) -> int32_t {
        auto it = vc.token_to_id.find(tag);
        return it != vc.token_to_id.end() ? it->second : -1;
    };
    const int32_t tok_eot = vocab_id("<|endoftext|>");
    const int32_t tok_slash_s = vocab_id("</s>");
    const int32_t tok_bare_eos = vocab_id("[eos]");
    const int32_t tok_im_end = vocab_id("<|im_end|>");

    // size the KV cache: auto from the RAM still free now that the weights are loaded, unless
    // --ctx overrides it (the weights' own footprint is already subtracted by the kernel here)
    const int64_t n_ctx = ctx_set ? ctx_override
                                  : auto_n_ctx(m, hp, ggml_backend_buffer_get_size(ws.buffer));
    if (ctx_set) fprintf(stderr, "n_ctx: %lld (from --ctx)\n", (long long) n_ctx);

    kv_cache kv = init_kv_cache(m, hp, backend, n_ctx);
    ssm_state ss = init_ssm_state(hp, backend); // no-op struct for every arch except nemotron_h
    decode_session ds = init_decode_session(backend, hp, flash_attn);

    fprintf(stderr, "%s loaded (%s, %lld layers, backend=%s%s). Type a message and press Enter "
                     "(Ctrl-D or 'exit' to quit; a pasted block is sent as one message, or type "
                     "/paste and finish with /end; '/help' lists the local commands).\n",
            model_path.c_str(), hp.arch_name.c_str(), (long long) hp.n_layer, backend_name.c_str(),
            hp.supports_thinking ? (enable_thinking ? ", thinking=on" : ", thinking=off") : "");
    if (!n_set) {
        fprintf(stderr, "note: -n not set: each reply gets the room left in the KV cache after "
                        "its prompt, minus a 64-token EOS margin\n");
    }
    if (search_enabled) {
        fprintf(stderr, "web search: enabled -- the model may emit '[SEARCH: query]'; the CLI runs "
                        "curl | python3 and feeds the results back (cap ~%zu chars, %d searches/reply)\n",
                SEARCH_RESULT_MAX, max_searches);
    }

    // reasoning models emit their think trace as ordinary tokens; this filter hides it (or dims
    // it with --show-thinking) and is token-boundary safe -- see think_filter.h
    thinking_filter tf;
    tf.mode = show_thinking ? thinking_filter::MODE_DIMMED : thinking_filter::MODE_HIDDEN;

    std::vector<int32_t> pending;
    if (hp.bos_id >= 0) pending.push_back((int32_t) hp.bos_id); // some tokenizers have none
    if (!system_prompt.empty()) {
        append_chat_turn(hp, vc, pending, "system", system_prompt);
    }

    int64_t n_past = 0;
    std::vector<int32_t> history; // recent tokens for repeat_penalty; KV cache handles real context

    // role+text mirror of the live conversation, maintained for /save and /load only --
    // generation still goes straight into the KV cache. pending_attachments holds the file
    // blocks queued by /load-file until the next real message consumes them.
    std::vector<chat_message> messages;
    if (!system_prompt.empty()) messages.push_back({ "system", system_prompt });
    std::string pending_attachments;

    // ask the terminal to bracket pastes (\x1b[200~ .. \x1b[201~) so a multi-line block arrives
    // as one message; see read_user_message. Restored right before returning.
    const bool stdin_is_tty = isatty(STDIN_FILENO) != 0;
    if (stdin_is_tty) fputs("\x1b[?2004h", stderr);

    while (true) {
        std::string line;
        if (!read_user_message(line)) break; // EOF (Ctrl-D), or exit/quit
        if (line.empty()) continue;

        // local slash commands: only a single-line message can be one (a multi-line paste that
        // merely starts with '/' is still a message), and an unknown '/...' falls through to the
        // model like any other text. None of these ever reach the model.
        if (line.find('\n') == std::string::npos && line[0] == '/') {
            std::string arg;

            if (match_command(line, "/load-file", arg)) {
                if (arg.empty()) {
                    fprintf(stderr, "[usage: /load-file <path>]\n");
                    continue;
                }
                std::string content, err;
                if (!read_whole_file(arg, content, err)) {
                    fprintf(stderr, "[load-file: %s]\n", err.c_str());
                    continue;
                }
                if (!pending_attachments.empty()) pending_attachments += "\n\n";
                pending_attachments += "File " + arg + " contents:\n===== BEGIN FILE =====\n" +
                                       content + "\n===== END FILE =====";
                fprintf(stderr, "[attached '%s' (%zu bytes) -- appended to the next message]\n",
                        arg.c_str(), content.size());
                if ((int64_t) content.size() > n_ctx * 3) {
                    fprintf(stderr, "[warning: that is likely more than n_ctx=%lld tokens can "
                                    "hold -- the next turn may not fit]\n", (long long) n_ctx);
                }
                continue; // the file text lives in pending_attachments until the next message
            }

            if (match_command(line, "/save", arg)) {
                const std::string path = arg.empty() ? "chat_history.json" : arg;
                std::string err;
                if (!save_chat_history(path, messages, err)) {
                    fprintf(stderr, "[save: %s]\n", err.c_str());
                } else {
                    fprintf(stderr, "[saved %zu message%s to '%s']\n",
                            messages.size(), messages.size() == 1 ? "" : "s", path.c_str());
                }
                continue;
            }

            if (match_command(line, "/load", arg)) {
                if (arg.empty()) {
                    fprintf(stderr, "[usage: /load <filename>]\n");
                    continue;
                }
                std::vector<chat_message> loaded;
                std::string err;
                if (!load_chat_history(arg, loaded, err)) {
                    fprintf(stderr, "[load: %s]\n", err.c_str());
                    continue;
                }

                // tokenize the whole history first, so a too-long file is rejected without
                // disturbing the live context
                std::vector<int32_t> toks;
                if (hp.bos_id >= 0) toks.push_back((int32_t) hp.bos_id);
                for (const chat_message & msg : loaded) {
                    append_chat_turn(hp, vc, toks, msg.role, msg.content);
                }
                if ((int64_t) toks.size() >= n_ctx) {
                    fprintf(stderr, "[load: history is %lld tokens, which does not fit n_ctx=%lld "
                                    "-- not loaded]\n",
                            (long long) toks.size(), (long long) n_ctx);
                    continue;
                }

                // commit: drop any pending startup tokens / attachments, rewind the KV position,
                // then re-prefill from scratch. Overwriting [0, toks.size()) is the clear -- the
                // graph's K/V views are narrowed to n_past+n_new every step, so stale slots past
                // the new end are never attended to. The recurrent state (nemotron_h / qwen35),
                // however, is a running accumulator that cannot be rewound, so it is zeroed
                // explicitly (a no-op for every other architecture: ss.buffer is null there).
                pending.clear();
                pending_attachments.clear();
                n_past = 0;
                if (ss.buffer) ggml_backend_buffer_clear(ss.buffer, 0);
                if (!toks.empty()) {
                    prefill(m, hp, kv, ds, backend, 0, toks, n_vocab,
                            backend_name == "vulkan", &ss);
                }
                n_past = (int64_t) toks.size();
                history = toks; // repeat-penalty window mirrors the rebuilt context
                messages = std::move(loaded); // live history replaced; the file's raw text string
                                              // was already released inside load_chat_history()
                fprintf(stderr, "[loaded %zu message%s (%lld tokens) from '%s' -- context rebuilt]\n",
                        messages.size(), messages.size() == 1 ? "" : "s",
                        (long long) n_past, arg.c_str());
                continue;
            }

            if (match_command(line, "/help", arg)) {
                print_chat_commands();
                continue;
            }
            // unknown '/...': fall through and send it as an ordinary message
        }

        // a file queued by /load-file rides along with this message
        if (!pending_attachments.empty()) {
            line += "\n\n";
            line += pending_attachments;
            pending_attachments.clear();
        }

        // confirm on stderr what was just consumed as the user message: the terminal echoes an
        // arriving paste with the exact same visual style as model output, so without this marker
        // a pasted message reads as "the model started talking by itself"
        const size_t n_lines = 1 + (size_t) std::count(line.begin(), line.end(), '\n');
        fprintf(stderr, "[user message: %zu line%s, %zu chars]\n",
                n_lines, n_lines == 1 ? "" : "s", line.size());

        messages.push_back({ "user", line });
        append_chat_turn(hp, vc, pending, "user", line);
        append_generation_prompt(hp, vc, pending, enable_thinking);

        // reply budget: the room left after the prompt, minus one slot reserved for the
        // terminator token the loop below commits after the reply body. An explicit -n is
        // clamped to that room with a warning; without -n, the reply auto-fills the room but
        // keeps a 64-token EOS margin for the model to emit its end-of-turn token
        const int64_t prompt_total = n_past + (int64_t) pending.size();
        const int64_t reply_room = n_ctx - prompt_total - 1;
        if (reply_room < 1) {
            fprintf(stderr, "\n[context window full (%lld tokens) -- restart the chat]\n",
                    (long long) n_ctx);
            break;
        }
        int64_t reply_budget;
        if (!n_set) {
            reply_budget = std::max<int64_t>(n_ctx - prompt_total - 64, 1);
        } else {
            reply_budget = max_reply_tokens;
            if (reply_budget > reply_room) {
                fprintf(stderr, "warning: -n %d exceeds the room left in n_ctx (%lld tok); "
                                "clamping the reply to %lld\n",
                        max_reply_tokens, (long long) reply_room, (long long) reply_room);
                reply_budget = reply_room;
            }
        }

        auto t0 = std::chrono::steady_clock::now();
        // chunked on Vulkan -- see prefill()'s comment in model.cpp -- to dodge a mid-size-batch
        // shader-selection cliff that a short chat turn (very often 9-30 tokens) lands in constantly
        std::vector<float> logits = prefill(m, hp, kv, ds, backend, n_past, pending, n_vocab,
                                             backend_name == "vulkan", &ss);
        n_past += (int64_t) pending.size();
        auto t1 = std::chrono::steady_clock::now();
        double prefill_s = std::chrono::duration<double>(t1 - t0).count();
        double prefill_tps = pending.size() / prefill_s;
        size_t n_prefill = pending.size();
        history.insert(history.end(), pending.begin(), pending.end());
        pending.clear(); // this turn's tokens are now committed to the KV cache via n_past

        int32_t next = sample(smp, logits, history);
        render_filter_reset(tf); // new assistant turn: no half-parsed tag may leak across turns
        if (generation_prompt_opens_think(hp, enable_thinking)) {
            render_filter_begin_thinking(tf); // nemotron_h: the prompt already emitted the open tag
        }
        int n_reply = 0;
        std::string reply_text; // raw detokenized text of the current reply segment, for /save
        auto t2 = std::chrono::steady_clock::now();

        // Reply generation, with the optional [SEARCH: ...] tool call (--search): when the model
        // asks for a search, generation pauses, perform_search() runs, a "[SEARCH RESULTS for ...]"
        // block is injected into the live context and generation resumes (up to max_searches
        // searches per reply, hard-capped). Without --search this is exactly the old loop. Each
        // segment and the tool exchange are mirrored into `messages` for /save; /load re-templates
        // that mirror, so the reloaded context is equivalent, not byte-identical (the mirror always
        // had that caveat).
        int searches_done = 0;
        double search_wait_s = 0.0; // wall time inside perform_search(), kept out of the decode t/s
        bool turn_over = false;
        while (!turn_over) {
            // bytes withheld from the console because they may still become "[SEARCH: "
            std::string render_hold;
            std::string search_query;
            bool search_wanted = false;

            while (n_reply < reply_budget && next != tok_end && next != hp.eos_id && next != tok_eot &&
                   next != tok_im_end && next != tok_slash_s && next != tok_bare_eos) {
                const std::string piece = vc.detok({ next });
                reply_text += piece; // the raw text keeps everything; only the console hides markers
                history.push_back(next);

                if (!search_enabled) {
                    render_filter_thinking(tf, piece); // hides/dims reasoning blocks on the console
                } else {
                    render_hold += piece;
                    size_t marker_start = 0;
                    const search_scan_t sr = search_scan(render_hold, search_query, marker_start, false);
                    const size_t render_len = (sr == SEARCH_SCAN_NONE) ? render_hold.size() : marker_start;
                    if (render_len > 0) render_filter_thinking(tf, render_hold.substr(0, render_len));
                    render_hold.erase(0, render_len);
                    if (sr == SEARCH_SCAN_MARKER) {
                        render_hold.clear(); // the marker itself is never displayed
                        search_wanted = true;
                    }
                }

                std::vector<float> step_logits = forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab, &ss);
                n_past += 1;
                n_reply++;
                if (search_wanted) break; // the marker token is committed; generation pauses here
                next = sample(smp, step_logits, history);
            }

            if (!search_wanted) {
                // end of reply: a marker that never got its ']' (the model hit EOS mid-marker)
                // still counts as a call
                if (!render_hold.empty()) {
                    size_t marker_start = 0;
                    if (search_scan(render_hold, search_query, marker_start, true) == SEARCH_SCAN_MARKER) {
                        search_wanted = true;
                    } else {
                        render_filter_thinking(tf, render_hold);
                    }
                    render_hold.clear();
                }
                if (!search_wanted) {
                    // commit the terminator itself to the cache so history matches the chat
                    // template exactly for the next turn's context
                    forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab, &ss);
                    n_past += 1;
                    messages.push_back({ "assistant", reply_text });
                    turn_over = true;
                    break;
                }
                // an unterminated marker: its last token (usually EOS) is deliberately NOT
                // committed -- the reply continues after the search below
            }

            // ---- tool call: pause, search, feed the results back, resume ----
            messages.push_back({ "assistant", reply_text }); // the segment exactly as emitted
            reply_text.clear();

            if (searches_done >= max_searches) {
                fprintf(stderr, "\n[search: round limit (%d) reached -- ending the reply]\n",
                        max_searches);
            } else {
                searches_done++;
                std::string results;
                fprintf(stderr, "\n[Web Search %d/%d]: %s ...\n",
                        searches_done, max_searches, search_query.c_str());
                const auto search_t0 = std::chrono::steady_clock::now();
                const bool searched = perform_search(search_query, results);
                search_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - search_t0).count();

                // a failed search still injects a note -- the model must know no results arrived,
                // so it can apologize or retry with a different query instead of guessing
                std::string note;
                if (!searched) {
                    note = search_failed_block(search_query);
                    fprintf(stderr, "[Web Search %d: no usable results -- feeding a failure note]\n",
                            searches_done);
                } else {
                    note = search_results_block(search_query, results);
                }

                // feed the block into the live context and keep generating. The injected text is
                // the rawest possible tool exchange -- no template tokens are fabricated mid-turn,
                // so this works for every architecture's chat format. perform_search() already
                // capped the results at SEARCH_RESULT_MAX chars; fit whatever is left to the token
                // room actually available (65-token EOS margin kept), so a small n_ctx can't
                // overflow the KV cache.
                const int64_t inject_budget = n_ctx - n_past - 65;
                const std::string inject = (inject_budget >= 16)
                    ? search_fit_to_tokens(vc, "\n" + note, inject_budget) : std::string();
                std::vector<int32_t> inject_toks = inject.empty() ? std::vector<int32_t>()
                                                                  : vc.encode(inject);
                if (!inject_toks.empty() && (int64_t) inject_toks.size() <= inject_budget) {
                    // mirror the tool exchange in /save's message array -- only once it actually
                    // enters the live context, so the mirror stays aligned with it (the mirror
                    // stores the fitted block, i.e. exactly what was injected)
                    messages.push_back({ "user", inject.substr(1) });
                    std::vector<float> resume_logits = prefill(m, hp, kv, ds, backend, n_past, inject_toks,
                                                               n_vocab, backend_name == "vulkan", &ss);
                    n_past += (int64_t) inject_toks.size();
                    history.insert(history.end(), inject_toks.begin(), inject_toks.end());

                    // the injected tokens ate KV room: keep the turn's 64-token EOS margin honest
                    const int64_t max_future = n_ctx - n_past - 65;
                    reply_budget = std::min<int64_t>(reply_budget,
                                                     n_reply + std::max<int64_t>(max_future, 1));
                    next = sample(smp, resume_logits, history);
                    continue;
                }
                fprintf(stderr, "[search: no room left in n_ctx for the results -- ending the reply]\n");
            }

            // the search was not performed (round limit or no room): close the turn cleanly by
            // committing the template's own end-of-turn token, so the next turn starts from a
            // complete template state
            if (tok_end >= 0) {
                forward_step(m, hp, kv, ds, backend, n_past, { tok_end }, n_vocab, &ss);
                n_past += 1;
            }
            turn_over = true;
        }
        auto t3 = std::chrono::steady_clock::now();
        double decode_s = std::max(std::chrono::duration<double>(t3 - t2).count() - search_wait_s, 1e-6);

        render_filter_finish(tf); // flush any withheld tail; close dim colouring if left open
        if (!show_thinking && tf.hidden_chars > 0) {
            fprintf(stderr, "\n[thinking hidden: %lld chars -- drop --hide-thinking to display it]\n",
                    tf.hidden_chars);
        }

        fprintf(stderr, "\n[prefill %zu tok in %.2fs (%.1f tok/s), reply %d tok in %.2fs (%.1f tok/s)]\n",
                n_prefill, prefill_s, prefill_tps, n_reply, decode_s, n_reply / decode_s);
    }

    free_decode_session(ds);
    free_ssm_state(ss);
    ggml_backend_buffer_free(kv.buffer);
    ggml_free(kv.ctx);
    ggml_backend_buffer_free(ws.buffer);
    ggml_free(ws.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    if (stdin_is_tty) fputs("\x1b[?2004l", stderr);
    return 0;
}
