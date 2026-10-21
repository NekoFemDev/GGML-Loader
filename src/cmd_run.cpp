// `ggml-loader run <model.gguf> [-p/--prompt "text"] [-n/--n-predict N] [--backend cpu|vulkan]`:
// one-shot raw completion (no chat template, except for --edit-file below), with prefill/decode
// timing -- the mode used to benchmark against llama-cli/llama-bench on the same prompt, and cpu
// vs vulkan against each other.
//
// Reply budgeting: without -n, max_new_tokens defaults to n_ctx - prompt_tokens - 64 (a
// 64-token safety margin so the model has room to emit EOS); an explicit -n that exceeds the
// room left in n_ctx is warned about and clamped to it, not fatal. Decoding stops early at the
// first EOS / end-of-turn token (<|im_end|>, </s>, [eos], ...), and --edit-file additionally
// stops as soon as the rewrite's closing ``` fence has been emitted (trailing prose and the
// bare-fence degeneration loop observed on llama-3.2-3b are simply never decoded).
//
// `--edit-file <path> [--edit-on]`: wraps <path>'s contents + the -p instruction into a
// "rewrite this whole file" prompt, runs completion, and (with --edit-on) backs the original up
// to <path>.original and overwrites <path> with the model's output -- but only if the output
// passes a basic sanity check (non-trivial length and non-fence content, balanced
// braces/parens/brackets), since these small local models have no diff/code-edit fine-tuning and
// raw full-file rewrites can truncate or hallucinate. Without --edit-on the rewrite is just
// printed, nothing is touched on disk.
//
// Unlike plain run (raw completion, no chat template), --edit-file is an instruction-following
// task: its prompt is encoded with the model's own chat template, because instruct-tuned models
// fed a bare raw prompt don't answer at all (observed on nemotron_h: immediate EOS, 0 decoded
// tokens) or degenerate into a bare-fence loop (llama-3.2-3b). --thinking/--no-thinking control
// the reasoning trace for this mode (default: off -- a think trace costs minutes at these decode
// speeds); --show-thinking/--hide-thinking control whether that trace is printed in dim gray or
// hidden on stdout (default: hidden). Either way the trace is stripped before both the rewrite
// extraction and the closing-fence early stop, so a fenced draft inside the reasoning can never
// be mistaken for the rewrite. Plain run ignores all four flags and stays a raw-completion
// benchmark.
//
// --search turns the decode loop into a multi-step web-search agent loop with the same tool
// contract as chat (web_search.h): when the model emits "[SEARCH: query]", decoding pauses, the
// search runs through popen (curl | python3), a "[SEARCH RESULTS for ...]: ..." block (or a
// "[SEARCH FAILED ...]" note when nothing came back) is prefilled into the live context and
// decoding resumes -- up to --max-searches rounds (default 3, hard-capped so a runaway model
// cannot loop forever). Results are truncated to ~2000 chars, and further when the KV-cache room
// demands it. Off by default; without --search the decode path is byte-identical to before.
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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [model.gguf] [-p/--prompt \"text\"] [-n/--n-predict N] [--backend cpu|vulkan] [--flash-attn]\n"
        "           [--temp F] [--top-k N] [--top-p F] [--repeat-penalty F] [--repeat-last-n N] [--seed N]\n"
        "           [--edit-file path] [--edit-on] [--thinking | --no-thinking]\n"
        "           [--show-thinking | --hide-thinking] [--ctx N] [--search [--max-searches N]]\n"
        "  no -p: BOS-seeded unconditional free-run\n"
        "  temp<=0 (default): greedy/deterministic, same as before these flags existed\n"
        "  -n/--n-predict N: max new tokens (optional). Default: n_ctx - prompt - 64, i.e. the\n"
        "             room left after the prompt minus a 64-token EOS margin; an explicit N that\n"
        "             exceeds the room left in n_ctx is clamped to it with a warning\n"
        "  generation stops early at the model's EOS / end-of-turn token; --edit-file also stops\n"
        "             once the rewrite's closing ``` fence appears\n"
        "  --edit-file path: wrap path's contents + -p into a whole-file-rewrite prompt, encoded\n"
        "             with the model's own chat template (instruct models need it -- a bare raw\n"
        "             prompt gets an immediate EOS out of nemotron_h)\n"
        "  --edit-on: with --edit-file, back path up to path.original and overwrite path with\n"
        "             the model's output if it passes a basic sanity check (dry-run without it)\n"
        "  --thinking/--no-thinking: --edit-file only, reasoning models only (default: off);\n"
        "             plain run ignores both and stays a raw-completion benchmark\n"
        "  --show-thinking/--hide-thinking: --edit-file only: print the model's think trace in dim\n"
        "             gray instead of hiding it (default: hidden). Either way the trace is never\n"
        "             extracted as the rewrite and never counts toward the closing-fence stop\n"
        "  --search: multi-step web-search tool call (off by default): the model may emit\n"
        "             '[SEARCH: <query>]', the CLI pauses decoding, runs curl | python3 (DuckDuckGo's\n"
        "             html endpoint, ~15s cap) and prefills '[SEARCH RESULTS for <query>]: ...' back\n"
        "             into the live context (up to --max-searches rounds per prompt, default 3; each\n"
        "             result truncated to ~2000 chars, further if n_ctx demands it)\n"
        "  --max-searches N: hard cap on searches per prompt (default 3; implies --search)\n"
        "  --ctx N: KV cache context size. Default: auto-sized at load time from the RAM free\n"
        "             after the weights are loaded, clamped to the model's trained context length\n"
        "             (see the n_ctx=auto report line)\n",
        argv0);
}

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static bool write_file(const std::string & path, const std::string & content) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f << content;
    return (bool) f;
}

// heuristic-only sanity check: these are small raw-completion models with no code-edit
// fine-tuning, so a full-file rewrite can come back truncated (hit -n before finishing) or
// otherwise malformed. Rather than trust it blindly, require it be a non-trivial fraction of the
// original file's size and have balanced braces/parens/brackets before ever overwriting anything.
static bool looks_like_plausible_code(const std::string & generated, size_t original_len) {
    if (generated.empty()) return false;
    if (original_len > 0 && generated.size() < original_len / 3) return false;

    // Walk line by line: a line whose first non-blank char is a backtick is a fence delimiter
    // ("```" or "```lang") and contributes no code. Everything else contributes brace balance and
    // "substantive" (non-whitespace) content. A degeneration loop of bare fences has trivially
    // balanced braces, so balance alone would pass it -- the substance floor is what rejects it
    // (observed live: llama-3.2-3b emitting 150 lines of ``` under greedy raw-completion).
    long curly = 0, paren = 0, square = 0;
    size_t substantive = 0;
    size_t line_start = 0;
    while (line_start <= generated.size()) {
        size_t line_end = generated.find('\n', line_start);
        if (line_end == std::string::npos) line_end = generated.size();
        size_t first = generated.find_first_not_of(" \t\r", line_start);
        bool fence_line = first != std::string::npos && first < line_end && generated[first] == '`';
        if (!fence_line) {
            for (size_t i = line_start; i < line_end; i++) {
                char c = generated[i];
                if (c == '{') curly++; else if (c == '}') curly--;
                else if (c == '(') paren++; else if (c == ')') paren--;
                else if (c == '[') square++; else if (c == ']') square--;
                if (c != ' ' && c != '\t' && c != '\r') substantive++;
            }
        }
        if (line_end == generated.size()) break;
        line_start = line_end + 1;
    }
    if (original_len > 0 && substantive < original_len / 4) return false;
    if (original_len == 0 && substantive == 0) return false;
    return curly == 0 && paren == 0 && square == 0;
}

// Extracts the model's intended file rewrite from its output, handling the common small-model
// failure modes:
//   1. Model wraps the rewrite in a ```lang ... ``` fence.
//   2. Model first writes narrative text ("Sure! Here's the improved version:") or a
//      simplified draft, *then* the actual rewrite, with or without a fence.
//   3. Model produces multiple fence-wrapped blocks (a "simple" version and an "improved"
//      one); we want the LAST non-empty one since that's the canonical answer.
//   4. Model drifts after the rewrite and emits trailing ``` tokens (with no body between them)
//      because it missed eos; we must not let those trailing bare fences become our "last
//      fence pair" and clobber the real content.
//   5. Model copies the prompt's own BEGIN/END-FILE delimiter and writes its answer between
//      its own pair of those markers (observed: small models see the prompt as a "fill in
//      between these markers" template and dutifully produce their own). Recognize this and
//      take the LAST ===== BEGIN FILE ===== / ===== END FILE ===== pair's body.
//
// Strategy: try the strategies in order of specificity (most specific first).
//   a) If the output contains ===== END FILE =====, take everything between the last
//      ===== END FILE ===== and the matching ===== BEGIN FILE ===== just before it (case 5).
//   b) Else find the LAST non-empty ``` fence pair (cases 1, 3, 4).
//   c) Else fall back to "rewrite starts after the first blank line following prose" (case 2).
//   d) Final fallback: return input unchanged; brace-balance sanity check catches the worst.
static std::string extract_edit_rewrite(const std::string & s) {
    auto is_blank = [](const std::string & body) {
        return body.find_first_not_of(" \t\r\n") == std::string::npos;
    };

    // (a) ===== BEGIN FILE ===== / ===== END FILE ===== pattern (case 5). Walk all marker
    // positions, find the LAST "===== END FILE =====" and pair it with the nearest
    // "===== BEGIN FILE =====" to its left.
    auto find_marker = [&s](const std::string & marker, size_t start) -> size_t {
        return s.find(marker, start);
    };
    size_t scan_from = 0;
    size_t last_end = std::string::npos;
    while (true) {
        size_t p = find_marker("===== END FILE =====", scan_from);
        if (p == std::string::npos) break;
        last_end = p;
        scan_from = p + 1; // advance past this match to look for another
    }
    if (last_end != std::string::npos) {
        // find the nearest ===== BEGIN FILE ===== to the left of last_end
        size_t begin = s.rfind("===== BEGIN FILE =====", last_end);
        if (begin != std::string::npos) {
            size_t body_start = s.find('\n', begin);
            body_start = (body_start == std::string::npos) ? begin : body_start + 1;
            // body is [body_start, last_end)
            std::string body = s.substr(body_start, last_end - body_start);
            if (!body.empty() && body.back() == '\n') body.pop_back();
            if (!is_blank(body)) return body;
        }
    }

    // (b) Last non-empty ``` fence pair (cases 1, 3, 4)
    std::vector<size_t> fences;
    for (size_t i = 0; i + 2 < s.size(); ) {
        if (s[i] == '`' && s[i+1] == '`' && s[i+2] == '`') {
            fences.push_back(i);
            i += 3;
        } else {
            i++;
        }
    }
    // Try every (open, close) pair, rightmost-pairs first, and return the first one with a
    // non-empty body. Greedy rightward pairing: for each close, the nearest open to its left.
    for (size_t k = fences.size(); k-- > 0; ) {
        size_t close = fences[k];
        size_t open = std::string::npos;
        for (size_t j = k; j-- > 0; ) { open = fences[j]; break; }
        if (open == std::string::npos) continue;
        size_t body_start = s.find('\n', open);
        if (body_start == std::string::npos || body_start >= close) {
            body_start = open + 3;
        } else {
            body_start += 1;
        }
        std::string body = s.substr(body_start, close - body_start);
        if (!body.empty() && body.back() == '\n') body.pop_back();
        if (!is_blank(body)) return body;
    }

    // (c) "rewrite starts after the first blank line following prose" fallback (case 2)
    bool saw_prose = false;
    size_t i = 0;
    while (i < s.size()) {
        size_t line_end = s.find('\n', i);
        std::string line = s.substr(i, (line_end == std::string::npos) ? std::string::npos : line_end - i);
        size_t first = line.find_first_not_of(" \t\r");
        bool fence_line = first != std::string::npos && line[first] == '`';
        if (first == std::string::npos) {
            if (saw_prose) {
                size_t rewrite_start = (line_end == std::string::npos) ? s.size() : line_end + 1;
                while (rewrite_start < s.size() && s[rewrite_start] == '\n') rewrite_start++;
                std::string body = s.substr(rewrite_start);
                if (!is_blank(body)) return body;
                break;
            }
        } else if (!fence_line) {
            // real prose -- a fence-only line is neither prose nor a rewrite boundary (otherwise a
            // bare-fence degeneration loop would make even line one look like "prose")
            saw_prose = true;
        }
        if (line_end == std::string::npos) break;
        i = line_end + 1;
    }
    return s; // give up -- return as-is
}

// --edit-file early-stop helper (see the decode loop): the model was asked to wrap the rewrite in
// a single fenced code block ("three backticks on the open and close lines"), so once a *second*
// line whose first non-blank character is a ``` run shows up, the block has closed and everything
// the model would keep writing is trailing commentary (or a bare-fence degeneration loop).
// Counted over the whole text generated so far rather than per token, because a fence can be
// split across token boundaries ("``" + "`"); the string is small, so the rescan is cheap. A
// false positive is possible when the rewritten file itself contains fence lines -- the
// looks_like_plausible_code() check before any disk write is the backstop there.
static int count_code_fence_lines(const std::string & s) {
    int count = 0;
    size_t line = 0;
    while (line <= s.size()) {
        size_t end = s.find('\n', line);
        if (end == std::string::npos) end = s.size();
        size_t first = s.find_first_not_of(" \t\r", line);
        if (first != std::string::npos && first < end && s.compare(first, 3, "```") == 0) count++;
        if (end == s.size()) break;
        line = end + 1;
    }
    return count;
}

int cmd_run(int argc, char ** argv) {
    std::string model_path = "Phi-4-mini-instruct-Q4_K_M.gguf";
    bool model_path_set = false;
    std::string prompt;
    std::string backend_name = "cpu";
    int max_new_tokens = 0; // user -n, or auto-sized from n_ctx - prompt - 64 once tokenized
    bool n_set = false;
    bool flash_attn = false;
    std::string edit_file;
    bool edit_on = false;
    bool enable_thinking = false; // --edit-file only: off by default (see the file-header comment)
    bool show_thinking = false;   // --edit-file only: trace hidden by default; --show-thinking dims it
    bool ctx_set = false;
    int64_t ctx_override = 0;
    bool search_enabled = false; // --search: multi-step web-search loop around the decode (see header)
    int  max_searches = SEARCH_DEFAULT_MAX_ROUNDS; // --max-searches: hard cap per prompt
    sampler_params sp;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-p" || a == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((a == "-n" || a == "--n-predict") && i + 1 < argc) {
            max_new_tokens = atoi(argv[++i]);
            n_set = true;
        } else if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "--flash-attn") {
            flash_attn = true;
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
        } else if (a == "--ctx" && i + 1 < argc) {
            ctx_override = atoll(argv[++i]);
            ctx_set = true;
        } else if (a == "--search") {
            search_enabled = true;
        } else if (a == "--max-searches" && i + 1 < argc) {
            max_searches = atoi(argv[++i]);
            search_enabled = true; // the cap only means something with the tool call enabled
        } else if (a == "--edit-file" && i + 1 < argc) {
            edit_file = argv[++i];
        } else if (a == "--edit-on") {
            edit_on = true;
        } else if (a == "--thinking") {
            enable_thinking = true;
        } else if (a == "--no-thinking") {
            enable_thinking = false;
        } else if (a == "--show-thinking") {
            show_thinking = true;
        } else if (a == "--hide-thinking") {
            show_thinking = false;
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

    if (ctx_set && ctx_override < 64) {
        fprintf(stderr, "error: --ctx must be >= 64 (got %lld)\n", (long long) ctx_override);
        return 1;
    }

    if (search_enabled && max_searches < 1) {
        fprintf(stderr, "error: --max-searches must be >= 1\n");
        return 1;
    }

    if (edit_on && edit_file.empty()) {
        fprintf(stderr, "error: --edit-on requires --edit-file <path>\n");
        return 1;
    }

    std::string edit_original_content;
    if (!edit_file.empty()) {
        if (!read_file(edit_file, edit_original_content)) {
            fprintf(stderr, "error: --edit-file: could not read '%s'\n", edit_file.c_str());
            return 1;
        }
        std::string instruction = prompt.empty() ? "Improve this file." : prompt;
        // Prompt shape: short instruction, file in a plain-text delimiter (NOT a ``` fence --
        // putting ``` in the prompt confuses small models into treating the *whole prompt* as
        // one giant code block to "complete" inside, observed as a looped trivial rewrite under
        // greedy decoding), short instruction. Narrative/extra-fence output is then handled
        // robustly in extract_edit_rewrite() rather than forbidden in the prompt, because
        // verbose rule lists themselves trigger the failure mode (qwen2.5-coder-3b and qwen3-8b
        // both degenerate into repeating an invented "Do not change the file's ..." rule
        // forever when given a long bulleted instruction set).
        // NOTE: no size estimate for max_new_tokens here anymore: without an explicit -n the
        // budget is auto-sized to the room left in n_ctx, after the prompt is tokenized (below)
        prompt = instruction + "\n\n"
                 "File `" + edit_file + "`:\n"
                 "===== BEGIN FILE =====\n" + edit_original_content + "\n===== END FILE =====\n\n"
                 "Reply with the complete updated file in a single fenced code block (three backticks "
                 "on the open and close lines), no other text.";
    }

    // --search: advertise the tool contract in the prompt itself. Plain run appends it to the raw
    // prompt; --edit-file's chat-templated prompt was assembled above and gets it appended too, so
    // both completion paths can trigger the search loop below.
    if (search_enabled) {
        prompt += "\n\n";
        prompt += SEARCH_TOOL_NOTE;
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

    // Early-stop set: decoding ends the moment any of these is sampled. hp.eos_id, plus the
    // literal end-of-generation markers the supported families actually emit as their own tokens
    // -- ChatML/qwen "<|im_end|>", llama3 "<|eot_id|>", phi3 "<|end|>", a literal "</s>" /
    // "[eos]", and "<|endoftext|>" (some models emit it distinctly from hp.eos_id -- see
    // cmd_chat.cpp). Guarded lookups: a vocab without the token contributes nothing, so raw
    // completion on non-chat GGUFs is unaffected. The chat-templated --edit-file mode
    // additionally pulls the architecture's own turn terminator, the authoritative source for
    // the same ids on the supported families.
    std::vector<int32_t> stop_ids;
    auto add_stop = [&stop_ids](int32_t id) { if (id >= 0) stop_ids.push_back(id); };
    auto vocab_id = [&vc](const char * tag) -> int32_t {
        auto it = vc.token_to_id.find(tag);
        return it != vc.token_to_id.end() ? it->second : -1;
    };
    add_stop((int32_t) hp.eos_id);
    add_stop(vocab_id("<|endoftext|>"));
    add_stop(vocab_id("<|im_end|>"));
    add_stop(vocab_id("<|eot_id|>"));
    add_stop(vocab_id("<|end|>"));
    add_stop(vocab_id("</s>"));
    add_stop(vocab_id("[eos]"));
    if (!edit_file.empty()) add_stop(turn_end_token(hp, vc));

    // --edit-file + --thinking on a nemotron_h-style template: the generation prompt itself opens
    // the think block (see append_generation_prompt), so the first generated tokens are reasoning
    // with no opening tag anywhere in the output. The console filter and the extraction stripper
    // below must start in thinking state to match -- lockstep with append_generation_prompt via
    // generation_prompt_opens_think (see model.h).
    const bool prompt_opens_think = !edit_file.empty() &&
                                    generation_prompt_opens_think(hp, enable_thinking);

    // size the KV cache: auto from the RAM still free now that the weights are loaded, unless
    // --ctx overrides it (the weights' own footprint is already subtracted by the kernel here)
    const int64_t n_ctx = ctx_set ? ctx_override
                                  : auto_n_ctx(m, hp, ggml_backend_buffer_get_size(ws.buffer));
    if (ctx_set) fprintf(stderr, "n_ctx: %lld (from --ctx)\n", (long long) n_ctx);

    std::vector<int32_t> prompt_tokens;
    if (hp.bos_id >= 0) prompt_tokens.push_back((int32_t) hp.bos_id); // some tokenizers have none
    if (!edit_file.empty()) {
        // instruction-following turn: encode with the model's own chat template (see the
        // file-header comment for why a bare raw prompt fails on instruct models)
        append_chat_turn(hp, vc, prompt_tokens, "user", prompt);
        append_generation_prompt(hp, vc, prompt_tokens, enable_thinking);
        fprintf(stderr, "edit-file prompt: %zu tokens (chat template, thinking %s)\n",
                prompt_tokens.size(), enable_thinking ? "on" : "off");
    } else if (!prompt.empty()) {
        std::vector<int32_t> enc = vc.encode(prompt);
        prompt_tokens.insert(prompt_tokens.end(), enc.begin(), enc.end());
        fprintf(stderr, "prompt: \"%s\" -> %zu tokens (round-trip: \"%s\")\n",
                prompt.c_str(), enc.size(), vc.detok(enc).c_str());
    }

    // Reply budget. Hard limit: prompt + reply must fit n_ctx (the decode loop writes one KV
    // slot per generated token, so the last writable slot is n_ctx-1). Without an explicit -n,
    // fill the remaining room but keep a 64-token margin for the model to emit EOS. An explicit
    // -n that exceeds the room is warned about and clamped to it, never fatal.
    const int64_t ctx_room = n_ctx - (int64_t) prompt_tokens.size();
    if (ctx_room < 1) {
        fprintf(stderr, "error: prompt (%zu tokens) does not fit in n_ctx=%lld; raise --ctx\n",
                prompt_tokens.size(), (long long) n_ctx);
        return 1;
    }
    if (!n_set) {
        max_new_tokens = (int) std::max<int64_t>(ctx_room - 64, 1);
        fprintf(stderr, "note: -n not set, using max_new_tokens=%d (n_ctx %lld - prompt %zu - 64)\n",
                max_new_tokens, (long long) n_ctx, prompt_tokens.size());
    } else if ((int64_t) max_new_tokens > ctx_room) {
        fprintf(stderr, "warning: -n %d exceeds the room left in n_ctx (%lld: prompt %zu of "
                        "%lld); clamping to %lld\n",
                max_new_tokens, (long long) ctx_room, prompt_tokens.size(), (long long) n_ctx,
                (long long) ctx_room);
        max_new_tokens = (int) ctx_room;
    }

    fprintf(stderr, "backend=%s n_embd=%lld n_head=%lld n_head_kv=%lld n_layer=%lld n_ff=%lld n_rot=%lld n_vocab=%lld\n",
            backend_name.c_str(), (long long) hp.n_embd, (long long) hp.n_head, (long long) hp.n_head_kv,
            (long long) hp.n_layer, (long long) hp.n_ff, (long long) hp.n_rot, (long long) n_vocab);

    kv_cache kv = init_kv_cache(m, hp, backend, n_ctx);
    ssm_state ss = init_ssm_state(hp, backend); // no-op struct for every arch except nemotron_h
    decode_session ds = init_decode_session(backend, hp, flash_attn);
    fprintf(stderr, "KV cache: %lld ctx, f16 -> %.0f MiB (%.1f KiB/token)\n",
            (long long) n_ctx, (double) kv_cache_bytes(m, hp, n_ctx) / (1024.0 * 1024.0),
            (double) kv_cache_bytes(m, hp, 1) / 1024.0);
    if (search_enabled) {
        fprintf(stderr, "web search: enabled -- the model may emit '[SEARCH: query]'; the CLI runs "
                        "curl | python3 and feeds the results back (cap ~%zu chars, %d searches/prompt)\n",
                SEARCH_RESULT_MAX, max_searches);
    }

    // prefill: process the whole prompt (BOS + encoded text) in one batch (chunked on Vulkan --
    // see prefill()'s comment in model.cpp -- to dodge a mid-size-batch shader-selection cliff)
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = prefill(m, hp, kv, ds, backend, 0, prompt_tokens, n_vocab,
                                         backend_name == "vulkan", &ss);
    auto t1 = std::chrono::steady_clock::now();
    double prefill_s = std::chrono::duration<double>(t1 - t0).count();

    std::vector<int32_t> all_tokens = prompt_tokens;
    int64_t n_past = (int64_t) prompt_tokens.size();
    int32_t next = sample(smp, logits, all_tokens);

    fprintf(stderr, "prefill: %zu tokens in %.3fs (%.2f tok/s)\n",
            prompt_tokens.size(), prefill_s, prompt_tokens.size() / prefill_s);
    fprintf(stderr, "generating (KV-cached, up to %d tokens)...\n", max_new_tokens);

    // --edit-file: the text generated so far, so the decode loop can stop at the closing fence
    std::string edit_generated;
    const bool stop_at_closing_fence = !edit_file.empty();

    // --edit-file: a reasoning model's think trace is hidden (default) or shown in dim gray on
    // stdout; the raw text still accumulates in edit_generated, and the trace is stripped before
    // the fence count / extraction below see it (see render_filter_strip_thinking)
    thinking_filter tf;
    tf.mode = show_thinking ? thinking_filter::MODE_DIMMED : thinking_filter::MODE_HIDDEN;
    if (stop_at_closing_fence && prompt_opens_think) {
        render_filter_begin_thinking(tf); // the prompt already emitted the opening tag
    }

    // decode: one new token at a time, timed separately from prefill for a pp/tg-style comparison.
    // With --search this is wrapped into a multi-step tool loop: when the model emits
    // "[SEARCH: query]", decoding pauses, perform_search() runs, a "[SEARCH RESULTS for ...]"
    // block is prefilled into the live context and decoding resumes -- up to max_searches rounds
    // per prompt (hard-capped, so the loop always terminates). Without --search the loop below is
    // the old single-pass decode, byte-identical.
    int n_decoded = 0;
    int searches_done = 0;
    double search_wait_s = 0.0; // wall time inside perform_search(), kept out of the decode t/s

    // one place that decides where a decoded piece goes: the think filter + extraction buffer
    // (--edit-file) or the raw stdout stream (plain run)
    auto emit_piece = [&](const std::string & text) {
        if (text.empty()) return;
        if (stop_at_closing_fence) {
            edit_generated += text;
            render_filter_thinking(tf, text); // think trace hidden / dimmed on stdout only
        } else {
            fputs(text.c_str(), stdout); // plain run: raw completion, byte-identical output
            fflush(stdout);
        }
    };

    auto t2 = std::chrono::steady_clock::now();
    while (true) {
        // bytes withheld from the console because they may still become "[SEARCH: " (only used
        // with --search; the scanner is token-boundary safe, see web_search.h)
        std::string render_hold;
        std::string search_query;
        bool search_wanted = false;

        // this segment's budget: everything left of -n, and -- once a search has injected results
        // into the KV cache -- never past the room left in n_ctx minus a 65-token EOS margin. For
        // the first segment this equals max_new_tokens exactly (the no-search path is unchanged).
        const int64_t segment_left = std::min<int64_t>(max_new_tokens - n_decoded,
                                                       n_ctx - n_past - (searches_done > 0 ? 65 : 0));
        if (segment_left < 1) {
            // the -n budget just ran out (the old for-loop's silent end); only an actual KV-cache
            // overflow needs a note here, and that can only happen after a search injection
            if (max_new_tokens - n_decoded >= 1) {
                fprintf(stderr, "[context window full (%lld tokens) -- ending generation]\n",
                        (long long) n_ctx);
            }
            break;
        }

        for (int64_t step = 0; step < segment_left; step++) {
            if (std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end()) {
                fprintf(stderr, "[eos]\n");
                break;
            }
            all_tokens.push_back(next);
            n_decoded++;

            const std::string piece = vc.detok({ next });
            if (!search_enabled) {
                emit_piece(piece);
            } else {
                // hold the tail that could still become a "[SEARCH: " call; everything before it
                // is ordinary output (a marker itself is never displayed nor extracted)
                render_hold += piece;
                size_t marker_start = 0;
                const search_scan_t sr = search_scan(render_hold, search_query, marker_start, false);
                const size_t render_len = (sr == SEARCH_SCAN_NONE) ? render_hold.size() : marker_start;
                if (render_len > 0) emit_piece(render_hold.substr(0, render_len));
                render_hold.erase(0, render_len);
                if (sr == SEARCH_SCAN_MARKER) {
                    render_hold.clear();
                    search_wanted = true;
                }
            }

            if (stop_at_closing_fence) {
                // stop as soon as the model has closed its fenced code block (a second fence line):
                // everything past it is trailing commentary (or a bare-fence degeneration loop), and
                // the extraction pass below only ever wants the last complete block anyway. The
                // fence line itself is kept in the output so extract_edit_rewrite() sees a closed pair.
                // Counted on the think-stripped text: a draft fenced block *inside* the reasoning
                // trace must not end decoding before the real rewrite has been emitted.
                if (count_code_fence_lines(render_filter_strip_thinking(edit_generated,
                                                                        prompt_opens_think)) >= 2) {
                    fprintf(stderr, "\n[stop: rewrite's closing ``` fence]\n");
                    break;
                }
            }

            std::vector<float> step_logits = forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab, &ss);
            n_past += 1;
            if (search_wanted) break; // the marker's last token is committed; pause and search below
            next = sample(smp, step_logits, all_tokens);
        }

        // segment ended: flush whatever is still withheld. An unterminated marker (the model ran
        // into EOS / the budget mid-marker) still counts as a call, same as chat.
        if (search_enabled && !render_hold.empty()) {
            size_t marker_start = 0;
            if (search_scan(render_hold, search_query, marker_start, true) == SEARCH_SCAN_MARKER) {
                search_wanted = true;
            } else {
                emit_piece(render_hold);
            }
            render_hold.clear();
        }

        if (!search_wanted) break; // EOS / closing fence / budget: the reply is complete

        // ---- tool call: pause, search, feed the results back, resume ----
        if (searches_done >= max_searches) {
            fprintf(stderr, "\n[search: round limit (%d) reached -- ending generation]\n", max_searches);
            break;
        }
        searches_done++;
        std::string results;
        fprintf(stderr, "\n[Web Search %d/%d]: %s ...\n", searches_done, max_searches, search_query.c_str());
        const auto search_t0 = std::chrono::steady_clock::now();
        const bool searched = perform_search(search_query, results);
        search_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - search_t0).count();

        // a failed search still injects a note -- the model must know no results arrived, so it
        // can apologize or retry with a different query instead of guessing
        std::string note;
        if (!searched) {
            note = search_failed_block(search_query);
            fprintf(stderr, "[Web Search %d: no usable results -- feeding a failure note]\n", searches_done);
        } else {
            note = search_results_block(search_query, results);
        }

        // feed the block into the live context (KV cache) and keep decoding. perform_search()
        // already capped the results at SEARCH_RESULT_MAX chars; fit whatever is left to the token
        // room actually available (65-token EOS margin kept), so a small n_ctx can't overflow the
        // KV cache. The injected tokens stay out of all_tokens -- that vector mirrors the model's
        // own output (and the --edit-file extraction), not the context.
        const int64_t inject_budget = n_ctx - n_past - 65;
        const std::string inject = (inject_budget >= 16)
            ? search_fit_to_tokens(vc, "\n" + note, inject_budget) : std::string();
        std::vector<int32_t> inject_toks = inject.empty() ? std::vector<int32_t>() : vc.encode(inject);
        if (inject_toks.empty() || (int64_t) inject_toks.size() > inject_budget) {
            fprintf(stderr, "[search: no room left in n_ctx for the results -- ending generation]\n");
            break;
        }
        std::vector<float> resume_logits = prefill(m, hp, kv, ds, backend, n_past, inject_toks, n_vocab,
                                                    backend_name == "vulkan", &ss);
        n_past += (int64_t) inject_toks.size();
        next = sample(smp, resume_logits, all_tokens);
        // loop back: continue the same reply with the results now in context
    }
    auto t3 = std::chrono::steady_clock::now();
    double decode_s = std::max(std::chrono::duration<double>(t3 - t2).count() - search_wait_s, 1e-6);
    render_filter_finish(tf); // flush any withheld tail; close dim colouring if left open
    if (stop_at_closing_fence && !show_thinking && tf.hidden_chars > 0) {
        fprintf(stderr, "\n[thinking hidden: %lld chars -- pass --show-thinking to display it]\n",
                tf.hidden_chars);
    }
    fputc('\n', stdout);

    fprintf(stderr, "decode: %d tokens in %.3fs (%.2f tok/s)\n", n_decoded, decode_s,
            n_decoded / decode_s);
    fprintf(stderr, "\nfull output:\n%s\n", vc.detok(all_tokens).c_str());

    if (!edit_file.empty()) {
        std::vector<int32_t> generated_tokens(all_tokens.begin() + prompt_tokens.size(), all_tokens.end());
        // with --search, use the piece stream instead of the token slice: the marker text of a
        // "[SEARCH: ...]" tool call was withheld from it (a call is never rewrite content), while
        // the slice of committed tokens still contains the marker. Without --search the two are
        // the same text.
        std::string raw = search_enabled ? edit_generated : vc.detok(generated_tokens);
        // --thinking: drop the reasoning trace before extraction -- fences / BEGIN-END markers
        // quoted inside a think block (draft rewrites, restated prompts) must not win over the
        // final answer. No-op when no think tags were emitted.
        std::string answer_text = render_filter_strip_thinking(raw, prompt_opens_think);
        std::string generated = extract_edit_rewrite(answer_text);
        // Show what was extracted vs the model output -- this is the difference between what
        // the model literally said and what we'll actually write to disk, so misbehaving prompts
        // (where extraction picks the wrong block) are easy to diagnose.
        fprintf(stderr, "\n[extract: kept %zu/%zu chars of model output]\n",
                generated.size(), answer_text.size());
        if (answer_text.size() != raw.size()) {
            fprintf(stderr, "[extract: dropped %zu chars of thinking]\n",
                    raw.size() - answer_text.size());
        }

        if (!edit_on) {
            fprintf(stderr, "\n[--edit-file dry run: not writing '%s', pass --edit-on to apply]\n",
                    edit_file.c_str());
        } else {
            std::string backup_path = edit_file + ".original";
            if (!write_file(backup_path, edit_original_content)) {
                fprintf(stderr, "error: could not write backup '%s', aborting edit (original file untouched)\n",
                        backup_path.c_str());
            } else if (!looks_like_plausible_code(generated, edit_original_content.size())) {
                fprintf(stderr,
                    "\nwarning: generated output failed the sanity check (too short, fence-only, and/or\n"
                    "unbalanced braces/parens/brackets -- likely truncated, looped, or malformed). NOT\n"
                    "overwriting '%s'. Original is safe (also backed up at '%s'); try again with a\n"
                    "different prompt or model (or raise -n if the rewrite was simply truncated).\n",
                    edit_file.c_str(), backup_path.c_str());
            } else if (!write_file(edit_file, generated)) {
                fprintf(stderr, "error: could not write '%s' (backup is safe at '%s')\n",
                        edit_file.c_str(), backup_path.c_str());
            } else {
                fprintf(stderr, "\nedit applied: '%s' overwritten (%zu bytes), original backed up to '%s' (%zu bytes)\n",
                        edit_file.c_str(), generated.size(), backup_path.c_str(), edit_original_content.size());
            }
        }
    }

    free_decode_session(ds);
    free_ssm_state(ss);
    ggml_backend_buffer_free(kv.buffer);
    ggml_free(kv.ctx);
    ggml_backend_buffer_free(ws.buffer);
    ggml_free(ws.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
