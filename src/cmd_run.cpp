// `ggml-loader run <model.gguf> [-p/--prompt "text"] [-n/--n-predict N] [--backend cpu|vulkan]`:
// one-shot raw completion (no chat template), with prefill/decode timing -- the mode used to
// benchmark against llama-cli/llama-bench on the same prompt, and cpu vs vulkan against each other.
//
// `--edit-file <path> [--edit-on]`: wraps <path>'s contents + the -p instruction into a
// "rewrite this whole file" prompt, runs completion, and (with --edit-on) backs the original up
// to <path>.original and overwrites <path> with the model's output -- but only if the output
// passes a basic sanity check (non-trivial length, balanced braces/parens/brackets), since these
// small local models have no diff/code-edit fine-tuning and raw full-file rewrites can truncate
// or hallucinate. Without --edit-on the rewrite is just printed, nothing is touched on disk.
#include "commands.h"
#include "model.h"
#include "ollama_model.h"

#include "ggml-backend.h"
#include "gguf.h"

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
        "           [--edit-file path] [--edit-on]\n"
        "  no -p: BOS-seeded unconditional free-run\n"
        "  temp<=0 (default): greedy/deterministic, same as before these flags existed\n"
        "  --edit-file path: wrap path's contents + -p into a whole-file-rewrite prompt\n"
        "  --edit-on: with --edit-file, back path up to path.original and overwrite path with\n"
        "             the model's output if it passes a basic sanity check (dry-run without it)\n",
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
    long curly = 0, paren = 0, square = 0;
    for (char c : generated) {
        if (c == '{') curly++; else if (c == '}') curly--;
        else if (c == '(') paren++; else if (c == ')') paren--;
        else if (c == '[') square++; else if (c == ']') square--;
    }
    return curly == 0 && paren == 0 && square == 0;
}

// small local models routinely ignore "no code fences" and wrap the rewrite in a ```lang ... ```
// block anyway (sometimes followed by rambling repeated ``` tokens once past the real content,
// having missed eos). Strip a leading fence and cut at the *first* matching close, discarding
// anything after it, rather than trusting the model's own stop point.
static std::string strip_code_fence(const std::string & s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos || s.compare(start, 3, "```") != 0) return s;
    size_t nl = s.find('\n', start);
    if (nl == std::string::npos) return s;
    size_t close = s.find("```", nl + 1);
    std::string body = (close == std::string::npos) ? s.substr(nl + 1) : s.substr(nl + 1, close - (nl + 1));
    // trim a single trailing newline left by the fence line, if present
    if (!body.empty() && body.back() == '\n') body.pop_back();
    return body;
}

int cmd_run(int argc, char ** argv) {
    std::string model_path = "Phi-4-mini-instruct-Q4_K_M.gguf";
    bool model_path_set = false;
    std::string prompt;
    std::string backend_name = "cpu";
    int max_new_tokens = 32;
    bool n_set = false;
    bool flash_attn = false;
    std::string edit_file;
    bool edit_on = false;
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
        } else if (a == "--edit-file" && i + 1 < argc) {
            edit_file = argv[++i];
        } else if (a == "--edit-on") {
            edit_on = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!model_path_set) {
            model_path = a;
            model_path_set = true;
        }
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
        prompt = "Here is the file `" + edit_file + "`:\n```\n" + edit_original_content +
                 "\n```\n\n" + instruction +
                 "\n\nOutput the complete updated file and nothing else, no explanation, no code fences.\n";
        if (!n_set) {
            // rough token estimate (~3 bytes/token) plus headroom for growth (e.g. added debug code)
            int64_t est = (int64_t) (edit_original_content.size() / 3) + 256;
            if (est > N_CTX_MAX) est = N_CTX_MAX;
            max_new_tokens = (int) est;
        }
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
    // some models occasionally emit the literal "<|endoftext|>" token distinct from hp.eos_id;
    // treat it as a stop signal too if the vocab has it (see cmd_chat.cpp for where this was found)
    auto eot_it = vc.token_to_id.find("<|endoftext|>");
    const int32_t tok_eot = eot_it != vc.token_to_id.end() ? eot_it->second : -1;

    std::vector<int32_t> prompt_tokens;
    if (hp.bos_id >= 0) prompt_tokens.push_back((int32_t) hp.bos_id); // some tokenizers have none
    if (!prompt.empty()) {
        std::vector<int32_t> enc = vc.encode(prompt);
        prompt_tokens.insert(prompt_tokens.end(), enc.begin(), enc.end());
        fprintf(stderr, "prompt: \"%s\" -> %zu tokens (round-trip: \"%s\")\n",
                prompt.c_str(), enc.size(), vc.detok(enc).c_str());
    }

    if ((int64_t) prompt_tokens.size() + max_new_tokens > N_CTX_MAX) {
        fprintf(stderr, "error: prompt (%zu) + max_new_tokens (%d) exceeds the %lld-token KV cache\n",
                prompt_tokens.size(), max_new_tokens, (long long) N_CTX_MAX);
        return 1;
    }

    fprintf(stderr, "backend=%s n_embd=%lld n_head=%lld n_head_kv=%lld n_layer=%lld n_ff=%lld n_rot=%lld n_vocab=%lld\n",
            backend_name.c_str(), (long long) hp.n_embd, (long long) hp.n_head, (long long) hp.n_head_kv,
            (long long) hp.n_layer, (long long) hp.n_ff, (long long) hp.n_rot, (long long) n_vocab);

    kv_cache kv = init_kv_cache(hp, backend);
    ssm_state ss = init_ssm_state(hp, backend); // no-op struct for every arch except nemotron_h
    decode_session ds = init_decode_session(backend, flash_attn);
    const int64_t head_dim = hp.n_embd_head;
    const double kv_cache_mib =
        (double)(hp.n_layer * 2 * N_CTX_MAX * hp.n_head_kv * head_dim * 2) / (1024.0 * 1024.0);
    fprintf(stderr, "KV cache: %lld ctx x %lld layers, f16 -> %.0f MiB\n",
            (long long) N_CTX_MAX, (long long) hp.n_layer, kv_cache_mib);

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

    // decode: one new token at a time, timed separately from prefill for a pp/tg-style comparison
    int n_decoded = 0;
    auto t2 = std::chrono::steady_clock::now();
    for (int step = 0; step < max_new_tokens; step++) {
        if (next == hp.eos_id || next == tok_eot) {
            fprintf(stderr, "[eos]\n");
            break;
        }
        all_tokens.push_back(next);
        n_decoded++;

        std::string piece = vc.detok({ next });
        fputs(piece.c_str(), stdout);
        fflush(stdout);

        std::vector<float> step_logits = forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab, &ss);
        n_past += 1;
        next = sample(smp, step_logits, all_tokens);
    }
    auto t3 = std::chrono::steady_clock::now();
    double decode_s = std::chrono::duration<double>(t3 - t2).count();
    fputc('\n', stdout);

    fprintf(stderr, "decode: %d tokens in %.3fs (%.2f tok/s)\n", n_decoded, decode_s,
            n_decoded / decode_s);
    fprintf(stderr, "\nfull output:\n%s\n", vc.detok(all_tokens).c_str());

    if (!edit_file.empty()) {
        std::vector<int32_t> generated_tokens(all_tokens.begin() + prompt_tokens.size(), all_tokens.end());
        std::string generated = strip_code_fence(vc.detok(generated_tokens));

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
                    "\nwarning: generated output failed the sanity check (too short and/or unbalanced\n"
                    "braces/parens/brackets -- likely truncated or malformed). NOT overwriting '%s'.\n"
                    "Original is safe (also backed up at '%s'); try again with a larger -n or a\n"
                    "different prompt.\n", edit_file.c_str(), backup_path.c_str());
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
