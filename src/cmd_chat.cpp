// `ggml-loader chat <model.gguf> [--system "text"]`: interactive multi-turn chat.
// Chat formatting (append_chat_turn/append_generation_prompt/turn_end_token) is
// architecture-specific -- see model.cpp -- so this file doesn't hardcode any model family's
// template. Multi-turn history lives entirely in the persistent KV cache -- each turn only
// encodes/prefills the newest user message, not the whole conversation.
#include "commands.h"
#include "model.h"
#include "ollama_model.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [model.gguf] [--system \"text\"] [--backend cpu|vulkan] [--flash-attn]\n"
        "           [--temp F] [--top-k N] [--top-p F] [--repeat-penalty F] [--repeat-last-n N] [--seed N]\n"
        "           [--thinking | --no-thinking]\n"
        "  temp<=0 (default): greedy/deterministic, same as before these flags existed\n"
        "  --thinking/--no-thinking only matter for reasoning models (auto-detected from the\n"
        "  model's own chat_template); default is --thinking (let the model decide whether to\n"
        "  reason). --no-thinking forces an empty <think></think> block to suppress it.\n", argv0);
}

int cmd_chat(int argc, char ** argv) {
    std::string model_path = "Phi-4-mini-instruct-Q4_K_M.gguf";
    bool model_path_set = false;
    std::string system_prompt;
    std::string backend_name = "cpu";
    bool flash_attn = false;
    bool enable_thinking = true;
    sampler_params sp;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "--flash-attn") {
            flash_attn = true;
        } else if (a == "--thinking") {
            enable_thinking = true;
        } else if (a == "--no-thinking") {
            enable_thinking = false;
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
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!model_path_set) {
            model_path = a;
            model_path_set = true;
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
    const int32_t tok_end = turn_end_token(hp, vc);
    // some models (observed on qwen3) occasionally emit the literal "<|endoftext|>" token mid-turn
    // instead of the chat template's own end-of-turn token; treat it as a stop signal too if the
    // vocab has it, rather than letting generation run past it into repetition
    auto eot_it = vc.token_to_id.find("<|endoftext|>");
    const int32_t tok_eot = eot_it != vc.token_to_id.end() ? eot_it->second : -1;

    kv_cache kv = init_kv_cache(hp, backend);
    ssm_state ss = init_ssm_state(hp, backend); // no-op struct for every arch except nemotron_h
    decode_session ds = init_decode_session(backend, flash_attn);

    fprintf(stderr, "%s loaded (%s, %lld layers, backend=%s%s). Type a message and press Enter "
                     "(Ctrl-D or 'exit' to quit).\n",
            model_path.c_str(), hp.arch_name.c_str(), (long long) hp.n_layer, backend_name.c_str(),
            hp.supports_thinking ? (enable_thinking ? ", thinking=on" : ", thinking=off") : "");

    std::vector<int32_t> pending;
    if (hp.bos_id >= 0) pending.push_back((int32_t) hp.bos_id); // some tokenizers have none
    if (!system_prompt.empty()) {
        append_chat_turn(hp, vc, pending, "system", system_prompt);
    }

    int64_t n_past = 0;
    const int max_reply_tokens = 512;
    std::vector<int32_t> history; // recent tokens for repeat_penalty; KV cache handles real context

    while (true) {
        printf("> ");
        fflush(stdout);
        std::string line;
        if (!std::getline(std::cin, line)) break; // EOF (Ctrl-D)
        if (line == "exit" || line == "quit") break;
        if (line.empty()) continue;

        append_chat_turn(hp, vc, pending, "user", line);
        append_generation_prompt(hp, vc, pending, enable_thinking);

        if (n_past + (int64_t) pending.size() + max_reply_tokens > N_CTX_MAX) {
            fprintf(stderr, "\n[context window full (%lld tokens) -- restart to continue]\n",
                    (long long) N_CTX_MAX);
            break;
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
        int n_reply = 0;
        auto t2 = std::chrono::steady_clock::now();
        while (n_reply < max_reply_tokens && next != tok_end && next != hp.eos_id && next != tok_eot) {
            std::string piece = vc.detok({ next });
            fputs(piece.c_str(), stdout);
            fflush(stdout);
            history.push_back(next);

            std::vector<float> step_logits = forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab, &ss);
            n_past += 1;
            n_reply++;
            next = sample(smp, step_logits, history);
        }
        // commit the terminator itself to the cache so history matches the chat template
        // exactly for the next turn's context
        forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab, &ss);
        n_past += 1;
        auto t3 = std::chrono::steady_clock::now();
        double decode_s = std::chrono::duration<double>(t3 - t2).count();

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
    return 0;
}
