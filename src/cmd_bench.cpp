// `ggml-loader bench <model.gguf> [--backend cpu|vulkan] [-p n_prompt] [-n n_gen] [-r repetitions]`:
// repeated, averaged pp/tg throughput benchmark on synthetic tokens (not real text) -- matches
// llama-bench's pp128/tg64 methodology so the numbers are directly comparable. One untimed
// warmup cycle runs first so one-time costs (e.g. lazily-compiled Vulkan pipelines on first use)
// don't leak into the reported stats.
#include "commands.h"
#include "model.h"
#include "ollama_model.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [model.gguf] [--backend cpu|vulkan] [-p n_prompt] [-n n_gen] [-r repetitions] [--flash-attn]\n"
        "  defaults: -p 128 -n 64 -r 3, flash-attn off\n", argv0);
}

struct stats { double mean; double stddev; };

static stats compute_stats(const std::vector<double> & xs) {
    double sum = 0.0;
    for (double x : xs) sum += x;
    double mean = sum / (double) xs.size();
    double var = 0.0;
    for (double x : xs) var += (x - mean) * (x - mean);
    var /= xs.size() > 1 ? (double)(xs.size() - 1) : 1.0;
    return { mean, sqrt(var) };
}

// runs one pp(n_pp)+tg(n_tg) cycle from a clean n_past=0 and returns {pp_tok_per_s, tg_tok_per_s}
static std::pair<double, double> run_cycle(const model & m, const hparams & hp, kv_cache & kv,
                                            ssm_state & ss, decode_session & ds, ggml_backend_t backend,
                                            int n_pp, int n_tg, int64_t n_vocab) {
    std::vector<int32_t> pp_tokens(n_pp);
    for (int i = 0; i < n_pp; i++) pp_tokens[i] = (int32_t)(i % (n_vocab - 1)) + 1; // avoid id 0

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = forward_step(m, hp, kv, ds, backend, 0, pp_tokens, n_vocab, &ss);
    auto t1 = std::chrono::steady_clock::now();
    double pp_s = std::chrono::duration<double>(t1 - t0).count();

    int64_t n_past = n_pp;
    int32_t next = argmax(logits);
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_tg; i++) {
        std::vector<float> step_logits = forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab, &ss);
        n_past += 1;
        next = argmax(step_logits);
    }
    auto t3 = std::chrono::steady_clock::now();
    double tg_s = std::chrono::duration<double>(t3 - t2).count();

    return { n_pp / pp_s, n_tg / tg_s };
}

int cmd_bench(int argc, char ** argv) {
    std::string model_path = "Phi-4-mini-instruct-Q4_K_M.gguf";
    bool model_path_set = false;
    std::string backend_name = "cpu";
    int n_pp = 128;
    int n_tg = 64;
    int n_reps = 3;
    bool flash_attn = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "-p" && i + 1 < argc) {
            n_pp = atoi(argv[++i]);
        } else if (a == "-n" && i + 1 < argc) {
            n_tg = atoi(argv[++i]);
        } else if (a == "-r" && i + 1 < argc) {
            n_reps = atoi(argv[++i]);
        } else if (a == "--flash-attn") {
            flash_attn = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!model_path_set) {
            model_path = a;
            model_path_set = true;
        }
    }

    if ((int64_t) n_pp + n_tg > N_CTX_MAX) {
        fprintf(stderr, "error: -p (%d) + -n (%d) exceeds the %lld-token KV cache\n",
                n_pp, n_tg, (long long) N_CTX_MAX);
        return 1;
    }

    model_path = resolve_model_ref(model_path);
    ggml_backend_t backend = init_backend(backend_name, 8);

    struct gguf_context * gctx = nullptr;
    weights_store ws = load_weights(model_path.c_str(), backend, &gctx);

    hparams hp = load_hparams(gctx);
    model   m  = load_model(ws.ctx, hp);
    const int64_t n_vocab = m.token_embd->ne[1];

    kv_cache kv = init_kv_cache(hp, backend);
    ssm_state ss = init_ssm_state(hp, backend); // no-op struct for every arch except nemotron_h
    decode_session ds = init_decode_session(backend, flash_attn);

    fprintf(stderr, "model=%s arch=%s backend=%s flash_attn=%d n_layer=%lld -- pp%d, tg%d, %d reps (+1 warmup)\n",
            model_path.c_str(), hp.arch_name.c_str(), backend_name.c_str(), (int) flash_attn,
            (long long) hp.n_layer, n_pp, n_tg, n_reps);

    fprintf(stderr, "warmup...\n");
    run_cycle(m, hp, kv, ss, ds, backend, n_pp, n_tg, n_vocab); // untimed, absorbs first-use costs

    std::vector<double> pp_tps, tg_tps;
    for (int rep = 0; rep < n_reps; rep++) {
        auto [pp, tg] = run_cycle(m, hp, kv, ss, ds, backend, n_pp, n_tg, n_vocab);
        pp_tps.push_back(pp);
        tg_tps.push_back(tg);
        fprintf(stderr, "  rep %d/%d: pp %.2f tok/s, tg %.2f tok/s\n", rep + 1, n_reps, pp, tg);
    }

    stats pp_stats = compute_stats(pp_tps);
    stats tg_stats = compute_stats(tg_tps);

    printf("\n| model      | backend | test  |          t/s |\n");
    printf("| ---------- | ------- | ----- | ------------ |\n");
    printf("| %-10s | %-7s | pp%-3d | %6.2f ± %5.2f |\n",
           hp.arch_name.c_str(), backend_name.c_str(), n_pp, pp_stats.mean, pp_stats.stddev);
    printf("| %-10s | %-7s | tg%-3d | %6.2f ± %5.2f |\n",
           hp.arch_name.c_str(), backend_name.c_str(), n_tg, tg_stats.mean, tg_stats.stddev);

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
