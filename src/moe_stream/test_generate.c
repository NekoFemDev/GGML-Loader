/*
 * test_generate.c — end-to-end token generation with the scalar CPU model.
 *
 * Usage:
 *   ./moe_generate <gguf> [--cache-mb N] [--threads N] [--ctx N] [--ngen N]
 *                         [--direct] [--ids "1,2,3"] [prompt...]
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <unistd.h>

#include "autotune.h"
#include "compute_backend.h"
#include "qwen_model.h"
#include "tokenizer.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char *path     = "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf";
    const char *prompt   = "def fibonacci(n):\n";
    const char *ids_arg  = NULL;
    size_t cache_mb      = 512;
    int    threads       = 4;
    int    max_seq       = 256;
    int    ngen          = 8;
    int    direct        = 0;
    int    legacy        = 0;
    int    use_vulkan    = 0;
    int    auto_cfg      = 0;
    int    npos          = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cache-mb") && i + 1 < argc) {
            cache_mb = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            max_seq = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ngen") && i + 1 < argc) {
            ngen = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ids") && i + 1 < argc) {
            ids_arg = argv[++i];
        } else if (!strcmp(argv[i], "--direct")) {
            direct = 1;
        } else if (!strcmp(argv[i], "--legacy")) {
            legacy = 1;
        } else if (!strcmp(argv[i], "--vulkan")) {
            use_vulkan = 1;
        } else if (!strcmp(argv[i], "--auto")) {
            auto_cfg = 1;
        } else if (strncmp(argv[i], "--", 2) != 0) {
            if (npos == 0) {
                path = argv[i];
            } else if (npos == 1) {
                prompt = argv[i];
            }
            npos++;
        }
    }

    if (auto_cfg) {
        long pages = sysconf(_SC_PHYS_PAGES);
        long psize = sysconf(_SC_PAGESIZE);
        if (pages > 0 && psize > 0) {
            size_t ram = (size_t)pages * (size_t)psize;
            calculate_optimal_cache(ram, (int *)&cache_mb, &max_seq);
            fprintf(stderr, "[autotune] RAM %zu MiB -> cache %zu MiB, ctx %d\n",
                    ram / (1024 * 1024), cache_mb, max_seq);
        }
    }

    tok_t tok;
    if (tok_load(&tok, path) != 0) {
        fprintf(stderr, "failed to load tokenizer\n");
        return 1;
    }
    printf("vocab: %u tokens\n", tok_count(&tok));

    qwen_model_t *m = qwen_load(path, cache_mb * 1024 * 1024, threads, max_seq, direct);
    if (!m) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    int ids[512];
    int n = 0;

    if (ids_arg) {
        char *copy = strdup(ids_arg);
        char *save = NULL;
        char *p = strtok_r(copy, ",", &save);
        while (p && n < 512) {
            ids[n++] = atoi(p);
            p = strtok_r(NULL, ",", &save);
        }
        free(copy);
    } else {
        n = tok_encode(&tok, prompt, ids, 512);
        if (n <= 0) {
            fprintf(stderr, "tokenize failed\n");
            return 1;
        }
    }

    printf("\nprompt: \"%s\"\n", ids_arg ? "(explicit ids)" : prompt);
    printf("tokens (%d): ", n);
    for (int i = 0; i < n; i++) {
        printf("%d ", ids[i]);
        if (i < 12) {
            char buf[128];
            int k = tok_decode(&tok, ids[i], buf, sizeof(buf) - 1);
            if (k > 0) {
                buf[k] = '\0';
                printf("[%s] ", buf);
            }
        }
    }
    printf("\n\n");

    if (use_vulkan) {
        qwen_set_backend(m, compute_backend_vulkan());
        fprintf(stderr, "[backend] %s\n", qwen_backend_name(m));
    }

    if (n + ngen >= max_seq) {
        fprintf(stderr, "prompt + generation exceeds context\n");
        return 1;
    }

    qwen_reset(m);
    const double t0 = now_ms();
    int prc;
    if (legacy) {
        prc = 0;
        for (int i = 0; i < n && prc == 0; i++) {
            prc = qwen_forward(m, ids[i]);
        }
    } else {
        prc = qwen_prefill(m, (const uint32_t *)ids, n);
    }
    const double t1 = now_ms();
    if (prc != 0) {
        fprintf(stderr, "prefill failed: %d\n", prc);
        return 1;
    }
    printf("prefill%s: %d tokens in %.1f ms (%.1f ms/token)\n\n",
           legacy ? " (legacy, token-by-token)" : " (layer sweep)",
           n, t1 - t0, (t1 - t0) / n);

    printf("generated:\n---\n");
    double tg0 = now_ms();
    int    emitted = 0;

    for (int k = 0; k < ngen; k++) {
        int next = qwen_argmax(m);
        if (next < 0) {
            break;
        }
        char buf[256];
        int  nb = tok_decode(&tok, next, buf, sizeof(buf) - 1);
        if (nb > 0) {
            buf[nb] = '\0';
            fputs(buf, stdout);
        }
        emitted++;
        if (qwen_pos(m) >= max_seq) {
            break;
        }
        double tf = now_ms();
        if (qwen_forward(m, next) != 0) {
            break;
        }
        fprintf(stderr, "[gen %d] id=%-7d %6.0f ms\n", k, next, now_ms() - tf);
    }
    double tg1 = now_ms();

    printf("\n---\n");
    printf("generated %d tokens in %.1f s (%.2f tok/s, %.0f ms/token)\n\n",
           emitted, (tg1 - tg0) / 1000.0,
           emitted * 1000.0 / (tg1 - tg0), (tg1 - tg0) / emitted);

    qwen_stats_t st;
    qwen_get_stats(m, &st);
    printf("experts fetched: %llu (%.2f GiB), forward %.0f ms/token\n",
           (unsigned long long)st.experts_fetched,
           (double)st.expert_bytes / (1024.0 * 1024.0 * 1024.0),
           qwen_last_ms(m));

    qwen_free(m);
    tok_free(&tok);
    return 0;
}
