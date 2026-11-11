/*
 * chat.c — interactive ChatML chat, pure C, no CGO.
 *
 * Uses the engine directly: tokenizer.c, qwen_model.c, and the selected compute
 * backend. This replaces the Go front-end for CLI use; the CGO bridge remains
 * available for an HTTP/API layer but is no longer on the critical path.
 *
 *   ./moe-chat <model.gguf> [--backend vulkan|cpu] [--ctx N] [--cache-mb N]
 *                            [--threads N] [--max-new N] [--system "..."]
 *                            [--no-direct] [--debug]
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "autotune.h"
#include "compute_backend.h"
#include "memtrack.h"
#include "qwen_model.h"
#include "tokenizer.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LINE_MAX_LEN  8192
#define PROMPT_MAX    65536
#define MAX_IDS       16384

#define COLOR_BOLD  "\033[1m"
#define COLOR_DIM   "\033[2m"
#define COLOR_OFF   "\033[0m"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void usage(void)
{
    fprintf(stderr,
        "usage: moe-chat <model.gguf> [flags]\n"
        "\n"
        "  --backend vulkan|cpu   math backend (default vulkan)\n"
        "  --cache-mb N           expert cache MiB (default: auto)\n"
        "  --ctx N                context length   (default: auto)\n"
        "  --threads N            I/O workers      (default 6)\n"
        "  --max-new N            max tokens per reply (default 512)\n"
        "  --system \"...\"         system prompt\n"
        "  --no-direct            disable O_DIRECT\n"
        "  --debug                tokenizer diagnostics\n");
}

static int parse_int(const char *s, int dflt)
{
    char *end = NULL;
    long  v   = strtol(s, &end, 10);
    return (end && *end == '\0' && v >= 0) ? (int)v : dflt;
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *system     = NULL;
    const char *backend    = "vulkan";
    int         cache_mb   = 0;
    int         ctx_len    = 0;
    int         threads    = 6;
    int         max_new    = 512;
    int         no_direct  = 0;
    int         debug      = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            usage();
            return 0;
        } else if (!strcmp(a, "--backend") && i + 1 < argc) {
            backend = argv[++i];
        } else if (!strcmp(a, "--cache-mb") && i + 1 < argc) {
            cache_mb = parse_int(argv[++i], 0);
        } else if (!strcmp(a, "--ctx") && i + 1 < argc) {
            ctx_len = parse_int(argv[++i], 0);
        } else if (!strcmp(a, "--threads") && i + 1 < argc) {
            threads = parse_int(argv[++i], 6);
        } else if (!strcmp(a, "--max-new") && i + 1 < argc) {
            max_new = parse_int(argv[++i], 512);
        } else if (!strcmp(a, "--system") && i + 1 < argc) {
            system = argv[++i];
        } else if (!strcmp(a, "--no-direct")) {
            no_direct = 1;
        } else if (!strcmp(a, "--debug")) {
            debug = 1;
        } else if (a[0] != '-' && !model_path) {
            model_path = a;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a);
            usage();
            return 2;
        }
    }
    if (!model_path) {
        usage();
        return 2;
    }

    /* Auto defaults come from the same budgeter the Go CLI used. */
    int auto_cache = 0, auto_ctx = 0;
    calculate_optimal_cache(0, &auto_cache, &auto_ctx);
    const int avail_mb = (int)(moe_mem_available() / (1024 * 1024));
    if (cache_mb == 0) {
        cache_mb = auto_cache;
    }
    if (ctx_len == 0) {
        ctx_len = auto_ctx;
    }
    printf("[Auto-Cache] Using %d MB for experts, %d context length based on "
           "%d MB available RAM\n", cache_mb, ctx_len, avail_mb);

    qwen_model_t *m = qwen_load(model_path, (size_t)cache_mb * 1024 * 1024,
                                threads, ctx_len, !no_direct);
    if (!m) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    if (!strcmp(backend, "vulkan")) {
        qwen_set_backend(m, compute_backend_vulkan());
    }
    printf("[backend] %s, vocab %d, ctx %d\n",
           qwen_backend_name(m), qwen_vocab_size(m), qwen_max_seq(m));

    tok_t tok;
    if (tok_load(&tok, model_path) != 0) {
        fprintf(stderr, "failed to load tokenizer\n");
        qwen_free(m);
        return 1;
    }

    int end_ids[8];
    int n_end = tok_encode(&tok, "<|im_end|>", end_ids, 8);
    if (n_end <= 0) {
        fprintf(stderr, "cannot encode <|im_end|>\n");
        return 1;
    }
    const int im_end = end_ids[0];
    if (debug) {
        printf("[debug] <|im_end|> -> id %d (%d token(s))\n", im_end, n_end);
    }

    char  *line   = (char *)malloc(LINE_MAX_LEN);
    char  *prompt = (char *)malloc(PROMPT_MAX);
    int   *ids    = (int *)malloc(MAX_IDS * sizeof(int));
    if (!line || !prompt || !ids) {
        fprintf(stderr, "[OOM] chat buffers\n");
        return 1;
    }

    int first = 1;
    for (;;) {
        printf("\n%suser>%s ", COLOR_BOLD, COLOR_OFF);
        fflush(stdout);

        if (!fgets(line, LINE_MAX_LEN, stdin)) {
            printf("\n");
            break;
        }
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) {
            continue;
        }
        if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) {
            break;
        }
        if (!strcmp(line, "/reset")) {
            qwen_reset(m);
            first = 1;
            printf("%s[context reset]%s\n", COLOR_DIM, COLOR_OFF);
            continue;
        }

        int wrote;
        if (first) {
            wrote = system
                ? snprintf(prompt, PROMPT_MAX,
                           "<|im_start|>system\n%s<|im_end|>\n"
                           "<|im_start|>user\n%s<|im_end|>\n"
                           "<|im_start|>assistant\n", system, line)
                : snprintf(prompt, PROMPT_MAX,
                           "<|im_start|>user\n%s<|im_end|>\n"
                           "<|im_start|>assistant\n", line);
            first = 0;
        } else {
            /* The previous turn ended with <|im_end|> in the KV cache. */
            wrote = snprintf(prompt, PROMPT_MAX,
                             "\n<|im_start|>user\n%s<|im_end|>\n"
                             "<|im_start|>assistant\n", line);
        }
        if (wrote < 0 || wrote >= PROMPT_MAX) {
            fprintf(stderr, "prompt too long\n");
            continue;
        }

        int n_ids = tok_encode(&tok, prompt, ids, MAX_IDS);
        if (n_ids <= 0) {
            fprintf(stderr, "tokenize failed\n");
            continue;
        }
        if (qwen_pos(m) + n_ids + max_new >= qwen_max_seq(m)) {
            fprintf(stderr, "\n%s[context full: %d/%d; use /reset]%s\n",
                    COLOR_DIM, qwen_pos(m), qwen_max_seq(m), COLOR_OFF);
            continue;
        }
        if (debug) {
            printf("%s[debug] turn %d tokens, pos %d%s\n",
                   COLOR_DIM, n_ids, qwen_pos(m), COLOR_OFF);
        }

        const double t_pre0 = now_ms();
        if (qwen_prefill(m, (const uint32_t *)ids, n_ids) != 0) {
            fprintf(stderr, "prefill failed\n");
            continue;
        }
        const double t_pre1 = now_ms();

        printf("%sassistant>%s ", COLOR_BOLD, COLOR_OFF);
        fflush(stdout);

        int    last    = -1;
        int    stopped = 0;
        int    emitted = 0;
        for (int k = 0; k < max_new; k++) {
            const int id = qwen_argmax(m);
            if (id < 0) {
                break;
            }
            if (id == im_end) {
                last    = id;
                stopped = 1;
                break;
            }

            char text[256];
            int  n = tok_decode(&tok, id, text, sizeof(text));
            if (n > 0) {
                fwrite(text, 1, (size_t)n, stdout);
                fflush(stdout);
            }
            last = id;
            emitted++;

            if (k + 1 >= max_new) {
                break;   /* last emitted; forwarded after the loop */
            }
            if (qwen_forward(m, id) != 0) {
                break;
            }
        }
        const double t_dec1 = now_ms();
        printf("\n");
        printf("%s[prefill] %d tok in %.1f s (%.1f tok/s)  |  "
               "decode %d tok in %.1f s (%.2f tok/s)%s\n",
               COLOR_DIM, n_ids, (t_pre1 - t_pre0) / 1000.0,
               n_ids * 1000.0 / (t_pre1 - t_pre0),
               emitted, (t_dec1 - t_pre1) / 1000.0,
               emitted > 0 ? emitted * 1000.0 / (t_dec1 - t_pre1) : 0.0,
               COLOR_OFF);

        /* Close the turn: push the last emitted token (the C loop stops
         * before forwarding it), then <|im_end|>, so the next prompt is
         * contiguous ChatML. */
        if (last >= 0 && qwen_pos(m) < qwen_max_seq(m)) {
            if (!stopped) {
                qwen_forward(m, last);
            }
            if (qwen_pos(m) < qwen_max_seq(m)) {
                qwen_forward(m, im_end);
            }
        } else if (qwen_pos(m) < qwen_max_seq(m)) {
            qwen_forward(m, im_end);
        }
    }

    free(line);
    free(prompt);
    free(ids);
    tok_free(&tok);
    qwen_free(m);
    return 0;
}
