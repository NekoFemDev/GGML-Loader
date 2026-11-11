/*
 * test_moe_stream.c — verification + end-to-end benchmark against the real GGUF.
 *
 * Checks:
 *   1. GGUF index resolves expert tensors and the 884736-byte stride
 *   2. O_DIRECT and buffered reads are byte-identical
 *   3. LRU round-trip, duplicate collapse, pin accounting
 *   4. eviction under pressure
 *   5. streaming fetch (callback) delivers every unique expert once
 *
 * Benchmark:
 *   prefill : 100-token prompt, full 48-layer sweep, all 128 experts/layer
 *   decode  : 10 generated tokens, top-8 per layer, LRU-resident
 *
 * Usage: moe_stream_test [gguf] [--direct] [--cache-mb N] [--threads N]
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "expert_loader.h"
#include "gguf_index.h"
#include "lru_cache.h"
#include "moe_stream.h"

#define PREFILL_TOKENS 100
#define DECODE_TOKENS  10

static uint64_t fnv1a(const void *buf, size_t n)
{
    const uint8_t *p = (const uint8_t *)buf;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ---- deterministic Zipf router ----------------------------------------- */

static void zipf_cdf(double *cdf, int n, double s)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += 1.0 / pow((double)i + 1.0, s);
        cdf[i] = sum;
    }
    for (int i = 0; i < n; i++) {
        cdf[i] /= sum;
    }
}

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t x = (*state += 0x9E3779B97F4A7C15ull);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static uint32_t zipf_sample(const double *cdf, int n, uint64_t *state)
{
    double u = (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (uint32_t)lo;
}

/* routing_map[layer][token][k] = expert id */
static uint32_t *build_routing_map(const moe_plan_t *plan, int tokens,
                                   const double *cdf, uint64_t seed)
{
    size_t topk = (size_t)plan->n_expert_used;
    size_t n    = (size_t)plan->n_layers * (size_t)tokens * topk;
    uint32_t *rm = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!rm) {
        return NULL;
    }

    uint64_t st = seed;
    for (int l = 0; l < plan->n_layers; l++) {
        for (int t = 0; t < tokens; t++) {
            for (size_t k = 0; k < topk; k++) {
                size_t at = ((size_t)l * tokens + (size_t)t) * topk + k;
                rm[at] = zipf_sample(cdf, (int)plan->n_expert, &st);
            }
        }
    }
    return rm;
}

/* ---- helpers ------------------------------------------------------------ */

static void weights_from_expert(const moe_expert_t *e, moe_weights_t *w)
{
    w->gate       = e->gate;
    w->gate_bytes = e->gate_bytes;
    w->up         = e->up;
    w->up_bytes   = e->up_bytes;
    w->down       = e->down;
    w->down_bytes = e->down_bytes;
}

typedef struct {
    int      expect;
    int      seen;
    uint64_t sum;
    int      bad_slot;
} stream_probe_t;

static int stream_probe_cb(void *user, const moe_expert_t *e)
{
    stream_probe_t *p = (stream_probe_t *)user;
    p->seen++;
    p->sum ^= fnv1a(e->gate, e->gate_bytes);
    if (e->slot < 0 || e->slot >= p->expect) {
        p->bad_slot = 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char *path = "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf";
    size_t cache_mb = 1024;
    int    threads  = 4;
    int    direct   = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--direct")) {
            direct = 1;
        } else if (!strcmp(argv[i], "--cache-mb") && i + 1 < argc) {
            cache_mb = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else {
            path = argv[i];
        }
    }

    printf("=== moe_stream verification + benchmark ===\n");
    printf("gguf: %s\n", path);
    printf("cache: %zu MiB, threads: %d, direct-io: %s\n\n",
           cache_mb, threads, direct ? "on" : "off");

    /* ---- 1. index + geometry ---- */
    gguf_index_t idx;
    int rc = gguf_index_load(&idx, path);
    if (rc != 0) {
        fprintf(stderr, "FAIL: gguf_index_load: %d\n", rc);
        return 1;
    }
    printf("[index] arch=%s layers=%u embd=%u ff_exp=%u experts=%u top%u\n",
           idx.arch, idx.n_layer, idx.n_embd, idx.n_ff_exp, idx.n_expert, idx.n_expert_used);
    printf("[index] tensors=%u data_start=%" PRIu64 " (mod 4096 = %" PRIu64 ")\n",
           idx.count, idx.data_start, idx.data_start % 4096);

    moe_plan_t plan;
    rc = moe_plan_build(&idx, &plan);
    if (rc != 0) {
        fprintf(stderr, "FAIL: moe_plan_build: %d\n", rc);
        return 1;
    }
    moe_plan_print(&plan);

    if (plan.n_layers != 48 || plan.n_expert != 128 || plan.n_expert_used != 8) {
        fprintf(stderr, "FAIL: geometry mismatch\n");
        return 1;
    }
    if (plan.layers[0].gate.stride != 884736 || plan.layers[0].expert_bytes != 2654208) {
        fprintf(stderr, "FAIL: unexpected expert geometry\n");
        return 1;
    }
    printf("[index] geometry OK: 884736 B/slice, 2654208 B/expert (2.53 MiB)\n\n");

    /* ---- 2. O_DIRECT vs buffered equivalence ---- */
    {
        const int layer = 11, expert = 7;
        const moe_layer_t *ml = &plan.layers[layer];
        uint64_t off = ml->gate.base + (uint64_t)expert * ml->gate.stride;
        size_t   len = (size_t)ml->gate.stride;

        uint8_t *a = (uint8_t *)malloc(len);
        uint8_t *b = (uint8_t *)malloc(len);
        expert_io_t bio, dio;
        memset(&bio, 0, sizeof(bio));
        memset(&dio, 0, sizeof(dio));
        bio.fd = -1;
        dio.fd = -1;

        if (!a || !b) {
            fprintf(stderr, "FAIL: oom\n");
            return 1;
        }
        if (expert_io_open(&bio, path, 0) != 0) {
            fprintf(stderr, "FAIL: buffered open\n");
            return 1;
        }
        int have_direct = expert_io_open(&dio, path, 1) == 0 && dio.odirect;
        if (!have_direct) {
            fprintf(stderr, "NOTE: O_DIRECT unavailable on this fs, checking buffered only\n");
        }

        if (expert_io_read(&bio, off, len, a) != 0) {
            fprintf(stderr, "FAIL: buffered read\n");
            return 1;
        }
        if (have_direct) {
            double t0 = now_ms();
            if (expert_io_read(&dio, off, len, b) != 0) {
                fprintf(stderr, "FAIL: O_DIRECT read\n");
                return 1;
            }
            double t1 = now_ms();
            if (memcmp(a, b, len) != 0) {
                fprintf(stderr, "FAIL: direct/buffered mismatch\n");
                return 1;
            }
            printf("[read] blk.11 expert 7 gate: %zu B, direct==buffered, %.2f ms\n",
                   len, t1 - t0);
        }
        printf("[read] gate checksum = %016" PRIx64 "\n\n", fnv1a(a, len));

        expert_io_close(&bio);
        expert_io_close(&dio);
        free(a);
        free(b);
    }

    /* ---- 3. cache round-trip + duplicate collapse ---- */
    moe_ctx_t *ctx = moe_open(path, cache_mb * 1024 * 1024, threads, direct);
    if (!ctx) {
        fprintf(stderr, "FAIL: moe_open\n");
        return 1;
    }

    uint32_t experts[5] = { 1, 7, 42, 99, 7 };
    moe_expert_t out[MOE_MAX_TOP_K];

    memset(out, 0, sizeof(out));
    rc = moe_layer_fetch(ctx, 11, experts, 5, out);
    if (rc != 5) {
        fprintf(stderr, "FAIL: fetch: %d\n", rc);
        return 1;
    }
    uint64_t sum1 = 0;
    for (int i = 0; i < 5; i++) {
        sum1 ^= fnv1a(out[i].gate, out[i].gate_bytes) ^ fnv1a(out[i].down, out[i].down_bytes);
    }
    printf("[fetch] layer 11, 5 ids (1 dup): checksum %016" PRIx64 "\n", sum1);
    if (out[1].entry != out[4].entry) {
        fprintf(stderr, "FAIL: duplicate id did not dedupe\n");
        return 1;
    }
    moe_release(ctx, out, 5);

    memset(out, 0, sizeof(out));
    rc = moe_layer_fetch(ctx, 11, experts, 5, out);
    if (rc != 5) {
        fprintf(stderr, "FAIL: refetch: %d\n", rc);
        return 1;
    }
    uint64_t sum2 = 0;
    for (int i = 0; i < 5; i++) {
        sum2 ^= fnv1a(out[i].gate, out[i].gate_bytes) ^ fnv1a(out[i].down, out[i].down_bytes);
    }
    moe_release(ctx, out, 5);
    if (sum1 != sum2) {
        fprintf(stderr, "FAIL: cache returned different bytes\n");
        return 1;
    }
    printf("[fetch] refetch identical (%016" PRIx64 "), cache hit path OK\n", sum2);

    /* ---- 4. streaming API ---- */
    {
        uint32_t ids[8] = { 5, 5, 9, 17, 33, 65, 100, 127 };
        stream_probe_t p;
        memset(&p, 0, sizeof(p));
        p.expect = 8;

        rc = moe_layer_fetch_stream(ctx, 20, ids, 8, stream_probe_cb, &p);
        if (rc != 7) { /* 8 ids, one duplicate -> 7 unique */
            fprintf(stderr, "FAIL: stream returned %d, expected 7\n", rc);
            return 1;
        }
        if (p.seen != 7 || p.bad_slot) {
            fprintf(stderr, "FAIL: stream delivered %d, bad_slot=%d\n", p.seen, p.bad_slot);
            return 1;
        }
        printf("[stream] 8 ids (1 dup) -> %d callbacks, checksum %016" PRIx64 "\n\n",
               p.seen, p.sum);
    }

    /* ---- 5. eviction pressure ---- */
    moe_ctx_t *tiny = moe_open(path, 16 * 1024 * 1024, 2, direct);
    if (!tiny) {
        fprintf(stderr, "FAIL: tiny open\n");
        return 1;
    }
    for (int rep = 0; rep < 4; rep++) {
        memset(out, 0, sizeof(out));
        uint32_t ids[3] = { (uint32_t)(rep * 3), (uint32_t)(rep * 3 + 1),
                            (uint32_t)(rep * 3 + 2) };
        rc = moe_layer_fetch(tiny, 0, ids, 3, out);
        if (rc != 3) {
            fprintf(stderr, "FAIL: tiny fetch: %d\n", rc);
            return 1;
        }
        moe_release(tiny, out, 3);
    }
    moe_stats_t ts;
    moe_get_stats(tiny, &ts);
    printf("[evict] 16MiB cache, 12 distinct experts: %zu evictions, "
           "%zu TinyLFU bypasses, %zu resident\n\n",
           ts.evictions, ts.bypassed, ts.cache_entries);
    if (ts.evictions + ts.bypassed == 0) {
        fprintf(stderr, "FAIL: expected eviction or admission rejection\n");
        return 1;
    }
    moe_close(tiny);

    /* ---- 6. benchmark: prefill layer sweep ---- */
    double *cdf = (double *)malloc(sizeof(double) * (size_t)plan.n_expert);
    if (!cdf) {
        return 1;
    }
    zipf_cdf(cdf, (int)plan.n_expert, 1.2);

    uint32_t *rm = build_routing_map(&plan, PREFILL_TOKENS, cdf, 0xC0FFEEull);
    if (!rm) {
        return 1;
    }

    moe_stats_t before, after;
    moe_get_stats(ctx, &before);

    printf("[prefill] %d-token prompt, all layers, unrouted experts skipped\n",
           PREFILL_TOKENS);
    double t0 = now_ms();
    rc = moe_prefill_layer_sweep(ctx, PREFILL_TOKENS, rm);
    double t1 = now_ms();
    if (rc != 0) {
        fprintf(stderr, "FAIL: prefill sweep: %d\n", rc);
        return 1;
    }

    moe_get_stats(ctx, &after);
    double pf_ms    = t1 - t0;
    double pf_bytes = (double)(after.prefill_bytes - before.prefill_bytes);
    double pf_gib   = pf_bytes / (1024.0 * 1024.0 * 1024.0);

    printf("[prefill] %.2f s total, %.2f GiB read, %.0f MB/s, %llu experts\n",
           pf_ms / 1000.0, pf_gib, pf_bytes / (pf_ms / 1000.0) / 1e6,
           (unsigned long long)(after.prefill_experts - before.prefill_experts));
    printf("[prefill] %.1f ms/layer, %.1f experts/s, %.2f ms/expert\n",
           pf_ms / plan.n_layers,
           (double)(after.prefill_experts - before.prefill_experts) / (pf_ms / 1000.0),
           pf_ms / (double)(after.prefill_experts - before.prefill_experts));

    /* ---- 7. benchmark: decode ---- */
    printf("\n[decode] %d tokens, top-8 per layer, cache %.0f MiB\n",
           DECODE_TOKENS, (double)cache_mb);

    moe_stats_t d0s;
    moe_get_stats(ctx, &d0s);

    double d0 = now_ms();
    for (int tok = 0; tok < DECODE_TOKENS; tok++) {
        uint64_t route_state = 0x9E3779B9ull + (uint64_t)tok * 0x100000001B3ull;
        for (int l = 0; l < plan.n_layers; l++) {
            uint32_t ids[8];
            for (int k = 0; k < 8; k++) {
                ids[k] = zipf_sample(cdf, (int)plan.n_expert, &route_state);
            }
            memset(out, 0, sizeof(out));
            rc = moe_layer_fetch(ctx, l, ids, 8, out);
            if (rc != 8) {
                fprintf(stderr, "FAIL: decode layer %d: %d\n", l, rc);
                return 1;
            }
            uint32_t token_id = (uint32_t)tok;
            for (int k = 0; k < 8; k++) {
                moe_weights_t w;
                weights_from_expert(&out[k], &w);
                moe_compute(ctx, l, out[k].expert, &w, &token_id, 1);
            }
            moe_release(ctx, out, 8);
        }
    }
    double d1 = now_ms();

    moe_stats_t d1s;
    moe_get_stats(ctx, &d1s);
    double dec_ms  = d1 - d0;
    double dec_sec = dec_ms / 1000.0;
    double dec_gib = (double)(d1s.bytes_read - d0s.bytes_read) / (1024.0 * 1024.0 * 1024.0);

    printf("[decode] %.2f s total, %.1f ms/token, %.3f tok/s\n",
           dec_sec, dec_ms / DECODE_TOKENS, DECODE_TOKENS / dec_sec);
    printf("[decode] %.2f MiB read/token, %.2f GiB total\n",
           (double)(d1s.bytes_read - d0s.bytes_read) / DECODE_TOKENS / (1024.0 * 1024.0),
           dec_gib);

    /* ---- 8. summary ---- */
    printf("\n");
    moe_print_stats(ctx);

    moe_stats_t fin;
    moe_get_stats(ctx, &fin);
    printf("\n[summary]\n");
    printf("  prefill : %d tokens in %.2f s  -> %.1f tok/s prompt processing\n",
           PREFILL_TOKENS, pf_ms / 1000.0, PREFILL_TOKENS / (pf_ms / 1000.0));
    printf("  decode  : %d tokens in %.2f s  -> %.3f tok/s generation\n",
           DECODE_TOKENS, dec_sec, DECODE_TOKENS / dec_sec);
    printf("  prefill read %.2f GiB = %.1f%% of the 15.49 GiB expert set (%llu of %d experts routed)\n",
           pf_gib, pf_gib / 15.49 * 100.0,
           (unsigned long long)(fin.prefill_experts),
           fin.n_layers * fin.n_expert);
    printf("  decode  hit %.1f%% over the run\n",
           (double)fin.hits / (double)(fin.hits + fin.misses) * 100.0);
    printf("\nNote: dummy_gemm does no real math. decode tok/s is an I/O ceiling.\n");

    free(rm);
    free(cdf);
    moe_close(ctx);
    gguf_index_free(&idx);

    printf("\nall checks passed\n");
    return 0;
}
