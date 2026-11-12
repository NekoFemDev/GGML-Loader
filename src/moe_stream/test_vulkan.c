/*
 * test_vulkan.c — shader oracle: compare the Vulkan matmul against the CPU
 * dequantiser on real rows from the GGUF.
 *
 * For each quantised type we pick a 2-D tensor, load row 0, and compare
 *   vec_dot(type, row, x)          (CPU reference)
 * against
 *   backend->matmul(..., ne1=1)    (Vulkan shader)
 * A shader bug in the 6-bit scale packing or a nibble order shows up here
 * immediately, long before the full model produces garbage.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "compute_backend.h"
#include "dequant.h"
#include "expert_loader.h"
#include "gguf_index.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int         compute_vulkan_is_ready(void);
extern const char *compute_vulkan_device_name(void);

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return ((float)(rng >> 40) / 8388608.0f) - 1.0f; /* [-1,1) */
}

static const gguf_tensor_t *find_2d(const gguf_index_t *idx, uint32_t type,
                                    uint64_t min_ne0)
{
    for (uint32_t i = 0; i < idx->count; i++) {
        const gguf_tensor_t *t = &idx->tensors[i];
        if (t->type == type && t->n_dims >= 2 && t->dims[0] >= min_ne0) {
            return t;
        }
    }
    return NULL;
}

static const char *type_name(uint32_t t)
{
    switch (t) {
        case GGML_T_F32:  return "f32";
        case GGML_T_Q2_K: return "q2_K";
        case GGML_T_Q3_K: return "q3_K";
        case GGML_T_Q4_K: return "q4_K";
        case GGML_T_Q5_K: return "q5_K";
        case GGML_T_Q6_K: return "q6_K";
        default:          return "?";
    }
}

static int test_row(const compute_backend_t *be, expert_io_t *io,
                    const gguf_index_t *idx, uint32_t type)
{
    const gguf_tensor_t *t = find_2d(idx, type, 512);
    if (!t) {
        printf("  %-5s: no 2-D tensor of this type, skipped\n", type_name(type));
        return 1;
    }

    const uint64_t ne0 = t->dims[0];
    const uint64_t rb  = ggml_type_row_bytes(type, ne0);
    if (rb == 0) {
        printf("  %-5s: unsupported geometry\n", type_name(type));
        return 0;
    }

    uint8_t *row = (uint8_t *)malloc(rb);
    float   *x   = (float *)malloc(ne0 * sizeof(float));
    if (!row || !x) {
        return 0;
    }

    int rc = expert_io_read(io, idx->data_start + t->offset, rb, row);
    if (rc != 0) {
        printf("  %-5s: read failed %d\n", type_name(type), rc);
        free(row);
        free(x);
        return 0;
    }

    for (uint64_t i = 0; i < ne0; i++) {
        x[i] = frand();
    }

    const float cpu = vec_dot(type, row, x, ne0);
    float       gpu = 0.0f;
    be->matmul(be->self, type, row, 0, ne0, 1, x, &gpu);

    const float denom = fabsf(cpu) > 1e-3f ? fabsf(cpu) : 1.0f;
    const float rel   = fabsf(gpu - cpu) / denom;
    const int   ok    = rel < 1e-3f;

    printf("  %-5s %-28s ne0=%-5llu cpu=%10.6f gpu=%10.6f rel=%.2e %s\n",
           type_name(type), t->name, (unsigned long long)ne0,
           cpu, gpu, rel, ok ? "OK" : "FAIL");

    free(row);
    free(x);
    return ok;
}

static void bench_full(const compute_backend_t *be, expert_io_t *io,
                       const gguf_index_t *idx)
{
    const gguf_tensor_t *t = NULL;
    uint64_t nb = 0;
    for (uint32_t i = 0; i < idx->count; i++) {
        const gguf_tensor_t *c = &idx->tensors[i];
        if (c->type != GGML_T_Q4_K || c->n_dims < 2 || c->dims[0] < 512) continue;
        uint64_t ne = 1;
        for (uint32_t d = 0; d < c->n_dims; d++) ne *= c->dims[d];
        uint64_t bytes = ggml_tensor_nbytes(c->type, ne);
        if (bytes >= (1u << 20) && bytes <= (16u << 20)) {
            t = c; nb = bytes; break;
        }
    }
    if (!t) {
        return;
    }

    const uint64_t ne0 = t->dims[0];
    const uint64_t ne1 = t->dims[1];

    uint8_t *w = (uint8_t *)malloc(nb);
    float   *x = (float *)malloc(ne0 * sizeof(float));
    float   *y = (float *)malloc(ne1 * sizeof(float));
    float   *y2 = (float *)malloc(ne1 * sizeof(float));
    if (!w || !x || !y || !y2) {
        free(w); free(x); free(y); free(y2);
        return;
    }
    if (expert_io_read(io, idx->data_start + t->offset, nb, w) != 0) {
        free(w); free(x); free(y); free(y2);
        return;
    }
    for (uint64_t i = 0; i < ne0; i++) {
        x[i] = frand();
    }

    const double c0 = now_ms();
    const compute_backend_t *sc = compute_backend_scalar();
    sc->matmul(NULL, t->type, w, 0, ne0, ne1, x, y);
    const double c1 = now_ms();

    be->matmul(be->self, t->type, w, 0, ne0, ne1, x, y2);
    const double g1 = now_ms();

    double maxrel = 0.0;
    for (uint64_t i = 0; i < ne1; i++) {
        const double d = fabs((double)y2[i] - y[i]) /
                         (fabs((double)y[i]) > 1e-3 ? fabs((double)y[i]) : 1.0);
        if (d > maxrel) {
            maxrel = d;
        }
    }

    printf("\n  full %s [%llu x %llu] %.1f MiB\n", t->name,
           (unsigned long long)ne1, (unsigned long long)ne0,
           (double)nb / (1024.0 * 1024.0));
    printf("  cpu %8.1f ms   gpu %8.1f ms   speedup %.2fx   max rel %.2e %s\n",
           c1 - c0, g1 - c1, (c1 - c0) / (g1 - c1 > 0 ? g1 - c1 : 1e-9),
           maxrel, maxrel < 1e-3 ? "OK" : "FAIL");

    free(w); free(x); free(y); free(y2);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char *path = argc > 1 ? argv[1]
        : "../../Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf";

    gguf_index_t idx;
    if (gguf_index_load(&idx, path) != 0) {
        fprintf(stderr, "index load failed: %s\n", path);
        return 1;
    }

    expert_io_t io;
    memset(&io, 0, sizeof(io));
    io.fd = -1;
    if (expert_io_open(&io, path, 0) != 0) {
        fprintf(stderr, "open failed\n");
        return 1;
    }

    const compute_backend_t *be = compute_backend_vulkan();

    /* --- f32 sanity on a small matrix --- */
    {
        enum { N = 64, K = 128 };
        float *W = (float *)malloc(sizeof(float) * N * K);
        float *X = (float *)malloc(sizeof(float) * K);
        float *Y = (float *)malloc(sizeof(float) * N);
        for (int i = 0; i < N * K; i++) W[i] = frand();
        for (int i = 0; i < K; i++)     X[i] = frand();

        be->matmul(be->self, GGML_T_F32, (const uint8_t *)W, 0, K, N, X, Y);
        if (!compute_vulkan_is_ready()) {
            printf("SKIP: no Vulkan compute device; backend fell back to scalar\n");
            free(W); free(X); free(Y);
            expert_io_close(&io);
            gguf_index_free(&idx);
            return 0;
        }
        printf("device: %s\n", compute_vulkan_device_name());
        printf("[f32] 64x128\n");

        int ok = 1;
        for (int r = 0; r < N; r++) {
            float acc = 0.0f;
            for (int i = 0; i < K; i++) {
                acc += W[r * K + i] * X[i];
            }
            float denom = fabsf(acc) > 1e-3f ? fabsf(acc) : 1.0f;
            if (fabsf(Y[r] - acc) / denom > 1e-4f) {
                printf("  row %d cpu=%f gpu=%f FAIL\n", r, acc, Y[r]);
                ok = 0;
                break;
            }
        }
        printf("  %s\n", ok ? "OK" : "FAIL");

        /* dispatch overhead: same tiny matmul 50x, compute is negligible */
        const double o0 = now_ms();
        for (int i = 0; i < 50; i++) {
            be->matmul(be->self, GGML_T_F32, (const uint8_t *)W, 0, K, N, X, Y);
        }
        const double o1 = now_ms();
        printf("  submit+wait: %.2f ms per dispatch (64x128 is ~8k MACs)\n",
               (o1 - o0) / 50.0);

        free(W); free(X); free(Y);
        if (!ok) return 1;
    }

    printf("\n[quantised rows, GPU vs vec_dot]\n");
    int ok = 1;
    ok &= test_row(be, &io, &idx, GGML_T_Q2_K);
    ok &= test_row(be, &io, &idx, GGML_T_Q3_K);
    ok &= test_row(be, &io, &idx, GGML_T_Q4_K);
    ok &= test_row(be, &io, &idx, GGML_T_Q5_K);
    ok &= test_row(be, &io, &idx, GGML_T_Q6_K);

    /* --- batch mechanics: 8 dispatches, same W, different y --- */
    {
        const gguf_tensor_t *tb = find_2d(&idx, GGML_T_Q4_K, 2048);
        if (tb) {
            const uint64_t ne0 = tb->dims[0];
            const uint64_t ne1 = 768;             /* gate-sized */
            const uint64_t rb  = ggml_type_row_bytes(tb->type, ne0);
            const size_t   nb  = (size_t)rb * ne1;

            uint8_t *w = (uint8_t *)malloc(nb);
            float   *x = (float *)malloc(ne0 * sizeof(float));
            float   *y[8];
            for (int i = 0; i < 8; i++) y[i] = (float *)malloc(ne1 * sizeof(float));

            expert_io_read(&io, idx.data_start + tb->offset, nb, w);
            for (uint64_t i = 0; i < ne0; i++) x[i] = frand();

            /* 8 distinct weight buffers, 8 distinct keys (the MoE pattern) */
            uint8_t *w8[8];
            for (int i = 0; i < 8; i++) {
                w8[i] = (uint8_t *)malloc(nb);
                expert_io_read(&io, idx.data_start + tb->offset + (uint64_t)i * nb,
                               nb, w8[i]);
            }

            be->begin_batch(be->self);
            for (int i = 0; i < 8; i++) {
                be->matmul(be->self, tb->type, w8[i],
                           0x1234000000000000ull + (uint64_t)i, ne0, ne1, x, y[i]);
            }
            be->end_batch(be->self);

            double worst = 0.0;
            for (int i = 0; i < 8; i++) {
                float *ref = (float *)malloc(ne1 * sizeof(float));
                compute_backend_scalar()->matmul(NULL, tb->type, w8[i], 0, ne0, ne1, x, ref);
                for (uint64_t r = 0; r < ne1; r++) {
                    double d = fabs((double)y[i][r] - ref[r]) /
                               (fabs((double)ref[r]) > 1e-3 ? fabs((double)ref[r]) : 1.0);
                    if (d > worst) worst = d;
                }
                free(ref);
            }
            printf("\n[batch x8] same W, 8 outputs: max rel %.2e %s\n",
                   worst, worst < 1e-3 ? "OK" : "FAIL");
        }
    }

    bench_full(be, &io, &idx);

    expert_io_close(&io);
    gguf_index_free(&idx);

    compute_vulkan_profile_report();
    printf("\n%s\n", ok ? "all shader checks passed" : "SHADER MISMATCH");
    return ok ? 0 : 1;
}
