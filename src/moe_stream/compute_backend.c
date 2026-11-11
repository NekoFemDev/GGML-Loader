#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "compute_backend.h"

#include "dequant.h"

#include <math.h>
#include <string.h>

/* ---- scalar backend ----------------------------------------------------- */

static void sc_matmul(void *self, uint32_t type, const uint8_t *w, uint64_t wkey,
                      uint64_t ne0, uint64_t ne1, const float *x, float *y)
{
    (void)self;
    (void)wkey;
    const uint64_t rb = ggml_type_row_bytes(type, ne0);
    for (uint64_t o = 0; o < ne1; o++) {
        y[o] = vec_dot(type, w + o * rb, x, ne0);
    }
}

static void sc_matmul_batch(void *self, uint32_t type, const uint8_t *w,
                            uint64_t wkey, uint64_t ne0, uint64_t ne1,
                            const float *x, const uint32_t *idx, int n_tokens,
                            float *y)
{
    (void)self;
    (void)wkey;
    const uint64_t rb = ggml_type_row_bytes(type, ne0);
    for (int t = 0; t < n_tokens; t++) {
        const float *xt = x + (size_t)(idx ? idx[t] : (uint32_t)t) * ne0;
        float       *yt = y + (size_t)t * ne1;
        for (uint64_t o = 0; o < ne1; o++) {
            yt[o] = vec_dot(type, w + o * rb, xt, ne0);
        }
    }
}

static void sc_rmsnorm(void *self, float *out, const float *x, const float *w,
                       int n, float eps)
{
    (void)self;
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    const float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale * w[i];
    }
}

static void sc_rope(void *self, float *v, int n, const float *cs, const float *sn)
{
    (void)self;
    const int half = n / 2;
    for (int i = 0; i < half; i++) {
        const float a = v[i];
        const float b = v[i + half];
        v[i]        = a * cs[i] - b * sn[i];
        v[i + half] = b * cs[i] + a * sn[i];
    }
}

static void sc_softmax(void *self, float *x, int n)
{
    (void)self;
    float mx = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > mx) {
            mx = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv;
    }
}

static void sc_swiglu(void *self, const float *gate, const float *up,
                      float *out, int n)
{
    (void)self;
    for (int i = 0; i < n; i++) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

static void sc_begin_batch(void *self) { (void)self; }
static int  sc_end_batch(void *self)   { (void)self; return 0; }

static const compute_backend_t g_scalar = {
    .name    = "scalar",
    .self    = NULL,
    .matmul  = sc_matmul,
    .matmul_batch = sc_matmul_batch,
    .rmsnorm = sc_rmsnorm,
    .rope    = sc_rope,
    .softmax = sc_softmax,
    .swiglu  = sc_swiglu,
    .begin_batch = sc_begin_batch,
    .end_batch   = sc_end_batch,
    .sync    = NULL,
};

const compute_backend_t *compute_backend_scalar(void)
{
    return &g_scalar;
}

/* ---- Vulkan backend (stub) ---------------------------------------------- *
 *
 * Integration points, in the order they will matter:
 *
 *   vk_matmul      the hot path (94% of expert bytes). Keep weights in a
 *                  VkBuffer, dequantise in-shader (q4_K block -> 256 floats is
 *                  a 144-byte read per workgroup). Cache buffers by
 *                  (ptr,len) so a streamed expert is uploaded once.
 *   vk_swiglu      fuse into the gate/up shader to avoid a round trip.
 *   attention      currently a CPU loop over 128-dim dots; a fused
 *                  flash-attention-style shader is the second big win.
 *   sync           vkQueueSubmit + vkQueueWaitIdle, or a fence per batch.
 *
 * Mali note: tile-based GPUs punish per-dispatch submits. Batch all matmuls of
 * one layer into one command buffer and submit at the layer boundary; that is
 * why `sync` exists separately from the ops.
 */

#ifndef MOE_ENABLE_VULKAN
/* CPU-forwarding stub used when built without -DMOE_ENABLE_VULKAN. */
static void vk_matmul(void *self, uint32_t type, const uint8_t *w, uint64_t wkey,
                      uint64_t ne0, uint64_t ne1, const float *x, float *y)
{
    sc_matmul(self, type, w, wkey, ne0, ne1, x, y);
}

static void vk_begin_batch(void *self) { (void)self; }
static int  vk_end_batch(void *self)   { (void)self; return 0; }

static void vk_rmsnorm(void *self, float *out, const float *x, const float *w,
                       int n, float eps)
{
    sc_rmsnorm(self, out, x, w, n, eps);
}

static void vk_rope(void *self, float *v, int n, const float *cs, const float *sn)
{
    sc_rope(self, v, n, cs, sn);
}

static void vk_softmax(void *self, float *x, int n)
{
    sc_softmax(self, x, n);
}

static void vk_swiglu(void *self, const float *gate, const float *up,
                      float *out, int n)
{
    sc_swiglu(self, gate, up, out, n);
}

static void vk_sync(void *self)
{
    (void)self;
}

static const compute_backend_t g_vulkan = {
    .name    = "vulkan-stub",
    .self    = NULL,
    .matmul  = vk_matmul,
    .matmul_batch = sc_matmul_batch,
    .rmsnorm = vk_rmsnorm,
    .rope    = vk_rope,
    .softmax = vk_softmax,
    .swiglu  = vk_swiglu,
    .begin_batch = vk_begin_batch,
    .end_batch   = vk_end_batch,
    .sync    = vk_sync,
};
#endif /* !MOE_ENABLE_VULKAN */

#ifndef MOE_ENABLE_VULKAN
int   compute_vulkan_arena_init(size_t b, size_t s) { (void)b; (void)s; return -1; }
void *compute_vulkan_arena_alloc(void *u, size_t s) { (void)u; (void)s; return NULL; }
void  compute_vulkan_arena_free(void *u, void *p, size_t s) { (void)u; (void)p; (void)s; }
void  compute_vulkan_arena_shutdown(void) { }
void  compute_vulkan_cache_clear(void) { }
#endif

const compute_backend_t *compute_backend_vulkan(void)
{
#ifdef MOE_ENABLE_VULKAN
    extern const compute_backend_t *compute_backend_vulkan_impl(void);
    return compute_backend_vulkan_impl();
#else
    return &g_vulkan;
#endif
}
