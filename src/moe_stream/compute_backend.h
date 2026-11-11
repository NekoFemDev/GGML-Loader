/*
 * compute_backend.h — swappable math backend.
 *
 * The forward pass keeps all control flow (layer order, residuals, routing) in
 * C and delegates only the arithmetic to a backend vtable. Scalar today,
 * Vulkan next.
 *
 * Why not put whole blocks behind the vtable? Because the expert buffers come
 * from the streamer and the Vulkan backend needs to own their upload/staging
 * lifetime. Keeping `matmul` as the unit lets the backend cache a VkBuffer per
 * VkDeviceMemory range and still lets C decide *when* an expert is resident.
 *
 * Threading: the scalar backend is called from one thread. A Vulkan backend may
 * enqueue work; it must be complete before `matmul` returns unless it also
 * implements `sync` and the caller brackets with it. The reference loop is
 * synchronous, so a correct backend is always safe.
 */
#ifndef MOE_COMPUTE_BACKEND_H
#define MOE_COMPUTE_BACKEND_H

#include <stddef.h>
#include <stdint.h>

typedef struct compute_backend compute_backend_t;

struct compute_backend {
    const char *name;
    void       *self;      /* device/context for this backend instance */

    /* y[ne1] = W[ne1][ne0] * x[ne0]. W is ggml-quantised, row-major,
     * row_bytes = ggml_type_row_bytes(type, ne0).
     *
     * `wkey` is a stable identity for the weight buffer, assigned by the
     * model. 0 means "no residency"; otherwise the backend may cache the
     * weights under this key. Keys let the Vulkan backend keep weights in
     * VRAM even though the streamer reuses host pointers between experts. */
    void (*matmul)(void *self, uint32_t type, const uint8_t *w, uint64_t wkey,
                   uint64_t ne0, uint64_t ne1, const float *x, float *y);

    /* Batched GEMM: y[t][o] = sum_k W[o][k] * x[row(t)][k], row(t) = idx[t]
     * or t when idx is NULL. x is [>=max(row)+1][ne0], y is [n_tokens][ne1].
     * Records n_tokens dispatches (or one tiled kernel) into the open batch. */
    void (*matmul_batch)(void *self, uint32_t type, const uint8_t *w, uint64_t wkey,
                         uint64_t ne0, uint64_t ne1,
                         const float *x, const uint32_t *idx, int n_tokens,
                         float *y);

    void (*rmsnorm)(void *self, float *out, const float *x, const float *w,
                    int n, float eps);
    void (*rope)(void *self, float *v, int n, const float *cs, const float *sn);
    void (*softmax)(void *self, float *x, int n);
    void (*swiglu)(void *self, const float *gate, const float *up,
                   float *out, int n);

    /* Record dispatches without submitting. Between begin_batch() and
     * end_batch() the caller must not read any output. end_batch submits the
     * whole batch with one fence wait. Both may be NULL (scalar). */
    void (*begin_batch)(void *self);
    int  (*end_batch)(void *self);

    /* Flush queued GPU work. No-op for scalar. */
    void (*sync)(void *self);
};

/* Stateless CPU implementation. Safe to share across models. */
const compute_backend_t *compute_backend_scalar(void);

/* Vulkan stub: currently forwards to scalar so the integration is testable.
 * Every entry point marks the exact vkCmd* call site. */
const compute_backend_t *compute_backend_vulkan(void);

/* Single mapping shared by the expert LRU and the GPU. Defined by the Vulkan
 * backend when built, weak stubs otherwise. */
int   compute_vulkan_arena_init(size_t bytes, size_t slot_size);
void *compute_vulkan_arena_alloc(void *user, size_t size);
void  compute_vulkan_arena_free(void *user, void *ptr, size_t size);
void  compute_vulkan_arena_shutdown(void);
void  compute_vulkan_cache_clear(void);
void  compute_vulkan_profile_report(void);
void  compute_vulkan_profile_reset(void);

#endif /* MOE_COMPUTE_BACKEND_H */
