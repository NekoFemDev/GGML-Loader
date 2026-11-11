/*
 * moe_stream.h — per-layer expert fetch pipeline + prefill layer sweep.
 *
 * Two access patterns, one storage layer:
 *
 *   DECODE  (token at a time)   moe_layer_fetch[_stream]()
 *           top-8 of 128 experts per layer, served from an LRU.
 *
 *   PREFILL (prompt at a time)  moe_prefill_layer_sweep()
 *           all 128 experts of a layer, each read exactly once, applied to
 *           every token routed to it, then discarded. Bypasses the LRU.
 *
 * Routing map layout (prefill):
 *
 *   routing_map[((layer * num_tokens) + token) * top_k + k] = expert_id
 *
 *   row-major, layer-major, uint32_t. top_k = plan.n_expert_used (8 here).
 *   Size = n_layers * num_tokens * top_k * sizeof(uint32_t).
 *
 *   In a real forward pass the routing for layer L is only known after
 *   attention(L), so the useful entry point is moe_prefill_layer() called once
 *   per layer. moe_prefill_layer_sweep() is the batch/benchmark form that takes
 *   the whole map.
 */
#ifndef MOE_STREAM_H
#define MOE_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "expert_loader.h"
#include "gguf_index.h"
#include "lru_cache.h"

#define MOE_MAX_TOP_K 16

typedef struct moe_ctx moe_ctx_t;

/* Borrowed view of one expert's three quantised matrices. */
typedef struct {
    const uint8_t *gate;
    size_t         gate_bytes;
    const uint8_t *up;
    size_t         up_bytes;
    const uint8_t *down;
    size_t         down_bytes;
} moe_weights_t;

typedef struct {
    lru_entry_t *entry;      /* pinned; release with moe_release() */
    uint32_t     expert;
    int          slot;       /* index in the caller's requested array */
    uint8_t     *gate;
    uint8_t     *up;
    uint8_t     *down;
    size_t       gate_bytes;
    size_t       up_bytes;
    size_t       down_bytes;
} moe_expert_t;

/* Called once per expert as soon as its weights are resident.
 * Return non-zero to stop the fetch (-ECANCELED). The pointer is only valid
 * for the duration of the call; the entry is unpinned afterwards. */
typedef int (*moe_expert_cb)(void *user, const moe_expert_t *expert);

/* Replace dummy_gemm with the real gate/up/down matmuls. */
typedef void (*moe_gemm_fn)(void *user, int layer, uint32_t expert,
                            const moe_weights_t *w,
                            const uint32_t *token_ids, int n_tokens);

typedef struct {
    int      n_layers;
    int      n_expert;
    int      n_expert_used;
    size_t   cache_capacity;
    size_t   cache_bytes;
    size_t   cache_entries;
    size_t   hits;
    size_t   misses;
    size_t   evictions;
    size_t   rejected;
    size_t   admitted;
    size_t   bypassed;
    uint64_t bytes_read;      /* cache-fill reads only */
    uint64_t prefill_bytes;   /* layer-sweep reads (bypass the cache) */
    uint64_t prefill_experts;
    uint64_t prefill_tokens;
    int      io_errors;
} moe_stats_t;

moe_ctx_t *moe_open(const char *gguf_path, size_t cache_bytes, int nthreads, int odirect);
void       moe_close(moe_ctx_t *ctx);

/* ---- decode path -------------------------------------------------------- */

/* Blocking. out[i] corresponds to experts[i]; duplicates share one entry with
 * an extra pin. Dispatch order is sorted by physical file offset. */
int  moe_layer_fetch(moe_ctx_t *ctx, int layer, const uint32_t *experts, int n,
                     moe_expert_t *out);
void moe_release(moe_ctx_t *ctx, moe_expert_t *out, int n);

/* Pipelined. cb fires as each expert becomes resident, in completion order,
 * so the GEMM for expert 0 can run while experts 1..7 are still on the bus.
 * Duplicate ids are collapsed and delivered once. */
int  moe_layer_fetch_stream(moe_ctx_t *ctx, int layer, const uint32_t *experts,
                            int n, moe_expert_cb cb, void *user);

/* Non-blocking: enqueue reads for experts likely to be needed later. Returns
 * the number of jobs queued (0 if the queue is full or all are resident).
 * Prefetched entries follow the same TinyLFU admission rules as normal fills
 * and are unpinned immediately, so they can be evicted without stalling. */
int moe_prefetch(moe_ctx_t *ctx, int layer, const uint32_t *ids, int n);

/* Install a backing allocator for expert entries (the Vulkan arena). Must be
 * called before the first expert is fetched. Returns 0 on success. */
int moe_set_cache_allocator(moe_ctx_t *ctx,
                            void *(*alloc)(void *user, size_t size),
                            void  (*freefn)(void *user, void *ptr, size_t size),
                            void *user);

/* ---- prefill path ------------------------------------------------------- */

/* One layer, routing is [num_tokens][top_k] for THIS layer.
 * Call this from the forward pass after attention(L) produces the router ids. */
int moe_prefill_layer(moe_ctx_t *ctx, int layer, int num_tokens,
                      const uint32_t *routing_layer);

/* All layers. routing_map is [n_layers][num_tokens][top_k]. */
int moe_prefill_layer_sweep(moe_ctx_t *ctx, int num_tokens,
                            const uint32_t *routing_map);

/* Same sweep, but `fn` performs the math instead of the context's dummy gemm.
 * `fn` is called once per routed expert, after its weights are resident, with
 * the list of token ids routed to it. */
int moe_prefill_layer_with(moe_ctx_t *ctx, int layer, int num_tokens,
                           const uint32_t *routing,
                           moe_gemm_fn fn, void *user);

/* ---- compute hook ------------------------------------------------------- */

void moe_set_gemm(moe_ctx_t *ctx, moe_gemm_fn fn, void *user);
void moe_compute(moe_ctx_t *ctx, int layer, uint32_t expert,
                 const moe_weights_t *w, const uint32_t *token_ids, int n_tokens);

/* Placeholder: touches a sample of each slice, does no real math. */
void moe_dummy_gemm(void *user, int layer, uint32_t expert,
                    const moe_weights_t *w, const uint32_t *token_ids, int n_tokens);

/* Read-only views for layers that sit above the streamer (the model). */
const gguf_index_t *moe_index(const moe_ctx_t *ctx);
const moe_plan_t   *moe_plan(const moe_ctx_t *ctx);

void moe_get_stats(moe_ctx_t *ctx, moe_stats_t *out);
void moe_print_stats(moe_ctx_t *ctx);

#endif /* MOE_STREAM_H */
