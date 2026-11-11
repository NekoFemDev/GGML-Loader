/*
 * qwen_model.h — scalar/vulkan-swappable CPU forward pass for qwen3moe.
 */
#ifndef MOE_QWEN_MODEL_H
#define MOE_QWEN_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "compute_backend.h"

typedef struct qwen_model qwen_model_t;

qwen_model_t *qwen_load(const char *gguf_path, size_t cache_bytes, int threads,
                        int max_seq, int odirect);
void          qwen_free(qwen_model_t *m);

/* Install a math backend. NULL restores the scalar path. Takes effect on the
 * next forward/prefill; call before the first token. */
void qwen_set_backend(qwen_model_t *m, const compute_backend_t *be);

int  qwen_vocab_size(const qwen_model_t *m);
int  qwen_n_layers(const qwen_model_t *m);
int  qwen_pos(const qwen_model_t *m);      /* next KV position */
int  qwen_max_seq(const qwen_model_t *m);

/* Decode one token at the current position; advances pos. */
int  qwen_forward(qwen_model_t *m, int token);

/* Prefill a prompt with the layer-sweep: per layer, each routed expert is
 * read once and applied to every token routed to it. Fills the KV cache for
 * all n positions and leaves logits for the last token. */
int  qwen_prefill(qwen_model_t *m, const uint32_t *tokens, int n);

/* Clear the KV cache and rewind to position 0. */
void qwen_reset(qwen_model_t *m);

const char  *qwen_backend_name(const qwen_model_t *m);
const float *qwen_logits(const qwen_model_t *m);
int          qwen_argmax(const qwen_model_t *m);
double       qwen_last_ms(const qwen_model_t *m);

typedef struct {
    uint64_t experts_fetched;
    uint64_t expert_bytes;
    uint64_t matmul_calls;
} qwen_stats_t;
void qwen_get_stats(qwen_model_t *m, qwen_stats_t *out);

#endif /* MOE_QWEN_MODEL_H */
