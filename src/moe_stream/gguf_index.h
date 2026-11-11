/*
 * gguf_index.h — minimal read-only GGUF v3 index.
 *
 * We deliberately do not link libggml here. The streaming layer needs exactly
 * three things: the tensor directory (name -> absolute file offset), the GGML
 * type geometry (so a tensor's byte length can be computed without reading it),
 * and the per-layer expert tensor triples. Parsing that once at startup is a
 * few milliseconds and removes any dependency on the runtime's model loader.
 */
#ifndef MOE_GGUF_INDEX_H
#define MOE_GGUF_INDEX_H

#include <stddef.h>
#include <stdint.h>

#define GGUF_MAX_NAME 160
#define MOE_MAX_LAYERS 256

typedef struct {
    char     name[GGUF_MAX_NAME];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;      /* relative to the data section */
} gguf_tensor_t;

typedef struct {
    gguf_tensor_t *tensors;
    uint32_t       count;
    uint64_t       data_start;
    uint32_t       alignment;
    char           arch[32];

    /* copied out of metadata for convenience */
    uint32_t       n_layer;
    uint32_t       n_embd;
    uint32_t       n_ff_exp;
    uint32_t       n_expert;
    uint32_t       n_expert_used;
    uint32_t       n_head;
    uint32_t       n_head_kv;
    uint32_t       head_dim;
    uint32_t       n_vocab;
    float          rope_theta;
    float          eps;
} gguf_index_t;

typedef struct {
    uint64_t base;        /* absolute offset of expert 0's bytes */
    uint64_t stride;      /* bytes between consecutive experts */
    uint32_t n_expert;
    uint32_t ne0;         /* input dim  (contiguous) */
    uint32_t ne1;         /* output dim */
    uint32_t type;        /* ggml type id */
    int      present;
} moe_tensor_t;

typedef struct {
    moe_tensor_t gate;    /* ffn_gate_exps */
    moe_tensor_t up;      /* ffn_up_exps   */
    moe_tensor_t down;    /* ffn_down_exps */
    uint64_t     expert_bytes;
} moe_layer_t;

typedef struct {
    moe_layer_t  layers[MOE_MAX_LAYERS];
    int          n_layers;
    uint32_t     n_expert;
    uint32_t     n_expert_used;
    uint32_t     n_embd;
    uint32_t     n_ff_exp;
    uint64_t     total_expert_bytes;
    uint64_t     max_expert_bytes;   /* dynamic quants vary per layer */
    const gguf_index_t *idx;
} moe_plan_t;

int  gguf_index_load(gguf_index_t *idx, const char *path);
void gguf_index_free(gguf_index_t *idx);
const gguf_tensor_t *gguf_find(const gguf_index_t *idx, const char *name);

/* Absolute file offset + byte length of one expert slice inside a 3-D
 * [.., .., n_expert] tensor. Returns 0 or -errno. */
int moe_tensor_span(const gguf_index_t *idx, const gguf_tensor_t *t,
                    uint32_t expert, uint64_t *abs_off, uint64_t *bytes);

int  moe_plan_build(const gguf_index_t *idx, moe_plan_t *plan);
void moe_plan_print(const moe_plan_t *plan);

/* bytes per expert for a given ggml type, or 0 if the type is unsupported. */
uint64_t ggml_tensor_nbytes(uint32_t type, uint64_t nelements);

#endif /* MOE_GGUF_INDEX_H */
