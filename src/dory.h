// dory (NVIDIA Nemotron "Dory") -- looped hybrid architecture: hparams/weights/KV-cache data
// structures + a forward-pass graph builder for the GGML-Loader engine.
//
// All details below are verified against this checkpoint's own metadata/tensor listing AND the
// architecture's reference inference implementation (llama.cpp PR #30306 -- the spec NVIDIA's
// model card points at). Several are non-obvious normalized-GPT (nGPT) behaviors, not vanilla
// transformer defaults:
//
// 1. Hybrid blocks, alternating across the whole depth: attention, MLP, attention, MLP, ...
//    72 physical blocks = 36 attention + 36 MLP (verified: 36*8 + 36*6 + 3 global = 507
//    tensors). dory.layer_pattern carries one char per physical block:
//      '-' -> MLP block:  ffn_gate, ffn_up, ffn_down, ffn_gate_scale, ffn_up_scale,
//                         residual_alpha
//      'S' -> sliding-window attention (window n_swa; rope base 1e4)
//      '*' -> full attention (every 8th block; rope base 1e7)
//    attention block: attn_q, attn_k, attn_v, attn_output, attn_gate, attn_qk_scale,
//    attn_gate_scale, residual_alpha.
//    dory.rope.profile repeats the same split numerically: 1 -> base 1 (the '*' blocks),
//    2 -> base 2 (the 'S' blocks).
//
// 2. Normalized-GPT math. There are no norm weights because every normalization is weightless
//    L2: norm(x) = x / sqrt(sum(x^2) + 1e-6). All learned "scale" tensors are per-channel
//    multipliers (some squared in the graph, see 4), and the residual update is a *spherical
//    interpolation* between the normalized block input and output, not an addition.
//
// 3. Looping and input reinjection. Blocks [input_start, input_start+recurrent) run num_loops
//    times; at the start of every recurrent pass the state is recombined with the input
//    stack's output: h = norm(h + input_state). Only the recurrent range loops.
//
// 4. Exact op sequence of one block (reference-verified; the only place to look if output ever
//    degrades):
//      residual = cur
//      MLP:    g = Wg cur; u = Wu cur
//              cur = L2( silu(g * ffn_gate_scale*sqrt(n_embd)) * (u * ffn_up_scale*sqrt(n_embd)) )
//              cur = Wd cur
//      attn:   q,k,v = Wq/Wk/Wv cur          (per-head 3D [hd, n_head, n_tokens])
//              q,k = rope(theta by profile, NEOX, full width)
//              q = L2(q) * (attn_qk_scale   * sqrt(hd)      / init_std)^2   <- SQUARED
//              k = L2(k)                                                    <- unscaled
//              gate = sigmoid(Wgate cur * (attn_gate_scale * sqrt(gate_init)/init_std)^2)
//              cur = L2(attn(q, k, v, scale = 1/sqrt(hd)) * gate)
//              cur = Wo cur
//      update: a = L2(residual); b = L2(cur)
//              alpha = |residual_alpha * alpha_init/init_std|
//              cur = L2(a + (b - a) * alpha)
//    Head: no final norm; logits = output(cur) * (output_scale / init_std).
//
// ---------------------------------------------------------------------------------------------
// KV-cache logic -- the two modes, and why last-loop is the RAM win:
//
// One ring per (attention block, loop slot): k/v tensors shaped
// [n_embd_head(_v), n_head_kv, n_cells, n_slots]. Cells are addressed by absolute position
// modulo n_cells (cell = pos % n_cells); no shifting or memmove exists on any path.
//
//   * 'S' sliding block: n_cells = n_swa (1024) -- any position older than one window can never
//     be attended again, so the ring is fixed at 4 MiB per (layer, slot), independent of n_ctx.
//   * '*' full block:    n_cells = n_ctx (runtime context, clamped to n_ctx_train) -- these 9
//     layers do need the whole context; at n_ctx * 4 KiB per slot they dominate cache cost.
//
//   * DORY_KV_PER_LOOP (metadata default): n_slots == num_loops on recurrent blocks. Loop k
//     reads/writes its own ring, so each loop attends to its own history.
//   * DORY_KV_LAST_LOOP (RAM-saving override): n_slots == 1 everywhere. All loops of the
//     recurrent range alias the same cells, so each loop iteration overwrites the previous
//     one in place; the ring always holds the most recent loop's K/V for every position.
//
// Ring wrap needs no special attention path: attention reads the whole ring and a mask (see
// dory_fill_attn_mask) marks a cell invisible when its newest write lies in the query's future
// (possible during prefill, where a later batch token already overwrote the cell) or when the
// cell was never written yet. Two masks exist because the two block kinds have different ring
// geometry; each is shared by every block and loop iteration of its kind.
// ---------------------------------------------------------------------------------------------
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Upper bound this module supports; this checkpoint has 72 blocks.
#define DORY_MAX_BLOCK_COUNT 128

// Graph/context budget for one full forward pass. A dory forward executes 16 + 40*num_loops +
// 16 blocks (each ~12-22 ops plus views) -- several times a vanilla-transformer graph, so
// ggml_new_graph()'s 2048-node default is not enough. This constant covers both the graph's
// node budget and the context's tensor-object budget in dory_forward_ctx_bytes().
#define DORY_GRAPH_SIZE 8192

// Where a block sits in the execution schedule. Only RECURRENT blocks are re-run by the loop.
enum dory_layer_role {
    DORY_ROLE_INPUT,
    DORY_ROLE_RECURRENT,
    DORY_ROLE_OUTPUT,
};

// KV-cache strategy for the recurrent range -- see the header comment above.
enum dory_kv_mode {
    DORY_KV_PER_LOOP,  // file default: private ring per loop iteration (num_loops x RAM)
    DORY_KV_LAST_LOOP, // override: one shared ring, overwritten in place      (1 x RAM)
};

struct dory_hparams {
    // --- topology (dory.*) ---
    uint32_t block_count;           // 72 (total physical blocks in the file)
    uint32_t input_layer_count;     // 16 -> blocks [0, 16)
    uint32_t recurrent_layer_count; // 40 -> blocks [16, 56), run num_loops times
    uint32_t output_layer_count;    // 16 -> blocks [56, 72)
    uint32_t num_loops;             // 4 in this file (README recommends overriding to 2)

    // --- widths ---
    uint32_t n_embd;                // embedding_length (2560)
    uint32_t n_ff;                  // feed_forward_length (7168)
    uint32_t n_head;                // attention.head_count (16)
    uint32_t n_head_kv;             // attention.head_count_kv (4, GQA)
    uint32_t n_embd_head;           // attention.key_length   (256, per-head q/k width)
    uint32_t n_embd_head_v;         // attention.value_length (256, per-head v width)
    uint32_t n_rot;                 // rope.dimension_count   (256, full width)

    // --- context / attention ---
    uint32_t n_ctx_train;           // context_length (262144)
    uint32_t n_swa;                 // attention.sliding_window (1024; window of the 'S' blocks)

    // --- rope ---
    float    rope_freq_base;        // rope.freq_base   (1e7; profile-value-1 blocks)
    float    rope_freq_base_2;      // rope.freq_base_2 (1e4; profile-value-2 blocks)
    bool     rope_neox;             // NEOX (split-half). Verified: the reference groups DORY
                                    // with the NEOX rotaries, not the interleaved ones.
    bool     has_rope_profile;      // true when dory.rope.profile was present in the file
    uint8_t  rope_profile[DORY_MAX_BLOCK_COUNT]; // raw per-BLOCK values 1/2 (see point 1)

    // --- nGPT constants from the file (used by the forward pass's scale factors!) ---
    float    init_std;              // dory.init_std (0.0177) -- every learned scale is divided
                                    // by it in the graph (see point 4)
    float    alpha_init;            // dory.alpha_init (0.05) -- residual interpolation factor
    float    gate_init;             // dory.gate_init (= sqrt(n_embd) = 50.596) -- gate logit scale

    float    rms_eps;               // (no norm weights exist; kept for the L2 norm's eps path)

    // --- layer kinds / provenance ---
    // dory.layer_pattern: one char per physical block: '-' = MLP, 'S' = sliding, '*' = full.
    char     layer_pattern[DORY_MAX_BLOCK_COUNT];
    uint32_t n_pattern;             // == block_count when the string was well-formed
    enum dory_kv_mode kv_mode_file; // dory.recurrent_kv_cache_mode as recorded in the file
};

// Parses every dory.* key this module uses from a loaded gguf context (the same one handed to
// ggml_init for the weights). Returns false (with a message on stderr) if the file isn't a dory
// model or required metadata is missing/inconsistent -- including the nGPT constants
// (init_std / alpha_init / gate_init must be > 0, the reference implementation requires them).
bool dory_load_hparams(struct dory_hparams * hp, const struct gguf_context * ctx);

struct dory_layer {
    uint32_t             index;     // physical block number as stored in the file
    enum dory_layer_role role;
    char                 pattern;   // '-' | 'S' | '*' (layer_pattern char for this block)
    bool                 full_attn; // true for '*' blocks (global attention, ring = n_ctx)

    // Every block has this one: the nGPT residual interpolation factor, see point 4.
    struct ggml_tensor * residual_alpha;   // BF16 [n_embd]

    // Attention half -- NULL when this block is an MLP block.
    struct ggml_tensor * attn_q;           // Q5_K [n_embd, n_head    * n_embd_head]
    struct ggml_tensor * attn_k;           // Q5_K [n_embd, n_head_kv * n_embd_head]
    struct ggml_tensor * attn_v;           // Q6_K [n_embd, n_head_kv * n_embd_head_v]
    struct ggml_tensor * attn_output;      // Q5_K [n_head * n_embd_head, n_embd]
    struct ggml_tensor * attn_gate;        // Q5_K [n_embd, n_head * n_embd_head]
    struct ggml_tensor * attn_qk_scale;    // BF16 [n_head * n_embd_head]
    struct ggml_tensor * attn_gate_scale;  // BF16 [n_head * n_embd_head]

    // MLP half -- NULL when this block is an attention block.
    struct ggml_tensor * ffn_gate;         // Q5_K [n_embd, n_ff]
    struct ggml_tensor * ffn_up;           // Q5_K [n_embd, n_ff]
    struct ggml_tensor * ffn_down;         // Q6_K [n_ff, n_embd]
    struct ggml_tensor * ffn_gate_scale;   // BF16 [n_ff]
    struct ggml_tensor * ffn_up_scale;     // BF16 [n_ff]
};

struct dory_model {
    struct dory_hparams hparams;
    struct ggml_tensor * token_embd;   // [n_embd, n_vocab]
    struct ggml_tensor * output;       // lm head; NULL => tied to token_embd
    struct ggml_tensor * output_scale; // [n_vocab], optional; the logit scale (point 4)
    struct dory_layer    layers[DORY_MAX_BLOCK_COUNT];
};

// Binds every tensor this architecture needs from `wctx` (the context holding the loaded
// weights), resolves the layer pattern onto the blocks, and fails hard on missing tensors
// (stderr + exit(1), repo convention).
void dory_load_model(struct dory_model * m, const struct dory_hparams * hp,
                     struct ggml_context * wctx);

struct dory_kv_entry {
    struct ggml_tensor * k; // F16 [n_embd_head,   n_head_kv, n_cells, n_slots]; NULL for MLP blocks
    struct ggml_tensor * v; // F16 [n_embd_head_v, n_head_kv, n_cells, n_slots]
    uint32_t             n_cells;     // ring size: n_swa for 'S', n_ctx for '*'
    uint32_t             n_slots;     // 1, or num_loops for per-loop recurrent blocks
    uint32_t             layer_index;
    bool                 recurrent;
};

struct dory_kv_cache {
    enum dory_kv_mode    mode;
    uint32_t             n_ctx;           // runtime context length actually used
    uint32_t             n_cells_sliding; // ring size of 'S' blocks (n_swa)
    uint32_t             n_cells_full;    // ring size of '*' blocks (n_ctx)
    uint32_t             n_cells_min;     // smallest ring across entries -- prefill chunk limit
    bool                 has_sliding;
    bool                 has_full;
    struct dory_kv_entry entries[DORY_MAX_BLOCK_COUNT]; // indexed by physical block index
};

// Creates the per-block K/V ring tensors in `ctx` (the caller owns a backend buffer for them,
// same contract as init_kv_cache() in model.cpp). `n_ctx` is the runtime context length used to
// size global-attention rings; pass 0 to fall back to the trained context length. Tensors
// intentionally start uninitialized: dory_fill_attn_mask() makes never-written cells unreadable,
// so zeroing is not required.
void dory_kv_cache_init(struct dory_kv_cache * kv, const struct dory_model * m,
                        struct ggml_context * ctx, enum dory_kv_mode mode, uint32_t n_ctx);

// Bytes the cache will occupy at `mode` / `n_ctx`; mirrors dory_kv_cache_init() exactly. For
// this checkpoint (27 sliding + 9 global attention blocks, 15/5 of them recurrent, 1024-cell
// sliding rings = 4 MiB/slot, global rings = n_ctx * 4 KiB/slot):
//   sliding: per_loop 288 MiB -> last_loop 108 MiB (at num_loops=4; scales with num_loops)
//   global : per_loop 24 slots -> last_loop 9 slots (scales with n_ctx)
size_t dory_kv_cache_bytes(const struct dory_model * m, enum dory_kv_mode mode, uint32_t n_ctx);

// Affine decomposition of the cache cost, for the loader's auto context sizing:
//   bytes(n_ctx) == base + per_ctx * n_ctx
// where `base` is the fixed sliding-ring footprint (independent of n_ctx) and `per_ctx` the
// per-context-token footprint of the global-attention rings. The generic budget/per-token
// division does not fit a cache whose sliding part does not grow with the context, so
// model.cpp's auto_n_ctx() uses this instead. Either output may be NULL.
void dory_kv_cost(const struct dory_model * m, enum dory_kv_mode mode,
                  size_t * base, size_t * per_ctx);

// Fills a host-side [n_cells, n_tokens] F16 attention mask for a batch at absolute positions
// [pos0, pos0 + n_tokens): 0.0f = cell visible, -INFINITY = masked. Pass the ring size of the
// block kind being masked (n_cells_sliding / n_cells_full). A mask of one kind is identical for
// every block and every loop iteration of that kind, which is why one mask tensor per kind is
// shared by the whole graph.
void dory_fill_attn_mask(ggml_fp16_t * dst, uint32_t n_cells, int64_t pos0, int64_t n_tokens);

struct dory_forward {
    struct ggml_tensor * tokens;     // I32 [n_tokens]                    -- host-fill: token ids
    struct ggml_tensor * positions;  // I32 [n_tokens]                    -- host-fill: pos0 + i
    struct ggml_tensor * mask;       // F16 [n_cells_sliding, n_tokens]   -- host-fill; NULL if
                                     // the model has no sliding blocks
    struct ggml_tensor * mask_full;  // F16 [n_cells_full,    n_tokens]   -- host-fill; NULL if
                                     // the model has no global blocks
    struct ggml_tensor * logits;     // F32 [n_vocab, n_tokens]
    struct ggml_cgraph * gf;
};

// Builds the complete forward graph:
//   input blocks -> num_loops x recurrent blocks -> output blocks -> lm head -> logit scale
//   (with the per-loop reinjection h = L2(h + input_state) and the nGPT residual updates).
// K/V writes into `kv` are expanded into `gf` as they are built, so they execute before the
// attention that reads them (same write-then-attend ordering as model.cpp's build_graph).
//
// Usage (mirrors forward_step() in model.cpp):
//   1. ctx = ggml_init({ .mem_size = dory_forward_ctx_bytes(), .no_alloc = true });
//   2. f = dory_forward_pass(ctx, &m, &kv, n_past, n_tokens);
//   3. ggml_gallocr_alloc_graph(galloc, f.gf);
//   4. ggml_backend_tensor_set() for tokens / positions / mask / mask_full (masks via
//      dory_fill_attn_mask() with kv.n_cells_sliding / kv.n_cells_full);
//   5. ggml_backend_graph_compute(backend, f.gf);
//   6. read the logits of the last token back from f.logits.
//
// Prefill must be chunked to n_tokens <= kv->n_cells_min (asserted): a batch wider than a ring
// would overwrite cells the same batch still has to attend to.
struct dory_forward dory_forward_pass(struct ggml_context * ctx, const struct dory_model * m,
                                      struct dory_kv_cache * kv, int64_t pos0, int64_t n_tokens);

// Context bytes one dory_forward_pass() needs (tensor objects + graph), for ggml_init().
size_t dory_forward_ctx_bytes(void);

#ifdef __cplusplus
}
#endif
