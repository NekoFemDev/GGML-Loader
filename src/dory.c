// dory -- implementation. See dory.h for the architecture notes, the tensor inventory, the
// normalized-GPT (nGPT) block math, and the KV-cache design. Plain C11 on top of ggml/gguf.
#include "dory.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================================
// gguf metadata helpers -- typed reads that tolerate the common integer spellings so a
// re-converted checkpoint doesn't fail just because a count was stored as i64 instead of u32.
// =============================================================================================

static uint32_t dory_u32(const struct gguf_context * ctx, const char * key, uint32_t fallback) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return fallback;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return (uint32_t) gguf_get_val_i64(ctx, id);
        default:               return fallback;
    }
}

static float dory_f32_meta(const struct gguf_context * ctx, const char * key, float fallback) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return fallback;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(ctx, id);
        default:                return fallback;
    }
}

static const char * dory_str(const struct gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return NULL;
    }
    return gguf_get_val_str(ctx, id);
}

// =============================================================================================
// hparams
// =============================================================================================

bool dory_load_hparams(struct dory_hparams * hp, const struct gguf_context * ctx) {
    memset(hp, 0, sizeof(*hp));

    const char * arch = dory_str(ctx, "general.architecture");
    if (arch == NULL || strcmp(arch, "dory") != 0) {
        fprintf(stderr, "dory: general.architecture = \"%s\", expected \"dory\"\n",
                arch != NULL ? arch : "(missing)");
        return false;
    }

    // topology: three consecutive ranges that must tile the file [0, block_count)
    hp->block_count           = dory_u32(ctx, "dory.block_count", 0);
    hp->input_layer_count     = dory_u32(ctx, "dory.input_layer_count", 0);
    hp->recurrent_layer_count = dory_u32(ctx, "dory.recurrent_layer_count", 0);
    hp->output_layer_count    = dory_u32(ctx, "dory.output_layer_count", 0);
    hp->num_loops             = dory_u32(ctx, "dory.num_loops", 1);
    {
        // Runtime override (DORY_LOOPS=N): k drives the decode cost directly -- every forward
        // step executes 16 + 40*k + 16 blocks, and the recurrent weights are re-read from RAM
        // on every loop (no on-chip cache reuse at this size), so k=4 costs ~2x the per-token
        // bandwidth of k=2. The checkpoint ships 4; the model card recommends 2 for speed.
        const char * env = getenv("DORY_LOOPS");
        if (env != NULL && env[0] != '\0') {
            const long v = atol(env);
            if (v < 1 || v > 5) {
                // >4 exceeds what DORY_GRAPH_SIZE (8192 nodes) was budgeted for; 5 is the
                // largest value that still fits with margin
                fprintf(stderr, "dory: DORY_LOOPS=%s is out of range [1, 5]\n", env);
                return false;
            }
            fprintf(stderr, "dory: num_loops %u -> %ld (DORY_LOOPS)\n", hp->num_loops, v);
            hp->num_loops = (uint32_t) v;
        }
    }

    // widths
    hp->n_embd        = dory_u32(ctx, "dory.embedding_length", 0);
    hp->n_ff          = dory_u32(ctx, "dory.feed_forward_length", 0);
    hp->n_head        = dory_u32(ctx, "dory.attention.head_count", 0);
    hp->n_head_kv     = dory_u32(ctx, "dory.attention.head_count_kv", hp->n_head);
    hp->n_embd_head   = dory_u32(ctx, "dory.attention.key_length",
                                 hp->n_head != 0 ? hp->n_embd / hp->n_head : 0);
    hp->n_embd_head_v = dory_u32(ctx, "dory.attention.value_length", hp->n_embd_head);
    hp->n_rot         = dory_u32(ctx, "dory.rope.dimension_count", hp->n_embd_head);

    // context / attention
    hp->n_ctx_train = dory_u32(ctx, "dory.context_length", 0);
    hp->n_swa       = dory_u32(ctx, "dory.attention.sliding_window", 0);

    // rope
    hp->rope_freq_base   = dory_f32_meta(ctx, "dory.rope.freq_base", 10000.0f);
    hp->rope_freq_base_2 = dory_f32_meta(ctx, "dory.rope.freq_base_2", hp->rope_freq_base);
    hp->init_std         = dory_f32_meta(ctx, "dory.init_std", 0.0f);
    hp->alpha_init       = dory_f32_meta(ctx, "dory.alpha_init", 0.0f);
    hp->gate_init        = dory_f32_meta(ctx, "dory.gate_init", 0.0f);
    hp->rms_eps          = dory_f32_meta(ctx, "dory.attention.layer_norm_rms_epsilon", 1e-5f);

    // The reference implementation groups DORY with the split-half (NEOX) rotaries, not the
    // interleaved ones. Kept overridable via an optional key for future variants; if a
    // conversion ever turns out to scramble positions, this is the flag to flip.
    hp->rope_neox = true;
    const char * rope_layout = dory_str(ctx, "dory.rope.mode");
    if (rope_layout != NULL && strcmp(rope_layout, "normal") == 0) {
        hp->rope_neox = false;
    }

    // dory.rope.profile: one value per physical BLOCK (72 here), 1 or 2:
    //   1 -> rope_freq_base   (1e7) -- the global-attention ('*') blocks
    //   2 -> rope_freq_base_2 (1e4) -- the sliding ('S') blocks
    // Keep the raw value -- a != 0 boolean test would collapse the two profiles into one.
    const int64_t prof_id = gguf_find_key(ctx, "dory.rope.profile");
    if (prof_id >= 0 && gguf_get_arr_type(ctx, prof_id) == GGUF_TYPE_INT32) {
        const size_t    n    = gguf_get_arr_n(ctx, prof_id);
        const int32_t * prof = (const int32_t *) gguf_get_arr_data(ctx, prof_id);
        for (uint32_t il = 0; il < hp->block_count && (size_t) il < n; il++) {
            hp->rope_profile[il] = (uint8_t) prof[il];
        }
        hp->has_rope_profile = n > 0;
    }

    // dory.layer_pattern: ONE CHARACTER PER PHYSICAL BLOCK -- '-' = MLP block, 'S' = sliding
    // attention, '*' = full attention. (The dashes are layer entries, not separators: the
    // string length equals block_count for this checkpoint.) dory_load_model() maps it onto
    // the blocks; when absent it falls back to tensor presence + rope profile.
    const char * pattern = dory_str(ctx, "dory.layer_pattern");
    if (pattern != NULL) {
        const size_t n = strlen(pattern);
        if (n == hp->block_count && strspn(pattern, "S*-") == n) {
            memcpy(hp->layer_pattern, pattern, n);
            hp->n_pattern = (uint32_t) n;
        } else {
            fprintf(stderr, "dory: layer_pattern \"%s\" is not %u chars of [S*-]; ignoring\n",
                    pattern, hp->block_count);
        }
    }

    // runtime KV-cache mode recorded in the file (the caller may still ask for last-loop)
    hp->kv_mode_file = DORY_KV_PER_LOOP;
    const char * kv_mode = dory_str(ctx, "dory.recurrent_kv_cache_mode");
    if (kv_mode != NULL) {
        if (strcmp(kv_mode, "per_loop") == 0) {
            hp->kv_mode_file = DORY_KV_PER_LOOP;
        } else if (strcmp(kv_mode, "last_loop") == 0) {
            hp->kv_mode_file = DORY_KV_LAST_LOOP;
        } else {
            fprintf(stderr, "dory: unknown recurrent_kv_cache_mode \"%s\"\n", kv_mode);
        }
    }

    // validation
    if (hp->block_count == 0 || hp->n_embd == 0 || hp->n_ff == 0 || hp->n_head == 0) {
        fprintf(stderr, "dory: missing core metadata "
                        "(block_count / embedding_length / feed_forward_length / head_count)\n");
        return false;
    }
    if (hp->block_count > DORY_MAX_BLOCK_COUNT) {
        fprintf(stderr, "dory: block_count %u exceeds DORY_MAX_BLOCK_COUNT (%d)\n",
                hp->block_count, DORY_MAX_BLOCK_COUNT);
        return false;
    }
    if (hp->input_layer_count + hp->recurrent_layer_count + hp->output_layer_count != hp->block_count) {
        fprintf(stderr, "dory: input+recurrent+output layer counts (%u+%u+%u) != block_count (%u)\n",
                hp->input_layer_count, hp->recurrent_layer_count, hp->output_layer_count,
                hp->block_count);
        return false;
    }
    if (hp->num_loops == 0) {
        fprintf(stderr, "dory: num_loops must be >= 1\n");
        return false;
    }
    if (hp->n_head_kv == 0 || hp->n_head % hp->n_head_kv != 0) {
        fprintf(stderr, "dory: head_count (%u) must be a multiple of head_count_kv (%u)\n",
                hp->n_head, hp->n_head_kv);
        return false;
    }
    // the nGPT constants are functional -- the forward pass divides by init_std and takes
    // sqrt(gate_init); the reference implementation requires all three to be present and > 0
    if (!(hp->init_std > 0.0f) || !(hp->alpha_init > 0.0f) || !(hp->gate_init > 0.0f)) {
        fprintf(stderr, "dory: init_std/alpha_init/gate_init must be present and > 0 "
                        "(got %g / %g / %g)\n", hp->init_std, hp->alpha_init, hp->gate_init);
        return false;
    }

    return true;
}

// =============================================================================================
// model / weight binding
// =============================================================================================

static enum dory_layer_role dory_role_of(const struct dory_hparams * hp, uint32_t il) {
    if (il < hp->input_layer_count) {
        return DORY_ROLE_INPUT;
    }
    if (il < hp->input_layer_count + hp->recurrent_layer_count) {
        return DORY_ROLE_RECURRENT;
    }
    return DORY_ROLE_OUTPUT;
}

static struct ggml_tensor * dory_blk_tensor(struct ggml_context * ctx, uint32_t il,
                                            const char * suffix) {
    char name[GGML_MAX_NAME];
    snprintf(name, sizeof(name), "blk.%u.%s.weight", il, suffix);
    return ggml_get_tensor(ctx, name);
}

static void dory_require(struct ggml_tensor * t, uint32_t il, const char * suffix) {
    if (t == NULL) {
        fprintf(stderr, "dory: required tensor blk.%u.%s.weight is missing\n", il, suffix);
        exit(1);
    }
}

void dory_load_model(struct dory_model * m, const struct dory_hparams * hp,
                     struct ggml_context * wctx) {
    memset(m, 0, sizeof(*m));
    m->hparams = *hp;

    m->token_embd = ggml_get_tensor(wctx, "token_embd.weight");
    if (m->token_embd == NULL) {
        fprintf(stderr, "dory: token_embd.weight is missing\n");
        exit(1);
    }
    m->output       = ggml_get_tensor(wctx, "output.weight");       // optional: tied if absent
    m->output_scale = ggml_get_tensor(wctx, "output_scale.weight"); // optional

    assert(m->token_embd->ne[0] == (int64_t) hp->n_embd);

    const bool have_pattern = (hp->n_pattern == hp->block_count);
    bool warned_mismatch = false;

    for (uint32_t il = 0; il < hp->block_count; il++) {
        struct dory_layer * l = &m->layers[il];
        l->index = il;
        l->role  = dory_role_of(hp, il);

        // Attention and MLP halves are bound independently; the pattern (or, failing that,
        // tensor presence) decides which one this block is.
        l->attn_q = dory_blk_tensor(wctx, il, "attn_q");
        if (l->attn_q != NULL) {
            l->attn_k          = dory_blk_tensor(wctx, il, "attn_k");
            l->attn_v          = dory_blk_tensor(wctx, il, "attn_v");
            l->attn_output     = dory_blk_tensor(wctx, il, "attn_output");
            l->attn_gate       = dory_blk_tensor(wctx, il, "attn_gate");
            l->attn_qk_scale   = dory_blk_tensor(wctx, il, "attn_qk_scale");
            l->attn_gate_scale = dory_blk_tensor(wctx, il, "attn_gate_scale");
            dory_require(l->attn_k,          il, "attn_k");
            dory_require(l->attn_v,          il, "attn_v");
            dory_require(l->attn_output,     il, "attn_output");
            dory_require(l->attn_gate,       il, "attn_gate");
            dory_require(l->attn_qk_scale,   il, "attn_qk_scale");
            dory_require(l->attn_gate_scale, il, "attn_gate_scale");

            assert(l->attn_q->ne[1] == (int64_t) hp->n_head * hp->n_embd_head);
            assert(l->attn_k->ne[1] == (int64_t) hp->n_head_kv * hp->n_embd_head);
            assert(l->attn_output->ne[0] == (int64_t) hp->n_head * hp->n_embd_head);
        }

        l->ffn_gate = dory_blk_tensor(wctx, il, "ffn_gate");
        if (l->ffn_gate != NULL) {
            l->ffn_up         = dory_blk_tensor(wctx, il, "ffn_up");
            l->ffn_down       = dory_blk_tensor(wctx, il, "ffn_down");
            l->ffn_gate_scale = dory_blk_tensor(wctx, il, "ffn_gate_scale");
            l->ffn_up_scale   = dory_blk_tensor(wctx, il, "ffn_up_scale");
            dory_require(l->ffn_up,         il, "ffn_up");
            dory_require(l->ffn_down,       il, "ffn_down");
            dory_require(l->ffn_gate_scale, il, "ffn_gate_scale");
            dory_require(l->ffn_up_scale,   il, "ffn_up_scale");

            assert(l->ffn_gate->ne[1] == (int64_t) hp->n_ff);
        }

        if (l->attn_q == NULL && l->ffn_gate == NULL) {
            fprintf(stderr, "dory: blk.%u carries neither attention nor MLP tensors\n", il);
            exit(1);
        }

        // resolve the block kind: pattern first (it is the architecture's own declaration),
        // else 'S' attention / '-' MLP by presence
        if (have_pattern) {
            const char pc = hp->layer_pattern[il];
            l->pattern = pc;
            if ((pc == '-') == (l->attn_q != NULL) && !warned_mismatch) {
                fprintf(stderr, "dory: layer_pattern disagrees with tensor presence at blk.%u"
                                " ('%c' vs '%s'); using tensor presence for the kind\n",
                        il, pc, l->attn_q != NULL ? "attention" : "mlp");
                warned_mismatch = true;
            }
            l->full_attn = (pc == '*') && l->attn_q != NULL;
        } else {
            l->pattern   = l->attn_q != NULL ? 'S' : '-';
            // no pattern string: the rope profile carries the same split (value 1 = the '*'
            // blocks, value 2 = the 'S' blocks)
            l->full_attn = (l->attn_q != NULL) &&
                           (hp->has_rope_profile ? (hp->rope_profile[il] == 1) : false);
        }

        l->residual_alpha = dory_blk_tensor(wctx, il, "residual_alpha");
        dory_require(l->residual_alpha, il, "residual_alpha");
    }

    // Backfill the rope profile when the explicit array was absent: '*' blocks -> 1 (1e7),
    // 'S' blocks -> 2 (1e4). Uses the m copy so the caller's hparams stay untouched.
    if (!hp->has_rope_profile) {
        for (uint32_t il = 0; il < hp->block_count; il++) {
            const struct dory_layer * l = &m->layers[il];
            if (l->attn_q != NULL) {
                m->hparams.rope_profile[il] = l->full_attn ? 1 : 2;
            }
        }
    }
}

// =============================================================================================
// KV cache
// =============================================================================================

// Slot count for one entry. Only recurrent attention blocks can ever need more than one ring:
// they are the ones executed num_loops times, and only per_loop mode keeps those rings private.
static uint32_t dory_kv_slots_of(const struct dory_layer * l, const struct dory_hparams * hp,
                                 enum dory_kv_mode mode) {
    if (l->attn_q != NULL && l->role == DORY_ROLE_RECURRENT && mode == DORY_KV_PER_LOOP) {
        return hp->num_loops;
    }
    return 1;
}

static uint32_t dory_ctx_len(const struct dory_model * m, uint32_t n_ctx) {
    const struct dory_hparams * hp = &m->hparams;

    uint32_t ctx_len = n_ctx > 0 ? n_ctx : hp->n_ctx_train;
    if (ctx_len == 0) {
        ctx_len = 4096;
    }
    if (hp->n_ctx_train > 0 && ctx_len > hp->n_ctx_train) {
        ctx_len = hp->n_ctx_train;
    }
    return ctx_len;
}

static void dory_kv_layout(const struct dory_model * m, uint32_t n_ctx,
                           uint32_t * out_sliding, uint32_t * out_full, uint32_t * out_ctx) {
    const struct dory_hparams * hp = &m->hparams;
    const uint32_t ctx_len = dory_ctx_len(m, n_ctx);

    *out_ctx     = ctx_len;
    // 'S' sliding blocks can never attend past one window, so their rings are fixed at n_swa;
    // '*' global blocks need the whole runtime context.
    *out_sliding = hp->n_swa > 0 ? hp->n_swa : ctx_len;
    *out_full    = ctx_len;
}

// Affine cost decomposition -- keep in lockstep with dory_kv_cache_init's per-entry sizing.
void dory_kv_cost(const struct dory_model * m, enum dory_kv_mode mode,
                  size_t * base, size_t * per_ctx) {
    const struct dory_hparams * hp = &m->hparams;

    size_t b = 0, p = 0;
    for (uint32_t il = 0; il < hp->block_count; il++) {
        const struct dory_layer * l = &m->layers[il];
        if (l->attn_q == NULL) {
            continue;
        }
        const size_t per_cell = ((size_t) hp->n_embd_head + hp->n_embd_head_v) *
                                hp->n_head_kv * sizeof(ggml_fp16_t);
        const size_t slots = dory_kv_slots_of(l, hp, mode);
        if (l->full_attn || hp->n_swa == 0) {
            p += per_cell * slots;                  // ring size tracks n_ctx
        } else {
            b += per_cell * hp->n_swa * slots;      // ring size is the fixed window
        }
    }
    if (base    != NULL) *base    = b;
    if (per_ctx != NULL) *per_ctx = p;
}

void dory_kv_cache_init(struct dory_kv_cache * kv, const struct dory_model * m,
                        struct ggml_context * ctx, enum dory_kv_mode mode, uint32_t n_ctx) {
    const struct dory_hparams * hp = &m->hparams;

    memset(kv, 0, sizeof(*kv));
    kv->mode = mode;
    dory_kv_layout(m, n_ctx, &kv->n_cells_sliding, &kv->n_cells_full, &kv->n_ctx);
    assert(kv->n_cells_sliding > 0 && kv->n_cells_full > 0);

    kv->n_cells_min = UINT32_MAX;

    for (uint32_t il = 0; il < hp->block_count; il++) {
        const struct dory_layer * l = &m->layers[il];
        struct dory_kv_entry * e = &kv->entries[il];

        e->layer_index = il;
        e->recurrent   = (l->role == DORY_ROLE_RECURRENT);

        if (l->attn_q == NULL) {
            continue; // MLP block: no attention state
        }

        e->n_cells = l->full_attn ? kv->n_cells_full : kv->n_cells_sliding;
        e->n_slots = dory_kv_slots_of(l, hp, mode);

        // Ring per (block, slot). last-loop collapses n_slots to 1 for recurrent blocks, which
        // is exactly the "all loops overwrite the same memory" trade declared in dory.h.
        e->k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,
                                  hp->n_embd_head, hp->n_head_kv, e->n_cells, e->n_slots);
        e->v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,
                                  hp->n_embd_head_v, hp->n_head_kv, e->n_cells, e->n_slots);

        char name[GGML_MAX_NAME];
        snprintf(name, sizeof(name), "dory.kv.%u.k", il);
        ggml_set_name(e->k, name);
        snprintf(name, sizeof(name), "dory.kv.%u.v", il);
        ggml_set_name(e->v, name);

        if (e->n_cells < kv->n_cells_min) {
            kv->n_cells_min = e->n_cells;
        }
        kv->has_sliding |= !l->full_attn;
        kv->has_full    |=  l->full_attn;
    }

    if (kv->n_cells_min == UINT32_MAX) {
        kv->n_cells_min = kv->n_cells_sliding; // model with no attention blocks: keep it sane
    }
}

size_t dory_kv_cache_bytes(const struct dory_model * m, enum dory_kv_mode mode, uint32_t n_ctx) {
    size_t base = 0, per_ctx = 0;
    dory_kv_cost(m, mode, &base, &per_ctx);
    return base + per_ctx * (size_t) dory_ctx_len(m, n_ctx);
}

// floor-mod, the single addressing primitive for the ring (C's % keeps the sign otherwise).
static int64_t dory_mod(int64_t a, int64_t n) {
    const int64_t r = a % n;
    return r < 0 ? r + n : r;
}

// Slot (4th tensor dim) used by a given loop iteration.
//   per-loop recurrent block: slot = loop index -> a private ring per loop.
//   everything else (input/output blocks; all blocks in last-loop mode): slot = 0, so every
//   loop writes the same cells and only the most recent loop's K/V survives per position.
static uint32_t dory_kv_slot(const struct dory_kv_entry * e, uint32_t loop_idx) {
    return e->n_slots > 1 ? loop_idx : 0;
}

struct dory_kv_run {
    uint32_t off; // first cell in the ring
    uint32_t len; // number of cells
};

// A batch at positions [pos0, pos0+n) lands on cells pos0%n .. (pos0+n-1)%n, which is at most
// two contiguous runs: [start, n_cells) then [0, ...). No shifting/memmove on any path.
static int dory_kv_runs(int64_t pos0, uint32_t n, uint32_t n_cells, struct dory_kv_run runs[2]) {
    const uint32_t start = (uint32_t) dory_mod(pos0, (int64_t) n_cells);
    if (start + n <= n_cells) {
        runs[0].off = start;
        runs[0].len = n;
        return 1;
    }
    runs[0].off = start;
    runs[0].len = n_cells - start;
    runs[1].off = 0;
    runs[1].len = n - runs[0].len;
    return 2;
}

// Copies this batch's K (post-RoPE) and V into the ring: one ggml_cpy per contiguous run, so at
// most two per tensor. The cpy nodes are expanded into `gf` immediately -- before the
// flash-attn node that reads the same ring is created -- which is what makes write-then-attend
// ordering explicit in the graph (ggml executes in build order for independent nodes).
static void dory_kv_write(struct ggml_context * ctx, struct ggml_cgraph * gf,
                          const struct dory_kv_entry * e, uint32_t slot,
                          struct ggml_tensor * k_new, struct ggml_tensor * v_new,
                          int64_t pos0, uint32_t n_tokens) {
    assert(e->k != NULL && e->v != NULL);
    assert((int64_t) k_new->ne[2] == (int64_t) n_tokens);
    assert((int64_t) v_new->ne[2] == (int64_t) n_tokens);
    assert((int64_t) slot < (int64_t) e->n_slots);

    struct dory_kv_run runs[2];
    const int n_runs = dory_kv_runs(pos0, n_tokens, (uint32_t) e->k->ne[2], runs);

    uint32_t tok = 0;
    for (int r = 0; r < n_runs; r++) {
        const uint32_t len = runs[r].len;

        const size_t k_src_off = (size_t) tok * k_new->nb[2];
        const size_t v_src_off = (size_t) tok * v_new->nb[2];
        const size_t k_dst_off = (size_t) slot * e->k->nb[3] + (size_t) runs[r].off * e->k->nb[2];
        const size_t v_dst_off = (size_t) slot * e->v->nb[3] + (size_t) runs[r].off * e->v->nb[2];

        struct ggml_tensor * k_src = ggml_view_3d(ctx, k_new, k_new->ne[0], k_new->ne[1], len,
                                                  k_new->nb[1], k_new->nb[2], k_src_off);
        struct ggml_tensor * v_src = ggml_view_3d(ctx, v_new, v_new->ne[0], v_new->ne[1], len,
                                                  v_new->nb[1], v_new->nb[2], v_src_off);
        struct ggml_tensor * k_dst = ggml_view_3d(ctx, e->k, e->k->ne[0], e->k->ne[1], len,
                                                  e->k->nb[1], e->k->nb[2], k_dst_off);
        struct ggml_tensor * v_dst = ggml_view_3d(ctx, e->v, e->v->ne[0], e->v->ne[1], len,
                                                  e->v->nb[1], e->v->nb[2], v_dst_off);

        // F32 K/V source -> F16 ring: ggml_cpy converts the dtype as it copies.
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k_src, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v_src, v_dst));

        tok += len;
    }
}

// [n_embd_head(_v), n_head_kv, n_cells, 1] view of one slot's ring -- what flash-attn reads
// after the permute to [..., n_cells, n_head_kv, 1].
static struct ggml_tensor * dory_kv_view(struct ggml_context * ctx, struct ggml_tensor * t,
                                         uint32_t slot) {
    return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], 1,
                        t->nb[1], t->nb[2], t->nb[3], (size_t) slot * t->nb[3]);
}

// =============================================================================================
// attention mask
// =============================================================================================

// Shared mask for one block kind, [n_cells, n_tokens] row-major (dst[query * n_cells + cell]),
// 0.0f = visible, -INFINITY = masked. Valid for every attention block and loop iteration of
// that kind (identical ring geometry; the loop index does not change the geometry).
//
// A cell is visible to a query at absolute position p iff the newest write into that cell is a
// real position not lying in p's future:
//
//   last_write = head - ((head - cell) mod n_cells)      head = last position this batch writes
//   visible    = (last_write >= 0) && (last_write <= p)
//
// The first clause excludes never-written cells (session start; the modulo would otherwise
// alias them onto future positions); the second excludes cells a LATER token of this same
// batch has already overwritten -- the one ring-aliasing hazard of a shared window, where a
// naive causal mask would leak future tokens into earlier queries during prefill. With
// n_cells == the ring's window, "last_write <= p" also implies the cell lies inside the
// window (the ring only ever holds the newest n_cells writes), so no separate window bound
// is needed.
void dory_fill_attn_mask(ggml_fp16_t * dst, uint32_t n_cells, int64_t pos0, int64_t n_tokens) {
    assert(n_cells >= 1 && n_tokens >= 1);

    const int64_t n    = (int64_t) n_cells;
    const int64_t head = pos0 + n_tokens - 1;

    for (int64_t q = 0; q < n_tokens; q++) {
        const int64_t p = pos0 + q;
        for (int64_t c = 0; c < n; c++) {
            const int64_t last_write = head - dory_mod(head - c, n);
            const bool    visible    = (last_write >= 0) && (last_write <= p);
            dst[q * n + c] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }
}

// =============================================================================================
// graph helpers
// =============================================================================================

// nGPT L2 normalization: norm(x) = x / sqrt(sum(x^2) + 1e-6). The reference implementation
// expresses it as rms_norm(x, eps/ne0) * 1/sqrt(ne0), which is algebraically identical -- the
// two ops below are exactly that. Applied over ne0 (per head for 3D Q/K tensors, per token for
// 2D activation matrices). This is why the checkpoint has no norm weights at all.
static struct ggml_tensor * dory_l2_norm(struct ggml_context * ctx, struct ggml_tensor * x) {
    const float n = (float) x->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f / n), 1.0f / sqrtf(n));
}

// The per-channel scale tensors (qk/gate/ffn scales, residual_alpha, output_scale) are BF16 in
// the file while activations are F32, and ggml's elementwise ops want matching dtypes -- cast
// on use. These tensors are tiny (<= 16 KB), so the cast cost is noise.
static struct ggml_tensor * dory_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

// Per-block RoPE base: profile value 2 uses the secondary base (1e4, the 'S' blocks),
// everything else the primary (1e7, the '*' blocks). See dory_load_hparams.
static float dory_rope_freq_base(const struct dory_hparams * hp, uint32_t il) {
    return hp->rope_profile[il] == 2 ? hp->rope_freq_base_2 : hp->rope_freq_base;
}

// =============================================================================================
// block builders
//
// One block = its half (attention or MLP) plus the shared nGPT residual interpolation. Exact
// op-for-op match to the reference implementation -- see point 4 of the dory.h comment for the
// full recipe; the comments here only explain the non-obvious factors.
// =============================================================================================

static struct ggml_tensor * dory_build_attn_block(
        struct ggml_context * ctx,
        const struct dory_model * m,
        struct dory_kv_cache * kv,
        struct ggml_cgraph * gf,
        const struct dory_layer * l,
        struct ggml_tensor * cur,            // [n_embd, n_tokens]
        struct ggml_tensor * positions,      // I32 [n_tokens]
        struct ggml_tensor * mask_swa_f16,   // F16 [n_cells_sliding, n_tokens]
        struct ggml_tensor * mask_full_f16,  // F16 [n_cells_full,    n_tokens]
        int64_t pos0,
        int64_t n_tokens,
        uint32_t loop_idx) {
    const struct dory_hparams * hp = &m->hparams;

    const int64_t hd     = hp->n_embd_head;
    const int64_t hd_v   = hp->n_embd_head_v;
    const int64_t nh     = hp->n_head;
    const int64_t nk     = hp->n_head_kv;

    struct dory_kv_entry * e = &kv->entries[l->index];
    assert(e->k != NULL); // the cache must cover every attention block

    // Each kind gets its own mask; both are pre-cast F16 and shared across blocks/loops.
    struct ggml_tensor * mask_f16 = l->full_attn ? mask_full_f16 : mask_swa_f16;
    assert(mask_f16 != NULL);

    // Q/K/V straight from the block input -- no pre-norm: nGPT keeps the state normalized at
    // block boundaries (the residual update below is what re-normalizes).
    struct ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l->attn_q, cur), hd, nh, n_tokens);
    struct ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l->attn_k, cur), hd, nk, n_tokens);
    struct ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l->attn_v, cur), hd_v, nk, n_tokens);

    // RoPE first, full width, NEOX (split-half), base selected per block by the profile.
    const int   rope_mode = hp->rope_neox ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;
    const float theta     = dory_rope_freq_base(hp, l->index);
    q = ggml_rope_ext(ctx, q, positions, NULL, (int) hp->n_rot, rope_mode, (int) hp->n_ctx_train,
                      theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, (int) hp->n_rot, rope_mode, (int) hp->n_ctx_train,
                      theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // nGPT QK handling, after rope: q is L2-normalized then multiplied by the SQUARED learned
    // per-channel scale (scaled by sqrt(hd)/init_std before squaring); k is only L2-normalized.
    struct ggml_tensor * sq = ggml_sqr(ctx, ggml_scale(ctx, dory_f32(ctx, l->attn_qk_scale),
                                                        sqrtf((float) hd) / hp->init_std));
    q = ggml_mul(ctx, dory_l2_norm(ctx, q), ggml_reshape_3d(ctx, sq, hd, nh, 1));
    k = dory_l2_norm(ctx, k);

    // attention output gate, projected from the same block input; the gate logits carry a
    // squared learned scale, so the trained gate is close to binary
    struct ggml_tensor * gate = ggml_mul_mat(ctx, l->attn_gate, cur);       // [nh*hd, n_tokens]
    struct ggml_tensor * sg = ggml_sqr(ctx, ggml_scale(ctx, dory_f32(ctx, l->attn_gate_scale),
                                                        sqrtf(hp->gate_init) / hp->init_std));
    gate = ggml_sigmoid(ctx, ggml_mul(ctx, gate, sg));

    // Write first (expands the cpy nodes into gf), then attend over the ring.
    const uint32_t slot = dory_kv_slot(e, loop_idx);
    dory_kv_write(ctx, gf, e, slot, k, v, pos0, (uint32_t) n_tokens);

    struct ggml_tensor * K = ggml_permute(ctx, dory_kv_view(ctx, e->k, slot), 0, 2, 1, 3); // [hd, n_cells, nk]
    struct ggml_tensor * V = ggml_permute(ctx, dory_kv_view(ctx, e->v, slot), 0, 2, 1, 3);
    struct ggml_tensor * Q = ggml_permute(ctx, q, 0, 2, 1, 3); // [hd, n_tokens, nh]

    const float kq_scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, mask_f16, kq_scale, 0.0f, 0.0f);
    struct ggml_tensor * out = ggml_cont_2d(ctx, attn, nh * hd_v, n_tokens);

    // gated attention output, L2-normalized before the output projection
    out = dory_l2_norm(ctx, ggml_mul(ctx, out, gate));
    return ggml_mul_mat(ctx, l->attn_output, out); // [n_embd, n_tokens]
}

static struct ggml_tensor * dory_build_ffn_block(
        struct ggml_context * ctx,
        const struct dory_hparams * hp,
        const struct dory_layer * l,
        struct ggml_tensor * cur) { // [n_embd, n_tokens]
    // no pre-norm (see the attention block); per-projection learned scales, both multiplied by
    // sqrt(n_embd) -- NOT squared, unlike the attention scales
    const float f = sqrtf((float) hp->n_embd);
    struct ggml_tensor * g = ggml_mul_mat(ctx, l->ffn_gate, cur); // [n_ff, n_tokens]
    struct ggml_tensor * u = ggml_mul_mat(ctx, l->ffn_up,   cur); // [n_ff, n_tokens]
    g = ggml_mul(ctx, g, ggml_scale(ctx, dory_f32(ctx, l->ffn_gate_scale), f));
    u = ggml_mul(ctx, u, ggml_scale(ctx, dory_f32(ctx, l->ffn_up_scale),   f));

    // silu gating (hidden_act = "silu"), L2-normalized before the down projection
    struct ggml_tensor * h = dory_l2_norm(ctx, ggml_mul(ctx, ggml_silu(ctx, g), u));
    return ggml_mul_mat(ctx, l->ffn_down, h); // [n_embd, n_tokens]
}

static struct ggml_tensor * dory_build_block(
        struct ggml_context * ctx,
        const struct dory_model * m,
        struct dory_kv_cache * kv,
        struct ggml_cgraph * gf,
        const struct dory_layer * l,
        struct ggml_tensor * cur,
        struct ggml_tensor * positions,
        struct ggml_tensor * mask_swa_f16,
        struct ggml_tensor * mask_full_f16,
        int64_t pos0,
        int64_t n_tokens,
        uint32_t loop_idx) {
    const struct dory_hparams * hp = &m->hparams;
    struct ggml_tensor * residual = cur;

    // exactly one half per block in this architecture; both would simply run in sequence
    if (l->attn_q != NULL) {
        cur = dory_build_attn_block(ctx, m, kv, gf, l, cur, positions,
                                    mask_swa_f16, mask_full_f16, pos0, n_tokens, loop_idx);
    }
    if (l->ffn_gate != NULL) {
        cur = dory_build_ffn_block(ctx, hp, l, cur);
    }

    // nGPT spherical residual update: interpolate between the L2-normalized block input and the
    // L2-normalized block output (per-channel factor, non-negative), then L2 again.
    struct ggml_tensor * a = dory_l2_norm(ctx, residual);
    struct ggml_tensor * b = dory_l2_norm(ctx, cur);
    struct ggml_tensor * alpha = ggml_abs(ctx, ggml_scale(ctx, dory_f32(ctx, l->residual_alpha),
                                                          hp->alpha_init / hp->init_std));
    return dory_l2_norm(ctx, ggml_add(ctx, a, ggml_mul(ctx, ggml_sub(ctx, b, a), alpha)));
}

// =============================================================================================
// forward pass
// =============================================================================================

struct dory_forward dory_forward_pass(struct ggml_context * ctx, const struct dory_model * m,
                                      struct dory_kv_cache * kv, int64_t pos0, int64_t n_tokens) {
    const struct dory_hparams * hp = &m->hparams;

    // A batch wider than a ring would overwrite cells this same batch still has to attend to
    // (the write of token pos0+n aliases the cell of query pos0). Callers chunk prefill to
    // n_tokens <= n_cells_min, exactly like forward_step()'s ubatch handling in model.cpp.
    assert(n_tokens > 0 && n_tokens <= (int64_t) kv->n_cells_min);

    struct dory_forward f;
    memset(&f, 0, sizeof(f));
    f.gf = ggml_new_graph_custom(ctx, DORY_GRAPH_SIZE, false);

    f.tokens    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    f.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(f.tokens,    "dory.tokens");
    ggml_set_name(f.positions, "dory.positions");

    // One mask per block kind (different ring geometry), created directly in F16 and shared
    // by every flash-attn node of that kind. The full-context mask is [n_ctx, n_tokens] -- the
    // reason prefill chunks should stay small: its transient footprint is 4 bytes per
    // (cell, token) between the F16 tensor and the host staging vector.
    struct ggml_tensor * mask_swa_f16  = NULL;
    struct ggml_tensor * mask_full_f16 = NULL;
    if (kv->has_sliding) {
        f.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv->n_cells_sliding, n_tokens);
        ggml_set_name(f.mask, "dory.mask.swa");
        mask_swa_f16 = f.mask;
    }
    if (kv->has_full) {
        f.mask_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv->n_cells_full, n_tokens);
        ggml_set_name(f.mask_full, "dory.mask.full");
        mask_full_f16 = f.mask_full;
    }

    // Raw token embeddings, no scaling: the first block consumes them as-is and the residual
    // update normalizes the state from there on.
    struct ggml_tensor * cur = ggml_get_rows(ctx, m->token_embd, f.tokens); // [n_embd, n_tokens]

    uint32_t il;

    // ---- phase 1: input blocks, executed once ----------------------------------------------
    for (il = 0; il < hp->input_layer_count; il++) {
        cur = dory_build_block(ctx, m, kv, f.gf, &m->layers[il], cur, f.positions,
                               mask_swa_f16, mask_full_f16, pos0, n_tokens, 0);
    }

    // The input stack's output, reinjected at the start of every recurrent pass.
    struct ggml_tensor * input_h = cur;

    // ---- phase 2: recurrent blocks, num_loops passes over the SAME weights ------------------
    // Per-loop state lives entirely in the KV cache: in last-loop mode every iteration writes
    // the same cells (overwrite in place); in per-loop mode each iteration owns a ring slot.
    for (uint32_t loop = 0; loop < hp->num_loops; loop++) {
        const uint32_t rec_start = hp->input_layer_count;
        const uint32_t rec_end   = hp->input_layer_count + hp->recurrent_layer_count;
        for (il = rec_start; il < rec_end; il++) {
            if (il == rec_start) {
                // loop reinjection: h = L2(h + input_h). At loop 0 cur == input_h, so this
                // legitimately reduces to L2(2*cur) == L2(cur) -- kept unconditional to match
                // the reference exactly.
                cur = dory_l2_norm(ctx, ggml_add(ctx, cur, input_h));
            }
            cur = dory_build_block(ctx, m, kv, f.gf, &m->layers[il], cur, f.positions,
                                   mask_swa_f16, mask_full_f16, pos0, n_tokens, loop);
        }
        // end-of-recurrent-pass norm (the reference normalizes here; the interpolation already
        // leaves cur normalized, so this is idempotent -- kept for exact fidelity)
        cur = dory_l2_norm(ctx, cur);
    }

    // ---- phase 3: output blocks, executed once ---------------------------------------------
    for (il = hp->input_layer_count + hp->recurrent_layer_count; il < hp->block_count; il++) {
        cur = dory_build_block(ctx, m, kv, f.gf, &m->layers[il], cur, f.positions,
                               mask_swa_f16, mask_full_f16, pos0, n_tokens, 0);
    }

    // Head: NO final norm (the last block's residual update already normalized the state), and
    // the logits carry the per-vocab output scale divided by init_std.
    cur = ggml_mul_mat(ctx, m->output != NULL ? m->output : m->token_embd, cur); // [n_vocab, n_tokens]
    if (m->output_scale != NULL) {
        cur = ggml_mul(ctx, cur, ggml_scale(ctx, dory_f32(ctx, m->output_scale),
                                            1.0f / hp->init_std));
    }
    f.logits = cur;

    // Pulls in every node built above, K/V cache writes included (they were expanded during
    // the block builds, so they execute before the attention that reads them).
    ggml_build_forward_expand(f.gf, f.logits);
    return f;
}

size_t dory_forward_ctx_bytes(void) {
    return DORY_GRAPH_SIZE * ggml_tensor_overhead() +
           ggml_graph_overhead_custom(DORY_GRAPH_SIZE, false);
}
