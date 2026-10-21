#include "clip_text.h"
#include "sdxl_common.h"

#include "ggml-alloc.h"

#include <vector>

#include <cstdio>
#include <limits>

static clip_layer load_clip_layer(struct ggml_context * ctx, const std::string & p) {
    clip_layer l;
    l.ln1_w = sdxl_must_get(ctx, p + "layer_norm1.weight");
    l.ln1_b = sdxl_must_get(ctx, p + "layer_norm1.bias");
    l.ln2_w = sdxl_must_get(ctx, p + "layer_norm2.weight");
    l.ln2_b = sdxl_must_get(ctx, p + "layer_norm2.bias");
    l.q_w   = sdxl_must_get(ctx, p + "self_attn.q_proj.weight");
    l.q_b   = sdxl_must_get(ctx, p + "self_attn.q_proj.bias");
    l.k_w   = sdxl_must_get(ctx, p + "self_attn.k_proj.weight");
    l.k_b   = sdxl_must_get(ctx, p + "self_attn.k_proj.bias");
    l.v_w   = sdxl_must_get(ctx, p + "self_attn.v_proj.weight");
    l.v_b   = sdxl_must_get(ctx, p + "self_attn.v_proj.bias");
    l.out_w = sdxl_must_get(ctx, p + "self_attn.out_proj.weight");
    l.out_b = sdxl_must_get(ctx, p + "self_attn.out_proj.bias");
    l.fc1_w = sdxl_must_get(ctx, p + "mlp.fc1.weight");
    l.fc1_b = sdxl_must_get(ctx, p + "mlp.fc1.bias");
    l.fc2_w = sdxl_must_get(ctx, p + "mlp.fc2.weight");
    l.fc2_b = sdxl_must_get(ctx, p + "mlp.fc2.bias");
    return l;
}

clip_text_model load_clip_text(struct ggml_context * wctx, const std::string & prefix,
                                int n_layer, int hidden, int n_head, bool has_text_projection,
                                bool gelu_tanh) {
    (void) hidden;
    clip_text_model m;
    m.n_head    = n_head;
    m.gelu_tanh = gelu_tanh;
    m.token_embedding    = sdxl_must_get(wctx, prefix + "embeddings.token_embedding.weight");
    m.position_embedding = sdxl_must_get(wctx, prefix + "embeddings.position_embedding.weight");
    m.layers.reserve(n_layer);
    for (int i = 0; i < n_layer; i++) {
        m.layers.push_back(load_clip_layer(wctx, prefix + "encoder.layers." + std::to_string(i) + "."));
    }
    m.final_ln_w = sdxl_must_get(wctx, prefix + "final_layer_norm.weight");
    m.final_ln_b = sdxl_must_get(wctx, prefix + "final_layer_norm.bias");
    if (has_text_projection) {
        m.text_projection = sdxl_must_get(wctx, prefix + "text_projection");
    }
    return m;
}

// x: [hidden, 77, 1]. mask: [77,77] causal, additive. Runs through `n_run` layers (clip_skip
// handling: caller passes n_layer-clip_skip, or n_layer for full depth), then final_layer_norm only
// when apply_final_ln is set (see file header: the pooled path always does, the context path only
// when clip_skip<=0 i.e. full depth -- SDXL always uses clip_skip=2 for the context path, so in
// practice apply_final_ln is false there and true for the pooled path).
static struct ggml_tensor * clip_text_forward(struct ggml_context * ctx, const clip_text_model & m,
                                               struct ggml_tensor * x, struct ggml_tensor * mask,
                                               int n_run, bool apply_final_ln) {
    for (int i = 0; i < n_run; i++) {
        const clip_layer & l = m.layers[i];

        struct ggml_tensor * r = x;
        struct ggml_tensor * h = sdxl_layer_norm(ctx, x, l.ln1_w, l.ln1_b);
        struct ggml_tensor * q = sdxl_linear(ctx, l.q_w, l.q_b, h);
        struct ggml_tensor * k = sdxl_linear(ctx, l.k_w, l.k_b, h);
        struct ggml_tensor * v = sdxl_linear(ctx, l.v_w, l.v_b, h);
        h = sdxl_attention(ctx, q, k, v, m.n_head, mask);
        h = sdxl_linear(ctx, l.out_w, l.out_b, h);
        x = ggml_add(ctx, r, h);

        r = x;
        h = sdxl_layer_norm(ctx, x, l.ln2_w, l.ln2_b);
        h = sdxl_linear(ctx, l.fc1_w, l.fc1_b, h);
        h = m.gelu_tanh ? ggml_gelu(ctx, h) : ggml_gelu_quick(ctx, h);
        h = sdxl_linear(ctx, l.fc2_w, l.fc2_b, h);
        x = ggml_add(ctx, r, h);
    }
    if (apply_final_ln) {
        x = sdxl_layer_norm(ctx, x, m.final_ln_w, m.final_ln_b);
    }
    return x;
}

static struct ggml_tensor * embed(struct ggml_context * ctx, const clip_text_model & m,
                                   struct ggml_tensor * ids) {
    struct ggml_tensor * tok = ggml_get_rows(ctx, m.token_embedding, ids); // [hidden, 77]
    tok = ggml_reshape_3d(ctx, tok, tok->ne[0], tok->ne[1], 1);
    // position_embedding is Q8_0 in this checkpoint (same quantize-everything convention as the
    // VAE's norm/bias tensors, see sdxl_to_f32) -- ggml_add can't mix it with tok's F32 directly.
    return ggml_add(ctx, tok, sdxl_to_f32(ctx, m.position_embedding));
}

sdxl_text_condition clip_encode(const clip_text_model & clip_l, const clip_text_model & clip_g,
                                 ggml_backend_t backend, const std::vector<int32_t> & ids) {
    const int n_ctx = (int) ids.size(); // 77

    // bigG's own embedding lookup zeroes everything after the first EOS instead of repeating EOS as
    // padding -- a real, deliberate quirk of how this checkpoint's conversion tokenizes for bigG vs
    // CLIP-L (see get_learned_condition_common in reference/stable-diffusion.cpp's conditioner.hpp:
    // input_ids is captured before the fill-with-0 pass, input_ids2 after).
    std::vector<int32_t> ids_g = ids;
    int32_t eos_pos = n_ctx - 1;
    for (int i = 0; i < n_ctx; i++) {
        if (ids[i] == 49407) { eos_pos = i; break; }
    }
    for (int i = eos_pos + 1; i < n_ctx; i++) ids_g[i] = 0;

    // Three stacked forward passes (CLIP-L context, bigG context, bigG full-depth-for-pooled) over
    // up to 32 layers each -- comfortably exceeds ggml_new_graph()'s default 2048-node graph size
    // (hit that assert directly during testing), so size the graph explicitly.
    const size_t n_graph_nodes = 16384;
    size_t buf_size = ggml_tensor_overhead() * n_graph_nodes + ggml_graph_overhead_custom(n_graph_nodes, false);
    std::vector<uint8_t> graph_buf(buf_size);
    struct ggml_init_params gip = { graph_buf.size(), graph_buf.data(), /*.no_alloc=*/true };
    struct ggml_context * ctx = ggml_init(gip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_graph_nodes, false);

    struct ggml_tensor * ids_l_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ctx);
    struct ggml_tensor * ids_g_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ctx);
    struct ggml_tensor * mask_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ctx, n_ctx);

    // clip_skip=2: SDXL's fixed default (see file header) -- run all but the last 2 layers, no
    // final_layer_norm, for the cross-attention context.
    struct ggml_tensor * ctx_l = clip_text_forward(ctx, clip_l, embed(ctx, clip_l, ids_l_t), mask_t,
                                                    (int) clip_l.layers.size() - 2, false);
    struct ggml_tensor * ctx_g = clip_text_forward(ctx, clip_g, embed(ctx, clip_g, ids_g_t), mask_t,
                                                    (int) clip_g.layers.size() - 2, false);
    struct ggml_tensor * context = ggml_concat(ctx, ctx_l, ctx_g, 0); // [768+1280=2048, 77, 1]

    // pooled: bigG only, full depth + final_layer_norm, at the EOS token position, then projected.
    struct ggml_tensor * full_g = clip_text_forward(ctx, clip_g, embed(ctx, clip_g, ids_g_t), mask_t,
                                                      (int) clip_g.layers.size(), true);
    struct ggml_tensor * eos_hidden = ggml_view_1d(ctx, full_g, full_g->ne[0],
                                                    full_g->nb[1] * eos_pos); // [1280]
    struct ggml_tensor * pooled = sdxl_linear(ctx, clip_g.text_projection, nullptr, eos_hidden);

    ggml_build_forward_expand(gf, context);
    ggml_build_forward_expand(gf, pooled);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "error: CLIP graph allocation failed\n");
        exit(1);
    }

    ggml_backend_tensor_set(ids_l_t, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ids_g_t, ids_g.data(), 0, ids_g.size() * sizeof(int32_t));

    std::vector<float> mask_vec((size_t) n_ctx * n_ctx);
    for (int i1 = 0; i1 < n_ctx; i1++) {       // query position
        for (int i0 = 0; i0 < n_ctx; i0++) {   // key position
            mask_vec[(size_t) i1 * n_ctx + i0] = (i0 > i1) ? -std::numeric_limits<float>::infinity() : 0.0f;
        }
    }
    ggml_backend_tensor_set(mask_t, mask_vec.data(), 0, mask_vec.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    sdxl_text_condition out;
    out.context.resize(ggml_nelements(context));
    ggml_backend_tensor_get(context, out.context.data(), 0, out.context.size() * sizeof(float));
    out.pooled.resize(ggml_nelements(pooled));
    ggml_backend_tensor_get(pooled, out.pooled.data(), 0, out.pooled.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return out;
}
