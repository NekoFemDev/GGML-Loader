#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "qwen_model.h"

#include "compute_backend.h"
#include "dequant.h"
#include "expert_loader.h"
#include "gguf_index.h"
#include "moe_stream.h"
#include "memtrack.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint8_t *data;
    size_t   bytes;
    uint32_t type;
    uint64_t ne0;
    uint64_t ne1;
    uint64_t row_bytes;
    uint64_t key;      /* stable identity for backend residency */
} mtensor_t;

typedef struct {
    mtensor_t attn_norm;
    mtensor_t attn_q;
    mtensor_t attn_k;
    mtensor_t attn_v;
    mtensor_t attn_output;
    mtensor_t attn_q_norm;
    mtensor_t attn_k_norm;
    mtensor_t ffn_norm;
    mtensor_t ffn_gate_inp;
} qwen_layer_t;

struct qwen_model {
    moe_ctx_t   *ctx;
    gguf_index_t idx;
    moe_plan_t   plan;
    const compute_backend_t *be;

    int   n_layers;
    int   n_embd;
    int   n_head;
    int   n_head_kv;
    int   head_dim;
    int   n_ff_exp;
    int   n_expert;
    int   top_k;
    int   n_vocab;
    int   max_seq;
    int   pos;
    float eps;
    float rope_theta;

    qwen_layer_t *layers;
    mtensor_t     tok_embd;
    mtensor_t     out_norm;
    mtensor_t     out_w;

    float *k_cache;
    float *v_cache;
    float *rope_cos;
    float *rope_sin;

    float *x;
    float *xb;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *proj;
    float *gate;
    float *up;
    float *act;
    float *expert_out;
    float *router;
    float *scores;
    float *logits;

    float *e_gate;          /* [top_k][n_ff]  batched expert scratch */
    float *e_up;
    float *e_act;
    float *e_down;          /* [top_k][n_embd] */

    uint64_t     next_key;
    uint64_t     model_id;
    uint32_t    *prev_route;   /* [n_layers][top_k], previous token */
    uint32_t    *cur_route;
    int          have_prev;
    size_t       cache_bytes;
    double       last_ms;
    qwen_stats_t stats;
};

/* ---- math through the backend ------------------------------------------- */

static inline void mt_mul(struct qwen_model *m, const mtensor_t *t,
                          const float *x, float *y)
{
    m->be->matmul(m->be->self, t->type, t->data, t->key, t->ne0, t->ne1, x, y);
}

static inline void raw_mul(struct qwen_model *m, uint32_t type, const uint8_t *w,
                           uint64_t key, uint64_t ne0, uint64_t ne1,
                           const float *x, float *y)
{
    m->be->matmul(m->be->self, type, w, key, ne0, ne1, x, y);
}

/* Expert residency key: namespace bit 62, layer, expert, part (0 gate,
 * 1 up, 2 down). Global tensors use bit 63 and a monotonic id. */
static uint64_t g_model_seq = 1;

static inline uint64_t expert_key(struct qwen_model *m, int layer,
                                  uint32_t expert, int part)
{
    return 0x4000000000000000ull | (m->model_id << 48) |
           ((uint64_t)(uint32_t)layer << 32) |
           ((uint64_t)expert << 4) | (uint64_t)part;
}

static inline int batch_begin(struct qwen_model *m)
{
    if (m->be->begin_batch) {
        m->be->begin_batch(m->be->self);
        return 1;
    }
    return 0;
}

static inline int batch_end(struct qwen_model *m, int active)
{
    if (active && m->be->end_batch) {
        return m->be->end_batch(m->be->self);
    }
    return 0;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ---- loading ------------------------------------------------------------ */

static void mt_free(mtensor_t *t)
{
    if (t->data) {
        moe_mem_free(t->data, t->bytes);
    }
    memset(t, 0, sizeof(*t));
}

static int mt_load(struct qwen_model *m, expert_io_t *io, const char *name,
                   mtensor_t *t)
{
    memset(t, 0, sizeof(*t));

    const gguf_tensor_t *g = gguf_find(&m->idx, name);
    if (!g || g->n_dims < 1) {
        fprintf(stderr, "mt_load: missing tensor %s\n", name);
        return -ENOENT;
    }

    uint64_t ne = 1;
    for (uint32_t d = 0; d < g->n_dims; d++) {
        ne *= g->dims[d];
    }

    const uint64_t nb = ggml_tensor_nbytes(g->type, ne);
    if (nb == 0) {
        fprintf(stderr, "mt_load: unsupported type %u for %s\n", g->type, name);
        return -ENOTSUP;
    }

    void *p = moe_mem_aligned(4096, nb);
    if (!p) {
        return -ENOMEM;
    }
    int rc = expert_io_read(io, m->idx.data_start + g->offset, nb, p);
    if (rc != 0) {
        free(p);
        return rc;
    }

    t->data      = (uint8_t *)p;
    t->bytes     = (size_t)nb;
    t->type      = g->type;
    t->ne0       = g->dims[0];
    t->ne1       = g->n_dims >= 2 ? g->dims[1] : 1;
    t->row_bytes = nb / (t->ne1 ? t->ne1 : 1);
    t->key       = 0x8000000000000000ull | (m->model_id << 40) | (m->next_key++);
    return 0;
}

static float *alloc_f32(size_t n)
{
    return (float *)calloc(n, sizeof(float));
}

qwen_model_t *qwen_load(const char *gguf_path, size_t cache_bytes, int threads,
                        int max_seq, int odirect)
{
    if (!gguf_path || max_seq <= 0) {
        return NULL;
    }

    struct qwen_model *m = (struct qwen_model *)calloc(1, sizeof(*m));
    if (!m) {
        return NULL;
    }
    m->be       = compute_backend_scalar();
    m->model_id    = g_model_seq++;
    m->cache_bytes = cache_bytes;

    m->ctx = moe_open(gguf_path, cache_bytes, threads, odirect);
    if (!m->ctx) {
        fprintf(stderr, "qwen_load: moe_open failed\n");
        goto fail;
    }

    const gguf_index_t *gi = moe_index(m->ctx);
    const moe_plan_t   *gp = moe_plan(m->ctx);
    if (!gi || !gp) {
        goto fail;
    }
    m->idx  = *gi;
    m->plan = *gp;

    m->n_layers   = (int)m->idx.n_layer;
    m->n_embd     = (int)m->idx.n_embd;
    m->n_head     = (int)m->idx.n_head;
    m->n_head_kv  = (int)m->idx.n_head_kv;
    m->head_dim   = (int)m->idx.head_dim;
    m->n_ff_exp   = (int)m->idx.n_ff_exp;
    m->n_expert   = (int)m->idx.n_expert;
    m->top_k      = (int)m->idx.n_expert_used;
    m->n_vocab    = (int)m->idx.n_vocab;
    m->max_seq    = max_seq;
    m->pos        = 0;
    m->eps        = m->idx.eps > 0 ? m->idx.eps : 1e-6f;
    m->rope_theta = m->idx.rope_theta > 0 ? m->idx.rope_theta : 10000.0f;

    if (m->n_layers <= 0 || m->n_embd <= 0 || m->n_head <= 0 ||
        m->n_head_kv <= 0 || m->head_dim <= 0 || m->n_vocab <= 0 ||
        m->top_k <= 0 || m->top_k > MOE_MAX_TOP_K) {
        fprintf(stderr, "qwen_load: incomplete config\n");
        goto fail;
    }

    expert_io_t io;
    memset(&io, 0, sizeof(io));
    io.fd = -1;
    if (expert_io_open(&io, gguf_path, 0) != 0) {
        goto fail;
    }

    m->layers = (qwen_layer_t *)calloc((size_t)m->n_layers, sizeof(qwen_layer_t));
    if (!m->layers) {
        expert_io_close(&io);
        goto fail;
    }

    for (int i = 0; i < m->n_layers; i++) {
        qwen_layer_t *L = &m->layers[i];
        char n[96];
#define LOAD(field, suffix)                                                    \
        do {                                                                   \
            snprintf(n, sizeof(n), "blk.%d.%s", i, suffix);                    \
            if (mt_load(m, &io, n, &L->field) != 0) {                          \
                expert_io_close(&io);                                          \
                goto fail;                                                     \
            }                                                                  \
        } while (0)

        LOAD(attn_norm,    "attn_norm.weight");
        LOAD(attn_q,       "attn_q.weight");
        LOAD(attn_k,       "attn_k.weight");
        LOAD(attn_v,       "attn_v.weight");
        LOAD(attn_output,  "attn_output.weight");
        LOAD(attn_q_norm,  "attn_q_norm.weight");
        LOAD(attn_k_norm,  "attn_k_norm.weight");
        LOAD(ffn_norm,     "ffn_norm.weight");
        LOAD(ffn_gate_inp, "ffn_gate_inp.weight");
#undef LOAD
    }

    if (mt_load(m, &io, "token_embd.weight", &m->tok_embd) != 0 ||
        mt_load(m, &io, "output_norm.weight", &m->out_norm) != 0 ||
        mt_load(m, &io, "output.weight", &m->out_w) != 0) {
        expert_io_close(&io);
        goto fail;
    }
    expert_io_close(&io);

    const int nkv = m->n_head_kv * m->head_dim;

    const size_t kvn = (size_t)m->n_layers * (size_t)max_seq * (size_t)nkv;
    m->k_cache  = (float *)moe_mem_alloc(kvn * sizeof(float));
    m->v_cache  = (float *)moe_mem_alloc(kvn * sizeof(float));
    m->rope_cos = alloc_f32((size_t)max_seq * (size_t)(m->head_dim / 2));
    m->rope_sin = alloc_f32((size_t)max_seq * (size_t)(m->head_dim / 2));

    m->x          = alloc_f32((size_t)m->n_embd);
    m->xb         = alloc_f32((size_t)m->n_embd);
    m->q          = alloc_f32((size_t)(m->n_head * m->head_dim));
    m->k          = alloc_f32((size_t)nkv);
    m->v          = alloc_f32((size_t)nkv);
    m->attn       = alloc_f32((size_t)(m->n_head * m->head_dim));
    m->proj       = alloc_f32((size_t)m->n_embd);
    m->gate       = alloc_f32((size_t)m->n_ff_exp);
    m->up         = alloc_f32((size_t)m->n_ff_exp);
    m->act        = alloc_f32((size_t)m->n_ff_exp);
    m->expert_out = alloc_f32((size_t)m->n_embd);
    m->router     = alloc_f32((size_t)m->n_expert);
    m->scores     = alloc_f32((size_t)max_seq);
    m->logits     = alloc_f32((size_t)m->n_vocab);
    m->e_gate     = alloc_f32((size_t)m->top_k * (size_t)m->n_ff_exp);
    m->e_up       = alloc_f32((size_t)m->top_k * (size_t)m->n_ff_exp);
    m->e_act      = alloc_f32((size_t)m->top_k * (size_t)m->n_ff_exp);
    m->e_down     = alloc_f32((size_t)m->top_k * (size_t)m->n_embd);
    m->prev_route = (uint32_t *)calloc((size_t)m->n_layers * (size_t)m->top_k,
                                       sizeof(uint32_t));
    m->cur_route  = (uint32_t *)calloc((size_t)m->n_layers * (size_t)m->top_k,
                                       sizeof(uint32_t));

    if (!m->k_cache || !m->v_cache || !m->rope_cos || !m->rope_sin ||
        !m->x || !m->xb || !m->q || !m->k || !m->v || !m->attn || !m->proj ||
        !m->gate || !m->up || !m->act || !m->expert_out || !m->router ||
        !m->scores || !m->logits || !m->e_gate || !m->e_up || !m->e_act ||
        !m->e_down || !m->prev_route || !m->cur_route) {
        fprintf(stderr, "qwen_load: scratch allocation failed\n");
        goto fail;
    }

    const int half = m->head_dim / 2;
    for (int p = 0; p < max_seq; p++) {
        for (int i = 0; i < half; i++) {
            const double freq =
                1.0 / pow((double)m->rope_theta, (double)(2 * i) / (double)m->head_dim);
            const double a = (double)p * freq;
            m->rope_cos[(size_t)p * half + i] = (float)cos(a);
            m->rope_sin[(size_t)p * half + i] = (float)sin(a);
        }
    }

    fprintf(stderr,
            "[qwen] %d layers, embd %d, heads %d/%d x %d, experts %d top-%d, "
            "vocab %d, ctx %d, backend %s\n",
            m->n_layers, m->n_embd, m->n_head, m->n_head_kv, m->head_dim,
            m->n_expert, m->top_k, m->n_vocab, m->max_seq, m->be->name);
    return m;

fail:
    qwen_free(m);
    return NULL;
}

void qwen_free(qwen_model_t *m)
{
    if (!m) {
        return;
    }
    if (m->layers) {
        for (int i = 0; i < m->n_layers; i++) {
            qwen_layer_t *L = &m->layers[i];
            mt_free(&L->attn_norm);
            mt_free(&L->attn_q);
            mt_free(&L->attn_k);
            mt_free(&L->attn_v);
            mt_free(&L->attn_output);
            mt_free(&L->attn_q_norm);
            mt_free(&L->attn_k_norm);
            mt_free(&L->ffn_norm);
            mt_free(&L->ffn_gate_inp);
        }
        free(m->layers);
    }
    mt_free(&m->tok_embd);
    mt_free(&m->out_norm);
    mt_free(&m->out_w);

    if (m->k_cache) {
        const size_t kvn = (size_t)m->n_layers * (size_t)m->max_seq *
                           (size_t)m->n_head_kv * (size_t)m->head_dim;
        moe_mem_free(m->k_cache, kvn * sizeof(float));
        moe_mem_free(m->v_cache, kvn * sizeof(float));
    }
    free(m->rope_cos);
    free(m->rope_sin);
    free(m->x);
    free(m->xb);
    free(m->q);
    free(m->k);
    free(m->v);
    free(m->attn);
    free(m->proj);
    free(m->gate);
    free(m->up);
    free(m->act);
    free(m->expert_out);
    free(m->router);
    free(m->scores);
    free(m->logits);
    free(m->e_gate);
    free(m->e_up);
    free(m->e_act);
    free(m->e_down);
    free(m->prev_route);
    free(m->cur_route);

    if (m->be && m->be->name && !strcmp(m->be->name, "vulkan")) {
        compute_vulkan_arena_shutdown();
        compute_vulkan_cache_clear();
    }

    if (m->ctx) {
        moe_close(m->ctx);
    }
    free(m);
}

void qwen_set_backend(qwen_model_t *m, const compute_backend_t *be)
{
    if (!m) {
        return;
    }
    m->be = be ? be : compute_backend_scalar();

    if (m->be->name && !strcmp(m->be->name, "vulkan")) {
        if (compute_vulkan_arena_init(m->cache_bytes,
                                      (size_t)m->plan.max_expert_bytes) != 0) {
            fprintf(stderr, "[vk] arena unavailable; using per-call uploads\n");
            return;
        }
        if (moe_set_cache_allocator(m->ctx, compute_vulkan_arena_alloc,
                                    compute_vulkan_arena_free, NULL) != 0) {
            fprintf(stderr, "[vk] LRU not empty; arena unused\n");
            compute_vulkan_arena_shutdown();
        }
    }
}

/* ---- blocks ------------------------------------------------------------- */

static void attention_block(struct qwen_model *m, qwen_layer_t *L, int layer,
                            const float *xbn, float *xh, int pos)
{
    const int hd   = m->head_dim;
    const int nh   = m->n_head;
    const int nkv  = m->n_head_kv;
    const int grp  = nh / nkv;
    const int nkvd = nkv * hd;

    /* q, k, v are independent and share x: one command buffer, one submit. */
    static int nb_attn = -1;
    if (nb_attn < 0) nb_attn = getenv("MOE_NO_BATCH_ATTN") != NULL;
    int batched = nb_attn ? 0 : batch_begin(m);
    mt_mul(m, &L->attn_q, xbn, m->q);
    mt_mul(m, &L->attn_k, xbn, m->k);
    mt_mul(m, &L->attn_v, xbn, m->v);
    batch_end(m, batched);

    const float *qkw = (const float *)L->attn_q_norm.data;
    const float *kkw = (const float *)L->attn_k_norm.data;
    const float *cs  = m->rope_cos + (size_t)pos * (hd / 2);
    const float *sn  = m->rope_sin + (size_t)pos * (hd / 2);

    for (int h = 0; h < nh; h++) {
        m->be->rmsnorm(m->be->self, m->q + h * hd, m->q + h * hd, qkw, hd, m->eps);
        m->be->rope(m->be->self, m->q + h * hd, hd, cs, sn);
    }
    for (int h = 0; h < nkv; h++) {
        m->be->rmsnorm(m->be->self, m->k + h * hd, m->k + h * hd, kkw, hd, m->eps);
        m->be->rope(m->be->self, m->k + h * hd, hd, cs, sn);
    }

    float *kc = m->k_cache + ((size_t)layer * m->max_seq + pos) * nkvd;
    float *vc = m->v_cache + ((size_t)layer * m->max_seq + pos) * nkvd;
    memcpy(kc, m->k, (size_t)nkvd * sizeof(float));
    memcpy(vc, m->v, (size_t)nkvd * sizeof(float));

    const float  scale = 1.0f / sqrtf((float)hd);
    const size_t step  = (size_t)nkvd;
    const float *kbase = m->k_cache + (size_t)layer * m->max_seq * nkvd;
    const float *vbase = m->v_cache + (size_t)layer * m->max_seq * nkvd;

    for (int h = 0; h < nh; h++) {
        const int    kv = h / grp;
        const float *qh = m->q + h * hd;
        const float *ks = kbase + (size_t)kv * hd;
        const float *vs = vbase + (size_t)kv * hd;

        for (int t = 0; t <= pos; t++) {
            const float *kt = ks + (size_t)t * step;
            float s = 0.0f;
            for (int d = 0; d < hd; d++) {
                s += qh[d] * kt[d];
            }
            m->scores[t] = s * scale;
        }
        /* TODO(vulkan): replace this whole block with a fused attention
         * dispatch (scores + softmax + weighted V in one workgroup sweep). */
        m->be->softmax(m->be->self, m->scores, pos + 1);

        float *oh = m->attn + h * hd;
        memset(oh, 0, (size_t)hd * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float  p  = m->scores[t];
            const float *vt = vs + (size_t)t * step;
            for (int d = 0; d < hd; d++) {
                oh[d] += p * vt[d];
            }
        }
    }

    mt_mul(m, &L->attn_output, m->attn, m->proj);
    for (int i = 0; i < m->n_embd; i++) {
        xh[i] += m->proj[i];
    }
}

static void moe_route(struct qwen_model *m, qwen_layer_t *L, const float *xbn,
                      uint32_t *ids, float *w)
{
    mt_mul(m, &L->ffn_gate_inp, xbn, m->router);
    m->be->softmax(m->be->self, m->router, m->n_expert);

    for (int k = 0; k < m->top_k; k++) {
        ids[k] = 0;
        w[k]   = -1.0f;
    }

    int   bi[MOE_MAX_TOP_K];
    float bw[MOE_MAX_TOP_K];
    for (int k = 0; k < m->top_k; k++) {
        bi[k] = -1;
        bw[k] = -1.0f;
    }

    for (int e = 0; e < m->n_expert; e++) {
        const float v = m->router[e];
        if (v <= bw[m->top_k - 1]) {
            continue;
        }
        int j = m->top_k - 1;
        while (j > 0 && bw[j - 1] < v) {
            bw[j] = bw[j - 1];
            bi[j] = bi[j - 1];
            j--;
        }
        bw[j] = v;
        bi[j] = e;
    }

    float sum = 0.0f;
    for (int k = 0; k < m->top_k; k++) {
        sum += bw[k];
    }
    if (sum > 0.0f) {
        for (int k = 0; k < m->top_k; k++) {
            bw[k] /= sum;
        }
    }

    for (int k = 0; k < m->top_k; k++) {
        ids[k] = (uint32_t)bi[k];
        w[k]   = bw[k];
    }
}

static void moe_block(struct qwen_model *m, qwen_layer_t *L, int layer,
                      const float *xbn, float *xh)
{
    uint32_t ids[MOE_MAX_TOP_K];
    float    w[MOE_MAX_TOP_K];
    moe_route(m, L, xbn, ids, w);

    memcpy(m->cur_route + (size_t)layer * m->top_k, ids,
           (size_t)m->top_k * sizeof(uint32_t));

    moe_expert_t ex[MOE_MAX_TOP_K];
    memset(ex, 0, sizeof(ex));
    int rc = moe_layer_fetch(m->ctx, layer, ids, m->top_k, ex);
    if (rc != m->top_k) {
        fprintf(stderr, "moe_block: fetch layer %d failed %d\n", layer, rc);
        return;
    }

    const moe_layer_t *ml = &m->plan.layers[layer];
    const int nff = m->n_ff_exp;

    /* gate/up for all 8 experts share x: 16 dispatches, one submit. */
    static int nb_moe = -1, nb_gu = -1, nb_dn = -1;
    if (nb_moe < 0) {
        nb_moe = getenv("MOE_NO_BATCH_MOE") != NULL;
        nb_gu  = getenv("MOE_NO_BATCH_GATEUP") != NULL;
        nb_dn  = getenv("MOE_NO_BATCH_DOWN") != NULL;
    }
    int batched = (nb_moe || nb_gu) ? 0 : batch_begin(m);
    for (int k = 0; k < m->top_k; k++) {
        const uint32_t e = ex[k].expert;
        raw_mul(m, ml->gate.type, ex[k].gate, expert_key(m, layer, e, 0),
                ml->gate.ne0, ml->gate.ne1, xbn,
                m->e_gate + (size_t)k * nff);
        raw_mul(m, ml->up.type, ex[k].up, expert_key(m, layer, e, 1),
                ml->up.ne0, ml->up.ne1, xbn,
                m->e_up + (size_t)k * nff);
    }
    batch_end(m, batched);

    static int g_verify = -1;
    if (g_verify < 0) g_verify = getenv("MOE_VERIFY") != NULL;
    if (g_verify) {
        float *tmp = (float *)malloc((size_t)nff * sizeof(float));
        const compute_backend_t *sc = compute_backend_scalar();
        double worst = 0.0;
        int    wexp   = -1;
        for (int k = 0; k < m->top_k; k++) {
            sc->matmul(NULL, ml->gate.type, ex[k].gate, 0, ml->gate.ne0,
                       ml->gate.ne1, xbn, tmp);
            for (int i = 0; i < nff; i++) {
                double d = fabs((double)m->e_gate[(size_t)k * nff + i] - tmp[i]) /
                           (fabs((double)tmp[i]) > 1e-3 ? fabs((double)tmp[i]) : 1.0);
                if (d > worst) { worst = d; wexp = k; }
            }
        }
        fprintf(stderr, "[verify] layer %d gate worst %.3e (expert slot %d, id %u)\n",
                layer, worst, wexp, wexp >= 0 ? ex[wexp].expert : 0);

        double uworst = 0.0;
        int    uexp   = -1;
        for (int k = 0; k < m->top_k; k++) {
            sc->matmul(NULL, ml->up.type, ex[k].up, 0, ml->up.ne0,
                       ml->up.ne1, xbn, tmp);
            for (int i = 0; i < nff; i++) {
                double d = fabs((double)m->e_up[(size_t)k * nff + i] - tmp[i]) /
                           (fabs((double)tmp[i]) > 1e-3 ? fabs((double)tmp[i]) : 1.0);
                if (d > uworst) { uworst = d; uexp = k; }
            }
        }
        fprintf(stderr, "[verify] layer %d up   worst %.3e (expert slot %d, id %u)\n",
                layer, uworst, uexp, uexp >= 0 ? ex[uexp].expert : 0);
        free(tmp);
    }

    for (int k = 0; k < m->top_k; k++) {
        m->be->swiglu(m->be->self, m->e_gate + (size_t)k * nff,
                      m->e_up + (size_t)k * nff,
                      m->e_act + (size_t)k * nff, nff);
    }

    /* down projections are independent: one submit for all 8. */
    batched = (nb_moe || nb_dn) ? 0 : batch_begin(m);
    for (int k = 0; k < m->top_k; k++) {
        const uint32_t e = ex[k].expert;
        raw_mul(m, ml->down.type, ex[k].down, expert_key(m, layer, e, 2),
                ml->down.ne0, ml->down.ne1, m->e_act + (size_t)k * nff,
                m->e_down + (size_t)k * m->n_embd);
    }
    batch_end(m, batched);

    if (g_verify) {
        float *tmp = (float *)malloc((size_t)m->n_embd * sizeof(float));
        const compute_backend_t *sc = compute_backend_scalar();
        double worst = 0.0;
        int    wexp   = -1;
        for (int k = 0; k < m->top_k; k++) {
            sc->matmul(NULL, ml->down.type, ex[k].down, 0, ml->down.ne0,
                       ml->down.ne1, m->e_act + (size_t)k * nff, tmp);
            for (int i = 0; i < m->n_embd; i++) {
                double d = fabs((double)m->e_down[(size_t)k * m->n_embd + i] - tmp[i]) /
                           (fabs((double)tmp[i]) > 1e-3 ? fabs((double)tmp[i]) : 1.0);
                if (d > worst) { worst = d; wexp = k; }
            }
        }
        fprintf(stderr, "[verify] layer %d down worst %.3e (expert slot %d, id %u)\n",
                layer, worst, wexp, wexp >= 0 ? ex[wexp].expert : 0);
        free(tmp);
    }

    memset(m->expert_out, 0, (size_t)m->n_embd * sizeof(float));
    for (int k = 0; k < m->top_k; k++) {
        const float wk = w[k];
        const float *d = m->e_down + (size_t)k * m->n_embd;
        for (int i = 0; i < m->n_embd; i++) {
            m->expert_out[i] += wk * d[i];
        }
        m->stats.experts_fetched++;
        m->stats.expert_bytes += ex[k].gate_bytes + ex[k].up_bytes + ex[k].down_bytes;
    }

    moe_release(m->ctx, ex, m->top_k);

    for (int i = 0; i < m->n_embd; i++) {
        xh[i] += m->expert_out[i];
    }
}

/* ---- prefill: layer sweep ----------------------------------------------- */

#define MOE_PREFILL_CHUNK 16

/* For one layer: the set of routed experts and, per expert, the token ids that
 * route to it (the inverted index used by the batched GEMMs). */
static void build_prefill_lists(int n, int topk, const uint32_t *rid,
                                int n_expert, uint32_t *idx, int *cnt,
                                int *experts, int *nexp)
{
    unsigned char used[256] = { 0 };

    for (int i = 0; i < n * topk; i++) {
        if (rid[i] < 256u) {
            used[rid[i]] = 1;
        }
    }
    *nexp = 0;
    for (int e = 0; e < n_expert; e++) {
        if (used[e]) {
            experts[(*nexp)++] = e;
        }
    }
    for (int j = 0; j < *nexp; j++) {
        const uint32_t e = (uint32_t)experts[j];
        int c = 0;
        for (int t = 0; t < n; t++) {
            for (int k = 0; k < topk; k++) {
                if (rid[(size_t)t * topk + k] == e) {
                    idx[(size_t)j * n + c++] = (uint32_t)t;
                    break;
                }
            }
        }
        cnt[j] = c;
    }
}

int qwen_prefill(qwen_model_t *m, const uint32_t *tokens, int n)
{
    if (!m || !tokens || n <= 0) {
        return -EINVAL;
    }
    if (m->pos + n > m->max_seq) {
        return -ERANGE;
    }

    const size_t embd = (size_t)m->n_embd;
    const size_t topk = (size_t)m->top_k;

    float    *h   = (float *)calloc((size_t)n * embd, sizeof(float));
    float    *xb  = (float *)calloc((size_t)n * embd, sizeof(float));
    float    *acc = (float *)calloc((size_t)n * embd, sizeof(float));
    uint32_t *rid = (uint32_t *)malloc((size_t)n * topk * sizeof(uint32_t));
    float    *rw  = (float *)malloc((size_t)n * topk * sizeof(float));
    const size_t chunk = MOE_PREFILL_CHUNK;

    float    *pg  = (float *)calloc(chunk * (size_t)n * (size_t)m->n_ff_exp,
                                    sizeof(float));
    float    *pu  = (float *)calloc(chunk * (size_t)n * (size_t)m->n_ff_exp,
                                    sizeof(float));
    float    *pa  = (float *)calloc(chunk * (size_t)n * (size_t)m->n_ff_exp,
                                    sizeof(float));
    float    *pd  = (float *)calloc(chunk * (size_t)n * embd, sizeof(float));
    uint32_t *eidx = (uint32_t *)malloc((size_t)m->n_expert * (size_t)n *
                                        sizeof(uint32_t));
    int      *ecnt = (int *)malloc((size_t)m->n_expert * sizeof(int));
    int      *euse = (int *)malloc((size_t)m->n_expert * sizeof(int));
    int       rc  = 0;

    if (!h || !xb || !acc || !rid || !rw || !pg || !pu || !pa || !pd ||
        !eidx || !ecnt || !euse) {
        rc = -ENOMEM;
        goto out;
    }

    for (int t = 0; t < n; t++) {
        if (tokens[t] >= (uint32_t)m->n_vocab) {
            rc = -ERANGE;
            goto out;
        }
        dequant_row(m->tok_embd.type,
                    m->tok_embd.data + (uint64_t)tokens[t] * m->tok_embd.row_bytes,
                    h + (size_t)t * embd, (uint64_t)m->n_embd);
    }

    fprintf(stderr, "[prefill] sweep: %d tokens x %d layers\n", n, m->n_layers);

    for (int l = 0; l < m->n_layers; l++) {
        qwen_layer_t *L = &m->layers[l];

        /* 1. attention for every token, in causal order (fills the KV cache) */
        for (int t = 0; t < n; t++) {
            float *ht = h + (size_t)t * embd;
            float *xt = xb + (size_t)t * embd;
            m->be->rmsnorm(m->be->self, xt, ht, (const float *)L->attn_norm.data,
                           m->n_embd, m->eps);
            attention_block(m, L, l, xt, ht, m->pos + t);
        }

        /* 2. pre-FFN norm + routing for the whole batch */
        for (int t = 0; t < n; t++) {
            float *ht = h + (size_t)t * embd;
            float *xt = xb + (size_t)t * embd;
            m->be->rmsnorm(m->be->self, xt, ht, (const float *)L->ffn_norm.data,
                           m->n_embd, m->eps);
            moe_route(m, L, xt, rid + (size_t)t * topk, rw + (size_t)t * topk);
        }
        memcpy(m->prev_route + (size_t)l * m->top_k,
               rid + (size_t)(n - 1) * m->top_k,
               (size_t)m->top_k * sizeof(uint32_t));

        /* 3. expert sweep, batched across experts.
         *
         * All experts of a chunk are fetched (and therefore uploaded to VRAM)
         * up front, then gate/up for the whole chunk is recorded into ONE
         * command buffer and submitted once. swiglu runs on the CPU, then the
         * down projections get their own single submit. */
        memset(acc, 0, (size_t)n * embd * sizeof(float));

        int nexp = 0;
        build_prefill_lists(n, m->top_k, rid, m->n_expert, eidx, ecnt, euse,
                            &nexp);

        const moe_layer_t *ml = &m->plan.layers[l];

        for (int c0 = 0; c0 < nexp; c0 += (int)chunk) {
            const int cn = (nexp - c0 < (int)chunk) ? nexp - c0 : (int)chunk;

            uint32_t ids[MOE_PREFILL_CHUNK];
            for (int j = 0; j < cn; j++) {
                ids[j] = (uint32_t)euse[c0 + j];
            }

            moe_expert_t ex[MOE_PREFILL_CHUNK];
            memset(ex, 0, sizeof(ex));
            int frc = moe_layer_fetch(m->ctx, l, ids, cn, ex);
            if (frc != cn) {
                fprintf(stderr, "[prefill] fetch layer %d: %d\n", l, frc);
                rc = frc ? frc : -EIO;
                goto out;
            }

            int batched = batch_begin(m);
            for (int j = 0; j < cn; j++) {
                const uint32_t e  = ids[j];
                const uint32_t *ix = eidx + (size_t)(c0 + j) * n;
                m->be->matmul_batch(m->be->self, ml->gate.type, ex[j].gate,
                                    expert_key(m, l, e, 0), ml->gate.ne0,
                                    ml->gate.ne1, xb, ix, ecnt[c0 + j],
                                    pg + (size_t)j * n * m->n_ff_exp);
                m->be->matmul_batch(m->be->self, ml->up.type, ex[j].up,
                                    expert_key(m, l, e, 1), ml->up.ne0,
                                    ml->up.ne1, xb, ix, ecnt[c0 + j],
                                    pu + (size_t)j * n * m->n_ff_exp);
            }
            batch_end(m, batched);

            for (int j = 0; j < cn; j++) {
                for (int t = 0; t < ecnt[c0 + j]; t++) {
                    m->be->swiglu(
                        m->be->self,
                        pg + ((size_t)j * n + t) * m->n_ff_exp,
                        pu + ((size_t)j * n + t) * m->n_ff_exp,
                        pa + ((size_t)j * n + t) * m->n_ff_exp,
                        m->n_ff_exp);
                }
            }

            batched = batch_begin(m);
            for (int j = 0; j < cn; j++) {
                const uint32_t e = ids[j];
                m->be->matmul_batch(m->be->self, ml->down.type, ex[j].down,
                                    expert_key(m, l, e, 2), ml->down.ne0,
                                    ml->down.ne1,
                                    pa + (size_t)j * n * m->n_ff_exp,
                                    NULL, ecnt[c0 + j],
                                    pd + (size_t)j * n * embd);
            }
            batch_end(m, batched);

            for (int j = 0; j < cn; j++) {
                const uint32_t e  = ids[j];
                const uint32_t *ix = eidx + (size_t)(c0 + j) * n;
                const float    *d  = pd + (size_t)j * n * embd;

                for (int t = 0; t < ecnt[c0 + j]; t++) {
                    const int tok = (int)ix[t];
                    float wt = 0.0f;
                    for (int k = 0; k < m->top_k; k++) {
                        if (rid[(size_t)tok * m->top_k + k] == e) {
                            wt = rw[(size_t)tok * m->top_k + k];
                            break;
                        }
                    }
                    float       *ac = acc + (size_t)tok * embd;
                    const float *dt = d + (size_t)t * embd;
                    for (int i = 0; i < m->n_embd; i++) {
                        ac[i] += wt * dt[i];
                    }
                }
            }

            moe_release(m->ctx, ex, cn);
        }

        for (size_t i = 0; i < (size_t)n * embd; i++) {
            h[i] += acc[i];
        }
    }

    /* logits for the last position only: that is what generation consumes. */
    m->be->rmsnorm(m->be->self, m->xb, h + (size_t)(n - 1) * embd,
                   (const float *)m->out_norm.data, m->n_embd, m->eps);
    mt_mul(m, &m->out_w, m->xb, m->logits);

    m->pos += n;
    m->have_prev = 1;
    rc = 0;

out:
    free(h);
    free(xb);
    free(acc);
    free(rid);
    free(rw);
    free(pg);
    free(pu);
    free(pa);
    free(pd);
    free(eidx);
    free(ecnt);
    free(euse);
    return rc;
}

int qwen_forward(qwen_model_t *m, int token)
{
    if (!m || token < 0 || token >= m->n_vocab) {
        return -EINVAL;
    }
    if (m->pos >= m->max_seq) {
        return -ERANGE;
    }

    const double t0 = now_ms();
    const int    pos = m->pos;

    dequant_row(m->tok_embd.type,
                m->tok_embd.data + (uint64_t)token * m->tok_embd.row_bytes,
                m->x, (uint64_t)m->n_embd);

    for (int l = 0; l < m->n_layers; l++) {
        qwen_layer_t *L = &m->layers[l];

        /* Speculative lookahead: routing is strongly correlated between
         * adjacent tokens, so start reading layer L+1's experts from the
         * previous token's routing while this layer runs. Non-blocking; a
         * wrong prediction is dropped by TinyLFU admission. */
        if (m->have_prev && l + 1 < m->n_layers) {
            moe_prefetch(m->ctx, l + 1,
                         m->prev_route + (size_t)(l + 1) * m->top_k, m->top_k);
        }

        m->be->rmsnorm(m->be->self, m->xb, m->x,
                       (const float *)L->attn_norm.data, m->n_embd, m->eps);
        attention_block(m, L, l, m->xb, m->x, pos);

        m->be->rmsnorm(m->be->self, m->xb, m->x,
                       (const float *)L->ffn_norm.data, m->n_embd, m->eps);
        moe_block(m, L, l, m->xb, m->x);
    }

    m->be->rmsnorm(m->be->self, m->xb, m->x,
                   (const float *)m->out_norm.data, m->n_embd, m->eps);
    mt_mul(m, &m->out_w, m->xb, m->logits);

    m->pos++;
    m->last_ms = now_ms() - t0;

    uint32_t *tmp = m->prev_route;
    m->prev_route = m->cur_route;
    m->cur_route  = tmp;
    m->have_prev  = 1;
    return 0;
}

void qwen_reset(qwen_model_t *m)
{
    if (!m) {
        return;
    }
    const size_t nkv = (size_t)m->n_layers * m->max_seq * m->n_head_kv * m->head_dim;
    memset(m->k_cache, 0, nkv * sizeof(float));
    memset(m->v_cache, 0, nkv * sizeof(float));
    m->pos = 0;
}

/* ---- accessors ---------------------------------------------------------- */

const char *qwen_backend_name(const qwen_model_t *m)
{
    return m && m->be && m->be->name ? m->be->name : "";
}

const float *qwen_logits(const qwen_model_t *m) { return m ? m->logits : NULL; }

int qwen_argmax(const qwen_model_t *m)
{
    if (!m || !m->logits) {
        return -1;
    }
    int   best = 0;
    float bv   = m->logits[0];
    for (int i = 1; i < m->n_vocab; i++) {
        if (m->logits[i] > bv) {
            bv   = m->logits[i];
            best = i;
        }
    }
    return best;
}

int    qwen_vocab_size(const qwen_model_t *m) { return m ? m->n_vocab : 0; }
int    qwen_n_layers(const qwen_model_t *m)   { return m ? m->n_layers : 0; }
int    qwen_pos(const qwen_model_t *m)        { return m ? m->pos : 0; }
int    qwen_max_seq(const qwen_model_t *m)    { return m ? m->max_seq : 0; }
double qwen_last_ms(const qwen_model_t *m)    { return m ? m->last_ms : 0.0; }

void qwen_get_stats(qwen_model_t *m, qwen_stats_t *out)
{
    if (!m || !out) {
        return;
    }
    *out = m->stats;
}
