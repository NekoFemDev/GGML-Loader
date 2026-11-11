/*
 * moe_stream.c — decode fetch pipeline + prefill layer sweep.
 *
 * Threading model
 * ---------------
 *   N cache-fill workers : pop jobs, pread 3 slices into a pinned cache
 *                          entry, commit, signal progress.
 *   1 prefill reader     : a single dedicated thread used by the layer sweep
 *                          for double-buffered reads, so expert E+1 is on the
 *                          bus while expert E is being multiplied.
 *
 * The two never run at the same time (an API caller does prefill or decode,
 * not both), so they share the file handle without contention.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "moe_stream.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOE_QUEUE_CAP 64

typedef struct {
    moe_ctx_t   *ctx;
    int          layer;
    int          n;
    moe_expert_t out[MOE_MAX_TOP_K];
    int          done[MOE_MAX_TOP_K];
    int          delivered[MOE_MAX_TOP_K];
    int          remaining;
    int          rc;
} fetch_batch_t;

typedef struct {
    moe_ctx_t     *ctx;
    fetch_batch_t *batch;   /* NULL = prefetch */
    int          slot;      /* index inside the batch (unique experts) */
    int          orig;      /* index in the caller's requested array */
    uint32_t     expert;
    uint64_t     key;
    uint64_t     off[3];
    size_t       len[3];
    int          nparts;
} moe_job_t;

/* Single-slot request/ack reader for the prefill sweep. One outstanding job at
 * a time is all double buffering needs. */
typedef struct {
    pthread_t       tid;
    pthread_mutex_t mu;
    pthread_cond_t  cv_req;
    pthread_cond_t  cv_done;
    int             started;
    int             quit;
    int             req;
    int             rc;
    uint64_t        submitted;
    uint64_t        completed;
    uint64_t        off[3];
    size_t          len[3];
    int             nparts;
    uint8_t        *dst;
} pf_reader_t;

struct moe_ctx {
    expert_io_t  io;
    gguf_index_t idx;
    moe_plan_t   plan;
    lru_cache_t *cache;
    int          nthreads;

    pthread_t       *threads;
    pthread_mutex_t  mu;
    pthread_cond_t   cv_job;
    pthread_cond_t   cv_progress;
    moe_job_t        queue[MOE_QUEUE_CAP];
    int              qhead, qtail, qcount;
    int              shutdown;
    int              inflight;
    int              io_errors;
    int              last_error;
    uint64_t         bytes_read;

    fetch_batch_t   *batch;      /* one in-flight fetch, non-reentrant */

    moe_gemm_fn      gemm;
    void            *gemm_user;
    uint64_t         dummy_acc;

    pf_reader_t      reader;
    uint8_t         *pf_buf[2];
    uint64_t         prefill_bytes;
    uint64_t         prefill_experts;
    uint64_t         prefill_tokens;
};

static inline uint64_t moe_key(int layer, uint32_t expert)
{
    return ((uint64_t)(uint32_t)layer << 32) | expert;
}

/* ---- cache-fill jobs ---------------------------------------------------- */

static int run_job(moe_job_t *job)
{
    moe_ctx_t *ctx = job->ctx;
    size_t total = 0;
    for (int i = 0; i < job->nparts; i++) {
        total += job->len[i];
    }

    int admitted = 1;
    lru_entry_t *e = lru_cache_alloc2(ctx->cache, job->key, total, &admitted);
    if (!e) {
        return -ENOMEM;
    }

    uint8_t *dst = lru_entry_data(e);
    size_t   at  = 0;
    for (int i = 0; i < job->nparts; i++) {
        int rc = expert_io_read(&ctx->io, job->off[i], job->len[i], dst + at);
        if (rc != 0) {
            lru_cache_abort(ctx->cache, e);
            return rc;
        }
        at += job->len[i];
    }

    if (admitted) {
        int rc = lru_cache_commit(ctx->cache, e);
        if (rc == -EEXIST) {
            lru_cache_abort(ctx->cache, e);
            e = lru_cache_get(ctx->cache, job->key);
            if (!e) {
                return -EIO;
            }
        } else if (rc != 0) {
            lru_cache_abort(ctx->cache, e);
            return rc;
        }
    }

    if (job->batch) {
        moe_expert_t *ex = &job->batch->out[job->slot];
        ex->entry  = e;
        ex->expert = job->expert;
        ex->slot   = job->orig;
    } else {
        lru_entry_unpin(ctx->cache, e);   /* prefetch: leave it resident */
    }
    return 0;
}

/* Populate the borrowed gate/up/down view from a resident cache entry. */
static void fill_view(const moe_layer_t *ml, moe_expert_t *ex)
{
    uint8_t *d = lru_entry_data(ex->entry);
    ex->gate       = d;
    ex->gate_bytes = (size_t)ml->gate.stride;
    ex->up         = d + ml->gate.stride;
    ex->up_bytes   = (size_t)ml->up.stride;
    ex->down       = d + ml->gate.stride + ml->up.stride;
    ex->down_bytes = (size_t)ml->down.stride;
}

static void *worker_main(void *arg)
{
    moe_ctx_t *ctx = (moe_ctx_t *)arg;

    for (;;) {
        pthread_mutex_lock(&ctx->mu);
        while (ctx->qcount == 0 && !ctx->shutdown) {
            pthread_cond_wait(&ctx->cv_job, &ctx->mu);
        }
        if (ctx->qcount == 0 && ctx->shutdown) {
            pthread_mutex_unlock(&ctx->mu);
            break;
        }

        moe_job_t job = ctx->queue[ctx->qhead];
        ctx->qhead = (ctx->qhead + 1) % MOE_QUEUE_CAP;
        ctx->qcount--;
        pthread_mutex_unlock(&ctx->mu);

        int rc = run_job(&job);

        pthread_mutex_lock(&ctx->mu);
        if (rc != 0) {
            ctx->io_errors++;
            ctx->last_error = rc;
            if (job.batch && job.batch->rc == 0) {
                job.batch->rc = rc;
            }
        } else {
            if (job.batch) {
                job.batch->done[job.slot] = 1;
            }
            for (int i = 0; i < job.nparts; i++) {
                ctx->bytes_read += job.len[i];
            }
        }
        if (job.batch) {
            job.batch->remaining--;
            pthread_cond_broadcast(&ctx->cv_progress);
        }
        ctx->inflight--;
        pthread_mutex_unlock(&ctx->mu);
    }
    return NULL;
}

/* ---- fetch engine (shared by blocking + streaming) ---------------------- */

static int fetch_impl(moe_ctx_t *ctx, int layer, const uint32_t *ids,
                      const int *slots, int n, moe_expert_cb cb, void *user,
                      moe_expert_t *out)
{
    const moe_layer_t *ml = &ctx->plan.layers[layer];
    fetch_batch_t b;
    int miss[MOE_MAX_TOP_K];
    int nmiss = 0;

    memset(&b, 0, sizeof(b));
    b.ctx   = ctx;
    b.layer = layer;
    b.n     = n;

    /* Cache hits are available immediately and delivered first. */
    for (int i = 0; i < n; i++) {
        lru_entry_t *e = lru_cache_get(ctx->cache, moe_key(layer, ids[i]));
        if (e) {
            b.out[i].entry  = e;
            b.out[i].expert = ids[i];
            b.out[i].slot   = slots[i];
            b.done[i]       = 1;
        } else {
            miss[nmiss++] = i;
        }
    }

    /* Sort misses by physical offset. For this GGUF the expert base is
     * monotonic in expert id, so this collapses to an id sort — but sorting on
     * the computed offset stays correct if a future layout interleaves. */
    for (int a = 1; a < nmiss; a++) {
        int      v  = miss[a];
        uint64_t vo = ml->gate.base + (uint64_t)ids[v] * ml->gate.stride;
        int      j  = a - 1;
        while (j >= 0) {
            uint64_t uo = ml->gate.base + (uint64_t)ids[miss[j]] * ml->gate.stride;
            if (uo <= vo) {
                break;
            }
            miss[j + 1] = miss[j];
            j--;
        }
        miss[j + 1] = v;
    }

    pthread_mutex_lock(&ctx->mu);
    if (ctx->batch) {
        pthread_mutex_unlock(&ctx->mu);
        for (int i = 0; i < n; i++) {
            if (b.out[i].entry) {
                lru_entry_unpin(ctx->cache, b.out[i].entry);
            }
        }
        return -EBUSY;
    }
    ctx->batch = &b;
    b.remaining = nmiss;

    for (int k = 0; k < nmiss; k++) {
        int i = miss[k];

        moe_job_t *job = &ctx->queue[ctx->qtail];
        memset(job, 0, sizeof(*job));
        job->ctx    = ctx;
        job->batch  = &b;
        job->slot   = i;
        job->orig   = slots[i];
        job->expert = ids[i];
        job->key    = moe_key(layer, ids[i]);
        job->off[0] = ml->gate.base + (uint64_t)ids[i] * ml->gate.stride;
        job->len[0] = (size_t)ml->gate.stride;
        job->off[1] = ml->up.base + (uint64_t)ids[i] * ml->up.stride;
        job->len[1] = (size_t)ml->up.stride;
        job->off[2] = ml->down.base + (uint64_t)ids[i] * ml->down.stride;
        job->len[2] = (size_t)ml->down.stride;
        job->nparts = 3;

        ctx->qtail = (ctx->qtail + 1) % MOE_QUEUE_CAP;
        ctx->qcount++;
        ctx->inflight++;
        pthread_cond_signal(&ctx->cv_job);
    }

    int delivered = 0;
    int stop      = 0;

    while (delivered < n && !stop) {
        int idx = -1;
        for (int i = 0; i < n; i++) {
            if (b.done[i] && !b.delivered[i]) {
                idx = i;
                break;
            }
        }

        if (idx < 0) {
            if (b.remaining == 0) {
                break;
            }
            pthread_cond_wait(&ctx->cv_progress, &ctx->mu);
            continue;
        }

        b.delivered[idx] = 1;
        moe_expert_t ex  = b.out[idx];
        fill_view(ml, &ex);
        pthread_mutex_unlock(&ctx->mu);

        int cb_rc = cb ? cb(user, &ex) : 0;
        if (out) {
            out[slots[idx]] = ex;          /* keep pinned for the caller */
        } else {
            lru_entry_unpin(ctx->cache, ex.entry);
        }

        pthread_mutex_lock(&ctx->mu);
        delivered++;
        if (cb_rc != 0) {
            stop = 1;
        }
    }

    while (b.remaining > 0) {
        pthread_cond_wait(&ctx->cv_progress, &ctx->mu);
    }
    ctx->batch = NULL;
    int rc = b.rc;
    pthread_mutex_unlock(&ctx->mu);

    /* Anything completed but not delivered has to be released. */
    for (int i = 0; i < n; i++) {
        if (b.done[i] && !b.delivered[i] && b.out[i].entry) {
            lru_entry_unpin(ctx->cache, b.out[i].entry);
            b.out[i].entry = NULL;
        }
    }

    if (stop) {
        return -ECANCELED;
    }
    return rc != 0 ? rc : n;
}

/* ---- public decode API -------------------------------------------------- */

int moe_layer_fetch(moe_ctx_t *ctx, int layer, const uint32_t *experts, int n,
                    moe_expert_t *out)
{
    if (!ctx || !experts || !out || n <= 0) {
        return -EINVAL;
    }
    if (n > MOE_MAX_TOP_K) {
        return -E2BIG;
    }
    if (layer < 0 || layer >= ctx->plan.n_layers) {
        return -ERANGE;
    }

    uint32_t uniq[MOE_MAX_TOP_K];
    int      slots[MOE_MAX_TOP_K];
    int      first[MOE_MAX_TOP_K];
    int      nu = 0;

    memset(out, 0, sizeof(*out) * (size_t)n);

    for (int i = 0; i < n; i++) {
        if (experts[i] >= (uint32_t)ctx->plan.n_expert) {
            return -ERANGE;
        }
        int j;
        for (j = 0; j < nu; j++) {
            if (uniq[j] == experts[i]) {
                break;
            }
        }
        if (j == nu) {
            uniq[nu]  = experts[i];
            slots[nu] = i;
            nu++;
        }
        first[i] = slots[j];
    }

    int rc = fetch_impl(ctx, layer, uniq, slots, nu, NULL, NULL, out);
    if (rc < 0) {
        for (int i = 0; i < n; i++) {
            if (out[i].entry) {
                lru_entry_unpin(ctx->cache, out[i].entry);
                out[i].entry = NULL;
            }
        }
        return rc;
    }

    /* Aliases share the entry and take their own pin so moe_release() stays
     * balanced against the caller's array length. */
    for (int i = 0; i < n; i++) {
        if (first[i] != i) {
            out[i] = out[first[i]];
            lru_entry_pin(ctx->cache, out[i].entry);
        }
    }
    return n;
}

int moe_layer_fetch_stream(moe_ctx_t *ctx, int layer, const uint32_t *experts,
                           int n, moe_expert_cb cb, void *user)
{
    if (!ctx || !experts || !cb || n <= 0) {
        return -EINVAL;
    }
    if (n > MOE_MAX_TOP_K) {
        return -E2BIG;
    }
    if (layer < 0 || layer >= ctx->plan.n_layers) {
        return -ERANGE;
    }

    uint32_t uniq[MOE_MAX_TOP_K];
    int      slots[MOE_MAX_TOP_K];
    int      nu = 0;

    for (int i = 0; i < n; i++) {
        if (experts[i] >= (uint32_t)ctx->plan.n_expert) {
            return -ERANGE;
        }
        int j;
        for (j = 0; j < nu; j++) {
            if (uniq[j] == experts[i]) {
                break;
            }
        }
        if (j == nu) {
            uniq[nu]  = experts[i];
            slots[nu] = i;
            nu++;
        }
    }

    return fetch_impl(ctx, layer, uniq, slots, nu, cb, user, NULL);
}

void moe_release(moe_ctx_t *ctx, moe_expert_t *out, int n)
{
    if (!ctx || !out) {
        return;
    }
    for (int i = 0; i < n; i++) {
        if (out[i].entry) {
            lru_entry_unpin(ctx->cache, out[i].entry);
            out[i].entry = NULL;
        }
    }
}

/* ---- prefill layer sweep ------------------------------------------------ */

static void *pf_reader_main(void *arg)
{
    moe_ctx_t   *ctx = (moe_ctx_t *)arg;
    pf_reader_t *r   = &ctx->reader;

    for (;;) {
        pthread_mutex_lock(&r->mu);
        while (!r->req && !r->quit) {
            pthread_cond_wait(&r->cv_req, &r->mu);
        }
        if (r->quit) {
            pthread_mutex_unlock(&r->mu);
            break;
        }

        uint64_t seq = r->submitted;
        uint64_t off[3];
        size_t   len[3];
        int      nparts = r->nparts;
        uint8_t *dst    = r->dst;

        memcpy(off, r->off, sizeof(off));
        memcpy(len, r->len, sizeof(len));
        r->req = 0;
        pthread_mutex_unlock(&r->mu);

        int    rc = 0;
        size_t at = 0;
        for (int i = 0; i < nparts; i++) {
            rc = expert_io_read(&ctx->io, off[i], len[i], dst + at);
            if (rc != 0) {
                break;
            }
            at += len[i];
        }

        pthread_mutex_lock(&r->mu);
        r->rc        = rc;
        r->completed = seq;
        pthread_cond_broadcast(&r->cv_done);
        pthread_mutex_unlock(&r->mu);
    }
    return NULL;
}

static uint64_t pf_submit(moe_ctx_t *ctx, const uint64_t off[3], const size_t len[3],
                          int nparts, uint8_t *dst)
{
    pf_reader_t *r = &ctx->reader;

    pthread_mutex_lock(&r->mu);
    memcpy(r->off, off, sizeof(r->off));
    memcpy(r->len, len, sizeof(r->len));
    r->nparts = nparts;
    r->dst    = dst;
    r->req    = 1;
    uint64_t seq = ++r->submitted;
    pthread_cond_signal(&r->cv_req);
    pthread_mutex_unlock(&r->mu);
    return seq;
}

static int pf_wait(moe_ctx_t *ctx, uint64_t seq)
{
    pf_reader_t *r = &ctx->reader;

    pthread_mutex_lock(&r->mu);
    while (r->completed < seq) {
        pthread_cond_wait(&r->cv_done, &r->mu);
    }
    int rc = r->rc;
    pthread_mutex_unlock(&r->mu);
    return rc;
}

static int pf_ensure(moe_ctx_t *ctx)
{
    pf_reader_t *r = &ctx->reader;

    if (r->started) {
        return 0;
    }

    size_t eb = (size_t)ctx->plan.max_expert_bytes;
    for (int i = 0; i < 2; i++) {
        if (!ctx->pf_buf[i]) {
            void *p = NULL;
            if (posix_memalign(&p, 4096, eb) != 0) {
                return -ENOMEM;
            }
            ctx->pf_buf[i] = (uint8_t *)p;
        }
    }

    if (pthread_mutex_init(&r->mu, NULL) != 0) {
        return -ENOMEM;
    }
    if (pthread_cond_init(&r->cv_req, NULL) != 0) {
        return -ENOMEM;
    }
    if (pthread_cond_init(&r->cv_done, NULL) != 0) {
        return -ENOMEM;
    }
    if (pthread_create(&r->tid, NULL, pf_reader_main, ctx) != 0) {
        return -ENOMEM;
    }
    r->started = 1;
    return 0;
}

static void prefill_offsets(const moe_ctx_t *ctx, int layer, int expert,
                            uint64_t off[3], size_t len[3])
{
    const moe_layer_t *ml = &ctx->plan.layers[layer];

    off[0] = ml->gate.base + (uint64_t)expert * ml->gate.stride;
    len[0] = (size_t)ml->gate.stride;
    off[1] = ml->up.base + (uint64_t)expert * ml->up.stride;
    len[1] = (size_t)ml->up.stride;
    off[2] = ml->down.base + (uint64_t)expert * ml->down.stride;
    len[2] = (size_t)ml->down.stride;
}

static int next_nonempty(const int *count, int from, int n)
{
    for (int i = from; i < n; i++) {
        if (count[i] > 0) {
            return i;
        }
    }
    return -1;
}

int moe_prefill_layer_with(moe_ctx_t *ctx, int layer, int num_tokens,
                           const uint32_t *routing, moe_gemm_fn fn, void *user)
{
    if (!ctx || !routing || num_tokens <= 0) {
        return -EINVAL;
    }
    if (layer < 0 || layer >= ctx->plan.n_layers) {
        return -ERANGE;
    }

    const int n_expert = ctx->plan.n_expert;
    const int topk     = ctx->plan.n_expert_used > 0 ? (int)ctx->plan.n_expert_used : 8;
    const size_t total = (size_t)num_tokens * (size_t)topk;

    int      *count = (int *)calloc((size_t)n_expert + 1, sizeof(int));
    int      *start = (int *)malloc(((size_t)n_expert + 1) * sizeof(int));
    int      *fill  = (int *)malloc((size_t)n_expert * sizeof(int));
    uint32_t *list  = (uint32_t *)malloc(total * sizeof(uint32_t));
    int       rc    = 0;

    if (!count || !start || !fill || !list) {
        rc = -ENOMEM;
        goto out;
    }

    for (size_t i = 0; i < total; i++) {
        uint32_t e = routing[i];
        if (e >= (uint32_t)n_expert) {
            rc = -ERANGE;
            goto out;
        }
        count[e]++;
    }

    start[0] = 0;
    for (int e = 0; e < n_expert; e++) {
        start[e + 1] = start[e] + count[e];
    }
    memcpy(fill, start, (size_t)n_expert * sizeof(int));

    /* Inverted index: expert -> list of token ids routed to it. Counting sort,
     * O(num_tokens * top_k) per layer. */
    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < topk; k++) {
            uint32_t e = routing[(size_t)t * topk + k];
            list[fill[e]++] = (uint32_t)t;
        }
    }

    rc = pf_ensure(ctx);
    if (rc != 0) {
        goto out;
    }

    int e0 = next_nonempty(count, 0, n_expert);
    if (e0 < 0) {
        rc = 0;
        goto out;
    }

    uint64_t off[3];
    size_t   len[3];
    int      bi = 0;

    prefill_offsets(ctx, layer, e0, off, len);
    uint64_t seq = pf_submit(ctx, off, len, 3, ctx->pf_buf[bi]);

    for (int e = e0; e >= 0; ) {
        int next = next_nonempty(count, e + 1, n_expert);

        rc = pf_wait(ctx, seq);
        if (rc != 0) {
            break;
        }

        int cur = bi;
        bi ^= 1;
        if (next >= 0) {
            prefill_offsets(ctx, layer, next, off, len);
            seq = pf_submit(ctx, off, len, 3, ctx->pf_buf[bi]);
        }

        const moe_layer_t *ml = &ctx->plan.layers[layer];
        moe_weights_t w;
        w.gate       = ctx->pf_buf[cur];
        w.gate_bytes = (size_t)ml->gate.stride;
        w.up         = w.gate + ml->gate.stride;
        w.up_bytes   = (size_t)ml->up.stride;
        w.down       = w.up + ml->up.stride;
        w.down_bytes = (size_t)ml->down.stride;

        if (fn) {
            fn(user, layer, (uint32_t)e, &w, list + start[e], count[e]);
        }

        pthread_mutex_lock(&ctx->mu);
        ctx->prefill_experts++;
        ctx->prefill_bytes += ml->expert_bytes;
        pthread_mutex_unlock(&ctx->mu);

        e = next;
    }

out:
    free(count);
    free(start);
    free(fill);
    free(list);
    return rc;
}

int moe_prefill_layer(moe_ctx_t *ctx, int layer, int num_tokens,
                      const uint32_t *routing)
{
    return moe_prefill_layer_with(ctx, layer, num_tokens, routing,
                                  ctx ? ctx->gemm : NULL,
                                  ctx ? ctx->gemm_user : NULL);
}

int moe_prefill_layer_sweep(moe_ctx_t *ctx, int num_tokens,
                            const uint32_t *routing_map)
{
    if (!ctx || !routing_map || num_tokens <= 0) {
        return -EINVAL;
    }

    const int    topk   = ctx->plan.n_expert_used > 0 ? (int)ctx->plan.n_expert_used : 8;
    const size_t stride = (size_t)num_tokens * (size_t)topk;

    for (int l = 0; l < ctx->plan.n_layers; l++) {
        int rc = moe_prefill_layer(ctx, l, num_tokens, routing_map + (size_t)l * stride);
        if (rc != 0) {
            return rc;
        }
    }

    pthread_mutex_lock(&ctx->mu);
    ctx->prefill_tokens += (uint64_t)num_tokens;
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

/* ---- compute hook ------------------------------------------------------- */

void moe_dummy_gemm(void *user, int layer, uint32_t expert,
                    const moe_weights_t *w, const uint32_t *token_ids,
                    int n_tokens)
{
    (void)layer;
    (void)expert;
    (void)token_ids;

    /* Touch a sample of each slice so the read cannot be optimised away.
     * Deliberately cheap: the benchmark is measuring I/O, not FLOPs. */
    uint64_t h = 0xcbf29ce484222325ull;
    const size_t sample = 2048;

    for (int s = 0; s < 3; s++) {
        const uint8_t *p = s == 0 ? w->gate : (s == 1 ? w->up : w->down);
        size_t n = s == 0 ? w->gate_bytes : (s == 1 ? w->up_bytes : w->down_bytes);
        size_t take = n < sample ? n : sample;
        for (size_t i = 0; i < take; i++) {
            h ^= p[i];
            h *= 0x100000001b3ull;
        }
    }

    if (user) {
        *(uint64_t *)user = h ^ (uint64_t)n_tokens;
    }
}

void moe_set_gemm(moe_ctx_t *ctx, moe_gemm_fn fn, void *user)
{
    if (!ctx) {
        return;
    }
    ctx->gemm      = fn ? fn : moe_dummy_gemm;
    ctx->gemm_user = user;
}

void moe_compute(moe_ctx_t *ctx, int layer, uint32_t expert,
                 const moe_weights_t *w, const uint32_t *token_ids, int n_tokens)
{
    if (!ctx || !ctx->gemm) {
        return;
    }
    ctx->gemm(ctx->gemm_user, layer, expert, w, token_ids, n_tokens);
}

/* ---- lifecycle ---------------------------------------------------------- */

moe_ctx_t *moe_open(const char *gguf_path, size_t cache_bytes, int nthreads, int odirect)
{
    if (!gguf_path || cache_bytes == 0) {
        return NULL;
    }
    if (nthreads < 1) {
        nthreads = 1;
    }
    if (nthreads > 16) {
        nthreads = 16;
    }

    moe_ctx_t *ctx = (moe_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    ctx->nthreads  = nthreads;
    ctx->io.fd     = -1;
    ctx->gemm      = moe_dummy_gemm;
    ctx->gemm_user = &ctx->dummy_acc;

    if (gguf_index_load(&ctx->idx, gguf_path) != 0) {
        goto fail;
    }
    if (moe_plan_build(&ctx->idx, &ctx->plan) != 0) {
        goto fail;
    }
    if (expert_io_open(&ctx->io, gguf_path, odirect) != 0) {
        goto fail;
    }

    ctx->cache = lru_cache_create(cache_bytes);
    if (!ctx->cache) {
        goto fail;
    }
    /* Tail misses should not evict hot experts; opt out with MOE_NO_TINYLFU. */
    lru_cache_set_admission(ctx->cache, getenv("MOE_NO_TINYLFU") == NULL);

    if (pthread_mutex_init(&ctx->mu, NULL) != 0) {
        goto fail;
    }
    if (pthread_cond_init(&ctx->cv_job, NULL) != 0) {
        goto fail;
    }
    if (pthread_cond_init(&ctx->cv_progress, NULL) != 0) {
        goto fail;
    }

    ctx->threads = (pthread_t *)calloc((size_t)nthreads, sizeof(pthread_t));
    if (!ctx->threads) {
        goto fail;
    }

    for (int i = 0; i < nthreads; i++) {
        if (pthread_create(&ctx->threads[i], NULL, worker_main, ctx) != 0) {
            ctx->nthreads = i;
            goto fail;
        }
    }

    return ctx;

fail:
    moe_close(ctx);
    return NULL;
}

void moe_close(moe_ctx_t *ctx)
{
    if (!ctx) {
        return;
    }

    if (ctx->threads) {
        pthread_mutex_lock(&ctx->mu);
        ctx->shutdown = 1;
        pthread_cond_broadcast(&ctx->cv_job);
        pthread_mutex_unlock(&ctx->mu);

        for (int i = 0; i < ctx->nthreads; i++) {
            pthread_join(ctx->threads[i], NULL);
        }
        free(ctx->threads);
    }

    if (ctx->reader.started) {
        pthread_mutex_lock(&ctx->reader.mu);
        ctx->reader.quit = 1;
        pthread_cond_signal(&ctx->reader.cv_req);
        pthread_mutex_unlock(&ctx->reader.mu);
        pthread_join(ctx->reader.tid, NULL);
        pthread_mutex_destroy(&ctx->reader.mu);
        pthread_cond_destroy(&ctx->reader.cv_req);
        pthread_cond_destroy(&ctx->reader.cv_done);
    }

    free(ctx->pf_buf[0]);
    free(ctx->pf_buf[1]);
    lru_cache_destroy(ctx->cache);
    expert_io_close(&ctx->io);
    gguf_index_free(&ctx->idx);
    free(ctx);
}

/* ---- stats -------------------------------------------------------------- */

int moe_set_cache_allocator(moe_ctx_t *ctx,
                            void *(*alloc)(void *user, size_t size),
                            void  (*freefn)(void *user, void *ptr, size_t size),
                            void *user)
{
    if (!ctx || !ctx->cache) {
        return -EINVAL;
    }
    return lru_cache_set_allocator(ctx->cache, alloc, freefn, user);
}

int moe_prefetch(moe_ctx_t *ctx, int layer, const uint32_t *ids, int n)
{
    if (!ctx || !ids || n <= 0) {
        return -EINVAL;
    }
    if (layer < 0 || layer >= ctx->plan.n_layers) {
        return -ERANGE;
    }

    const moe_layer_t *ml = &ctx->plan.layers[layer];
    int queued = 0;

    pthread_mutex_lock(&ctx->mu);
    for (int i = 0; i < n; i++) {
        const uint32_t ex = ids[i];
        if (ex >= ml->gate.n_expert) {
            continue;
        }
        const uint64_t key = moe_key(layer, ex);
        if (lru_cache_contains(ctx->cache, key)) {
            continue;
        }
        if (ctx->qcount == MOE_QUEUE_CAP) {
            break;   /* never block the caller */
        }

        moe_job_t *job = &ctx->queue[ctx->qtail];
        memset(job, 0, sizeof(*job));
        job->ctx    = ctx;
        job->batch  = NULL;
        job->slot   = -1;
        job->orig   = -1;
        job->expert = ex;
        job->key    = key;
        job->off[0] = ml->gate.base + (uint64_t)ex * ml->gate.stride;
        job->len[0] = (size_t)ml->gate.stride;
        job->off[1] = ml->up.base + (uint64_t)ex * ml->up.stride;
        job->len[1] = (size_t)ml->up.stride;
        job->off[2] = ml->down.base + (uint64_t)ex * ml->down.stride;
        job->len[2] = (size_t)ml->down.stride;
        job->nparts = 3;

        ctx->qtail = (ctx->qtail + 1) % MOE_QUEUE_CAP;
        ctx->qcount++;
        ctx->inflight++;
        queued++;
    }
    if (queued) {
        pthread_cond_broadcast(&ctx->cv_job);
    }
    pthread_mutex_unlock(&ctx->mu);
    return queued;
}

const gguf_index_t *moe_index(const moe_ctx_t *ctx)
{
    return ctx ? &ctx->idx : NULL;
}

const moe_plan_t *moe_plan(const moe_ctx_t *ctx)
{
    return ctx ? &ctx->plan : NULL;
}

void moe_get_stats(moe_ctx_t *ctx, moe_stats_t *out)
{
    if (!ctx || !out) {
        return;
    }

    lru_stats_t s;
    lru_cache_stats(ctx->cache, &s);

    memset(out, 0, sizeof(*out));
    out->n_layers       = ctx->plan.n_layers;
    out->n_expert       = (int)ctx->plan.n_expert;
    out->n_expert_used  = (int)ctx->plan.n_expert_used;
    out->cache_capacity = s.capacity_bytes;
    out->cache_bytes    = s.bytes;
    out->cache_entries  = s.count;
    out->hits           = s.hits;
    out->misses         = s.misses;
    out->evictions      = s.evictions;
    out->rejected       = s.rejected;
    out->admitted       = s.admitted;
    out->bypassed       = s.bypassed;

    pthread_mutex_lock(&ctx->mu);
    out->bytes_read      = ctx->bytes_read;
    out->prefill_bytes   = ctx->prefill_bytes;
    out->prefill_experts = ctx->prefill_experts;
    out->prefill_tokens  = ctx->prefill_tokens;
    out->io_errors       = ctx->io_errors;
    pthread_mutex_unlock(&ctx->mu);
}

void moe_print_stats(moe_ctx_t *ctx)
{
    moe_stats_t s;
    lru_stats_t l;
    moe_get_stats(ctx, &s);
    lru_cache_stats(ctx->cache, &l);

    printf("cache: %zu entries, %.2f MiB / %.2f MiB, hit %.1f%% (%zu hit / %zu miss), "
           "%zu evictions, %zu rejected, %zu TinyLFU admits, %zu bypasses\n",
           s.cache_entries,
           (double)s.cache_bytes / (1024.0 * 1024.0),
           (double)s.cache_capacity / (1024.0 * 1024.0),
           lru_cache_hit_rate(&l) * 100.0,
           s.hits, s.misses, s.evictions, s.rejected, s.admitted, s.bypassed);
    printf("decode io: %.2f MiB read, %d errors\n",
           (double)s.bytes_read / (1024.0 * 1024.0), s.io_errors);
    printf("prefill   : %.2f GiB read, %llu experts, %llu prompt tokens\n",
           (double)s.prefill_bytes / (1024.0 * 1024.0 * 1024.0),
           (unsigned long long)s.prefill_experts,
           (unsigned long long)s.prefill_tokens);
}
