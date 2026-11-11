#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "moe_bridge.h"

#include "autotune.h"
#include "compute_backend.h"
#include "qwen_model.h"
#include "tokenizer.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct moe_session {
    qwen_model_t *m;
    tok_t         tok;
    char          err[256];
};

/* process-wide, for create() failures before a session exists */
static char g_err[256];

static void set_err(char *dst, size_t cap, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dst, cap, fmt, ap);
    va_end(ap);
}

const char *moe_bridge_last_error(void) { return g_err; }
const char *moe_session_error(const moe_session_t *s) { return s ? s->err : g_err; }

moe_session_t *moe_session_create(const moe_session_opts_t *opts)
{
    g_err[0] = '\0';

    if (!opts || !opts->gguf_path) {
        set_err(g_err, sizeof(g_err), "null options");
        return NULL;
    }
    if (opts->max_seq <= 0 || opts->cache_bytes == 0) {
        set_err(g_err, sizeof(g_err), "max_seq and cache_bytes must be > 0");
        return NULL;
    }

    moe_session_t *s = (moe_session_t *)calloc(1, sizeof(*s));
    if (!s) {
        set_err(g_err, sizeof(g_err), "out of memory");
        return NULL;
    }

    if (tok_load(&s->tok, opts->gguf_path) != 0) {
        set_err(g_err, sizeof(g_err), "tokenizer load failed: %s", opts->gguf_path);
        free(s);
        return NULL;
    }

    s->m = qwen_load(opts->gguf_path, opts->cache_bytes, opts->threads,
                     opts->max_seq, opts->odirect);
    if (!s->m) {
        set_err(g_err, sizeof(g_err), "model load failed: %s", opts->gguf_path);
        tok_free(&s->tok);
        free(s);
        return NULL;
    }

    return s;
}

void moe_session_destroy(moe_session_t *s)
{
    if (!s) {
        return;
    }
    qwen_free(s->m);
    tok_free(&s->tok);
    free(s);
}

const char *moe_session_backend(const moe_session_t *s)
{
    return s && s->m ? qwen_backend_name(s->m) : "";
}

int moe_session_vocab_size(const moe_session_t *s) { return s ? qwen_vocab_size(s->m) : 0; }
int moe_session_context(const moe_session_t *s)  { return s ? qwen_max_seq(s->m) : 0; }
int moe_session_pos(const moe_session_t *s)      { return s ? qwen_pos(s->m) : 0; }

int moe_session_reset(moe_session_t *s)
{
    if (!s || !s->m) {
        return -EINVAL;
    }
    qwen_reset(s->m);
    return 0;
}

int moe_session_encode(moe_session_t *s, const char *text, uint32_t *ids, int max_ids)
{
    if (!s || !text || !ids || max_ids <= 0) {
        return -1;
    }
    return tok_encode(&s->tok, text, (int *)ids, max_ids);
}

int moe_session_decode(moe_session_t *s, uint32_t id, char *buf, int cap)
{
    if (!s || !buf || cap <= 0) {
        return -1;
    }
    return tok_decode(&s->tok, (int)id, buf, (size_t)cap);
}

int moe_session_prefill(moe_session_t *s, const uint32_t *tokens, int n_tokens)
{
    if (!s || !s->m) {
        return -EINVAL;
    }
    return qwen_prefill(s->m, tokens, n_tokens);
}

int moe_session_forward(moe_session_t *s, uint32_t token)
{
    if (!s || !s->m) {
        return -EINVAL;
    }
    return qwen_forward(s->m, (int)token);
}

int moe_session_argmax(moe_session_t *s)
{
    return s && s->m ? qwen_argmax(s->m) : -1;
}

static int generate_impl(moe_session_t *s, int max_new, uint32_t *out_ids,
                         int out_cap, moe_token_cb cb, void *user)
{
    if (!s || !s->m || max_new <= 0) {
        return -EINVAL;
    }

    int n = 0;
    for (; n < max_new; n++) {
        const int id = qwen_argmax(s->m);
        if (id < 0) {
            break;
        }

        char text[256];
        int  len = tok_decode(&s->tok, id, text, sizeof(text));

        if (out_ids && n < out_cap) {
            out_ids[n] = (uint32_t)id;
        }
        if (cb && cb(user, (uint32_t)id, text, len) != 0) {
            n++;
            break;
        }

        if (n + 1 >= max_new) {
            n++;
            break;
        }
        if (qwen_pos(s->m) >= qwen_max_seq(s->m)) {
            n++;
            break;
        }
        if (qwen_forward(s->m, id) != 0) {
            n++;
            break;
        }
    }
    return n;
}

int moe_session_generate(moe_session_t *s, int max_new, uint32_t *out_ids,
                         int out_cap, moe_token_cb cb, void *user)
{
    return generate_impl(s, max_new, out_ids, out_cap, cb, user);
}

/* Implemented in Go with //export moe_go_token_callback. Weak so that plain C
 * binaries linking libmoe_bridge.a do not need the Go runtime. */
#if defined(__GNUC__) || defined(__clang__)
extern int moe_go_token_callback(void *user, uint32_t token_id,
                                 char *utf8, int len) __attribute__((weak));
#endif

static int go_thunk(void *user, uint32_t token_id, const char *utf8, int len)
{
#if defined(__GNUC__) || defined(__clang__)
    int (*fn)(void *, uint32_t, char *, int) = moe_go_token_callback;
    if (!fn) {
        return -1;
    }
    return fn(user, token_id, (char *)utf8, len);
#else
    (void)user;
    (void)token_id;
    (void)utf8;
    (void)len;
    return -1;
#endif
}

int moe_session_generate_go(moe_session_t *s, int max_new, uint32_t *out_ids,
                            int out_cap, uintptr_t go_handle)
{
    return generate_impl(s, max_new, out_ids, out_cap, go_thunk,
                         (void *)go_handle);
}

void moe_auto_config(int *out_cache_mb, int *out_ctx_len, int *out_avail_mb)
{
    int cache_mb = 0, ctx = 0;
    calculate_optimal_cache(0, &cache_mb, &ctx);
    if (out_cache_mb) {
        *out_cache_mb = cache_mb;
    }
    if (out_ctx_len) {
        *out_ctx_len = ctx;
    }
    if (out_avail_mb) {
        *out_avail_mb = (int)(moe_mem_available() / (1024 * 1024));
    }
}

/* Implemented in Go with //export go_token_callback. Weak so the C-only
 * binaries still link. */
#if defined(__GNUC__) || defined(__clang__)
extern void go_token_callback(int token_id) __attribute__((weak));
#endif

int moe_session_generate_raw(moe_session_t *s, int max_new)
{
    if (!s || !s->m || max_new <= 0) {
        return -EINVAL;
    }

    int n = 0;
    for (; n < max_new; n++) {
        const int id = qwen_argmax(s->m);
        if (id < 0) {
            break;
        }
#if defined(__GNUC__) || defined(__clang__)
        void (*fn)(int) = go_token_callback;
        if (fn) {
            fn(id);
        }
#endif
        if (n + 1 >= max_new) {
            n++;
            break;
        }
        if (qwen_pos(s->m) >= qwen_max_seq(s->m)) {
            n++;
            break;
        }
        if (qwen_forward(s->m, id) != 0) {
            n++;
            break;
        }
    }
    return n;
}

int moe_session_use_vulkan(moe_session_t *s)
{
    if (!s || !s->m) {
        return -EINVAL;
    }
    qwen_set_backend(s->m, compute_backend_vulkan());
    return 0;
}
