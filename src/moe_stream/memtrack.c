#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "memtrack.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t          g_limit;
static size_t          g_used;
static int             g_inited;

void moe_mem_init(void)
{
    pthread_mutex_lock(&g_mu);
    if (!g_inited) {
        g_limit  = MOE_DEFAULT_MEM_LIMIT;
        g_inited = 1;
        const char *env = getenv("MOE_MEM_LIMIT_MB");
        if (env) {
            long mb = strtol(env, NULL, 10);
            if (mb > 0) {
                g_limit = (size_t)mb * 1024 * 1024;
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
}

void moe_mem_set_limit(size_t bytes)
{
    moe_mem_init();
    pthread_mutex_lock(&g_mu);
    g_limit = bytes;
    pthread_mutex_unlock(&g_mu);
}

size_t moe_mem_limit(void)
{
    moe_mem_init();
    pthread_mutex_lock(&g_mu);
    size_t v = g_limit;
    pthread_mutex_unlock(&g_mu);
    return v;
}

size_t moe_mem_used(void)
{
    pthread_mutex_lock(&g_mu);
    size_t v = g_used;
    pthread_mutex_unlock(&g_mu);
    return v;
}

int moe_mem_reserve(size_t bytes)
{
    moe_mem_init();
    pthread_mutex_lock(&g_mu);
    const int ok = (bytes <= g_limit && g_used <= g_limit - bytes);
    if (ok) {
        g_used += bytes;
    } else {
        fprintf(stderr,
                "[OOM] Reached memory limit: %zu MiB in use + %zu MiB requested"
                " > %zu MiB cap\n",
                g_used / (1024 * 1024), bytes / (1024 * 1024),
                g_limit / (1024 * 1024));
    }
    pthread_mutex_unlock(&g_mu);
    return ok;
}

void moe_mem_release(size_t bytes)
{
    pthread_mutex_lock(&g_mu);
    if (bytes <= g_used) {
        g_used -= bytes;
    } else {
        g_used = 0;
    }
    pthread_mutex_unlock(&g_mu);
}

void *moe_mem_alloc(size_t bytes)
{
    if (!moe_mem_reserve(bytes)) {
        return NULL;
    }
    void *p = calloc(1, bytes);
    if (!p) {
        moe_mem_release(bytes);
    }
    return p;
}

void *moe_mem_aligned(size_t align, size_t bytes)
{
    if (!moe_mem_reserve(bytes)) {
        return NULL;
    }
    void *p = NULL;
    if (posix_memalign(&p, align, bytes) != 0) {
        moe_mem_release(bytes);
        return NULL;
    }
    return p;
}

void moe_mem_free(void *p, size_t bytes)
{
    if (!p) {
        return;
    }
    free(p);
    moe_mem_release(bytes);
}
