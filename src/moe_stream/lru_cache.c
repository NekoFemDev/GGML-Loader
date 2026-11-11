/*
 * lru_cache.c — byte-budgeted LRU.
 *
 * Thread safety: every public entry point takes an internal mutex. The lock is
 * only ever held for map/list surgery, never across I/O: lru_cache_alloc()
 * returns a pinned entry, the caller fills it outside the lock, then commits.
 * That keeps the critical section at tens of nanoseconds while eight workers
 * stream experts in parallel.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "lru_cache.h"

#include "memtrack.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define LRU_TOMBSTONE ((lru_entry_t *)(uintptr_t)1)
#define LRU_MIN_SLOTS 64u
#define LRU_ALIGN     4096u
#define LRU_SKETCH_W  4096u
#define LRU_SKETCH_ROWS 4u

struct lru_entry {
    uint64_t     key;
    size_t       size;
    uint32_t     pins;
    int          linked;      /* 1 once inserted into the hash table */
    int          accounted;   /* bytes reserved in the cache budget */
    uint8_t     *data;
    lru_entry_t *prev;
    lru_entry_t *next;
};

struct lru_cache {
    lru_entry_t **slots;
    size_t        nslots;     /* power of two */
    size_t        used;       /* live entries + tombstones */
    size_t        count;

    lru_entry_t  *head;       /* most recently used */
    lru_entry_t  *tail;       /* eviction candidate */

    size_t capacity_bytes;
    size_t bytes;             /* committed + reserved */

    lru_stats_t st;
    pthread_mutex_t mu;

    /* TinyLFU: 4-row count-min sketch of key frequencies. */
    uint8_t  *sketch;
    uint32_t  sketch_w;
    uint64_t  sketch_ops;
    uint64_t  sketch_limit;
    int       admission;

    lru_alloc_fn alloc_fn;
    lru_free_fn  free_fn;
    void        *alloc_user;
};

static uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static size_t next_pow2(size_t n)
{
    size_t p = LRU_MIN_SLOTS;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

static size_t find_slot_locked(const lru_cache_t *c, uint64_t key, int *found)
{
    size_t mask = c->nslots - 1;
    size_t i    = (size_t)mix64(key) & mask;
    size_t first_tomb = (size_t)-1;

    for (;;) {
        lru_entry_t *e = c->slots[i];
        if (!e) {
            *found = 0;
            return first_tomb != (size_t)-1 ? first_tomb : i;
        }
        if (e == LRU_TOMBSTONE) {
            if (first_tomb == (size_t)-1) {
                first_tomb = i;
            }
        } else if (e->key == key) {
            *found = 1;
            return i;
        }
        i = (i + 1) & mask;
    }
}

static void list_unlink(lru_cache_t *c, lru_entry_t *e)
{
    if (e->prev) {
        e->prev->next = e->next;
    } else {
        c->head = e->next;
    }
    if (e->next) {
        e->next->prev = e->prev;
    } else {
        c->tail = e->prev;
    }
    e->prev = e->next = NULL;
}

static void list_push_front(lru_cache_t *c, lru_entry_t *e)
{
    e->prev = NULL;
    e->next = c->head;
    if (c->head) {
        c->head->prev = e;
    }
    c->head = e;
    if (!c->tail) {
        c->tail = e;
    }
}

static void list_touch(lru_cache_t *c, lru_entry_t *e)
{
    if (c->head == e) {
        return;
    }
    list_unlink(c, e);
    list_push_front(c, e);
}

static void entry_free(lru_cache_t *c, lru_entry_t *e)
{
    if (!e) {
        return;
    }
    if (c && c->free_fn) {
        c->free_fn(c->alloc_user, e->data, e->size);
    } else {
        moe_mem_free(e->data, e->size);
    }
    free(e);
}

static int rehash_locked(lru_cache_t *c, size_t nslots)
{
    lru_entry_t **old   = c->slots;
    size_t        old_n = c->nslots;
    lru_entry_t **fresh = (lru_entry_t **)calloc(nslots, sizeof(*fresh));
    if (!fresh) {
        return -ENOMEM;
    }

    c->slots  = fresh;
    c->nslots = nslots;
    c->used   = 0;

    for (size_t i = 0; i < old_n; i++) {
        lru_entry_t *e = old[i];
        if (!e || e == LRU_TOMBSTONE) {
            continue;
        }
        int    found = 0;
        size_t s     = find_slot_locked(c, e->key, &found);
        c->slots[s]  = e;
        c->used++;
    }

    free(old);
    return 0;
}

static int evict_one_locked(lru_cache_t *c)
{
    /* Walk from the cold end. Pinned entries are skipped; in the steady state
     * at most top_k are pinned (the in-flight layer), so this is short. */
    lru_entry_t *e = c->tail;
    while (e && e->pins > 0) {
        e = e->prev;
    }
    if (!e) {
        return 0;
    }

    int    found = 0;
    size_t s     = find_slot_locked(c, e->key, &found);
    if (found) {
        c->slots[s] = LRU_TOMBSTONE;
    }

    list_unlink(c, e);
    c->bytes -= e->size;
    c->count--;
    c->st.evictions++;
    entry_free(c, e);
    return 1;
}

static uint64_t sk_hash(uint64_t key, unsigned seed)
{
    uint64_t x = key ^ (0x9E3779B97F4A7C15ull * (seed + 1u));
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static void sketch_inc(lru_cache_t *c, uint64_t key)
{
    if (!c->sketch) {
        return;
    }
    for (unsigned r = 0; r < LRU_SKETCH_ROWS; r++) {
        uint32_t i = (uint32_t)(sk_hash(key, r) & (c->sketch_w - 1u));
        uint8_t *p = &c->sketch[(size_t)r * c->sketch_w + i];
        if (*p < 15u) {
            (*p)++;
        }
    }
    c->sketch_ops++;
    if (c->sketch_ops >= c->sketch_limit) {
        for (size_t i = 0; i < (size_t)LRU_SKETCH_ROWS * c->sketch_w; i++) {
            c->sketch[i] >>= 1;
        }
        c->sketch_ops = 0;
    }
}

static uint8_t sketch_est(const lru_cache_t *c, uint64_t key)
{
    if (!c->sketch) {
        return 0;
    }
    uint8_t m = 255u;
    for (unsigned r = 0; r < LRU_SKETCH_ROWS; r++) {
        uint32_t i = (uint32_t)(sk_hash(key, r) & (c->sketch_w - 1u));
        uint8_t  v = c->sketch[(size_t)r * c->sketch_w + i];
        if (v < m) {
            m = v;
        }
    }
    return m;
}

static lru_entry_t *raw_entry(lru_cache_t *c, uint64_t key, size_t size,
                              int accounted)
{
    lru_entry_t *e = (lru_entry_t *)calloc(1, sizeof(*e));
    if (!e) {
        return NULL;
    }

    void *p = NULL;
    if (c->alloc_fn) {
        p = c->alloc_fn(c->alloc_user, size);
        if (!p) {
            free(e);
            return NULL;
        }
    } else if (posix_memalign(&p, LRU_ALIGN, size) != 0) {
        free(e);
        return NULL;
    }
    e->key       = key;
    e->size      = size;
    e->data      = (uint8_t *)p;
    e->pins      = 1;
    e->accounted = accounted;
    return e;
}

/* alloc + TinyLFU admission. Returns a pinned entry; *admitted says whether it
 * should be committed to the cache or is a one-shot bypass buffer. */
static lru_entry_t *alloc2_locked(lru_cache_t *c, uint64_t key, size_t size,
                                  int *admitted)
{
    *admitted = 1;

    if (size > c->capacity_bytes) {
        c->st.rejected++;
        return NULL;
    }

    if (c->admission) {
        sketch_inc(c, key);
    }

    if (c->bytes + size > c->capacity_bytes && c->admission) {
        lru_entry_t *victim = c->tail;
        while (victim && victim->pins > 0) {
            victim = victim->prev;
        }
        if (!victim) {
            c->st.rejected++;
            return NULL;
        }
        if (sketch_est(c, key) <= sketch_est(c, victim->key)) {
            /* loser: bypass the cache entirely */
            lru_entry_t *e = raw_entry(c, key, size, 0);
            if (!e) {
                return NULL;
            }
            c->st.pins++;
            c->st.bypassed++;
            *admitted = 0;
            return e;
        }
        c->st.admitted++;
        evict_one_locked(c);
    }

    while (c->bytes + size > c->capacity_bytes) {
        if (!evict_one_locked(c)) {
            c->st.rejected++;
            return NULL;
        }
    }

    lru_entry_t *e = NULL;
    for (int tries = 0; tries < 8 && !e; tries++) {
        e = raw_entry(c, key, size, 1);
        if (!e && !evict_one_locked(c)) {
            break;
        }
    }
    if (!e) {
        c->st.rejected++;
        return NULL;
    }
    c->st.pins++;

    /* Reserve bytes now, not at commit: parallel workers would otherwise all
     * observe the same c->bytes and overshoot the budget. */
    c->bytes += size;
    if (c->bytes > c->st.peak_bytes) {
        c->st.peak_bytes = c->bytes;
    }
    return e;
}

static int commit_locked(lru_cache_t *c, lru_entry_t *e)
{
    if (e->linked) {
        return -EINVAL;
    }

    if (c->used * 10 >= c->nslots * 7) {
        int rc = rehash_locked(c, c->nslots * 2);
        if (rc != 0) {
            return rc;
        }
    }

    int    found = 0;
    size_t s     = find_slot_locked(c, e->key, &found);
    if (found) {
        return -EEXIST;
    }

    int tomb = (c->slots[s] == LRU_TOMBSTONE);
    c->slots[s] = e;
    if (!tomb) {
        c->used++;
    }
    e->linked = 1;
    list_push_front(c, e);

    c->count++;
    c->st.insertions++;
    return 0;
}

static void abort_locked(lru_cache_t *c, lru_entry_t *e)
{
    /* alloc() left one pin (and one stat pin) on the entry; release both, and
     * give back the reservation taken at alloc time. */
    if (e->pins > 0) {
        e->pins--;
        if (c->st.pins > 0) {
            c->st.pins--;
        }
    }
    if (e->accounted && c->bytes >= e->size) {
        c->bytes -= e->size;
    }
    entry_free(c, e);
}

static lru_entry_t *get_locked(lru_cache_t *c, uint64_t key)
{
    int    found = 0;
    size_t s     = find_slot_locked(c, key, &found);
    if (!found) {
        c->st.misses++;
        return NULL;
    }

    lru_entry_t *e = c->slots[s];
    list_touch(c, e);
    e->pins++;
    c->st.pins++;
    c->st.hits++;
    return e;
}

/* ---- public API --------------------------------------------------------- */

lru_cache_t *lru_cache_create(size_t capacity_bytes)
{
    lru_cache_t *c = (lru_cache_t *)calloc(1, sizeof(*c));
    if (!c) {
        return NULL;
    }

    /* Size the table for a 0.7 load factor at an assumed 2.5 MiB payload. */
    size_t est = capacity_bytes / (2560u * 1024u) + 16u;
    c->nslots  = next_pow2(est * 2u);
    c->slots   = (lru_entry_t **)calloc(c->nslots, sizeof(*c->slots));
    if (!c->slots) {
        free(c);
        return NULL;
    }
    if (pthread_mutex_init(&c->mu, NULL) != 0) {
        free(c->slots);
        free(c);
        return NULL;
    }

    c->capacity_bytes = capacity_bytes;
    c->sketch = (uint8_t *)calloc((size_t)LRU_SKETCH_ROWS * LRU_SKETCH_W, 1);
    c->sketch_w = LRU_SKETCH_W;
    /* halve counters every ~10 tracked accesses per resident entry */
    c->sketch_limit = 10ull * (capacity_bytes / (2560ull * 1024ull) + 16ull);
    return c;
}

void lru_cache_destroy(lru_cache_t *c)
{
    if (!c) {
        return;
    }
    for (size_t i = 0; i < c->nslots; i++) {
        lru_entry_t *e = c->slots[i];
        if (e && e != LRU_TOMBSTONE) {
            entry_free(c, e);
        }
    }
    pthread_mutex_destroy(&c->mu);
    free(c->sketch);
    free(c->slots);
    free(c);
}

lru_entry_t *lru_cache_get(lru_cache_t *c, uint64_t key)
{
    if (!c) {
        return NULL;
    }
    pthread_mutex_lock(&c->mu);
    lru_entry_t *e = get_locked(c, key);
    pthread_mutex_unlock(&c->mu);
    return e;
}

int lru_cache_contains(lru_cache_t *c, uint64_t key)
{
    if (!c) {
        return 0;
    }
    pthread_mutex_lock(&c->mu);
    int    found = 0;
    (void)find_slot_locked(c, key, &found);
    pthread_mutex_unlock(&c->mu);
    return found;
}

lru_entry_t *lru_cache_alloc2(lru_cache_t *c, uint64_t key, size_t size,
                              int *admitted)
{
    int dummy = 1;
    if (!admitted) {
        admitted = &dummy;
    }
    if (!c || size == 0) {
        *admitted = 0;
        return NULL;
    }
    pthread_mutex_lock(&c->mu);
    lru_entry_t *e = alloc2_locked(c, key, size, admitted);
    pthread_mutex_unlock(&c->mu);
    return e;
}

lru_entry_t *lru_cache_alloc(lru_cache_t *c, uint64_t key, size_t size)
{
    int admitted = 1;
    return lru_cache_alloc2(c, key, size, &admitted);
}

int lru_cache_set_allocator(lru_cache_t *c, lru_alloc_fn a, lru_free_fn f,
                            void *user)
{
    if (!c) {
        return -1;
    }
    pthread_mutex_lock(&c->mu);
    if (c->count != 0) {
        pthread_mutex_unlock(&c->mu);
        return -1;   /* switching with live entries would mix allocators */
    }
    c->alloc_fn   = a;
    c->free_fn    = f;
    c->alloc_user = user;
    if (a) {
        c->admission = 0;   /* bypass needs a temp slot pool; see header */
    }
    pthread_mutex_unlock(&c->mu);
    return 0;
}

size_t lru_cache_count(lru_cache_t *c)
{
    if (!c) {
        return 0;
    }
    pthread_mutex_lock(&c->mu);
    size_t n = c->count;
    pthread_mutex_unlock(&c->mu);
    return n;
}

void lru_cache_set_admission(lru_cache_t *c, int on)
{
    if (!c) {
        return;
    }
    pthread_mutex_lock(&c->mu);
    c->admission = on ? 1 : 0;
    pthread_mutex_unlock(&c->mu);
}

int lru_cache_commit(lru_cache_t *c, lru_entry_t *e)
{
    if (!c || !e) {
        return -EINVAL;
    }
    pthread_mutex_lock(&c->mu);
    int rc = commit_locked(c, e);
    pthread_mutex_unlock(&c->mu);
    return rc;
}

void lru_cache_abort(lru_cache_t *c, lru_entry_t *e)
{
    if (!c || !e) {
        return;
    }
    pthread_mutex_lock(&c->mu);
    abort_locked(c, e);
    pthread_mutex_unlock(&c->mu);
}

lru_entry_t *lru_cache_put(lru_cache_t *c, uint64_t key, const void *data, size_t size)
{
    if (!c || !data) {
        return NULL;
    }

    pthread_mutex_lock(&c->mu);

    lru_entry_t *hit = get_locked(c, key);
    if (hit) {
        pthread_mutex_unlock(&c->mu);
        return hit;
    }

    int admitted = 1;
    lru_entry_t *e = alloc2_locked(c, key, size, &admitted);
    if (!e) {
        pthread_mutex_unlock(&c->mu);
        return NULL;
    }
    memcpy(e->data, data, size);

    int rc = commit_locked(c, e);
    if (rc == -EEXIST) {
        abort_locked(c, e);
        e = get_locked(c, key);
    } else if (rc != 0) {
        abort_locked(c, e);
        e = NULL;
    }

    pthread_mutex_unlock(&c->mu);
    return e;
}

void lru_entry_unpin(lru_cache_t *c, lru_entry_t *e)
{
    if (!c || !e) {
        return;
    }
    pthread_mutex_lock(&c->mu);
    if (e->pins > 0) {
        e->pins--;
        if (c->st.pins > 0) {
            c->st.pins--;
        }
    }
    if (e->pins == 0 && !e->linked) {
        entry_free(c, e);   /* TinyLFU bypass buffer, never in the table */
    }
    pthread_mutex_unlock(&c->mu);
}

void lru_entry_pin(lru_cache_t *c, lru_entry_t *e)
{
    if (!c || !e) {
        return;
    }
    pthread_mutex_lock(&c->mu);
    e->pins++;
    c->st.pins++;
    pthread_mutex_unlock(&c->mu);
}

uint8_t *lru_entry_data(lru_entry_t *e)       { return e ? e->data : NULL; }
size_t   lru_entry_size(const lru_entry_t *e) { return e ? e->size : 0; }
uint64_t lru_entry_key(const lru_entry_t *e)  { return e ? e->key : 0; }

void lru_cache_clear(lru_cache_t *c)
{
    if (!c) {
        return;
    }
    pthread_mutex_lock(&c->mu);
    lru_entry_t *e = c->tail;
    while (e) {
        lru_entry_t *prev = e->prev;
        if (e->pins == 0) {
            int    found = 0;
            size_t s     = find_slot_locked(c, e->key, &found);
            if (found) {
                c->slots[s] = LRU_TOMBSTONE;
            }
            list_unlink(c, e);
            c->bytes -= e->size;
            c->count--;
            entry_free(c, e);
        }
        e = prev;
    }
    pthread_mutex_unlock(&c->mu);
}

void lru_cache_stats(lru_cache_t *c, lru_stats_t *out)
{
    if (!c || !out) {
        return;
    }
    pthread_mutex_lock(&c->mu);
    *out                = c->st;
    out->capacity_bytes = c->capacity_bytes;
    out->bytes          = c->bytes;
    out->count          = c->count;
    out->pins           = c->st.pins;
    pthread_mutex_unlock(&c->mu);
}

void lru_cache_reset_stats(lru_cache_t *c)
{
    if (!c) {
        return;
    }
    pthread_mutex_lock(&c->mu);
    size_t cap = c->st.capacity_bytes;
    memset(&c->st, 0, sizeof(c->st));
    c->st.capacity_bytes = cap;
    pthread_mutex_unlock(&c->mu);
}

double lru_cache_hit_rate(const lru_stats_t *s)
{
    if (!s) {
        return 0.0;
    }
    size_t total = s->hits + s->misses;
    return total ? (double)s->hits / (double)total : 0.0;
}
