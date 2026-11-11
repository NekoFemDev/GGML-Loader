/*
 * lru_cache.h — byte-budgeted LRU over (layer, expert) keys.
 *
 * Implementation: open-addressed hash table (power-of-two, linear probing,
 * tombstoned deletes) + intrusive doubly linked list for recency. Both
 * operations are O(1) amortised. Data buffers are 4096-aligned so an O_DIRECT
 * path can target them directly without an extra copy on filesystems that can
 * satisfy the alignment.
 *
 * Pinning: a returned entry is pinned and cannot be evicted until unpinned.
 * The streaming layer pins all experts it is about to compute with, which is
 * what prevents the prefetcher from stealing a slot out from under the matmul.
 */
#ifndef MOE_LRU_CACHE_H
#define MOE_LRU_CACHE_H

#include <stddef.h>
#include <stdint.h>

typedef struct lru_entry lru_entry_t;

typedef struct {
    size_t capacity_bytes;
    size_t bytes;
    size_t count;
    size_t pins;
    size_t hits;
    size_t misses;
    size_t evictions;
    size_t insertions;
    size_t rejected;      /* put/alloc refused: everything evictable was pinned */
    size_t admitted;      /* TinyLFU: admitted over the victim */
    size_t bypassed;      /* TinyLFU: served without caching */
    size_t peak_bytes;
} lru_stats_t;

typedef struct lru_cache lru_cache_t;

lru_cache_t *lru_cache_create(size_t capacity_bytes);
void         lru_cache_destroy(lru_cache_t *c);

/* Hit: returns pinned entry, bumps recency. Miss: NULL. */
lru_entry_t *lru_cache_get(lru_cache_t *c, uint64_t key);
int          lru_cache_contains(lru_cache_t *c, uint64_t key);

/* TinyLFU admission: compare the incoming key's frequency against the LRU
 * victim. If it loses, the entry is allocated but never inserted, so the read
 * is served without polluting the cache. `admitted` reports which happened. */
void         lru_cache_set_admission(lru_cache_t *c, int on);

/* Pluggable backing store for entry data. The Vulkan arena installs one so
 * expert weights live in a single mapped VkBuffer instead of malloc + upload.
 * Must be called while the cache is empty. Admission is disabled with an
 * arena because a bypassed entry would need a temporary slot pool. */
typedef void *(*lru_alloc_fn)(void *user, size_t size);
typedef void  (*lru_free_fn)(void *user, void *ptr, size_t size);
int          lru_cache_set_allocator(lru_cache_t *c, lru_alloc_fn a,
                                     lru_free_fn f, void *user);
size_t       lru_cache_count(lru_cache_t *c);
lru_entry_t *lru_cache_alloc2(lru_cache_t *c, uint64_t key, size_t size,
                              int *admitted);

/* Allocate a pinned, not-yet-visible entry of `size` bytes. Evicts from the
 * cold end as needed. Fill entry->data, then commit (publishes to the table and
 * makes it visible to lru_cache_get) or abort. */
lru_entry_t *lru_cache_alloc(lru_cache_t *c, uint64_t key, size_t size);
int          lru_cache_commit(lru_cache_t *c, lru_entry_t *e);
void         lru_cache_abort(lru_cache_t *c, lru_entry_t *e);

/* Convenience: alloc + memcpy + commit. Returns pinned entry or NULL. */
lru_entry_t *lru_cache_put(lru_cache_t *c, uint64_t key, const void *data, size_t size);

void         lru_entry_unpin(lru_cache_t *c, lru_entry_t *e);
void         lru_entry_pin(lru_cache_t *c, lru_entry_t *e);
uint8_t     *lru_entry_data(lru_entry_t *e);
size_t       lru_entry_size(const lru_entry_t *e);
uint64_t     lru_entry_key(const lru_entry_t *e);

/* Drop every unpinned entry (used when the runtime switches task / frees RAM). */
void         lru_cache_clear(lru_cache_t *c);
void         lru_cache_stats(lru_cache_t *c, lru_stats_t *out);
void         lru_cache_reset_stats(lru_cache_t *c);
double       lru_cache_hit_rate(const lru_stats_t *s);

#endif /* MOE_LRU_CACHE_H */
