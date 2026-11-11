/*
 * memtrack.h — process-wide allocation budget for a 12 GB Android device.
 *
 * Every large allocation (resident weights, KV cache, expert arena, Vulkan
 * buffers) goes through here. Crossing the limit fails the allocation with a
 * clear message instead of letting the LMK kill the process silently.
 */
#ifndef MOE_MEMTRACK_H
#define MOE_MEMTRACK_H

#include <stddef.h>

#define MOE_DEFAULT_MEM_LIMIT (4500ull * 1024 * 1024)

void   moe_mem_init(void);                 /* reads MOE_MEM_LIMIT_MB once */
void   moe_mem_set_limit(size_t bytes);
size_t moe_mem_limit(void);
size_t moe_mem_used(void);

/* Returns 1 if the reservation fits, 0 otherwise (and prints an [OOM] line). */
int    moe_mem_reserve(size_t bytes);
void   moe_mem_release(size_t bytes);

/* Tracked replacements for malloc/posix_memalign/free. Return NULL on limit. */
void  *moe_mem_alloc(size_t bytes);
void  *moe_mem_aligned(size_t align, size_t bytes);
void   moe_mem_free(void *p, size_t bytes);

#endif /* MOE_MEMTRACK_H */
