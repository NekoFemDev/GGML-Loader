/*
 * autotune.h — strict memory budgeting for Android.
 *
 * Two corrections over the first version:
 *   1. it reads MemAvailable from /proc/meminfo (MemFree is meaningless on
 *      Android, where most of RAM is page cache);
 *   2. it uses the *actual* f32 KV size, 192 KiB/token, not the f16 96 KiB.
 *      Budgeting 96 KiB against an f32 cache doubles the KV allocation and is
 *      the most likely cause of the LMK kill.
 */
#ifndef MOE_AUTOTUNE_H
#define MOE_AUTOTUNE_H

#include <stddef.h>

#define MOE_OS_RESERVE_BYTES   (2000ull * 1024 * 1024)
#define MOE_RUNTIME_RESERVE    (1900ull * 1024 * 1024)  /* resident host+VRAM */

/* f32 KV: 48 layers x 4 kv heads x 128 dim x 2 (K,V) x 4 B = 192 KiB/token.
 * This would be 96 KiB/token if the KV cache were switched to f16. */
#define MOE_KV_BYTES_PER_TOKEN (192ull * 1024)

#define MOE_EXPERT_BYTES       (2654208ull)

void calculate_optimal_cache(size_t total_ram, int *out_cache_mb, int *out_ctx_len);

/* Exposed for the CLI to print the same decision inputs. */
size_t moe_mem_available(void);

#endif /* MOE_AUTOTUNE_H */
