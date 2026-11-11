#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "autotune.h"
#include "memtrack.h"

#include <stdio.h>
#include <string.h>

size_t moe_mem_available(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) {
        return 0;
    }

    char   line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) {
            break;
        }
    }
    fclose(f);
    return kb * 1024;
}

void calculate_optimal_cache(size_t total_ram, int *out_cache_mb, int *out_ctx_len)
{
    if (!out_cache_mb || !out_ctx_len) {
        return;
    }

    /* Prefer MemAvailable; fall back to the caller's total. */
    size_t avail = moe_mem_available();
    if (avail == 0) {
        avail = total_ram;
    }
    if (avail == 0) {
        *out_cache_mb = 512;
        *out_ctx_len  = 2048;
        return;
    }

    /* Never exceed the process hard cap, minus the memory that is always
     * resident (weights, Vulkan copies, staging). */
    moe_mem_init();
    size_t cap = moe_mem_limit();
    size_t cap_usable = cap > MOE_RUNTIME_RESERVE
        ? cap - MOE_RUNTIME_RESERVE
        : 0;

    size_t usable = avail > MOE_OS_RESERVE_BYTES
        ? avail - MOE_OS_RESERVE_BYTES
        : 0;
    if (usable > cap_usable) {
        usable = cap_usable;
    }
    if (usable == 0) {
        usable = 512ull * 1024 * 1024;   /* floor; caller's tracker will warn */
    }

    size_t cache_bytes = usable / 2;
    size_t kv_bytes    = usable - cache_bytes;

    long cache_mb = (long)(cache_bytes / (1024 * 1024));
    if (cache_mb < 512) {
        cache_mb = 512;
        fprintf(stderr,
                "[autotune] WARNING: only %zu MiB usable; forcing 512 MiB "
                "expert cache (accelerator may not fit)\n",
                usable / (1024 * 1024));
    }

    long ctx = (long)(kv_bytes / MOE_KV_BYTES_PER_TOKEN);
    if (ctx < 512) {
        ctx = 512;
    }
    if (ctx > 32768) {
        ctx = 32768;
    }

    *out_cache_mb = (int)cache_mb;
    *out_ctx_len  = (int)ctx;

    fprintf(stderr,
            "[autotune] avail %zu MiB, cap %zu MiB -> cache %d MiB, ctx %d "
            "(%.1f GiB KV)\n",
            avail / (1024 * 1024), cap / (1024 * 1024), *out_cache_mb, *out_ctx_len,
            (double)((size_t)*out_ctx_len * MOE_KV_BYTES_PER_TOKEN) /
                (1024.0 * 1024 * 1024));
}
