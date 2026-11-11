# Memory architecture and the LMK fix

The `--auto` run was killed by Android's low-memory killer. Three things
combined to cause it, and only one of them was the cache size.

## 1. Postmortem

| cause | effect |
|---|---|
| autotune assumed 96 KiB/token for KV | the KV cache is **f32**: 48 layers x 4 kv heads x 128 x 2 x 4 B = **192 KiB/token**. `ctx` was computed 2x too large; at 32768 tokens that is **6.3 GiB of KV alone** |
| host LRU + VRAM cache held the same experts | up to 1.5 GiB duplicated |
| `MemTotal` was used as "free" | on Android most of RAM is page cache; `MemTotal - 2.5 GiB` is not available memory |
| no allocation cap | the process grew until the LMK killed it, with no message |

## 2. `calculate_optimal_cache` (strict)

```c
void calculate_optimal_cache(size_t total_ram, int *out_cache_mb, int *out_ctx_len)
{
    size_t avail = moe_mem_available();       /* MemAvailable from /proc/meminfo */
    if (avail == 0) avail = total_ram;

    size_t cap        = moe_mem_limit();      /* hard process cap, default 4500 MiB */
    size_t cap_usable = cap > MOE_RUNTIME_RESERVE
                      ? cap - MOE_RUNTIME_RESERVE : 0;

    size_t usable = avail > MOE_OS_RESERVE_BYTES
                  ? avail - MOE_OS_RESERVE_BYTES : 0;
    if (usable > cap_usable) usable = cap_usable;

    size_t cache_bytes = usable / 2;
    size_t kv_bytes    = usable - cache_bytes;

    long cache_mb = cache_bytes / (1024 * 1024);
    if (cache_mb < 512) { cache_mb = 512; /* warning */ }

    long ctx = kv_bytes / MOE_KV_BYTES_PER_TOKEN;   /* 192 KiB, f32 */
    if (ctx < 512) ctx = 512;
    if (ctx > 32768) ctx = 32768;
    ...
}
```

- `MOE_OS_RESERVE_BYTES = 2000 MiB`
- `MOE_RUNTIME_RESERVE = 1900 MiB` — resident weights on host and in VRAM plus
  staging; without subtracting this, the 50/50 split still overcommits.
- `MOE_KV_BYTES_PER_TOKEN = 192 KiB`. The 96 KiB figure assumes f16 KV; using it
  against the f32 cache was the largest single contributor to the OOM. If the
  KV cache is moved to f16, change this constant to 96 KiB.

The function caps `usable` at `MOE_MEM_LIMIT_MB - 1900 MiB`, so it never
recommends settings the tracker will reject.

## 3. Memory tracker

`memtrack.c` keeps a process-wide `total_bytes_allocated` under a mutex with a
4.5 GiB default cap (`MOE_MEM_LIMIT_MB` overrides).

```c
int moe_mem_reserve(size_t bytes);      /* 0 + "[OOM] ..." line on overflow */
void *moe_mem_alloc(size_t bytes);      /* tracked calloc */
void *moe_mem_aligned(size_t a, size_t n);
void  moe_mem_free(void *p, size_t n);
```

Wired into the large allocations only: resident weights (`mt_load`), the KV
cache, the expert arena, the Vulkan resident-weight cache, and the default LRU
allocator. Scratch buffers are small and untracked.

On overflow the allocation fails, `qwen_load` returns NULL, the bridge returns
an error, and the process exits with a diagnostic instead of waiting for the
LMK:

```
[OOM] Reached memory limit: 733 MiB in use + 243 MiB requested > 900 MiB cap
failed to load model
```

## 4. Single allocation: the expert arena

The LRU no longer allocates entry data with `malloc`. It takes a pluggable
allocator:

```c
int lru_cache_set_allocator(lru_cache_t *c,
                            void *(*alloc)(void *user, size_t size),
                            void  (*freefn)(void *user, void *ptr, size_t size),
                            void *user);
```

The Vulkan backend creates **one** `HOST_VISIBLE | HOST_COHERENT` VkBuffer,
maps it once, and exposes fixed-size slots:

```
[vk] expert arena: 799 MiB, 294 slots x 2850816 B (host+GPU one mapping)
```

Flow:

1. The streamer `pread()`s gate/up/down **directly into the mapped slot**
   (no intermediate C buffer, no upload).
2. `matmul` checks whether the weight pointer is inside the arena. If so it
   binds the arena VkBuffer with `offset = ptr - base`; there is no copy.
3. LRU eviction calls the arena free, which clears the slot bit.

Resident (non-expert) weights are still malloc'd and uploaded once into the
keyed cache, because they are variable-sized and loaded before the backend is
selected. That is the remaining duplication (~0.95 GiB host + 0.95 GiB VRAM);
the next step is a second arena with a bump allocator for them.

One consequence: **TinyLFU admission is disabled when the arena is installed.**
A bypassed entry would need a temporary slot pool outside the arena; the
admission code stays available for the non-arena path.

## 5. Results (12 GB phone, 4.5 GB cap)

| configuration | prefill 5 tok | decode |
|---|---|---|
| before (no cap, 1.5 GiB VRAM cache) | 8.6 s | 0.41 tok/s |
| 512 MiB arena, manual | 10.3 s | 0.47 tok/s |
| auto @ 3.5 GiB cap (935 MiB arena) | 6.5 s | **0.57–0.64 tok/s** |

Output remains token-identical to the CPU backend (` Paris. The`).

## 6. Why 2.0 tok/s is still out of reach

Per token the GPU must move **1.83 GB** of quantised weights:

```
attention + router   557 MB
lm_head              255 MB
8 experts x 48       1021 MB
```

At the measured ~1.1 GB/s effective kernel throughput that is ~1.6 s, which is
exactly the observed decode. 2.0 tok/s (0.5 s) requires **3.7 GB/s** through the
GEMV kernel. Caching cannot remove this floor: even with 100% expert hits, the
557 MB of attention and 255 MB of lm_head are read every token.

The remaining lever is the kernel itself. The current shader assigns one
invocation per output row, so the 64 invocations in a workgroup read rows
1,152 bytes apart — uncoalesced. The next experiment is a workgroup that
processes 8 rows at a time with 8 lanes per row while still staging `x` once
per 64 rows; that keeps the `x` amortisation that made the 8x8 and 16-token
tilings lose, while making the weight reads contiguous.
