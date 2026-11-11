# Vulkan compute backend

Status: **implemented, validated on a Mali-G615 MC6, and measured.** Correct, but
not yet a net win end-to-end. The reason is quantified in §6 and is not the
shader — it is per-dispatch submission overhead.

```
Go / C model
    │  compute_backend_t vtable
    ▼
compute_vulkan.c        context, buffers, descriptors, pipelines, dispatch
    │
    ▼
matmul_f32.comp          y = W·x, f32
matmul_quant.comp        y = dequant(W)·x, q4_K / q5_K / q6_K in-shader
    │
    ▼
SPIR-V (embedded in vulkan_shaders.h)
```

---

## 1. Files

| file | role |
|---|---|
| `compute_vulkan.c` | instance/device/queue, command buffer, descriptor set, 2 pipelines, `vk_matmul` |
| `shaders/matmul_f32.comp` | f32 GEMV |
| `shaders/matmul_quant.comp` | quantised-weight GEMV (q4_K, q5_K, q6_K) |
| `shaders/build.sh` | `glslc` → `spirv-val` → `vulkan_shaders.h` |
| `shaders/spv2h.py` | embeds SPIR-V as `uint32_t[]` |
| `vulkan_shaders.h` | generated, checked in so a plain build has a fallback |
| `test_vulkan.c` | GPU-vs-CPU oracle on real GGUF rows + dispatch benchmark |

## 2. Setup

Minimal by design: a compute queue, one command buffer, one fence, three
storage buffers, two pipelines.

```c
VkApplicationInfo app = { .apiVersion = VK_API_VERSION_1_1 };
vkCreateInstance(...);

/* first physical device with a VK_QUEUE_COMPUTE_BIT family */
vkEnumeratePhysicalDevices(...);
vkGetPhysicalDeviceQueueFamilyProperties(...);

vkCreateDevice(...);                  /* one queue, no extensions */
vkCreateCommandPool(...);             /* RESET_COMMAND_BUFFER_BIT */
vkAllocateCommandBuffers(...);
vkCreateFence(... SIGNALED);

/* bindings: 0 = W(uint[]), 1 = x(float[]), 2 = y(float[]) */
vkCreateDescriptorSetLayout(...);     /* 3 x STORAGE_BUFFER, COMPUTE stage */
vkCreatePipelineLayout(...);          /* push constants: 16 bytes */
vkCreateShaderModule(...);            /* embedded SPIR-V */
vkCreateComputePipelines(...);        /* pipe_f32, pipe_q */
```

No validation layers, no swapchain, no surface. This is a headless compute
device; that is why it works from Termux without a window.

Buffers are `HOST_VISIBLE | HOST_COHERENT` and persistently mapped. On Mali
this is UMA memory, so the "upload" is a memcpy, not a PCIe transfer.

## 3. Shaders

### 3.1 f32

One invocation per output row, exactly as specified:

```glsl
layout(local_size_x = 64) in;
shared float xs[4096];

void main() {
    for (uint i = gl_LocalInvocationID.x; i < pc.ne0; i += 64u)
        xs[i] = x[i];
    barrier();

    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.ne1) return;

    uint base = row * pc.ne0;
    float acc = 0.0;
    for (uint i = 0; i < pc.ne0; ++i)
        acc += w[base + i] * xs[i];
    y[row] = acc;
}
```

`shared` staging matters: all 64 invocations in a workgroup read the same `x`,
so global traffic drops 64x. Measured improvement: ~12.3 ms → ~6.5 ms on a
4096x2048 q4_K matmul.

### 3.2 Quantised weights, dequantised in-shader

The weight SSBO is `uint[]`; the shader walks the ggml block layout:

| type | block | bytes | payload |
|---|---|---|---|
| q4_K | 256 | 144 | f16 d, f16 dmin, 12 scale/min bytes, 128 qs |
| q5_K | 256 | 176 | + 32 bytes of high bits |
| q6_K | 256 | 210 | 128 ql, 64 qh, 16 int8 scales, f16 d |

The 6-bit scale/min unpacking is identical to `get_scale_min_k4()`:

```glsl
void getScaleMin(uint j, uint base, out uint d, out uint m) {
    if (j < 4u) {
        d = byteAt(base + j) & 63u;
        m = byteAt(base + j + 4u) & 63u;
    } else {
        d = (byteAt(base + j + 4u) & 0x0Fu) | ((byteAt(base + j - 4u) >> 6u) << 4u);
        m = (byteAt(base + j + 4u) >> 4u)    | ((byteAt(base + j)      >> 6u) << 4u);
    }
}
```

The hot loop reads 32-bit words (eight q4_K elements per load) instead of bytes.

### 3.3 The alignment trap (cost me a debugging cycle)

q4_K blocks are 144 B and q5_K are 176 B — both multiples of 4, so
`w[byte_offset >> 2]` is safe. **q6_K blocks are 210 B**, so every odd block
starts at `offset % 4 == 2`. A direct word read returns shifted bytes and the
dot product is silently wrong (deterministically ~half the correct value).

Fix: an unaligned reader used for q6_K (and available for any future type):

```glsl
uint u32At(uint off) {
    uint sh  = (off & 3u) * 8u;
    uint idx = off >> 2u;
    uint lo  = w[idx] >> sh;
    uint hi  = sh == 0u ? 0u : (w[idx + 1u] << (32u - sh));
    return lo | hi;
}
```

This is the kind of bug the row-level GPU-vs-CPU oracle catches immediately,
which is why `test_vulkan.c` compares against `vec_dot` on real tensors rather
than eyeballing generated text.

## 4. Dispatch

```c
static void vk_matmul(void *self, uint32_t type, const uint8_t *w,
                      uint64_t ne0, uint64_t ne1, const float *x, float *y)
{
    /* pick pipeline by type; q*_K requires ne0 % 256 == 0 and ne0 <= 4096 */
    ensure_buffer(s, 0, row_bytes * ne1);
    ensure_buffer(s, 1, ne0 * 4);
    ensure_buffer(s, 2, ne1 * 4);

    memcpy(s->map[0], w, wbytes);
    memcpy(s->map[1], x, xbytes);
    if (s->desc_dirty) update_descriptors(s);

    struct { uint32_t ne0, ne1, type, pad; } pc = { ne0, ne1, type, 0 };

    vkResetFences(...); vkResetCommandBuffer(...);
    vkBeginCommandBuffer(...);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(...);
    vkCmdPushConstants(cmd, playout, COMPUTE, 0, 16, &pc);
    vkCmdDispatch(cmd, (ne1 + 63) / 64, 1, 1);
    vkEndCommandBuffer(...);
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);

    memcpy(y, s->map[2], ybytes);
}
```

The vtable is built at runtime so the CPU-side ops point straight at the
scalar implementation:

```c
be.name    = "vulkan";
be.matmul  = vk_matmul;      /* <-- GPU */
be.rmsnorm = sc->rmsnorm;    /* <-- CPU for now */
be.rope    = sc->rope;
be.softmax = sc->softmax;
be.swiglu  = sc->swiglu;
```

Fallback is automatic and permanent: if instance/device/pipeline creation fails,
`vk_init` sets `state = -1` and every subsequent matmul calls the scalar path.
Unsupported types (f16, unaligned geometries, `ne0 > 4096`) also fall back
per call.

## 5. Correctness

`./build/test_vulkan <gguf>` compares the shader against `vec_dot` on real rows:

```
device: Mali-G615 MC6
[f32] 64x128
  OK

[quantised rows, GPU vs vec_dot]
  q4_K  token_embd.weight    ne0=2048  cpu=-0.604341 gpu=-0.604343 rel=2.47e-06 OK
  q5_K  blk.1.attn_k.weight  ne0=2048  cpu= 0.206486 gpu= 0.206486 rel=1.15e-06 OK
  q6_K  output.weight        ne0=2048  cpu=-1.786066 gpu=-1.786064 rel=7.34e-07 OK
```

End-to-end generation with `--vulkan` produces the same text as the CPU backend.

## 6. Performance, and why it is not a win yet

Measured on the Mali-G615 MC6, 4.5 MiB q4_K matmul `[4096 x 2048]`:

| | time |
|---|---|
| CPU `vec_dot` | 6.5 ms |
| GPU dispatch | 6.4 ms |
| **submit + fence wait** | **0.81 ms per dispatch** (64x128, 8k MACs) |

The matmul itself is at parity. The problem is how many times it is called:

```
per token = 48 layers x (q, k, v, o = 4  +  router 1  +  8 experts x 3 = 24) + lm_head
          = 48 x 29 + 1
          ≈ 1392 dispatches
1392 x 0.81 ms ≈ 1.13 s of pure submission overhead per token
```

End-to-end, 5-token prompt + 2 generated tokens:

| backend | decode | 
|---|---|
| CPU scalar | 3.28 s/token |
| Vulkan matmul | 3.45 s/token |

So the GPU saves roughly the matmul time and loses it again to 1392 submit/wait
cycles. That is the whole story; no further shader tuning fixes it.

### Fix order

1. **One submit per layer.** This requires rmsnorm, rope, softmax, swiglu and
   attention to run on the GPU too, so a whole layer becomes one command buffer:
   48 submits/token instead of 1392 (saves ~1.1 s/token). This is the single
   highest-value change and the reason the vtable has `sync` as a separate op.
2. **Keep weights resident.** The streamer hands the backend reused host
   pointers, so a pointer-keyed cache is unsafe. The correct fix is for the LRU
   to allocate expert slots in `VkDeviceMemory` (or a shared host-visible arena
   registered with the backend) and tag each slot with a generation counter.
3. **Tiled kernel.** One workgroup per row, 64 threads cooperating over
   `ne0`, shared-memory weights tile, one output. The current per-row
   invocation is memory-latency-bound; a tiled version should reach 10x.
4. **Cooperative matrix.** Mali-G615 exposes `VK_KHR_cooperative_matrix` on some
   driver versions. Query it at init and only then build a `coopmat` variant;
   q4_K would need an on-the-fly dequant into shared f16 tiles. Do not make it
   the only path — the fallback must stay.

## 7. Build

```bash
# shaders (only when they change)
cd src/moe_stream/shaders && ./build.sh

# CPU-only build (Vulkan backend compiles to the scalar stub)
cd src/moe_stream && make clean && make all

# Vulkan build
cd src/moe_stream && make clean && make VULKAN=1 all build/test_vulkan
./build/test_vulkan ../../Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf
```

`vulkan_shaders.h` is a build dependency of `compute_vulkan.o`; without that
dependency a shader change silently keeps the old SPIR-V linked. That was a
real bug during development.

## 8. Platform notes

- Termux can open the Mali device through the Android Vulkan loader with no
  surface. `vkCreateInstance` needs no extensions here.
- The GPU is UMA: `HOST_VISIBLE | HOST_COHERENT` buffers are device-local, so
  the upload is a memcpy. There is no separate VRAM budget.
- `vkWaitForFences` per matmul is what costs 0.81 ms; batching is mandatory on
  mobile.
- The shader uses 16 KiB of shared memory (`xs[4096]`), which caps occupancy but
  is well under Mali's 32 KiB per-workgroup limit.
- 64 invocations per workgroup, `ne1` rows per dispatch; `vkCmdDispatch` limit
  is far above the 2374 groups needed for the 151,936-row lm_head.

---

# Phase 2 — batching, residency, tiling

## 1. Command batching

The vtable gained three things:

```c
void (*begin_batch)(void *self);
int  (*end_batch)(void *self);
void (*matmul)(void *self, uint32_t type, const uint8_t *w, uint64_t wkey,
               uint64_t ne0, uint64_t ne1, const float *x, float *y);
```

`wkey` is a stable tensor identity assigned by the model. It exists because the
streamer reuses host pointers between experts — a pointer-keyed VRAM cache would
return the previous expert's weights.

The model groups only the ops that are genuinely independent, because the CPU
still consumes intermediate results (attention, swiglu, top-k):

| group | dispatches | submits |
|---|---|---|
| q, k, v | 3 | 1 |
| attn_output | 1 | 1 |
| router | 1 | 1 |
| 8 x gate + 8 x up | 16 | 1 |
| 8 x down | 8 | 1 |

Per layer: **5 submits instead of 29**. Per token: ~240 instead of ~1392.

Vulkan side: `begin_batch` resets and begins one command buffer; `matmul`
uploads x into a per-slot region, resolves the weight buffer, records
bind/push/dispatch, and stashes the y pointer. `end_batch` submits once, waits
on one fence, then copies every recorded y back.

Slot staging is `VK_MAX_BATCH(32) x 4096 floats` for x and y. Outputs larger
than a slot (lm_head, 151,936 rows) take the old synchronous path.

## 2. Resident weights

`wcache_t` is a fixed-size table keyed by the 64-bit tensor key:

```c
typedef struct {
    uint64_t       key;
    VkBuffer       buf;
    VkDeviceMemory mem;
    size_t         size;
    uint64_t       used;    /* LRU clock */
    int            pinned;  /* touched in this batch */
} wcache_t;
```

- Lookup by key, not pointer, so an expert evicted from the host LRU and
  re-fetched later still hits its VRAM copy.
- Byte budget: 1 GiB default, `MOE_VK_BUDGET_MB` overrides.
- LRU eviction skips entries pinned in the current batch, so a 16-dispatch
  gate/up batch cannot evict its own weights. If everything is pinned, the
  batch flushes and retries once.

Per-token uploads drop from ~1.8 GiB (every matmul copying its weights) to
only the experts that miss both caches.

## 3. Tiled shader — implemented, measured, reverted

`shaders/matmul_quant_tiled.comp` (kept for reference) maps a workgroup to
8 rows x 8 lanes: lanes of one row own consecutive 256-element blocks, `x` is
staged in shared memory, and lane 0 reduces the 8 partials.

It is numerically correct (max rel 9.3e-05 on the full matmul) but **slower**:

| shader | 4096x2048 q4_K | decode |
|---|---|---|
| row-per-invocation (kept) | 6.4–9.5 ms | **0.41–0.42 tok/s** |
| 8x8 tiled (`matmul_quant_tiled.comp`) | 9.5 ms | 0.23 tok/s |

Why: for GEMV there is only one x vector, so "tiling" cannot reuse weights
across columns. The 8-row workgroup reloads x 8x more often (once per 8 rows
instead of per 64) and pays two barriers plus a reduction per 8 rows, while
the row shader already stages x once per 64 rows and has no reduction.
Tiling pays off for a real GEMM — i.e. batched prefill — not for decode.

The row shader's remaining weakness is weight coalescing (lane i reads row i's
blocks, 1152 bytes apart). Fixing that needs a different decomposition: a
workgroup per row, 32 lanes splitting the row's blocks, and a subgroup
reduction, with x already resident. That is the next shader to try.

## 4. Results (3 alternating runs, same prompt)

| backend | decode |
|---|---|
| CPU scalar | 0.30–0.31 tok/s |
| Vulkan, matmul-only (Phase 1) | 0.29 tok/s |
| **Vulkan, batched + resident** | **0.41–0.42 tok/s** |

- vs CPU: **1.36x**
- vs Phase 1 Vulkan: **1.45x**

Prefill (5 tokens) is 15.6 s on Vulkan vs 11.4 s CPU: the expert sweep is
host/I-O bound and still submits twice per expert (gate+up, down). Batching
across experts requires the streamer to keep several experts resident at once,
which is a streamer change, not a backend change.

`test_vulkan` still passes all row-level checks (q4_K 2.5e-06, q5_K 1.2e-06,
q6_K 7.3e-07).

## 5. The batching bug that took the longest

Symptom: batching made the model produce different text (" Paris(button"
instead of " Paris.") starting around layer 8. Per-expert comparisons showed
q4_K/q5_K/q6_K matmuls matching the CPU to ~1e-4 right up to the layer where
the output exploded.

Root cause: when the weight-cache table hit `VK_CACHE_MAX`, insertion called
`cache_drop(s, 0)` unconditionally. Entry 0 could be **pinned by the command
buffer currently being recorded**, so its `VkBuffer`/`VkDeviceMemory` was freed
while a recorded dispatch still referenced it. The memory was also still mapped
(freed without `vkUnmapMemory`), which is UB on its own.

Two fixes:
- table-full now uses the same `cache_victim()` scan as budget eviction and
  skips pinned entries; if all are pinned the call falls back to scalar;
- `cache_drop` unmaps before destroying, and `ensure_sbuf` unmaps sync buffers
  before recreation.

Why it looked budget-independent: with a 1 GiB budget the table filled to 256
entries before the byte budget was exhausted, so 512 MiB and 8 GiB behaved
identically. Enlarging `VK_CACHE_MAX` to 2048 made the symptom disappear, which
is what pointed at the table-full path rather than eviction.

Diagnostic switches kept in the tree (all off by default):
`MOE_VK_NO_BATCH`, `MOE_NO_BATCH_MOE`, `MOE_NO_BATCH_ATTN`,
`MOE_NO_BATCH_GATEUP`, `MOE_NO_BATCH_DOWN`, `MOE_VERIFY`, `MOE_VK_VERIFY`.

---

# Phase 3 — prefill batching across experts

## 1. What changed

`moe_prefill` no longer walks one expert at a time. Per layer it now:

1. builds the inverted index (expert -> token list) with `build_prefill_lists()`;
2. walks the routed experts in chunks of 16 (`MOE_PREFILL_CHUNK`);
3. fetches a whole chunk with `moe_layer_fetch()` so all its weights are
   uploaded to the VRAM cache before recording;
4. records gate+up for the whole chunk into **one** command buffer, submits
   once, runs swiglu on the CPU, then records all down projections into a
   second command buffer and submits once;
5. scatters the weighted results and releases the chunk.

Submits per layer: `2 * ceil(experts_used / 16)` instead of `2 * experts_used`.
For a 100-token prompt (~97 experts/layer) that is 14 submits instead of 194,
i.e. ~670 per prefill instead of ~9,300.

The new backend entry point:

```c
void (*matmul_batch)(void *self, uint32_t type, const uint8_t *w, uint64_t wkey,
                     uint64_t ne0, uint64_t ne1,
                     const float *x, const uint32_t *idx, int n_tokens,
                     float *y);   /* y[t] = W * x[idx[t]] */
```

The scalar backend loops `vec_dot`. Vulkan stages the gathered x rows into a
1 MiB per-slot region, then records one GEMV dispatch per token with
`x_off`/`y_off` push constants — all inside the open batch, so the whole chunk
costs one `vkQueueSubmit`.

## 2. Results

5-token prompt:

| prefill | time |
|---|---|
| Vulkan, per-expert submits (Phase 2) | 15.6 s |
| CPU token-by-token sweep | 12.96 s |
| **Vulkan, chunk-batched** | **8.57 s** |

32-token prompt:

| prefill | time |
|---|---|
| CPU token-by-token sweep | 65.8 s |
| Vulkan, per-expert submits (Phase 2) | 65.8 s |
| **Vulkan, chunk-batched** | **29.4 s** |

Decode is unchanged at 0.41–0.42 tok/s.

## 3. Tiled GEMM shader — implemented, correct, slower

`shaders/matmul_gemm.comp` is the requested prefill GEMM: one invocation per
output row, 16 token accumulators, `x` staged in shared memory as 16x64 tiles,
weights decoded once and reused across all 16 tokens.

It is numerically correct (same output text), but slower:

| prefill, 32 tokens | time |
|---|---|
| batched per-token GEMV (default) | **29.4 s** |
| 16-token tiled GEMM (`MOE_VK_GEMM=1`) | 37–47 s |

Why: 16 accumulators plus the q4_K/q5_K/q6_K decode path pushes register
pressure high enough to cut occupancy, and every one of the 12 workgroups per
gate matrix re-reads the shared x tiles. The simple GEMV keeps one accumulator
and amortises x over 64 rows per workgroup. This is the third tiling variant
measured on the G615 and the third to lose to the row-per-invocation shader;
the pattern is consistent enough to treat as a property of this GPU for
GEMV-shaped work. The shader is kept behind `MOE_VK_GEMM=1` for GPUs where the
trade-off differs (desktop/discrete, or much larger token batches), and for
when prefill becomes a true large-M GEMM.

## 4. Memory management

- Host: `moe_layer_fetch()` pins a chunk of 16 experts; `moe_release()` unpins
  them after the scatter. The host LRU (`--cache-mb`) recycles the slots.
- VRAM: `matmul_batch` resolves each expert through the keyed cache. Entries
  referenced by the current chunk are pinned, so recording 32 dispatches cannot
  evict the weights they use.
- The chunk bounds the simultaneous prefill working set to 16 experts x 3
  matrices (~130 MiB), well inside the 1 GiB default budget, so a prefill does
  not sweep the whole VRAM cache in one layer.
- After the chunk, entries are unpinned and the decode LRU reclaims them
  naturally; any expert that decode also uses stays hot. No explicit flush is
  needed or performed.
- Shader changes still require `shaders/build.sh`; `compute_vulkan.o` depends on
  `vulkan_shaders.h`, so `make` rebuilds it.

---

# Phase 4 — decode experiments

## lm_head residency (shipped)

`sync_matmul()` now takes `wkey` and resolves the weight through the keyed
cache. `lm_head` is 255 MiB (`ne1 = 151936`, larger than a batch slot), so it
used to take the synchronous path and `memcpy` all 255 MiB into the staging
buffer **every token**. It is now uploaded once; the staging buffers are used
only for x and y.

## Coalesced GEMV (shipped, gated, slower)

`matmul_coalesced.comp` maps a workgroup to 8 groups x 8 lanes. Group g walks
8 rows; lanes read consecutive 256-element blocks, and `x` is staged once per
64 rows. Numerically correct (max rel 4e-7).

End-to-end it loses:

| variant | decode @ 1024/2048 | 4.5 MiB matmul |
|---|---|---|
| row-per-invocation (default) | **0.78–0.86 tok/s** | 9.5 ms |
| coalesced 8x8 | 0.65–0.69 tok/s | 7.3 ms |
| no-shared x (diagnostic) | 0.78–0.82 tok/s | 8.2 ms |

The microbench favours coalescing but includes a 4.5 MiB upload and does not
reflect the real dispatch mix (many small `ne1` matmuls, cached weights).
The no-shared variant removes all 16 KiB of shared memory and measures the
same as the row shader, which rules out occupancy as the limiter. The
conclusion is that this GEMV is **latency/instruction bound**, not DRAM
bandwidth bound; coalescing reduces the number of independent row streams
without improving the resource that is actually saturated.

Both variants stay behind `MOE_VK_COALESCED=1` / `MOE_VK_NOSHARED=1`; the row
shader remains the default.

## Autotune and threads (shipped)

- `calculate_optimal_cache()` prefers **1800 MiB cache / 4096 ctx** when
  `cap_usable` and `MemAvailable` allow it, otherwise falls back to the 50/50
  split.
- CLI and test harness default to `--threads 6`.

## Net result

Decode went from 0.61 tok/s (auto 1100/5871, 4 threads, lm_head re-uploaded)
to **0.72–0.86 tok/s** at a fixed 1024/2048 budget (best observed 0.86). The
1.8 tok/s target was not reached; the remaining gap is kernel-bound, and the
three tiling/occupancy attempts each lost to the simple row shader on this GPU.

---

# Phase 5 — decode profiling and 128-thread workgroups

## Timestamp profiler

`VkQueryPool` timestamps are written around each command buffer; `end_batch`
reads them after the fence and accumulates:

- **gpu** — GPU execution time from the timestamps
- **submit/wait** — wall time in `vkQueueSubmit` + `vkWaitForFences` minus GPU
- **record/upload** — CPU time building the command buffer (x copies, descriptor
  updates, recording)

`compute_vulkan_profile_reset()` is called after prefill, so the report covers
decode only. A run at 1024/2048, 8 generated tokens:

```
[vk profile] batches=2169 dispatch=12537
             gpu=5263 ms (61%)  submit/wait=2467 ms (29%)  record/upload=846 ms (10%)
```

Per token: **241 batches** (5 per layer + lm_head), **~1.14 ms per submit/wait**.
So decode is now roughly 2/3 kernel and 1/3 submission latency — the batching
work removed the earlier 1,392-submit collapse, but 241 per token is still
expensive at this fence latency.

Why 241 and not fewer: qkv and o are separated by CPU attention; the router must
finish before expert selection; gate/up precede the CPU swiglu; down precedes
the residual. Each dependency forces a fence.

## 128-thread workgroups

`matmul_wg128.comp` is the row shader with `local_size_x = 128`: 128 rows per
workgroup, half as many workgroups, same per-row code. Correct (2.5e-06).
End-to-end it is neutral-to-slightly-worse:

| variant | decode | decode GPU time |
|---|---|---|
| row (64) | 0.82 tok/s | 5.26 s |
| row-128 | 0.78 tok/s | 5.62 s |

The GPU time itself did not improve, so the kernel is not short of resident
threads; more latency hiding does not help. Kept behind `MOE_VK_WG128=1`.

## What the profile says to do next

1. **Cut batches per token.** The cheapest reduction is moving the router matmul
   to the CPU: 128x2048 f32 is ~0.26 MMAC, ~0.1 ms scalar, and it removes one
   submit/wait per layer (48 per token). Attention on the GPU would merge qkv+o
   into one batch and save another 48.
2. **Fence latency.** 1.14 ms per batch is high for a compute queue; a bounded
   spin on `vkGetFenceStatus` before falling back to `vkWaitForFences` is worth
   measuring on this driver.
3. **Kernel throughput** remains 61% of decode. Coalesced, no-shared, 8x8 and
   16-token variants all lost to the row shader; the next candidate needs a
   different decomposition (e.g. one workgroup per row with a subgroup
   reduction), not another tiling of the same mapping.
