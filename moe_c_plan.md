# MoE Expert Streaming in C — AirLLM Reverse-Engineering + Design

Target: **Qwen3-Coder-30B-A3B-Instruct, UD-Q4_K_XL GGUF (16.45 GiB)** on a 12 GB
Android phone (Termux, aarch64), ~580 MB/s UFS.

Everything below was verified against the actual file in this directory, not
against assumptions. Working code is in `src/moe_stream/`, compiled and tested.

---

## 0. TL;DR — the five things that matter

1. **One expert is 2.53 MiB, not 21 MB.** 21.2 MB is the *eight-expert working
   set* for one layer. Three slices (gate/up/down) × 884,736 B = 2,654,208 B.
   This error would have inflated every cache-size decision by 8×. On the 13
   layers whose `ffn_down_exps` is q5_K the payload is 2,850,816 B — see §2.1.

2. **AirLLM has no LRU cache and does not use mmap or O_DIRECT for weights.**
   It streams whole layers with `safetensors.safe_open(...).get_tensor()`,
   frees them with `module.to('meta')`, and delegates reuse to the OS page
   cache. Its only concurrency is a **single-slot, one-layer-lookahead**
   prefetch on **one** worker thread.

3. **AirLLM's MoE path (this fork already has one) is strictly worse than its
   dense path** — expert loads happen synchronously inside the forward
   pre-hook, with no prefetch, no cache, and no eviction policy. There is
   nothing to copy there. You have to build the cache layer yourself.

4. **Decode is the easy half. Prefill is the wall.** A 1000-token prompt
   routed batch-wise touches ~all 128 experts per layer. The only sane
   prefill is a *layer sweep*: read all 128 experts of layer L once, apply to
   every token, discard, next layer. That is one full 16.4 GiB pass per prompt
   (~28 s at 580 MB/s) regardless of prompt length — versus petabytes if you
   stream per token.

5. **At 0% cache hit, decode costs 1,019 MB of I/O per token** (48 × 21.2 MB),
   i.e. **0.57 tok/s at 580 MB/s**. A 2–4 GB expert cache is not an
   optimisation, it is the difference between usable and not. Measured on this
   device: 24.7% hit at 1 GiB → 571 MiB read/token → 2.6 tok/s I/O ceiling.

---

## 1. AirLLM: what the code actually does

Repo layout note: the package directory is `air_llm/airllm/`, not
`airllm/airllm/`. The source of truth is `airllm-main/air_llm/airllm/`.

### 1.1 Memory management when loading a layer

`airllm_base.py:430 load_layer_to_cpu()` → `utils.py:444 load_layer()`:
`ModelPersister.get_model_persister().load_model(layer_name, local_path)`.
For safetensors that is `safetensors.safe_open(...).get_tensor(name)`, i.e.
**buffered reads through the page cache, one full copy into a PyTorch CPU
tensor**. For `.bin` checkpoints it is a full `torch.load(..., map_location='cpu')`.

| Mechanism | Used for weights? | Where |
|---|---|---|
| `torch.load` / `safe_open().get_tensor` | **yes** | `utils.py:444`, `utils.py:431` |
| `mmap` | **no** (only for Flash-Next n-gram embedding tables) | `utils.py:312 open_ngram_mmap_table` |
| `O_DIRECT` | **no** | — |
| `read`/`pread` directly | **no** | — |

Eviction is whole-module and immediate: after the forward hook runs,
`module.to('meta')` drops every parameter (`airllm_base.py:917`), then
`clean_memory()` (`utils.py:78`) calls `gc.collect()`, `malloc_trim(0)` and
`torch.cuda.empty_cache()`.

So the "cache" is: *the kernel's page cache, plus whatever hasn't been
garbage-collected yet*. There is no residency control and no hit accounting.

### 1.2 LRU / disk swapping

**There is no LRU, no ARC, no clock, no frequency tracking.** A grep for
`lru|LRU|cache` over the package finds only the n-gram mmap table and
`empty_cache()`. Reuse of a previously-read layer is entirely a function of
whether the kernel still has its pages.

The only stateful read-ahead is a **single slot**:

```python
# airllm_base.py:191
self._executor = ThreadPoolExecutor(max_workers=1) if self.prefetching else None
self._prefetch_future = None
self._prefetched_idx = None
```

```python
# airllm_base.py:889 — _pre_hook
state_dict = self._prefetch_future.result()      # consume the one prefetch
...
nxt = self._next_streamed_idx(idx)
self._prefetch_future = self._executor.submit(self._load_streamed_layer, nxt)
```

That is a depth-1 pipeline. It works for dense models because one layer is
~350 MB and the read dwarfs everything else; it is inadequate for MoE, where
the unit is 2.65 MB and the access pattern is data-dependent.

### 1.3 Overlapping compute with disk I/O

- **Dense path:** yes, one layer ahead, one background thread. Layer N's
  weights are read while layer N−1 computes on the GPU.
- **MoE expert path (`airllm_base.py:866`):** no. The expert's weights are
  read *inside* the module's forward pre-hook and released in the post-hook.
  Per-expert streaming in this fork is a memory optimisation, not a latency
  optimisation — the read is fully exposed on the critical path.
- `pin_memory()` is applied to layers under 2 GiB (`max_pinned_layer_bytes`,
  `:75`) so the H2D copy can be async; on a phone there is no discrete GPU, so
  this is irrelevant to you.

### 1.4 The parts worth stealing

- **Meta-device lifecycle.** Instantiate the graph with no storage, attach a
  hook that materialises a module just before it runs, drop it after. In C
  this becomes: a tensor descriptor table + a per-slot `ready` flag.
- **`load_layer_subset` idea (`utils.py:431`).** Read only the named tensors
  out of a shard. Your GGUF equivalent is offset arithmetic — already
  implemented in `gguf_index.c`.
- **Their own admission (comment at `airllm_base.py:780`):**
  *"for Kimi K3 a layer's experts are ~55GB expanded, of which a token touches
  ~1GB"* — the ratio is exactly why per-expert streaming pays off, and why a
  cache matters more than a prefetcher.

### 1.5 What does *not* transfer

| AirLLM (dense) | Your MoE case |
|---|---|
| Unit = whole layer (~341 MiB) | Unit = one expert (2.53 MiB) |
| 1-slot lookahead is enough | 8 reads/layer, 384/token, needs a pool |
| Reuse via page cache | Needs explicit budgeted cache — 16.4 GiB file on a 12 GB device means the page cache will thrash |
| No eviction policy | Eviction policy *is* the performance |
| Prefetch next layer | Cannot see next layer's routing (needs layer L output) |
| `torch.load` / safe_open | `pread` on a byte range inside a GGUF |

Also note: their per-expert path is gated on `SafetensorModelPersister`, and is
explicitly disabled when `compression is not None` (`airllm_base.py:803`). A
GGUF has neither concept — you are writing this from scratch, not porting it.

---

## 2. The real geometry of your model

Parsed from the GGUF header of
`Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf`:

```
version 3, 579 tensors, 44 metadata keys, alignment 32
architecture qwen3moe, block_count 48, embedding_length 2048
expert_count 128, expert_used_count 8, expert_ff 768
data_start 5,973,152   (≡ 1184 mod 4096)
file size 17,665,334,432 B = 16.447 GiB
```

Per-layer expert tensors are **3-D, contiguous along the expert axis**:

| tensor | dims (ne0,ne1,ne2) | total | per expert |
|---|---|---|---|
| `blk.N.ffn_gate_exps.weight` | 2048, 768, 128 | 113,246,208 | **884,736** |
| `blk.N.ffn_up_exps.weight` | 2048, 768, 128 | 113,246,208 | **884,736** |
| `blk.N.ffn_down_exps.weight` | 768, 2048, 128 | 113,246,208 | **884,736** |

Since ggml stores ne0 fastest, expert `e` occupies
`[base + e*884736, base + (e+1)*884736)`. No strided/gather reads needed —
one `pread` per slice.

**884,736 = 216 × 4096 exactly.** The *length* is always block-aligned.
The *base* is not: layer 0 gate starts at 562,299,040 (`≡ 160 mod 4096`),
layer 11 gate at 4,512,476,320 (`≡ 3232 mod 4096`). Hence any `O_DIRECT`
path needs an alignment window + bounce buffer. Implemented in
`expert_loader.c:expert_io_read()`.

Budget split:

```
experts      16,634,609,664 B = 15.492 GiB  (94.2%)
non-experts   1,024,751,616 B =  0.954 GiB  ( 5.8%)
  attention (48 layers)          518.5 MiB
  routers   (48 × f32 2048×128)   48.0 MiB
  token_embd + output            410.3 MiB
per layer: 330.50 MiB experts of 341.30 MiB total
```

Two consequences:

- **94% of the model is streamable.** The resident set (attention + norms +
  routers + embeddings + lm_head) is only **0.954 GiB**. Keep it flat in RAM.
- `ffn_gate_inp` (the router) is stored **f32**, 1 MiB per layer, 48 MiB
  total. That is 0.3% of the file and it is on the critical path for every
  token — keep all 48 resident. Do not quantise it without measuring.

There is **no shared expert** in this model
(`expert_shared_feed_forward_length = 0`), which simplifies the layer:
top-8 routed experts only.

### 2.1 This is a dynamic quant — expert size is not uniform

The file is Unsloth **UD**-Q4_K_XL: quantization varies per tensor. 131 of the
144 expert tensors are q4_K, but **13 are q5_K**:

```
blk.1,2,3,21,28,39,40,41,43,44,45,46,47  .ffn_down_exps.weight   q5_K
  -> 138,412,032 B / 128 experts = 1,081,344 B per expert slice
  -> layer payload 884,736 + 884,736 + 1,081,344 = 2,850,816 B (2.72 MiB)
```

Everything else is 2,654,208 B (2.53 MiB). Consequences:

- `moe_plan_t.max_expert_bytes` must size any fixed scratch buffer. Sizing
  prefill buffers from layer 0 only overflows on layer 1 — this is exactly the
  bug the ASan run caught during Phase 2.
- Cache slot count: 2 GiB / 2.71 MiB average ≈ 790 slots, not 810.
- `moe_plan_print()` now reports the largest layer payload explicitly.

---

## 3. Performance model (do this arithmetic before writing code)

```
I/O per token, zero cache:
  48 layers × 8 experts × 2,654,208 B = 1,019,215,872 B ≈ 1.02 GB
  @ 580 MB/s  → 1.76 s/token → 0.57 tok/s
```

With an expert cache of hit rate *h*:

```
read/token = 1.02 GB × (1 − h)
```

| cache | slots (2.53 MiB) | slots/layer | measured hit* | read/token | 580 MB/s | measured† |
|---|---|---|---|---|---|---|
| 256 MiB | 97 | 2.0 | 0.3% | 1.02 GB | 1.76 s | 0.52 s |
| 1 GiB | 395 | 8.2 | 24.7% | 571 MiB | 0.99 s | 0.38 s |
| 2 GiB | ~790 | 16.5 | ~40% (est.) | ~610 MiB | 1.05 s | — |
| 4 GiB | ~1580 | 33 | ~55% (est.) | ~460 MiB | 0.79 s | — |

\* synthetic Zipf(1.2) router trace, not your model's actual routing.
† this device, `O_DIRECT`, 4 workers, checksum only, **matmul excluded**.

The 256 MiB row is the important one: it is *worse than useless* because 2
slots per layer cannot hold a hot set, so every access misses and you paid for
the cache twice. **Do not ship below ~1 GiB.** The minimum that can hold one
hot expert per layer is `48 × 2.53 MiB ≈ 1.02 GiB`.

Thread scaling measured (1 GiB, `O_DIRECT`, same trace):

| workers | ms/token | speedup |
|---|---|---|
| 1 | 749.9 | 1.00× |
| 2 | 475.9 | 1.58× |
| 4 | 380.9 | 1.97× |
| 8 | 435.1 | 1.72× (contention) |

**Use 2–4 workers.** The UFS queue saturates and additional threads only add
lock traffic. This is worth knowing before you copy the "one thread per expert"
pattern from desktop projects.

### 3.1 The measurement trap

Buffered `pread` on a 16.4 GiB file with 12 GB of RAM means the kernel caches
the hot subset, and your benchmark measures **RAM bandwidth, not UFS**. In our
runs buffered was *slower* than `O_DIRECT` (416 vs 381 ms/token) because of
double-caching and `malloc_trim`-style pressure, but on a different cache state
it will look faster and lie to you. Benchmark with `--direct` only.

### 3.2 Prefill — the part most projects get wrong

For a batch of *B* tokens, the expected number of distinct experts touched in
one layer is `128 × (1 − (1 − 8/128)^B)`:

| B | distinct experts | bytes/layer | bytes for 48 layers |
|---|---|---|---|
| 1 | 8 | 21 MB | 1.0 GB |
| 8 | 52 | 137 MB | 6.6 GB |
| 32 | 112 | 296 MB | 14.2 GB |
| 128 | 128 | 339 MB | 16.3 GB |

A naive token-at-a-time prefill of a 1000-token prompt therefore reads
**~1 TB**. The fix is not a bigger cache, it is a different traversal:

```
PREFILL (layer sweep):
  for L in 0..47:
      for e in 0..127:            # read every expert exactly once
          load expert (L,e)       # 2.53 MiB
          for each token routed to e: gemm_into_accumulator
      discard all 128
  total = 15.5 GiB per prompt, independent of prompt length
       ≈ 27 s @ 580 MB/s (I/O only; overlaps with compute)
```

This is the single most important architectural decision in the project, and
it is invisible if you only ever test with one-token prompts.

---

## 4. Architecture for the C implementation

```
                    ┌──────────────────────────────┐
  token ──► router ─┤ top-8 expert ids, per layer   │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ moe_layer_fetch()            │
                    │  probe cache (pin on hit)    │
                    │  enqueue misses              │
                    └──────┬───────────────┬───────┘
                           │               │
                 ┌─────────▼────┐   ┌──────▼────────┐
                 │ worker 0..N  │   │ LRU (byte     │
                 │ pread slices │──►│ budget, pins) │
                 └──────────────┘   └───────────────┘
                           │
                    compute (gate/up/down GEMM)
                           │
                    unpin all 8
```

Design rules that came out of the measurements:

1. **Key the cache on `(layer, expert)`.** Expert 7 of layer 3 and expert 7 of
   layer 4 are unrelated weights. `key = (uint64)layer << 32 | expert`.
2. **Pin across the GEMM.** A prefetcher must never evict a slot that is about
   to be read by the compute kernel. Every fetch returns pinned entries;
   `moe_release()` unpins.
3. **Reserve bytes at alloc, not at commit.** Eight workers allocating in
   parallel otherwise all observe the same `bytes` counter and overshoot the
   budget. (This was a real bug in the first draft — see `lru_cache.c`
   `alloc_locked()`.)
4. **Fill outside the lock.** Allocate pinned → `pread` → commit. The mutex is
   held for map/list surgery only.
5. **One expert = one contiguous 2.53 MiB buffer** laid out `[gate|up|down]`,
   so the compute path gets three pointers with no per-tensor allocation.
6. **Don't trust LRU.** Measured 24.7% at 8 slots/layer where the Zipf top-8
   mass is ~55%. Every tail miss is admitted and evicts a hot entry. The fix
   is an admission filter (TinyLFU) or per-layer quotas — see §8.

### 4.1 The structs

```c
/* One expert's weights, contiguous. */
typedef struct {
    uint8_t *data;          /* 4096-aligned, 2,654,208 B */
    uint32_t gate_bytes;    /* 884,736 */
    uint32_t up_bytes;      /* 884,736 */
    uint32_t down_bytes;    /* 884,736 */
    uint8_t *gate;          /* data                    */
    uint8_t *up;            /* data + 884,736          */
    uint8_t *down;          /* data + 1,769,472        */
    uint32_t expert_id;
} ExpertBuffer;
```

The batch view the compute kernel consumes (8 × 2.53 MiB = **21.2 MB**, which
is where your original "~21MB" number came from):

```c
typedef struct {
    ExpertBuffer e[8];
    uint32_t     ids[8];
    int          n;
} ExpertBatch;
```

The cache entry is intrusive (hash table + LRU list share the allocation):

```c
struct lru_entry {
    uint64_t     key;       /* (layer << 32) | expert */
    size_t       size;      /* 2,654,208 */
    uint32_t     pins;      /* >0 forbids eviction */
    int          linked;
    uint8_t     *data;      /* 4096-aligned payload */
    lru_entry_t *prev, *next;
};
```

`lru_cache_t` = open-addressed table (power-of-two, linear probing, tombstoned
deletes) + head/tail list + byte budget + stats. All operations O(1).

### 4.2 `pread` vs `mmap` vs `O_DIRECT` — and the combination that doesn't exist

`mmap` and `O_DIRECT` are **mutually exclusive**. `mmap()` on an fd opened
with `O_DIRECT` is not supported on Linux (the mapping has no page-cache
backing to fault in). Anyone telling you to "mmap with O_DIRECT" is
describing two alternatives.

| | buffered `pread` | `pread` + `O_DIRECT` | `mmap` the whole 16.4 GiB |
|---|---|---|---|
| copy cost | 1 (kernel→user) | 1 (+1 bounce if unaligned) | 0 (page faults) |
| page-cache double use | **yes** | no | yes (it *is* the cache) |
| eviction control | yours | yours | kernel's |
| latency shape | steady | steady | page-fault stalls |
| alignment work | none | window + bounce | none |
| fits 12 GB device | cache thrashes page cache | **best** | kernel keeps ~4 GB, rest refaults |
| our result | 416 ms/token | **381 ms/token** | not built |

Use `O_DIRECT` for the cache fill path. Because every GGUF expert base is
misaligned mod 4096, `expert_io_read()` rounds the window outwards, reads into
a 4096-aligned bounce buffer, and `memcpy`s the interior — an extra 2.65 MB
copy (~0.25 ms at 10 GB/s) that is noise against a 4.6 ms read.

If you must use buffered reads, call `posix_fadvise(fd, off, len,
POSIX_FADV_DONTNEED)` after each fill (`expert_io_dontneed()`), otherwise the
kernel and your LRU both hold a copy and you have halved your effective RAM.

### 4.3 Alignment, from the actual file

```
data_start = 5,973,152        ≡ 1184 (mod 4096)
blk.0.ffn_gate_exps  base     ≡  160 (mod 4096)
blk.11.ffn_gate_exps base     ≡ 3232 (mod 4096)
slice length 884,736          ≡    0 (mod 4096)   ← always aligned
```

So: length is free, offset is not. The loader computes

```c
start = offset & ~(align - 1);
end   = (offset + size + align - 1) & ~(align - 1);
pread(fd, bounce, end - start, start);
memcpy(dst, bounce + (offset - start), size);
```

---

## 5. The code (in `src/moe_stream/`)

| file | purpose | lines |
|---|---|---|
| `expert_loader.{h,c}` | `pread` loop, `O_DIRECT` windowed read, fadvise | 249 |
| `lru_cache.{h,c}` | thread-safe byte-budgeted LRU, pinning, stats | 518 |
| `gguf_index.{h,c}` | GGUF v3 parser, tensor dir, expert span resolver | 452 |
| `moe_stream.{h,c}` | worker pool, per-layer fetch, dedupe, pinning | 401 |
| `test_moe_stream.c` | verification + benchmark harness | 320 |
| `Makefile` | `make && make test` | — |

### 5.1 `load_expert_to_buffer` — the function you asked for

```c
/* Buffered, thread-safe: pread carries its own offset, so N workers can share
 * one FILE* with no locking and no seek races. */
int load_expert_to_buffer(FILE *f, uint64_t offset, size_t size, uint8_t *buffer)
{
    if (!f || !buffer) return -EINVAL;
    if (size == 0)     return 0;

    int fd = fileno(f);          /* keep the FILE* API, bypass libc buffering */
    if (fd < 0) return -EINVAL;

    size_t got = 0;
    int rc = read_loop(fd, offset, buffer, size, &got);
    return rc;                    /* 0 or -errno */
}
```

`read_loop` handles `EINTR`, partial reads and EOF, and rejects offsets above
`INT64_MAX` before casting to `off_t`.

### 5.2 The `O_DIRECT` path

```c
int expert_io_read(expert_io_t *io, uint64_t offset, size_t size, void *dst)
{
    if (!io->odirect)
        return read_loop(io->fd, offset, dst, size, NULL);

    const size_t   a     = io->align;                        /* 4096 */
    const uint64_t start = offset & ~(uint64_t)(a - 1);
    uint64_t       end   = (offset + size + a - 1) & ~(uint64_t)(a - 1);
    if (end > io->file_size)
        end = (io->file_size + a - 1) & ~(uint64_t)(a - 1);

    size_t span = (size_t)(end - start);
    if (ensure_bounce(io, span) != 0) return -ENOMEM;

    size_t got = 0;
    int rc = read_loop(io->fd, start, io->bounce, span, &got);
    if (rc != 0 && got < (size_t)(offset - start) + size) return rc;

    memcpy(dst, io->bounce + (size_t)(offset - start), size);
    return 0;
}
```

Opening falls back cleanly when the filesystem rejects `O_DIRECT`
(exFAT/sdcardfs), which is the normal case for a model on external storage.

### 5.3 LRU core

```c
lru_entry_t *lru_cache_alloc(lru_cache_t *c, uint64_t key, size_t size)
{
    if (size > c->capacity_bytes) { c->st.rejected++; return NULL; }

    while (c->bytes + size > c->capacity_bytes)
        if (!evict_one_locked(c)) { c->st.rejected++; return NULL; }
        /* evict_one walks from the cold end, skipping pinned entries */

    ... posix_memalign(&p, 4096, size) ...
    e->pins = 1;
    c->bytes += size;            /* RESERVE now — see §4 rule 3 */
    return e;
}
```

Commit publishes to the hash table; abort returns the reservation. The public
API takes the mutex; the internal `_locked` variants do not, so `put()` can
compose get/alloc/commit atomically.

### 5.4 Per-layer fetch

```c
int moe_layer_fetch(moe_ctx_t *ctx, int layer, const uint32_t *experts, int n,
                    moe_expert_t *out)
{
    /* 1. probe cache, pin hits, record misses (dedupe repeated ids) */
    /* 2. enqueue one job per miss: 3 offset/length pairs, one allocation */
    /* 3. sleep on condvar until inflight == 0 */
    /* 4. verify all slots resolved, fill gate/up/down pointers */
}
```

Workers do `alloc → pread×3 → commit`, all outside the ctx lock; only the
queue and the completion counter are guarded. The queue is fixed-size (64),
so submission never allocates.

### 5.5 GGUF index

`gguf_index_load()` parses the header, skips metadata by type (strings,
arrays, fixed-width), stores the tensor directory, and computes
`data_start = align_up(header_end, alignment)`. `moe_plan_build()` resolves
the three expert tensors for all 48 layers and derives the stride:

```c
stride = ggml_tensor_nbytes(type, ne0*ne1*ne2) / ne2;   /* 884,736 */
```

`ggml_tensor_nbytes()` implements the block geometry table for F32/F16/BF16,
Q4_0…Q8_K, IQ1–IQ4, TQ1/TQ2, MXFP4 — and returns 0 (hard error) for anything
it doesn't know, rather than guessing a stride and silently corrupting every
expert read.

---

## 6. Verification results (this device, this file)

```
$ make && ./build/moe_stream_test ../../Qwen3-...gguf --cache-mb 1024 --threads 4 --direct

[index] arch=qwen3moe layers=48 embd=2048 ff_exp=768 experts=128 top8
[index] tensors=579 data_start=5973152 (mod 4096 = 1184)
moe plan: 48 layers x 128 experts (top-8), embd=2048, ff_exp=768
  expert payload: gate=884736 up=884736 down=884736 total=2654208 B
  all experts on disk: 15.19 GiB
[index] geometry OK: 884736 B/slice, 2654208 B/expert (2.53 MiB)

[read] blk.11 expert 7 gate: 884736 B, direct==buffered, 2.67 ms
[read] gate checksum = f8b2b70d1da92184

[fetch] layer 11, 5 ids (1 dup): checksum 391b7524d399ee71
[fetch] refetch identical, cache hit path OK
  [evict] 16MiB cache, 12 expert touches: 6 evictions, 6 entries resident

[sim] 4 tokens in 1523.6 ms (380.90 ms/token, 2.63 tok/s upper bound)
cache: 395 entries, 1022.16 MiB / 1024.00 MiB, hit 24.7% (290/1174)
io: 2283.94 MiB read, 0 errors

all checks passed
```

What is actually proven:

- GGUF parse against a real 16.4 GiB file, including 5.9 MB of metadata
  (151,936 tokens + 151,387 merges) skipped without reading tensor data.
- Expert stride 884,736 and payload 2,654,208 derived from the file, not
  hard-coded.
- `O_DIRECT` and buffered reads return byte-identical data (memcmp) on a
  misaligned offset.
- Cache hit returns identical bytes; duplicate ids in a batch share one entry.
- Eviction fires under pressure and the byte budget is respected across 8
  concurrent workers.
- 1,536 expert fetches through the pool with 0 I/O errors.
- Compiles with `gcc -std=c11 -Wall -Wextra -Wshadow -Werror`, zero warnings.
  (Termux's `gcc` is a clang 21 wrapper; it is also clean under real GCC
  semantics — no GNU extensions beyond `O_DIRECT`/`posix_fadvise`, both
  POSIX.)

Caveat: the router trace is synthetic Zipf(1.2). The hit-rate numbers are
indicative. **Capture a real routing trace before sizing the cache.**

---

## 7. Memory budget on a 12 GB phone

| item | size | note |
|---|---|---|
| non-expert weights (attention, norms, routers, embeddings, lm_head) | **0.95 GiB** | resident, never evicted |
| expert cache | **2.0 GiB** | ~790 slots, 16/layer |
| KV cache @ 4096 ctx | **0.375 GiB** | 48 × 4 kv heads × 128 × 2 × 2 B = 96 KiB/token |
| KV cache @ 8192 ctx | 0.75 GiB | |
| activations + scratch | ~0.2 GiB | |
| **total @ 4k** | **~3.5 GiB** | comfortable for a foreground Termux process |

KV cache is `48 × 4 × 128 × 2 (K,V) × 2 B = 98,304 B/token`. At 32k context
that is 3 GiB on its own — it will dominate the budget before the expert cache
does. Plan context length accordingly, or quantise the KV cache to q8_0.

Do **not** try 4 GiB of expert cache and 32k context simultaneously; Android
will kill the process around 5–6 GiB of RSS for a non-foreground app.

---

## 8. What to fix next, in priority order

Done in Phase 2 (see §11): prefill layer sweep, sorted dispatch, intra-layer
pipelining, end-to-end benchmark.

1. **Prefill reader pool.** The sweep currently uses one reader thread and
   measures ~960 MB/s, while the 4-worker decode path reaches ~1.5 GB/s on the
   same file. Reading 2-4 experts ahead into a ring would cut prefill from
   13.2 s to ~8.5 s. This is the single biggest remaining win.
2. **Admission policy.** 42.4% hit at 2 GiB with plain LRU. The Zipf top-8
   mass is ~55%, and every tail miss evicts a hot entry. TinyLFU or a per-layer
   quota should recover most of that gap at the same footprint.
3. **Real routing trace.** Everything about hit rate is still synthetic.
   Capture `(layer, expert)` for 200 tokens of real code and replay it.
4. **Real GEMM.** `dummy_gemm` touches 2 KiB per slice so the I/O cannot be
   optimised away, but it adds no compute. Once ggml's `mul_mat` runs on these
   buffers, decode will be compute-bound at 2-4 workers and the I/O will
   partially hide — the numbers here are an I/O ceiling, not a prediction.
5. **Consider Q3_K_XL experts.** 94% of the file is expert data; dropping
   4.5 → 3.4 bpw cuts the streaming volume ~25%.
## 9. Android / Termux gotchas

- **`io_uring` is not available to Android apps** (SELinux blocks it on
  production kernels). `pread` + a small thread pool is the correct design;
  don't spend a day on io_uring.
- **`O_DIRECT` works on f2fs** (observed here) but **fails on sdcardfs/exFAT**.
  Always keep the buffered fallback — `expert_io_open()` retries without
  `O_DIRECT` and reports which path is active.
- **Page cache is not free.** 12 GB RAM and a 16.4 GiB file means buffered
  reads will evict each other. Either use `O_DIRECT` or `fadvise(DONTNEED)`.
- **`off_t` is 64-bit on aarch64**, but define `_FILE_OFFSET_BITS=64` anyway if
  you ever build 32-bit.
- **Termux `gcc` is clang.** Fine for C11; just don't rely on GCC-specific
  builtins.
- **Don't mmap the whole file.** It looks elegant and then the kernel makes
  eviction decisions you cannot observe or control, on a device with no swap.

---

## 10. Build and run

```bash
cd src/moe_stream
make                      # gcc -std=c11 -O2 -Wall -Wextra -Wshadow -Werror
./build/moe_stream_test ../../Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
    --cache-mb 2048 --threads 4 --direct
```

The harness exits non-zero on any failed check, so it is usable as a smoke
test in CI or before a long run.

---

## 11. Phase 2 — prefill sweep and pipelined decode (implemented)

### 11.1 Routing map layout

```c
/* layer-major, row-major, uint32_t */
routing_map[((layer * num_tokens) + token) * top_k + k] = expert_id;
```

- `top_k` = `plan.n_expert_used` (8 here).
- Size = `n_layers * num_tokens * top_k * 4` bytes (153 KiB for 100 tokens).
- Values must be `< n_expert`; the sweep returns `-ERANGE` otherwise.

**Important:** in a real forward pass layer L's routing does not exist until
after attention(L). Do not precompute the whole map with a fake router. Call
the per-layer entry point from inside the layer loop:

```c
moe_prefill_layer(ctx, L, num_tokens, routing_for_layer_L);  /* [tokens][8] */
```

`moe_prefill_layer_sweep(ctx, num_tokens, map)` is the batch/benchmark form
that loops the 48 layers for you.

Internally each layer builds an **inverted index** (expert → list of token
ids) with a two-pass counting sort, O(tokens·top_k + 128), so the compute step
walks contiguous token lists instead of rescanning the routing map 128 times.

### 11.2 Prefill pipeline

```
for each layer:
    build inverted index
    submit read(expert e0)                      -> reader thread
    for e in routed experts:
        wait(e)                                 -> data resident
        submit read(e_next)                     -> overlaps with compute(e)
        gemm(expert e, tokens routed to e)
        discard e
```

Two 4096-aligned buffers of `max_expert_bytes` (2.72 MiB each, 5.4 MiB total)
are reused across all 48 layers. The LRU is bypassed entirely — prefill
touches each expert at most once, so caching it would only evict the decode
working set.

`dummy_gemm` touches 2 KiB of each slice (FNV) so the read cannot be
dead-code-eliminated, and does no real math. Swap it with:

```c
moe_set_gemm(ctx, my_gemm, my_user);
```

### 11.3 Sorted dispatch + completion-order delivery

`moe_layer_fetch()` now sorts the miss list by physical offset before
enqueueing (insertion sort, n ≤ 16). On this GGUF the expert base is monotonic
in expert id, so the sort collapses to an id sort — but computing it from the
offset keeps the code correct if a future layout interleaves.

Pipelining needs a non-blocking interface, so there are now two entry points
sharing one engine:

```c
/* blocking, out[i] <=> experts[i], duplicates share a pinned entry */
int moe_layer_fetch(ctx, layer, ids, n, out);

/* streaming: cb fires as each expert lands, in COMPLETION order */
int moe_layer_fetch_stream(ctx, layer, ids, n, cb, user);
```

The callback receives a pinned `moe_expert_t` whose `slot` field identifies
which entry of the caller's request finished, so the GEMM for expert 0 can run
while experts 1..7 are still on the bus. Entries are unpinned when the callback
returns; return non-zero to abort (`-ECANCELED`).

### 11.4 Benchmark (100-token prompt + 10 generated tokens)

Command:

```bash
./build/moe_stream_test ../../Qwen3-...gguf --cache-mb 2048 --threads 4 --direct
```

Measured on this device (UFS, `O_DIRECT`, one full run):

```
[prefill] 100-token prompt, all layers, unrouted experts skipped
[prefill] 11.67 s total, 11.83 GiB read, 1089 MB/s, 4690 experts
[prefill] 243.1 ms/layer, 401.9 experts/s, 2.49 ms/expert

[decode] 10 tokens, top-8 per layer, cache 2048 MiB
[decode] 4.11 s total, 411.0 ms/token, 2.433 tok/s
[decode] 436.73 MiB read/token, 4.26 GiB total

cache: 792 entries, 2047.88 MiB / 2048.00 MiB, hit 42.4%
       (1250 hit / 1701 miss), 909 evictions, 0 rejected
prefill   : 11.83 GiB read, 4690 experts, 100 prompt tokens
```

| phase | time | bytes | rate |
|---|---|---|---|
| prefill, 100 tokens | 11.7-13.2 s | 11.83 GiB | 960-1090 MB/s, 7.6-8.6 prompt tok/s |
| decode, 10 tokens @2 GiB | 3.5-4.1 s | 4.26 GiB | 352-411 ms/token, 2.43-2.84 tok/s |
| decode, 10 tokens @1 GiB | 4.0-4.1 s | 5.36 GiB | 402-412 ms/token, ~2.45 tok/s |

Run-to-run spread is ~15% on this phone (UFS thermal), so treat these as
ranges, not points. Both phases are I/O-bound under `dummy_gemm`.

Three things this measurement exposes:

1. **The sweep reads 11.83 GiB, not 15.49 GiB.** A 100-token Zipf batch routes
   to 4,690 of 6,144 experts; the untouched tail (24%) is skipped rather than
   read and multiplied by zero tokens. For a 1-token prompt the saving is 16x.
2. **Prefill is single-stream I/O bound at ~960 MB/s.** Decode's 4-worker path
   reaches ~1.5 GB/s. A reader pool is the next optimisation (§8.1).
3. **Plain LRU reaches 42.4% at 2 GiB** — better than the 24.7% measured at
   1 GiB, but still below the ~55% of traffic the hot set carries. The gap is
   admission policy, not capacity.

### 11.5 Two bugs this phase found (worth knowing before you port it)

- **Shared O_DIRECT bounce buffer.** `expert_io_t` originally owned one bounce
  buffer; four workers called `expert_io_read()` concurrently, so one thread
  `free()`d it while another was `memcpy`ing out of it. Now thread-local via
  `pthread_key_t` with a destructor (`expert_loader.c`).
- **Fixed-size prefill buffers sized from layer 0.** Invalid on a dynamic
  quant: 13 layers have q5_K down projections 196,608 bytes larger. Fixed by
  sizing from `plan.max_expert_bytes` (§2.1).

Both were caught by ASan and by a pre-read bounds guard, not by the functional
tests — the corruption was silent until it wasn't.

## 12. Phase 3 — scalar CPU forward pass (implemented)

New files: `dequant.{h,c}`, `qwen_model.{h,c}`, `tokenizer.{h,c}`,
`test_generate.c`, plus a `moe_generate` target in the Makefile.

### 12.1 Dequantization is the price of admission

Every weight in this GGUF is quantised. There is no path to "just do a matmul"
without decoding q4_K/q5_K/q6_K blocks first. Implemented from the ggml block
layouts:

| type | block | bytes | used by |
|---|---|---|---|
| q4_K | 256 | 144 | most expert + attention tensors |
| q5_K | 256 | 176 | 13 `ffn_down_exps` tensors (see §2.1) |
| q6_K | 256 | 210 | `output.weight`, some `attn_output` |
| q8_0 | 32 | 34 | (not present, supported for safety) |
| f32/f16 | 1 | 4/2 | norms, router |

The q4_K/q5_K 6-bit scale/min unpacking is the subtle part:

```c
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >> 4)   | ((q[j]   >> 6) << 4);
    }
}
```

Weights are never materialised as f32. `vec_dot(type, row, x, n)` dequantises
one 256-element block into a stack scratch and accumulates — peak extra memory
is 1 KiB per matmul.

### 12.2 The primitives

```c
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* split-half RoPE (HF rotate_half, ggml NEOX convention) */
static void rope_apply(float *v, int n, const float *cs, const float *sn)
{
    const int half = n / 2;
    for (int i = 0; i < half; i++) {
        const float a = v[i], b = v[i + half];
        v[i]        = a * cs[i] - b * sn[i];
        v[i + half] = b * cs[i] + a * sn[i];
    }
}

static void softmax(float *x, int n) { /* max-subtract, expf, normalise */ }
```

RoPE frequencies are precomputed once for all `max_seq` positions
(`cos/sin[position][64]`), so the forward path has no `powf` calls.

### 12.3 Attention (GQA + KV cache)

Layout is `k_cache[layer][pos][n_head_kv][head_dim]`, f32. With 4 KV heads
against 32 query heads, head `h` reads KV head `h / 8`.

```c
for (int h = 0; h < nh; h++) {
    const int kv = h / grp;
    const float *qh = m->q + h * hd;
    const float *ks = kbase + (size_t)kv * hd;
    const float *vs = vbase + (size_t)kv * hd;

    for (int t = 0; t <= pos; t++)
        m->scores[t] = dot(qh, ks + t * step, hd) * scale;

    softmax(m->scores, pos + 1);

    for (int t = 0; t <= pos; t++)
        for (int d = 0; d < hd; d++)
            oh[d] += m->scores[t] * vs[t * step + d];
}
```

`q_norm`/`k_norm` are applied per head *before* RoPE, matching HF Qwen3:

```c
for (int h = 0; h < nh; h++) {
    rmsnorm_inplace(m->q + h * hd, qkw, hd, m->eps);
    rope_apply(m->q + h * hd, hd, cs, sn);
}
```

### 12.4 MoE block

Router logits are f32 (`ffn_gate_inp`, 1 MiB/layer, kept resident), softmaxed
over 128, top-8 selected, then **renormalised** (`norm_topk_prob = true` for
Qwen3):

```c
matmul_tensor(&L->ffn_gate_inp, m->xb, m->router);   /* [2048] -> [128] */
softmax(m->router, 128);
topk8(m->router, idx, w);                            /* descending insertion */
for (k) w[k] /= sum(w);                              /* renormalise */

moe_layer_fetch(ctx, layer, ids, 8, ex);             /* the Phase 2 engine */

for (int k = 0; k < 8; k++) {
    matmul_raw(ml->gate.type, ex[k].gate, 2048, 768, m->xb, m->gate);
    matmul_raw(ml->up.type,   ex[k].up,   2048, 768, m->xb, m->up);
    for (i) m->act[i] = silu(m->gate[i]) * m->up[i]; /* SwiGLU */
    matmul_raw(ml->down.type, ex[k].down,  768, 2048, m->act, m->proj);
    for (i) m->expert_out[i] += w[k] * m->proj[i];
}
moe_release(ctx, ex, 8);
```

Note the per-expert `type`: gate/up are q4_K everywhere, but `down` is q5_K on
the 13 layers from §2.1. The matmul reads the type from `moe_tensor_t`, so no
special-casing is needed.

### 12.5 Tokenizer

`tokenizer.c` parses `tokenizer.ggml.tokens` (151,936 strings, byte-level GPT-2
encoding). Decoding is exact: each codepoint maps back to its source byte
(Ġ → 0x20). Encoding is **greedy longest-match**, not true BPE — it produces a
valid segmentation (and matches the real tokenizer on ordinary words) but is
not bit-identical. Use `--ids "750,75698,1445,1648"` when exactness matters.

### 12.6 Results

```
$ ./moe_generate ...gguf --cache-mb 512 --threads 4 --ctx 64 --ngen 6 \
      --direct "The capital of France is"

prompt: "The capital of France is"
tokens (5): 785 [The] 6722 [ capital] 315 [ of] 9625 [ France] 374 [ is]

generated:
---
 Paris. The capital of Belgium
---
generated 6 tokens in 18.4 s (0.33 tok/s, 3064 ms/token)
```

```
$ ./moe_generate ... "def fibonacci(n):"

generated:
---
    if n <=
---
generated 6 tokens in 18.7 s (0.32 tok/s, 3114 ms/token)
```

Two independent prompts producing correct continuations is strong evidence
that every stage is right: a mistake in RoPE layout, GQA head mapping,
`norm_topk_prob`, or the SwiGLU operand order produces fluent-looking noise or
immediate repetition, not ` Paris. The capital of Belgium`.

### 12.7 Performance, and where it goes

~3.1 s/token, single-threaded scalar. Per token the model touches:

| stage | MACs | weight bytes dequantised |
|---|---|---|
| attention (48 layers) | 0.91 G | ~518 MiB |
| MoE (48 × 8 experts) | 1.81 G | ~1.03 GiB |
| lm_head | 0.31 G | 255 MiB (q6_K) |
| **total** | **~3.0 G** | **~1.8 GiB** |

~2 GFLOP/s effective. The output projection alone is 255 MiB of dequant work
per token — 14% of the total, for one matmul. That is the argument for a
Vulkan path that keeps weights quantised in VRAM and dequantises in-shader.

### 12.8 Deliberate limitations

- Greedy sampling only (no temperature/top-p) — determinism for testing.
- Single-token prefill (the Phase 2 layer sweep is not wired into the model
  yet; it needs layer-by-layer routing, which requires attention to run first).
- No SIMD backends, no threading inside matmuls: the 4 workers only do I/O.
- KV cache is f32; q8_0 would halve it.
- No batched decode.

## Appendix A — AirLLM call graph (for reference)

```
generate()
 └─ transformers forward
     ├─ embed hook ──────── load_layer_to_cpu()            (dense)
     ├─ per decoder layer
     │   ├─ _pre_hook()  ── _load_streamed_layer()
     │   │                   ├─ load_layer_to_cpu()        whole layer
     │   │                   └─ or load_layer_subset()     non-expert keys
     │   │                 move_layer_to_device()
     │   │                 submit(prefetch next layer)     ← depth 1
     │   ├─ attention / MoE forward
     │   │   └─ per expert: _expert_pre_hook()
     │   │        └─ load_layer_subset(keys)               ← synchronous, no prefetch
     │   │          _expert_post_hook() → to('meta')
     │   └─ _post_hook() → to('meta') + clean_memory()
     └─ norm / lm_head hook
```

## Appendix B — Why `safe_open().get_tensor()` is not `mmap` of the weights

`safetensors` mmaps the shard file to *locate* tensors, then copies each
requested tensor into a new tensor object. The mapping is an implementation
detail of the reader; the returned weights are ordinary heap allocations. This
is why AirLLM's RSS is "one layer", not "the whole checkpoint", and why it
gets no zero-copy benefit. Your `pread`-into-a-pinned-slot design is strictly
closer to the metal — there is nothing to gain by imitating their reader.
