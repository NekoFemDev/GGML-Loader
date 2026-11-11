# GGML-Loader × MoE engine — Go integration plan

Status: **implemented and tested** in this tree.

```
Go (gobridge/moe.go)          scheduling, prompt, output
      │  flat C ABI (moe_bridge.h)
      ▼
moe_bridge.c                  session = model + tokenizer + KV cache
      │
      ▼
qwen_model.c                  forward pass, routing, KV cache, layer sweep
      │                 │
      ▼                 ▼
moe_stream.c        compute_backend.c
LRU + O_DIRECT      scalar C  ──►  Vulkan (stub, drop-in)
      │
      ▼
dequant.c             q4_K / q5_K / q6_K block decoders
```

The rule that keeps this maintainable: **C owns state and control flow, Go
owns orchestration, the backend owns arithmetic only.**

---

## 1. CGO integration

### 1.1 Why a flat C ABI and not `//export` callbacks everywhere

cgo has two strict rules that shape the design:

1. C may not retain a Go pointer after the call returns.
2. A Go file that uses `//export` may not put function **definitions** in its
   cgo preamble.

So the bridge exposes only:

- opaque handles (`moe_session_t *`),
- POD structs with fixed layout (`moe_session_opts_t`),
- caller-owned output arrays,
- one weak callback symbol for streaming.

No C function pointer is ever created on the Go side, which removes the most
common cgo crash class.

### 1.2 `moe_bridge.h` (excerpt)

```c
typedef struct moe_session moe_session_t;

typedef struct {
    const char *gguf_path;
    size_t      cache_bytes;
    int         threads;
    int         max_seq;
    int         odirect;
} moe_session_opts_t;

typedef int (*moe_token_cb)(void *user, uint32_t token_id,
                            const char *utf8, int len);

moe_session_t *moe_session_create(const moe_session_opts_t *opts);
void           moe_session_destroy(moe_session_t *s);

int moe_session_encode(moe_session_t *s, const char *text, uint32_t *ids, int max_ids);
int moe_session_decode(moe_session_t *s, uint32_t id, char *buf, int cap);

int moe_session_prefill(moe_session_t *s, const uint32_t *tokens, int n_tokens);
int moe_session_forward(moe_session_t *s, uint32_t token);

int moe_session_generate(moe_session_t *s, int max_new, uint32_t *out_ids,
                         int out_cap, moe_token_cb cb, void *user);
int moe_session_generate_go(moe_session_t *s, int max_new, uint32_t *out_ids,
                            int out_cap, uintptr_t go_handle);
```

`moe_session_generate_go` exists because Go cannot pass a function pointer to C
directly. It uses a **weak symbol**:

```c
/* implemented by //export in Go; NULL when linked from plain C */
extern int moe_go_token_callback(void *user, uint32_t id, char *utf8, int len)
    __attribute__((weak));

static int go_thunk(void *user, uint32_t id, const char *t, int len)
{
    int (*fn)(void *, uint32_t, char *, int) = moe_go_token_callback;
    if (!fn) return -1;
    return fn(user, id, (char *)t, len);
}
```

Because it is weak, `libmoe_bridge.a` still links into the C-only
`moe_generate` binary (the symbol is simply NULL there).

### 1.3 Go wrapper (excerpt)

```go
/*
#cgo CFLAGS: -I${SRCDIR}/../src/moe_stream
#cgo LDFLAGS: -L${SRCDIR}/../src/moe_stream/build -lmoe_bridge -lm -lpthread
#include <stdlib.h>
#include "moe_bridge.h"
*/
import "C"

func Open(path string, o Options) (*Session, error) {
    cpath := C.CString(path)
    defer C.free(unsafe.Pointer(cpath))

    var opts C.moe_session_opts_t
    opts.gguf_path   = cpath
    opts.cache_bytes = C.size_t(o.CacheMB) * 1024 * 1024
    opts.threads     = C.int(o.Threads)
    opts.max_seq     = C.int(o.MaxSeq)
    if o.ODirect { opts.odirect = 1 }

    p := C.moe_session_create(&opts)
    if p == nil { return nil, errors.New(C.GoString(C.moe_bridge_last_error())) }
    return &Session{p: p}, nil
}
```

### 1.4 Getting tokens back to Go: array vs callback

| | shared C array | callback (`cgo.Handle`) |
|---|---|---|
| code | `Generate` fills `[]C.uint32_t` | Go closure invoked per token |
| overhead | one cgo call total | one cgo call **per token** |
| streaming UI | needs a second pass | immediate |
| lifetime risk | none (array lives on the Go stack for the call) | handle must be deleted |
| best for | benchmark, batch, tests | chat UI, token-by-token display |

Both are implemented. The array path is the default; the callback path is used
when `onToken != nil`:

```go
ids := make([]C.uint32_t, maxNew)
h := cgo.NewHandle(onToken)
defer h.Delete()
n := C.moe_session_generate_go(s.p, C.int(maxNew),
        (*C.uint32_t)(unsafe.Pointer(&ids[0])), C.int(maxNew), C.uintptr_t(h))
```

```go
//export moe_go_token_callback
func moe_go_token_callback(user unsafe.Pointer, id C.uint32_t,
                           text *C.char, n C.int) C.int {
    h := cgo.Handle(uintptr(user))
    fn, _ := h.Value().(func(uint32, string))
    if fn != nil { fn(uint32(id), C.GoStringN(text, n)) }
    return 0
}
```

`cgo.Handle` is the supported way to pass a Go value through C: it is a
`uintptr`, not a pointer, so rule (1) is not violated. The callback never
outlives the `Generate` call, and the handle is deleted by `defer`.

### 1.5 Memory rules that matter here

- `C.CString` must be freed. Every call site uses `defer C.free`.
- Passing `&tokens[0]` (a `[]uint32`) to C is legal **for the duration of the
  call** because the slice contains no Go pointers and C does not retain it.
- The session handle is freed by `moe_session_destroy`; a `runtime.SetFinalizer`
  is a safety net, not the primary path — call `Close()`.
- Do not call `Prefill` and `Generate` concurrently on one session: the model
  has a single set of scratch buffers and one KV cache. One goroutine per
  session; use a mutex if you must share.

### 1.6 Build

```bash
# 1. C side
cd src/moe_stream && make lib        # -> build/libmoe_bridge.a

# 2. Go side
cd gobridge && CGO_ENABLED=1 go build ./...
CGO_ENABLED=1 go run ./cmd/generate \
    -model ../Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf -ngen 4
```

Observed output:

```
backend=scalar vocab=151936 ctx=256
prompt "The capital of France is" -> 5 tokens
[prefill] sweep: 5 tokens x 48 layers
prefill: 12.755901155s (2551 ms/token)

generated:
 Paris.

2 tokens in 3.314962847s (0.603 tok/s), ids=[12095 13]
```

---

## 2. Prefill: layer sweep

### 2.1 Why token-by-token prefill is wrong

Decode reads `48 × 8` experts per token. Prefill that way reads the same count
**per prompt token**:

| prompt | token-by-token I/O | layer sweep I/O |
|---|---|---|
| 5 tokens | ~5.1 GB | ~5.1 GB (union) |
| 11 tokens | ~11.2 GB | ~11.2 GB |
| 32 tokens | ~33 GB | ~15.5 GB |
| 100 tokens | ~102 GB | ~15.5 GB |

The sweep caps at the full expert set (15.5 GiB) no matter how long the prompt
is, because each expert is loaded once and applied to every token routed to it.

### 2.2 Hidden-state storage

```c
float    *h   = calloc(n * n_embd, sizeof(float));  /* [n][2048] hidden states */
float    *xb  = calloc(n * n_embd, sizeof(float));  /* [n][2048] normalised   */
float    *acc = calloc(n * n_embd, sizeof(float));  /* [n][2048] expert sum   */
uint32_t *rid = malloc(n * top_k * 4);              /* [n][8]   expert ids    */
float    *rw  = malloc(n * top_k * 4);              /* [n][8]   router weights*/
```

For 256 tokens that is 6.3 MB of activations — negligible next to a 512 MB
cache. The KV cache is separate and already sized by `max_seq`.

### 2.3 Ordering constraint

Within one layer, **attention must run for all tokens before routing**, because
the router consumes the post-attention residual. That gives the layer shape:

```
for layer l:
    for t:  xb[t] = rmsnorm(h[t], attn_norm)
            attention(l, t)            -> writes K/V at position pos+t, h[t] += proj

    for t:  xb[t] = rmsnorm(h[t], ffn_norm)
            (rid[t], rw[t]) = top8(router(xb[t]))

    memset(acc, 0)
    for expert e in 0..127:            # <- the sweep, inside moe_stream
        if no token routed to e: skip
        load e once                    # double-buffered read
        for t in tokens_routed_to_e:
            acc[t] += rw[t][e] * down(silu(gate(xb[t])) * up(xb[t]))

    for t:  h[t] += acc[t]
```

### 2.4 Implementation

The sweep itself lives in the streamer (`moe_stream.c`) and takes a callback
for the math, so the I/O pipeline (single reader, double buffer, skip-empty)
is reused rather than duplicated:

```c
int moe_prefill_layer_with(moe_ctx_t *ctx, int layer, int num_tokens,
                           const uint32_t *routing,
                           moe_gemm_fn fn, void *user);
```

The model supplies the callback:

```c
static void prefill_expert_cb(void *user, int layer, uint32_t expert,
                              const moe_weights_t *w,
                              const uint32_t *token_ids, int n_tokens)
{
    prefill_job_t *p = user;
    for (int j = 0; j < n_tokens; j++) {
        const int t = token_ids[j];
        const float *x = p->xb + t * m->n_embd;

        float wt = lookup_router_weight(p, t, expert);
        raw_mul(m, gate_type, w->gate, 2048, 768, x, m->gate);
        raw_mul(m, up_type,   w->up,   2048, 768, x, m->up);
        m->be->swiglu(m->be->self, m->gate, m->up, m->act, 768);
        raw_mul(m, down_type, w->down, 768, 2048, m->act, m->proj);

        for (int i = 0; i < m->n_embd; i++)
            p->acc[t * m->n_embd + i] += wt * m->proj[i];
    }
}
```

`moe_prefill_layer_with` builds an inverted index (`expert -> token list`) with
a counting sort, so the callback gets a contiguous list and never rescans the
routing table.

### 2.5 Measured

Same model, same 512–1024 MB cache, `O_DIRECT`, 4 I/O workers.

| prompt | sweep | legacy (token-by-token) | speedup |
|---|---|---|---|
| 5 tokens | 11.3 s | 17.2 s | 1.5× |
| 11 tokens | 23.2 s | 35.2 s | 1.5× |
| 32 tokens | 65.8 s | ~102 s (extrapolated) | ~1.6× |

Two honest observations:

1. The gain is **not** larger yet because scalar attention is now the dominant
   cost at long prompts (O(n²) in positions). The I/O advantage is real and
   bounded, but it is hidden behind attention math.
2. Once attention moves to the GPU backend, the sweep's bounded I/O becomes the
   visible win, and the 100-token case should approach the 15.5 GiB read time
   (~16 s at 1 GB/s) instead of ~100 s.

---

## 3. Compute backend abstraction

### 3.1 Struct

```c
typedef struct compute_backend compute_backend_t;

struct compute_backend {
    const char *name;
    void       *self;          /* VkDevice wrapper, etc. */

    /* y[ne1] = W[ne1][ne0] * x[ne0]; W is ggml-quantised, row-major */
    void (*matmul)(void *self, uint32_t type, const uint8_t *w,
                   uint64_t ne0, uint64_t ne1, const float *x, float *y);

    void (*rmsnorm)(void *self, float *out, const float *x, const float *w,
                    int n, float eps);
    void (*rope)(void *self, float *v, int n, const float *cs, const float *sn);
    void (*softmax)(void *self, float *x, int n);
    void (*swiglu)(void *self, const float *gate, const float *up,
                   float *out, int n);

    void (*sync)(void *self);  /* flush queued GPU work; no-op for scalar */
};
```

`matmul` is the unit of delegation on purpose: the Vulkan backend needs to own
the upload lifetime of each expert buffer, and the streamer is the only layer
that knows when an expert becomes resident.

### 3.2 Exact call sites

| location | call | why it is on the backend |
|---|---|---|
| `attention_block` | `matmul` × 4 (q, k, v, o) | 518 MiB of weights per token |
| `attention_block` | `rmsnorm` × (n_head + n_head_kv) | q/k norm per head |
| `attention_block` | `rope` × (n_head + n_head_kv) | cheap, but fused later |
| `attention_block` | `softmax` | will fold into a fused attention kernel |
| `attention_block` | inner score/V loops | **TODO(vulkan):** one fused dispatch |
| `moe_route` | `matmul` (router) + `softmax` | f32 [2048×128] |
| `moe_block` / `prefill_expert_cb` | `matmul` × 3 per expert | 94% of all weight bytes |
| `moe_block` / `prefill_expert_cb` | `swiglu` | fuse into the gate/up epilogue later |
| `qwen_forward` / `qwen_prefill` | `matmul` (lm_head) | 255 MiB q6_K per token |

Selected code:

```c
/* attention_block — projections */
mt_mul(m, &L->attn_q, xbn, m->q);
mt_mul(m, &L->attn_k, xbn, m->k);
mt_mul(m, &L->attn_v, xbn, m->v);

/* per-head norm + rope */
for (int h = 0; h < nh; h++) {
    m->be->rmsnorm(m->be->self, m->q + h*hd, m->q + h*hd, qkw, hd, m->eps);
    m->be->rope(m->be->self, m->q + h*hd, hd, cs, sn);
}

/* MoE expert */
raw_mul(m, ml->gate.type, ex[k].gate, ml->gate.ne0, ml->gate.ne1, xbn, m->gate);
raw_mul(m, ml->up.type,   ex[k].up,   ml->up.ne0,   ml->up.ne1,   xbn, m->up);
m->be->swiglu(m->be->self, m->gate, m->up, m->act, m->n_ff_exp);
raw_mul(m, ml->down.type, ex[k].down, ml->down.ne0, ml->down.ne1, m->act, m->proj);
```

### 3.3 Scalar backend

`compute_backend_scalar()` returns a stateless vtable. `matmul` is the same
fused dequant+dot loop used by the working prototype:

```c
static void sc_matmul(void *self, uint32_t type, const uint8_t *w,
                      uint64_t ne0, uint64_t ne1, const float *x, float *y)
{
    const uint64_t rb = ggml_type_row_bytes(type, ne0);
    for (uint64_t o = 0; o < ne1; o++)
        y[o] = vec_dot(type, w + o * rb, x, ne0);
}
```

### 3.4 Vulkan stub and the Mali plan

`compute_backend_vulkan()` returns a vtable whose entries currently forward to
scalar, each marked with the exact `vkCmd*` call that replaces it:

```c
static void vk_matmul(void *self, uint32_t type, const uint8_t *w,
                      uint64_t ne0, uint64_t ne1, const float *x, float *y)
{
    /* TODO(vulkan): bind quantised-weight SSBO + x/y SSBO,
     *               vkCmdDispatch(ceil(ne1/64), 1, 1). */
    sc_matmul(self, type, w, ne0, ne1, x, y);
}
```

Mali-specific notes that should shape the implementation:

- **Batch submits.** Tile-based GPUs punish per-dispatch submission. Build one
  command buffer per layer and submit at the layer boundary (that is what
  `sync` is for). A 48-layer token is ~200 dispatches; batching per layer keeps
  it at 48 submits.
- **Keep weights quantized.** Do not upload dequantized f32; q4_K is 144 bytes
  per 256 weights. Dequantize in-shader into shared memory.
- **Buffer cache keyed by host pointer.** The streamer hands a fresh pointer per
  expert slot; the backend should map `(ptr, len)` to a persistent `VkBuffer`
  and re-upload only when the pointer changes. Expert slots are reused, so this
  is a stable working set.
- **Attention kernel is the second win.** The current C loop is O(pos) per head
  with a 128-float dot; it dominates prefill at 32+ tokens.
- **No `vkQueueWaitIdle` per op.** Use a fence per layer batch.

### 3.5 Migration order

1. Implement `vk_matmul` for q4_K only; keep scalar for the rest.
   Verify token-for-token equality on `"The capital of France is"`.
2. Add q5_K/q6_K. Compare logits, not just argmax (argmax hides small errors).
3. Move `swiglu` into the matmul epilogue.
4. Move attention (scores + softmax + V) into one kernel.
5. Profile: expect lm_head and the 8 experts to dominate; attention second.

---

## 4. Files added/changed

| file | role |
|---|---|
| `src/moe_stream/compute_backend.h` | backend vtable |
| `src/moe_stream/compute_backend.c` | scalar + Vulkan stub |
| `src/moe_stream/moe_bridge.h/.c` | CGO-facing session API |
| `src/moe_stream/qwen_model.{h,c}` | backend dispatch + `qwen_prefill` |
| `src/moe_stream/moe_stream.{h,c}` | `moe_prefill_layer_with` |
| `gobridge/go.mod` | standalone Go module |
| `gobridge/moe.go` | cgo wrapper |
| `gobridge/cmd/generate/main.go` | runnable example |
| `src/moe_stream/Makefile` | `libmoe_bridge.a` target |

---

## 5. Gotchas

- **cgo and `//export`:** the preamble must not contain function definitions.
  That is why the callback thunk lives in `moe_bridge.c`, not in the Go file.
- **Weak symbols are ELF-only.** Fine for Android/Linux. On macOS use
  `__attribute__((weak_import))` and a NULL check.
- **Go finalizers are not deterministic.** Call `Close()`; the finalizer only
  prevents leaks in tests.
- **One session, one goroutine.** The model has shared scratch and one KV cache.
- **`C.CString` allocates.** Free it; do not pass a Go string directly to a
  function that stores the pointer.
- **Static lib must be rebuilt** after C changes; `go build` will not do it.
  Wire `make lib` into your build script or a `go:generate` line.
- **O_DIRECT on the model path**: `/data/data` (f2fs) works; an SD-card path
  will silently fall back to buffered reads, which is correct but doubles page
  cache pressure.
- **The tokenizer encode is approximate** (greedy longest-match). If exact
  tokenization matters, pass ids from a real tokenizer.

---

## 6. What is deliberately not done yet

- Vulkan is a stub; the vtable and call sites are final, the shaders are not.
- Prefill attention is token-by-token; it is correct but O(n²) in C.
- No sampling controls (temperature/top-p); generation is greedy.
- No request scheduler or continuous batching. One prompt at a time.
- KV cache is f32; q8_0 would halve it and is a natural next step once the
  backend owns attention.

---

## 9. Vulkan backend and Go

`make -C src/moe_stream VULKAN=1 all` produces a `libmoe_bridge.a` that calls
into Vulkan. When linking that library from Go you must add `-lvulkan` to the
cgo LDFLAGS (and the device must expose a compute queue). The default `make`
build keeps the scalar stub, so the Go module links with no extra flags.

Shader changes require `src/moe_stream/shaders/build.sh`; `compute_vulkan.o`
depends on the generated `vulkan_shaders.h`, so `make` rebuilds it correctly.
