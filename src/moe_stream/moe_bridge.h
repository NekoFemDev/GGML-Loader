/*
 * moe_bridge.h — stable C ABI for the Go integration.
 *
 * Everything here is plain C: no C++ types, no callbacks with Go pointers, no
 * structs whose layout Go has to model. The session owns the model, the
 * tokenizer and the KV cache.
 */
#ifndef MOE_BRIDGE_H
#define MOE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moe_session moe_session_t;

typedef struct {
    const char *gguf_path;
    size_t      cache_bytes;   /* expert LRU budget */
    int         threads;       /* I/O workers */
    int         max_seq;       /* KV capacity in tokens */
    int         odirect;       /* 0/1 */
} moe_session_opts_t;

/* Return non-zero from the callback to stop generation. `utf8` is not
 * NUL-terminated; it is exactly `len` bytes. */
typedef int (*moe_token_cb)(void *user, uint32_t token_id,
                            const char *utf8, int len);

/* On failure returns NULL and sets the thread-local bridge error string. */
moe_session_t *moe_session_create(const moe_session_opts_t *opts);
void           moe_session_destroy(moe_session_t *s);

const char *moe_bridge_last_error(void);
const char *moe_session_error(const moe_session_t *s);
const char *moe_session_backend(const moe_session_t *s);

int moe_session_vocab_size(const moe_session_t *s);
int moe_session_context(const moe_session_t *s);
int moe_session_pos(const moe_session_t *s);

int moe_session_reset(moe_session_t *s);

/* Approximate (greedy longest-match) encode. Returns token count or -1. */
int moe_session_encode(moe_session_t *s, const char *text,
                       uint32_t *ids, int max_ids);
/* Returns bytes written to buf, or -1. */
int moe_session_decode(moe_session_t *s, uint32_t id, char *buf, int cap);

/* Layer-sweep prefill. Positions advance by n_tokens. */
int moe_session_prefill(moe_session_t *s, const uint32_t *tokens, int n_tokens);

/* Single decode step at the current position. */
int moe_session_forward(moe_session_t *s, uint32_t token);
int moe_session_argmax(moe_session_t *s);

/* Generate up to max_new tokens. Fills out_ids[0..n-1] (capped at out_cap).
 * Pass cb=NULL for batch mode. Returns the number generated or -errno. */
int moe_session_generate(moe_session_t *s, int max_new, uint32_t *out_ids,
                         int out_cap, moe_token_cb cb, void *user);

/* cgo variant: `go_handle` is a runtime/cgo.Handle. Uses the Go symbol
 * moe_go_token_callback (weak, so non-Go binaries still link). */
int moe_session_generate_go(moe_session_t *s, int max_new, uint32_t *out_ids,
                            int out_cap, uintptr_t go_handle);

/* Raw token-id callback path (the "ASCII shortcut"): calls the Go symbol
 * go_token_callback(int) per token, no detokenisation. Kept for compatibility;
 * the chat uses the decoded callback above because token ids are vocabulary
 * indices, not character codes. */
int moe_session_generate_raw(moe_session_t *s, int max_new);

/* Auto configuration: calls calculate_optimal_cache() and reports the
 * MemAvailable figure used for the decision. Any out pointer may be NULL. */
void moe_auto_config(int *out_cache_mb, int *out_ctx_len, int *out_avail_mb);

/* Select the Vulkan backend stub. Returns 0 (always succeeds while stubbed). */
int moe_session_use_vulkan(moe_session_t *s);

#ifdef __cplusplus
}
#endif

#endif /* MOE_BRIDGE_H */
