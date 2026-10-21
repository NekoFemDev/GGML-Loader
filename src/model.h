// Shared model/tokenizer/inference machinery used by all ggml-loader subcommands (inspect, run,
// chat). See model.cpp for the implementation and the design notes on scope.
//
// Multi-architecture support: hparams/layer_weights carry a handful of per-architecture flags
// (qkv_fused, has_qkv_bias, has_qk_norm, ffn_fused, has_rope_scaling) set once at load time from
// `general.architecture`; build_graph()/load_model() branch on those flags at a few well-defined
// points (QKV projection, FFN, RoPE params) rather than having one code path per architecture.
// Whether the output projection is tied to token_embd is NOT one of these flags -- it's decided
// by load_model() from whether an `output.weight` tensor actually exists in the file, since that
// varies by model *size* within an architecture (e.g. small Qwen models tie, larger ones don't),
// not by architecture alone.
// Currently supported: phi3 (fused QKV, fused gate+up FFN, LongRoPE partial rotary), qwen2
// (separate QKV with bias, separate gate/up FFN, plain full RoPE), qwen3 (like qwen2 but no QKV
// bias, adds per-head QK-Norm before RoPE, and attention width can differ from n_embd/n_head --
// see n_embd_head), llama (like qwen3's flags minus QK-Norm: separate QKV, no bias, separate
// gate/up FFN, plain full RoPE), nemotron_h (hybrid attn/SSM/FFN blocks with Mamba2), qwen35
// (hybrid attention/Gated-DeltaNet mixers), and dory (looped nGPT hybrid: attention/MLP blocks,
// L2 normalization everywhere, per-loop KV cache -- see dory.h; its graph/weights/cache are
// delegated to the dory module rather than the shared layer machinery below). Each has its own
// chat template (see append_chat_turn's switch); llama's is
// "<|start_header_id|>role<|end_header_id|>\n\ntext<|eot_id|>", distinct from both phi3's and
// ChatML (dory and the qwen family are ChatML, with dory adding its think-block handling in
// append_generation_prompt). Adding another architecture means extending
// load_hparams/load_model and, if its chat template differs from all existing ones,
// append_chat_turn()/friends.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "dory.h"

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Fallback KV-cache context size, used only when automatic sizing can't read the system's
// available memory (/proc/meminfo missing/unparseable). Normally every subcommand sizes its KV
// cache at startup from the RAM actually free after the weights are loaded -- see auto_n_ctx().
constexpr int64_t N_CTX_DEFAULT = 4096;

enum class arch_t {
    PHI3,
    QWEN2,
    QWEN3,
    LLAMA,
    NEMOTRON_H,
    QWEN35,
    DORY,
};

// nemotron_h only: which mixer a block uses. Derived from two per-layer gguf arrays
// (attention.head_count_kv, feed_forward_length) exactly like llama.cpp's own hparams does --
// n_head_kv==0 && n_ff==0 => SSM (Mamba2), n_head_kv!=0 => attention (no FFN in that block),
// n_ff!=0 => plain FFN (no attn/ssm in that block). Every nemotron_h block is exactly ONE of
// these (unlike a normal transformer block, which always combines attn+ffn) -- confirmed both
// from this project's own tensor-presence inspection and against llama.cpp's upstream source.
enum class layer_kind_t { ATTN, SSM, FFN };

struct hparams {
    arch_t      arch;
    std::string arch_name; // gguf key prefix, e.g. "phi3", "qwen2"

    int64_t n_embd;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t n_embd_head;    // per-head Q/K/V width. Usually n_embd/n_head, but NOT always: qwen3
                             // decouples attention width from n_embd (attention.key_length is
                             // explicit and can differ), so this is read from that key when
                             // present rather than always derived.
    int64_t n_layer;
    int64_t n_ctx_train;    // trained context length (`{arch}.context_length` gguf key); 0 if absent
    int64_t n_ff;
    int64_t n_rot;          // rope dimensions to rotate (== n_embd_head for full rotary)
    bool    rope_neox;      // true: NeoX-style (split-half) rotation, used by phi3/qwen2/qwen3.
                             // false: "normal" (interleaved-pairs) rotation, used by llama --
                             // this is a real per-architecture distinction (confirmed against
                             // llama.cpp's llama_model_rope_type()), not a universal default;
                             // getting it wrong doesn't crash, it just silently scrambles
                             // position information (observed: fluent words, broken grammar).
    float   rms_eps;
    float   rope_freq_base;

    bool    has_rope_scaling; // phi3 LongRoPE; if false, plain RoPE (ext_factor=0)
    int64_t n_ctx_orig;       // only meaningful if has_rope_scaling
    float   rope_attn_factor; // only meaningful if has_rope_scaling

    bool qkv_fused;    // phi3: one attn_qkv weight. qwen2/qwen3: separate attn_q/k/v.
    bool has_qkv_bias; // qwen2: attn_q/k/v.bias present. phi3/qwen3: no bias.
    bool has_qk_norm;  // qwen3: attn_q_norm/attn_k_norm (RMSNorm per-head, before RoPE).
    bool ffn_fused;    // phi3: one ffn_up weight holding [gate;up]. qwen2/qwen3: separate ffn_gate/up.

    // nemotron_h only, below. Its blocks aren't a uniform attn+ffn pair (see layer_kind_t) so it
    // doesn't reuse the qkv_fused/ffn_fused/has_qk_norm flags above -- those stay at their default
    // (false) for this arch and are simply unused by build_graph's nemotron_h branch.
    bool no_rope = false; // true (nemotron_h only): attention layers apply NO positional encoding
                           // at all -- confirmed both by this gguf having no rope.freq_base key
                           // and against llama.cpp upstream (rope_type reports NONE for this arch
                           // family in real bug reports). Getting this wrong the other way (adding
                           // RoPE where the model expects none) would silently scramble attention
                           // the same way a wrong rope_neox value did for llama.
    std::vector<int64_t> n_head_kv_layer; // per-layer attention.head_count_kv, 0 where N/A
    std::vector<int64_t> n_ff_layer;      // per-layer feed_forward_length, 0 where N/A
    int64_t ssm_d_state = 0; // Mamba2 state size (nemotron_h.ssm.state_size)
    int64_t ssm_d_conv  = 0; // causal conv1d kernel width (nemotron_h.ssm.conv_kernel)
    int64_t ssm_n_group = 0; // B/C group count (nemotron_h.ssm.group_count)
    int64_t ssm_d_inner = 0; // SSM's own working width (nemotron_h.ssm.inner_size)
    int64_t ssm_n_head  = 0; // SSM head count == inner_size/head_dim (nemotron_h.ssm.time_step_rank,
                              // despite that gguf key's Mamba-1-derived name -- confirmed via this
                              // checkpoint's actual ssm_in output width: 2*d_inner + 2*n_group*
                              // d_state + this value == ssm_in.weight's out-features exactly)

    layer_kind_t layer_kind(int64_t il) const {
        if (arch != arch_t::NEMOTRON_H) return layer_kind_t::ATTN; // unused by other archs' code path
        if (n_ff_layer[il] != 0) return layer_kind_t::FFN;
        if (n_head_kv_layer[il] != 0) return layer_kind_t::ATTN;
        return layer_kind_t::SSM;
    }

    // qwen35 only: unlike nemotron_h's per-layer arrays, this arch's mixer choice is a fixed
    // interval rule read straight off its own gguf key (qwen35.full_attention_interval, this
    // checkpoint: 4) -- every block still has both a mixer AND an FFN (normal transformer
    // structure, unlike nemotron_h's single-mixer-per-block), only the mixer choice varies.
    // Confirmed against llama.cpp upstream's own fallback rule when no per-layer override array
    // is present (this gguf has none): layer il (0-indexed) is full attention iff (il+1) is a
    // multiple of the interval, otherwise it's the Gated DeltaNet linear-attention mixer.
    int64_t full_attention_interval = 0; // 0 => not applicable (every arch except qwen35)
    int64_t dn_n_head    = 0; // Gated DeltaNet head count (qwen35.ssm.group_count)
    int64_t dn_head_dim  = 0; // per-head width, q/k/v and the square recurrent state
                               // (qwen35.ssm.state_size -- confirmed via this checkpoint's own
                               // attn_qkv output width: 3 * dn_n_head * dn_head_dim == 6144 exactly)
    int64_t dn_d_conv    = 0; // causal conv1d kernel width (qwen35.ssm.conv_kernel)

    bool is_deltanet_layer(int64_t il) const {
        if (arch != arch_t::QWEN35 || full_attention_interval <= 0) return false;
        if (il == n_layer - 1) return false; // the last layer is always forced to full attention
                                              // (confirmed via this checkpoint's own tensor
                                              // presence: layer 24/25 breaks the plain modulo rule
                                              // that otherwise holds for every earlier layer) --
                                              // a known hybrid-architecture pattern (full context
                                              // mixing right before the output projection).
        return (il + 1) % full_attention_interval != 0;
    }

    int64_t bos_id;
    int64_t eos_id;

    // detected from the gguf's own tokenizer.chat_template (substring-sniffed for "<think>" --
    // no jinja engine here, see append_chat_turn's file-header comment) -- true for any reasoning
    // model regardless of architecture (confirmed present on both qwen35 and nemotron_h's actual
    // templates), not hardcoded per arch_t.
    bool supports_thinking = false;

    // dory only. The dory module owns this architecture's parameters (block topology, layer
    // pattern, rope profiles, nGPT constants); the generic scalar fields above are populated
    // from the same keys so shared code paths (printing, per-token sizing) keep working.
    struct dory_hparams dory;
};

hparams load_hparams(const gguf_context * ctx);

struct bpe_ranks {
    std::unordered_map<std::string, int> rank; // "left\x01right" -> merge priority (lower = earlier)
};

// GPT-2-style byte-level BPE tokenizer (encode) + detokenizer (decode). The pretokenizer
// implements the classic GPT-2 splitting rules (contractions/letter-run/digit-run/other-run/
// whitespace-run), not the more elaborate model-specific pretokenizers named by tokenizer.ggml.pre
// ("gpt-4o", "qwen2", ...). Byte-level BPE is lossless regardless of where the pretokenizer draws
// chunk boundaries, so this still encodes to valid vocab ids and round-trips correctly -- it just
// may not match the official tokenizer's exact token boundaries/count in edge cases.
struct vocab {
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<uint32_t, uint8_t> byte_decoder;
    bpe_ranks bpe;

    std::string detok(const std::vector<int32_t> & ids) const;
    std::vector<int32_t> encode(const std::string & text) const;

    // exact-match lookup for a control/special token (e.g. "<|im_start|>"); exits with an error
    // if the model's vocab doesn't have it, rather than mis-tokenizing it through BPE
    int32_t special(const char * tag) const;
};

vocab load_vocab(const gguf_context * ctx);

// Appends one chat-template turn for `role` ("system"/"user"/"assistant") to `out`, in this
// model's own instruct format (phi3: "<|role|>{text}<|end|>"; qwen2/ChatML:
// "<|im_start|>role\n{text}<|im_end|>\n").
void append_chat_turn(const hparams & hp, const vocab & vc, std::vector<int32_t> & out,
                       const std::string & role, const std::string & text);

// Appends the tokens that prompt the model to start its own turn (phi3: "<|assistant|>";
// qwen2: "<|im_start|>assistant\n"). For a reasoning model (hparams::supports_thinking) with
// enable_thinking=false, also appends a forced-empty "<think>\n\n</think>\n\n" block right after,
// suppressing reasoning -- matches how these models' own chat templates handle the disabled case
// (confirmed against qwen35's and nemotron_h's actual jinja). enable_thinking=true (the default)
// appends nothing extra: the model decides for itself whether to open a <think> block, same as
// real chat templates leave it when thinking is allowed. No-op for non-reasoning models regardless
// of the flag.
void append_generation_prompt(const hparams & hp, const vocab & vc, std::vector<int32_t> & out,
                               bool enable_thinking = true);

// True when append_generation_prompt() (for this model + enable_thinking) leaves the assistant
// turn *inside* an open think block: nemotron_h's template emits the opening tag itself when
// thinking is enabled (verified against this checkpoint's jinja), so the first generated tokens
// are reasoning and a render-side filter must start in thinking state (see think_filter.h).
// qwen3-style models instead generate their own opening tag, so this returns false for them.
// Must stay in lockstep with append_generation_prompt()'s per-arch branches.
bool generation_prompt_opens_think(const hparams & hp, bool enable_thinking);

// The token id that ends a turn in this model's chat template (phi3: "<|end|>"; qwen2: "<|im_end|>").
int32_t turn_end_token(const hparams & hp, const vocab & vc);

struct layer_weights {
    struct ggml_tensor * attn_norm;

    // fused (phi3) vs separate (qwen2) QKV -- only the relevant fields are set, per hparams::qkv_fused
    struct ggml_tensor * attn_qkv     = nullptr;
    struct ggml_tensor * attn_q       = nullptr;
    struct ggml_tensor * attn_k       = nullptr;
    struct ggml_tensor * attn_v       = nullptr;
    struct ggml_tensor * attn_q_bias  = nullptr;
    struct ggml_tensor * attn_k_bias  = nullptr;
    struct ggml_tensor * attn_v_bias  = nullptr;

    // per-head RMSNorm applied to Q/K before RoPE -- per hparams::has_qk_norm (qwen3)
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;

    struct ggml_tensor * attn_output;
    struct ggml_tensor * ffn_norm;

    // fused (phi3) vs separate (qwen2) FFN gate/up -- per hparams::ffn_fused. nemotron_h's FFN-kind
    // blocks set only ffn_up/ffn_down (ffn_gate stays null -- no gating, see layer_kind_t::FFN in
    // build_graph: plain ReLU() rather than SwiGLU, verified against llama.cpp upstream).
    struct ggml_tensor * ffn_up_gate = nullptr; // packed [gate;up], phi3
    struct ggml_tensor * ffn_gate    = nullptr; // qwen2
    struct ggml_tensor * ffn_up      = nullptr; // qwen2

    struct ggml_tensor * ffn_down = nullptr;

    // nemotron_h Mamba2 mixer -- only set when hparams::layer_kind(il) == SSM. Verified tensor
    // roles and the exact op sequence they feed (ssm_in split order z/xBC/dt, conv-then-silu over
    // the whole xBC chunk not just x, A used raw with no exp/negate, softplus happening inside
    // ggml_ssm_scan rather than the graph, D-skip+gating before the groupnorm) against llama.cpp's
    // own build_mamba2_layer, not guessed -- see build_graph's nemotron_h branch for the recipe.
    struct ggml_tensor * ssm_in       = nullptr; // [n_embd, 2*d_inner + 2*n_group*d_state + n_head]
    struct ggml_tensor * ssm_conv1d_w = nullptr; // [d_conv, d_inner + 2*n_group*d_state]
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b     = nullptr; // [n_head] bias added to dt before ggml_ssm_scan
    struct ggml_tensor * ssm_a        = nullptr; // [1, n_head] Mamba2 scalar-per-head decay, used raw
    struct ggml_tensor * ssm_d        = nullptr; // [1, n_head] skip-connection scale
    struct ggml_tensor * ssm_norm     = nullptr; // [d_inner/n_group, n_group] grouped RMSNorm weight
    struct ggml_tensor * ssm_out      = nullptr; // [d_inner, n_embd]

    // qwen35 Gated DeltaNet mixer (linear-attention layers only, see hparams::is_deltanet_layer).
    // Reuses attn_qkv above for the fused q/k/v in-projection (3*dn_n_head*dn_head_dim wide) and
    // ssm_conv1d_w/ssm_dt_b/ssm_a/ssm_norm/ssm_out above for their same roles as nemotron_h's
    // Mamba2 mixer -- ssm_conv1d_b stays null here (this arch's conv1d has no bias tensor at all,
    // confirmed via inspect, not an omission). attn_gate below is this arch-specific: a dedicated
    // output-gate projection (nemotron_h instead derives its z-gate from within ssm_in's own split).
    struct ggml_tensor * ssm_alpha = nullptr; // [n_embd, dn_n_head] -> decay gate pre-activation
    struct ggml_tensor * ssm_beta  = nullptr; // [n_embd, dn_n_head] -> mixing-rate gate pre-activation
    struct ggml_tensor * attn_gate = nullptr; // [n_embd, n_embd] output gate (qwen35 delta-net only)
};

struct model {
    struct ggml_tensor * token_embd;
    struct ggml_tensor * output;             // == token_embd if no separate output.weight tensor exists
    struct ggml_tensor * output_norm;
    struct ggml_tensor * rope_factors = nullptr; // phi3 LongRoPE only; nullptr => plain RoPE
    std::vector<layer_weights> layers;

    // dory only: the module binds this architecture's weights itself (attention/MLP block split,
    // per-channel scales) and owns its graph; `layers` above stays empty for this arch.
    struct dory_model dory;
};

model load_model(struct ggml_context * wctx, const hparams & hp);

// Loads every tensor from a GGUF file into memory owned by `backend` -- works for any backend,
// including GPU ones like Vulkan, unlike gguf_init_from_file(no_alloc=false) which only ever
// gives plain host (malloc) memory. Reads each tensor's raw bytes directly from the file at its
// recorded offset and uploads via ggml_backend_tensor_set(), rather than relying on gguf.c's own
// (CPU-only) bulk loading path.
struct weights_store {
    struct ggml_context * ctx = nullptr;    // tensor metadata + backend-owned data; pass to load_model()
    ggml_backend_buffer_t buffer = nullptr; // owns the tensor data; free with ggml_backend_buffer_free()
};
weights_store load_weights(const char * path, ggml_backend_t backend, struct gguf_context ** out_gguf);

// Same as load_weights(), but only allocates/reads tensors whose name starts with one of
// `prefixes` -- everything else in the file is skipped entirely (never allocated in the backend
// buffer, never read off disk). Lets a process that only needs e.g. the SDXL VAE decoder avoid
// paying for CLIP+UNet's memory too, so a staged encoder/UNet/decoder pipeline run as separate
// process invocations actually lowers peak memory instead of each stage loading the whole
// checkpoint anyway. An empty `prefixes` matches everything (same behavior as load_weights()).
weights_store load_weights_filtered(const char * path, ggml_backend_t backend,
                                     const std::vector<std::string> & prefixes,
                                     struct gguf_context ** out_gguf);

// Selects a backend by name ("cpu" or "vulkan"). Exits with an error on an unknown name or if
// vulkan is requested but this build/device doesn't have it. n_threads only affects "cpu".
ggml_backend_t init_backend(const std::string & name, int n_threads);

// persistent KV cache: one F16 [head_dim, n_head_kv, n_ctx] tensor per layer, per K/V, where
// n_ctx is chosen by the caller at startup (auto_n_ctx() or an explicit override). For
// nemotron_h (mixed layer kinds), every layer still gets a slot -- wasteful for its SSM/FFN-kind
// layers (no attention there at all) but those slots are sized minimally and simply never
// written/read; simpler than a sparse per-kind allocation. For dory the cache is delegated to
// the dory module instead (see the `dory` member below).
struct kv_cache {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<struct ggml_tensor *> k;
    std::vector<struct ggml_tensor *> v;

    // dory only: ring caches with per-loop slots live in the dory module (one ring per
    // attention block; sliding blocks sized by the window, global blocks by n_ctx). The
    // `k`/`v` vectors above stay empty for this arch; `ctx`/`buffer` still own the storage.
    struct dory_kv_cache dory;
};

kv_cache init_kv_cache(const model & m, const hparams & hp, ggml_backend_t backend, int64_t n_ctx);

// Bytes the KV cache will occupy at `n_ctx` positions, mirroring init_kv_cache's exact per-layer
// sizing (so the two must stay in lockstep) -- auto_n_ctx() divides a memory budget by this to
// turn it into a token count. For dory this delegates to the module's sizing: sliding rings are
// context-independent, so the cost is affine in n_ctx (base + per-token) and auto_n_ctx() solves
// that instead of dividing by a per-token constant.
size_t kv_cache_bytes(const model & m, const hparams & hp, int64_t n_ctx);

// Bytes init_ssm_state() will allocate for this model's recurrent state (nemotron_h / qwen35);
// 0 for every other arch. Context-independent -- the recurrent state doesn't grow with n_ctx.
size_t ssm_state_bytes(const hparams & hp);

// Automatic KV-cache context sizing. Reads MemAvailable from /proc/meminfo -- call it AFTER the
// weights are loaded, so the model's own footprint is already subtracted by the kernel -- then:
//   budget = (MemAvailable - fixed reserve) * keep-fraction - recurrent state,
//   n_ctx  = budget / KV-bytes-per-token, rounded down to a multiple of 64,
// and finally clamps to the model's trained context length (phi3-with-LongRoPE additionally to
// its original_context_length -- the only range this loader's rope factors implement). The fixed
// reserve + keep-fraction are the margin for compute buffers, driver/allocator fragmentation and
// everything else running on the phone; deliberately conservative, since Android's LMK doesn't
// give a process a second warning. Prints one line with the whole calculation. Falls back to
// N_CTX_DEFAULT when /proc/meminfo can't be read.
int64_t auto_n_ctx(const model & m, const hparams & hp, size_t weights_bytes);

// nemotron_h's Mamba2 recurrent state -- NOT position-indexed like kv_cache. Each SSM-kind
// layer gets a small fixed-size running accumulator (does not grow with context length) that
// MUST be updated exactly once per token in strict sequence order -- unlike attention's KV range,
// it can't be recomputed or randomly accessed. Sized for a single sequence (this loader never
// batches multiple chat sessions together), so `ids` is always the constant 1-element {0} tensor
// ggml_ssm_scan needs to select "sequence slot 0". Layers where layer_kind() != SSM get null
// entries (never touched).
struct ssm_state {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor * ids = nullptr; // I32 [1], always {0}
    std::vector<struct ggml_tensor *> conv;  // per layer: [d_conv-1, d_inner+2*n_group*d_state] or null
    std::vector<struct ggml_tensor *> state; // per layer: [d_state, head_dim, n_head] or null
};

ssm_state init_ssm_state(const hparams & hp, ggml_backend_t backend);
void free_ssm_state(ssm_state & ss);

// Reusable scratch state for forward_step: a persistent graph-building buffer and a persistent
// graph allocator, so a generation loop doesn't malloc/free them (and re-plan the activation
// buffer layout) on every single token -- ggml_gallocr_alloc_graph is designed to be called
// repeatedly with differently-shaped graphs (our KV range grows every decode step) and replans
// as needed, so one decode_session can be reused for an entire run/chat session.
struct decode_session {
    std::vector<uint8_t> graph_buf;
    ggml_gallocr_t galloc = nullptr;
    bool use_flash_attn = false; // ggml_flash_attn_ext instead of the manual mul_mat/softmax/
                                  // mul_mat sequence -- opt-in (default off, matching llama.cpp's
                                  // own -fa flag) so it can be A/B benchmarked before trusting it
                                  // not to regress this project's existing lead over llama.cpp.
};

decode_session init_decode_session(ggml_backend_t backend, const hparams & hp, bool use_flash_attn = false);
void free_decode_session(decode_session & ds);

// Runs one forward pass for `new_tokens`, appending them to the KV cache at [n_past, n_past +
// new_tokens.size()) and attending over the full [0, n_past + new_tokens.size()) cached range.
// Returns the logits (size n_vocab) for the *last* new token only.
// `ss` is only consulted for nemotron_h (must be non-null then); every other arch ignores it, so
// existing call sites (run/bench on the 4 uniform-transformer archs) don't need to pass anything.
std::vector<float> forward_step(const model & m, const hparams & hp, kv_cache & kv,
                                 decode_session & ds, ggml_backend_t backend, int64_t n_past,
                                 const std::vector<int32_t> & new_tokens, int64_t n_vocab,
                                 ssm_state * ss = nullptr);

// Same contract as forward_step() (logits for the last token only), but for a batch of tokens
// being prefilled all at once (as opposed to forward_step()'s single-decode-step callers, which
// should keep calling forward_step() directly). When `use_chunking` is true (pass this for the
// Vulkan backend only -- see the long comment at this function's definition in model.cpp), splits
// the batch into <=8-token forward_step() calls if that avoids a ggml-vulkan shader-selection
// performance cliff for medium-small batches; otherwise identical to one forward_step() call.
std::vector<float> prefill(const model & m, const hparams & hp, kv_cache & kv, decode_session & ds,
                            ggml_backend_t backend, int64_t n_past, const std::vector<int32_t> & tokens,
                            int64_t n_vocab, bool use_chunking, ssm_state * ss = nullptr);

int32_t argmax(const std::vector<float> & logits);

// Sampling: temp<=0 is greedy (== argmax, deterministic) -- this is the default everywhere
// (run/chat/bench) so nothing changes unless a caller explicitly opts in via CLI flags. Above
// that: temperature scaling, then top-k, then top-p (nucleus) narrow the candidate set before a
// weighted random draw. repeat_penalty (1.0 = disabled) downweights tokens seen in the last
// repeat_last_n tokens of `recent_tokens` -- applied even at temp<=0, since it doesn't need
// randomness to stop greedy decoding's "Paris is the capital of France. Paris is..." loops.
struct sampler_params {
    float    temp            = 0.0f;
    int      top_k           = 40;
    float    top_p           = 0.95f;
    float    repeat_penalty  = 1.0f;
    int      repeat_last_n   = 64;
    uint32_t seed            = 0; // 0 = seed from std::random_device
};

struct sampler {
    sampler_params params;
    std::mt19937 rng;
};

sampler init_sampler(const sampler_params & params);

// mutates `logits` in place (repetition penalty); returns the sampled token id
int32_t sample(sampler & smp, std::vector<float> & logits, const std::vector<int32_t> & recent_tokens);
