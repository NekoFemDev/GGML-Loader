#include "model.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int64_t kv_find(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        fprintf(stderr, "error: missing gguf key '%s'\n", key);
        exit(1);
    }
    return id;
}

static long long kv_int(const gguf_context * ctx, const char * key) {
    int64_t id = kv_find(ctx, key);
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (long long) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default:
            fprintf(stderr, "error: key '%s' is not an integer type\n", key);
            exit(1);
    }
}

static double kv_float(const gguf_context * ctx, const char * key) {
    int64_t id = kv_find(ctx, key);
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(ctx, id);
        default:
            fprintf(stderr, "error: key '%s' is not a float type\n", key);
            exit(1);
    }
}

static std::string kv_string(const gguf_context * ctx, const char * key) {
    int64_t id = kv_find(ctx, key);
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        fprintf(stderr, "error: key '%s' is not a string type\n", key);
        exit(1);
    }
    return gguf_get_val_str(ctx, id);
}

// nemotron_h's per-layer head_count_kv/feed_forward_length are gguf INT32 arrays, not scalars --
// this is what actually encodes each block's layer_kind_t (see model.h).
static std::vector<int64_t> kv_int_arr(const gguf_context * ctx, const char * key) {
    int64_t id = kv_find(ctx, key);
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id) != GGUF_TYPE_INT32) {
        fprintf(stderr, "error: key '%s' is not an int32 array\n", key);
        exit(1);
    }
    size_t n = gguf_get_arr_n(ctx, id);
    const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx, id);
    return std::vector<int64_t>(data, data + n);
}

hparams load_hparams(const gguf_context * ctx) {
    hparams h{};
    h.arch_name = kv_string(ctx, "general.architecture");
    if      (h.arch_name == "phi3")       h.arch = arch_t::PHI3;
    else if (h.arch_name == "qwen2")      h.arch = arch_t::QWEN2;
    else if (h.arch_name == "qwen3")      h.arch = arch_t::QWEN3;
    else if (h.arch_name == "llama")      h.arch = arch_t::LLAMA;
    else if (h.arch_name == "nemotron_h") h.arch = arch_t::NEMOTRON_H;
    else if (h.arch_name == "qwen35")     h.arch = arch_t::QWEN35;
    else {
        fprintf(stderr, "error: unsupported architecture '%s' "
                         "(supported: phi3, qwen2, qwen3, llama, nemotron_h, qwen35)\n", h.arch_name.c_str());
        exit(1);
    }

    auto pfx = [&](const char * suffix) { return h.arch_name + "." + suffix; };

    h.n_embd  = kv_int(ctx, pfx("embedding_length").c_str());
    h.n_head  = kv_int(ctx, pfx("attention.head_count").c_str());
    h.n_layer = kv_int(ctx, pfx("block_count").c_str());
    h.rms_eps = (float) kv_float(ctx, pfx("attention.layer_norm_rms_epsilon").c_str());

    if (h.arch == arch_t::NEMOTRON_H) {
        // no rope.freq_base key at all in this arch's gguf (see hparams::no_rope) -- every other
        // scalar key that the generic path below reads unconditionally (rope.freq_base,
        // attention.head_count_kv, feed_forward_length) is either absent or an array here, so this
        // arch gets its own branch rather than trying to force it through the shared path.
        h.no_rope         = true;
        h.n_head_kv_layer  = kv_int_arr(ctx, pfx("attention.head_count_kv").c_str());
        h.n_ff_layer       = kv_int_arr(ctx, pfx("feed_forward_length").c_str());
        std::string key_len_key = pfx("attention.key_length");
        h.n_embd_head = gguf_find_key(ctx, key_len_key.c_str()) >= 0
                            ? kv_int(ctx, key_len_key.c_str()) : h.n_embd / h.n_head;
        h.n_rot = h.n_embd_head; // unused: no_rope is true, RoPE is never applied for this arch
        h.n_head_kv = 1; // scalar unused by nemotron_h's own code path; kept nonzero so any
                          // generic-path code that might still read it (e.g. a future shared
                          // helper) doesn't divide by zero or size a 0-dim tensor
        h.has_rope_scaling = false;
        h.n_ctx_orig       = kv_int(ctx, pfx("context_length").c_str());
        h.rope_attn_factor = 1.0f;
        h.rope_freq_base   = 10000.0f; // unused (no_rope=true), just a harmless default

        h.ssm_d_state = kv_int(ctx, pfx("ssm.state_size").c_str());
        h.ssm_d_conv  = kv_int(ctx, pfx("ssm.conv_kernel").c_str());
        h.ssm_n_group = kv_int(ctx, pfx("ssm.group_count").c_str());
        h.ssm_d_inner = kv_int(ctx, pfx("ssm.inner_size").c_str());
        h.ssm_n_head  = kv_int(ctx, pfx("ssm.time_step_rank").c_str()); // see hparams.ssm_n_head's comment

        h.qkv_fused = h.has_qkv_bias = h.has_qk_norm = h.ffn_fused = false; // unused by this arch
        h.n_ff = 0; // unused (per-layer n_ff_layer is what matters); load_model doesn't read this scalar for nemotron_h
    } else {
        h.n_head_kv = kv_int(ctx, pfx("attention.head_count_kv").c_str());
        h.n_ff      = kv_int(ctx, pfx("feed_forward_length").c_str());
        h.rope_freq_base = (float) kv_float(ctx, pfx("rope.freq_base").c_str());

        // attention width isn't always n_embd/n_head (qwen3 decouples them); prefer the explicit
        // key_length key when the model provides one
        std::string key_len_key = pfx("attention.key_length");
        h.n_embd_head = gguf_find_key(ctx, key_len_key.c_str()) >= 0
                            ? kv_int(ctx, key_len_key.c_str())
                            : h.n_embd / h.n_head;

        std::string rot_key = pfx("rope.dimension_count");
        h.n_rot = gguf_find_key(ctx, rot_key.c_str()) >= 0 ? kv_int(ctx, rot_key.c_str()) : h.n_embd_head;

        std::string ctx_orig_key = pfx("rope.scaling.original_context_length");
        h.has_rope_scaling = gguf_find_key(ctx, ctx_orig_key.c_str()) >= 0;
        if (h.has_rope_scaling) {
            h.n_ctx_orig       = kv_int(ctx, ctx_orig_key.c_str());
            h.rope_attn_factor = (float) kv_float(ctx, pfx("rope.scaling.attn_factor").c_str());
        } else {
            h.n_ctx_orig       = kv_int(ctx, pfx("context_length").c_str()); // unused when ext_factor=0
            h.rope_attn_factor = 1.0f;
        }

        h.qkv_fused    = (h.arch == arch_t::PHI3);
        h.has_qkv_bias = (h.arch == arch_t::QWEN2);
        h.has_qk_norm  = (h.arch == arch_t::QWEN3 || h.arch == arch_t::QWEN35);
        h.ffn_fused    = (h.arch == arch_t::PHI3);
        h.rope_neox    = (h.arch != arch_t::LLAMA); // llama uses "normal" interleaved-pairs rotation

        if (h.arch == arch_t::QWEN35) {
            // every block still has a normal attn_norm+mixer+ffn_norm+ffn structure (unlike
            // nemotron_h) -- only the mixer choice (attention vs Gated DeltaNet) varies per layer,
            // via this fixed-interval rule (see hparams::is_deltanet_layer).
            h.full_attention_interval = kv_int(ctx, pfx("full_attention_interval").c_str());
            h.dn_n_head   = kv_int(ctx, pfx("ssm.group_count").c_str());
            h.dn_head_dim = kv_int(ctx, pfx("ssm.state_size").c_str()); // confirmed via this
                            // checkpoint's own tensor shapes: attn_qkv's 6144-wide output ==
                            // 3 * dn_n_head * dn_head_dim exactly (3*16*128)
            h.dn_d_conv   = kv_int(ctx, pfx("ssm.conv_kernel").c_str());
        }
    }

    // not every tokenizer has a BOS token at all (qwen35: no bos_token_id key, and
    // tokenizer.ggml.add_bos_token is explicitly false) -- -1 means "none", callers must check
    // rather than assume a real id.
    h.bos_id = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id") >= 0
                   ? kv_int(ctx, "tokenizer.ggml.bos_token_id") : -1;
    h.eos_id = kv_int(ctx, "tokenizer.ggml.eos_token_id");

    if (gguf_find_key(ctx, "tokenizer.chat_template") >= 0) {
        h.supports_thinking = kv_string(ctx, "tokenizer.chat_template").find("<think>") != std::string::npos;
    }
    return h;
}

// ---- GPT-2 byte-level BPE: shared byte<->unicode table, then decode (detokenize) and
// encode (pretokenize + BPE merge) built on top of it ----

// bytes 33-126, 161-172, 174-255 map to themselves; every other byte (control chars, space,
// 127-160, 173) maps to a synthetic codepoint starting at 256, in byte order. This is the
// standard GPT-2 bytes_to_unicode() table, needed both directions: decoder for detok, encoder
// for tokenize.
static void build_byte_tables(std::array<std::string, 256> & encoder,
                               std::unordered_map<uint32_t, uint8_t> & decoder) {
    std::vector<int> bs;
    for (int b = 33;  b <= 126; b++) bs.push_back(b);
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);

    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!in_bs[b]) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); i++) {
        uint32_t cp = (uint32_t) cs[i];
        decoder[cp] = (uint8_t) bs[i];

        std::string enc;
        if (cp < 0x80) {
            enc.push_back((char) cp);
        } else if (cp < 0x800) {
            enc.push_back((char)(0xC0 | (cp >> 6)));
            enc.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            enc.push_back((char)(0xE0 | (cp >> 12)));
            enc.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            enc.push_back((char)(0x80 | (cp & 0x3F)));
        }
        encoder[(uint8_t) bs[i]] = enc;
    }
}

static std::vector<uint32_t> utf8_decode(const std::string & s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char) s[i];
        uint32_t cp;
        int len;
        if      ((c & 0x80) == 0x00) { cp = c;        len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else                         { cp = c;        len = 1; }
        for (int k = 1; k < len && i + k < s.size(); k++) {
            cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

// GPT-2 style pretokenizer: contractions, then a maximal run of letters/digits/other-symbols
// (each optionally prefixed by one leading space), then whitespace runs.
static bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
static bool is_digit_(unsigned char c) { return c >= '0' && c <= '9'; }
static bool is_space_(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

static std::vector<std::string> pretokenize(const std::string & text) {
    static const char * contractions[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
    std::vector<std::string> out;
    size_t i = 0, n = text.size();
    while (i < n) {
        bool matched = false;
        for (const char * c : contractions) {
            size_t len = strlen(c);
            if (i + len <= n && text.compare(i, len, c) == 0) {
                out.push_back(text.substr(i, len));
                i += len;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        size_t start = i;
        if (text[i] == ' ' && i + 1 < n && text[i + 1] != ' ') {
            i++; // a single leading space attaches to the run that follows
        }

        unsigned char c = (unsigned char) text[i];
        size_t j = i;
        if (is_space_(c)) {
            while (j < n && is_space_((unsigned char) text[j])) j++;
        } else if (is_digit_(c)) {
            while (j < n && is_digit_((unsigned char) text[j])) j++;
        } else if (is_letter(c)) {
            while (j < n && is_letter((unsigned char) text[j])) j++;
        } else {
            while (j < n && !is_space_((unsigned char) text[j]) && !is_digit_((unsigned char) text[j])
                          && !is_letter((unsigned char) text[j])) j++;
        }
        out.push_back(text.substr(start, j - start));
        i = j;
    }
    return out;
}

static bpe_ranks load_bpe(const gguf_context * ctx) {
    bpe_ranks b;
    int64_t id = kv_find(ctx, "tokenizer.ggml.merges");
    size_t n = gguf_get_arr_n(ctx, id);
    for (size_t i = 0; i < n; i++) {
        std::string merge = gguf_get_arr_str(ctx, id, i);
        size_t sp = merge.find(' ');
        if (sp == std::string::npos) continue;
        std::string key = merge.substr(0, sp) + "\x01" + merge.substr(sp + 1);
        b.rank[key] = (int) i;
    }
    return b;
}

// merges byte-encoded symbols for one pretoken chunk until no adjacent pair has a known rank
static std::vector<std::string> bpe_merge(const std::vector<std::string> & symbols_in,
                                           const bpe_ranks & bpe) {
    std::vector<std::string> symbols = symbols_in;
    while (symbols.size() > 1) {
        int best_rank = -1;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            auto it = bpe.rank.find(symbols[i] + "\x01" + symbols[i + 1]);
            if (it != bpe.rank.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank < 0) break;
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
}

std::string vocab::detok(const std::vector<int32_t> & ids) const {
    std::string raw;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= tokens.size()) continue;
        for (uint32_t cp : utf8_decode(tokens[id])) {
            auto it = byte_decoder.find(cp);
            if (it != byte_decoder.end()) {
                raw.push_back((char) it->second);
            }
        }
    }
    return raw;
}

std::vector<int32_t> vocab::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    for (const std::string & chunk : pretokenize(text)) {
        std::vector<std::string> symbols;
        symbols.reserve(chunk.size());
        for (unsigned char b : chunk) {
            symbols.push_back(byte_encoder[b]);
        }
        for (const std::string & piece : bpe_merge(symbols, bpe)) {
            auto it = token_to_id.find(piece);
            if (it != token_to_id.end()) {
                ids.push_back(it->second);
            } else {
                // shouldn't happen for a well-formed vocab, but fail soft rather than crash
                fprintf(stderr, "warning: no vocab id for BPE piece '%s'\n", piece.c_str());
            }
        }
    }
    return ids;
}

vocab load_vocab(const gguf_context * ctx) {
    vocab v;
    build_byte_tables(v.byte_encoder, v.byte_decoder);
    v.bpe = load_bpe(ctx);
    int64_t id = kv_find(ctx, "tokenizer.ggml.tokens");
    size_t n = gguf_get_arr_n(ctx, id);
    v.tokens.reserve(n);
    for (size_t i = 0; i < n; i++) {
        v.tokens.emplace_back(gguf_get_arr_str(ctx, id, i));
        v.token_to_id[v.tokens.back()] = (int32_t) i;
    }
    return v;
}

int32_t vocab::special(const char * tag) const {
    auto it = token_to_id.find(tag);
    if (it == token_to_id.end()) {
        fprintf(stderr, "error: model vocab has no special token '%s'\n", tag);
        exit(1);
    }
    return it->second;
}

// ---- chat templates (one hardcoded format per architecture, not a Jinja engine -- see model.h) ----

static void append_encoded(const vocab & vc, std::vector<int32_t> & out, const std::string & text) {
    std::vector<int32_t> enc = vc.encode(text);
    out.insert(out.end(), enc.begin(), enc.end());
}

void append_chat_turn(const hparams & hp, const vocab & vc, std::vector<int32_t> & out,
                       const std::string & role, const std::string & text) {
    switch (hp.arch) {
        case arch_t::PHI3:
            out.push_back(vc.special(("<|" + role + "|>").c_str()));
            append_encoded(vc, out, text);
            out.push_back(vc.special("<|end|>"));
            break;
        case arch_t::LLAMA: // "<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"
            out.push_back(vc.special("<|start_header_id|>"));
            append_encoded(vc, out, role);
            out.push_back(vc.special("<|end_header_id|>"));
            append_encoded(vc, out, "\n\n");
            append_encoded(vc, out, text);
            out.push_back(vc.special("<|eot_id|>"));
            break;
        case arch_t::QWEN2: // ChatML: "<|im_start|>{role}\n{text}<|im_end|>\n"
        case arch_t::QWEN3:
        default:
            out.push_back(vc.special("<|im_start|>"));
            append_encoded(vc, out, role + "\n");
            append_encoded(vc, out, text);
            out.push_back(vc.special("<|im_end|>"));
            append_encoded(vc, out, "\n");
            break;
    }
}

// exact token if the vocab registers it (reasoning models virtually always do, for "<think>"/
// "</think>"), otherwise falls back to plain BPE-encoded text -- correct either way, just
// possibly non-canonical if a model happens not to register it as its own token.
static void append_special_or_text(const vocab & vc, std::vector<int32_t> & out, const std::string & tag) {
    auto it = vc.token_to_id.find(tag);
    if (it != vc.token_to_id.end()) out.push_back(it->second);
    else append_encoded(vc, out, tag);
}

void append_generation_prompt(const hparams & hp, const vocab & vc, std::vector<int32_t> & out,
                               bool enable_thinking) {
    switch (hp.arch) {
        case arch_t::PHI3:
            out.push_back(vc.special("<|assistant|>"));
            break;
        case arch_t::LLAMA:
            out.push_back(vc.special("<|start_header_id|>"));
            append_encoded(vc, out, "assistant");
            out.push_back(vc.special("<|end_header_id|>"));
            append_encoded(vc, out, "\n\n");
            break;
        case arch_t::QWEN2:
        case arch_t::QWEN3:
        default:
            out.push_back(vc.special("<|im_start|>"));
            append_encoded(vc, out, "assistant\n");
            break;
    }
    if (hp.supports_thinking && !enable_thinking) {
        // forces the model to skip reasoning -- matches how these models' own templates handle
        // enable_thinking=false (confirmed against qwen35's and nemotron_h's actual jinja)
        append_special_or_text(vc, out, "<think>");
        append_encoded(vc, out, "\n\n");
        append_special_or_text(vc, out, "</think>");
        append_encoded(vc, out, "\n\n");
    }
}

int32_t turn_end_token(const hparams & hp, const vocab & vc) {
    switch (hp.arch) {
        case arch_t::PHI3:  return vc.special("<|end|>");
        case arch_t::LLAMA: return vc.special("<|eot_id|>");
        case arch_t::QWEN2:
        case arch_t::QWEN3:
        default:             return vc.special("<|im_end|>");
    }
}

// ---- model weights ----

static struct ggml_tensor * must_get(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "error: missing tensor '%s'\n", name.c_str());
        exit(1);
    }
    return t;
}

model load_model(struct ggml_context * wctx, const hparams & hp) {
    model m;
    m.token_embd  = must_get(wctx, "token_embd.weight");
    m.output_norm = must_get(wctx, "output_norm.weight");
    // tied vs untied output varies by model *size* within an architecture (small variants often
    // tie, larger ones often don't), so this is decided by tensor presence, not an hparams flag
    struct ggml_tensor * output_w = ggml_get_tensor(wctx, "output.weight");
    m.output = output_w ? output_w : m.token_embd;

    if (hp.has_rope_scaling) {
        m.rope_factors = must_get(wctx, "rope_factors_short.weight"); // phi3 LongRoPE
    }

    m.layers.resize(hp.n_layer);

    if (hp.arch == arch_t::NEMOTRON_H) {
        // one mixer per block (attn XOR ssm XOR ffn, never combined -- see layer_kind_t), and a
        // single pre-mixer norm (attn_norm) rather than the usual two-norm attn+ffn pair -- this
        // gguf genuinely has no ffn_norm tensor at all, confirmed via inspect, not an oversight.
        for (int64_t il = 0; il < hp.n_layer; il++) {
            std::string p = "blk." + std::to_string(il) + ".";
            layer_weights & l = m.layers[il];
            l.attn_norm = must_get(wctx, p + "attn_norm.weight");
            switch (hp.layer_kind(il)) {
                case layer_kind_t::ATTN:
                    l.attn_q      = must_get(wctx, p + "attn_q.weight");
                    l.attn_k      = must_get(wctx, p + "attn_k.weight");
                    l.attn_v      = must_get(wctx, p + "attn_v.weight");
                    l.attn_output = must_get(wctx, p + "attn_output.weight");
                    break;
                case layer_kind_t::SSM:
                    l.ssm_in       = must_get(wctx, p + "ssm_in.weight");
                    l.ssm_conv1d_w = must_get(wctx, p + "ssm_conv1d.weight");
                    l.ssm_conv1d_b = must_get(wctx, p + "ssm_conv1d.bias");
                    l.ssm_dt_b     = must_get(wctx, p + "ssm_dt.bias");
                    l.ssm_a        = must_get(wctx, p + "ssm_a");
                    l.ssm_d        = must_get(wctx, p + "ssm_d");
                    l.ssm_out      = must_get(wctx, p + "ssm_out.weight");
                    l.ssm_norm     = ggml_get_tensor(wctx, (p + "ssm_norm.weight").c_str()); // optional
                    break;
                case layer_kind_t::FFN:
                    l.ffn_up   = must_get(wctx, p + "ffn_up.weight");
                    l.ffn_down = must_get(wctx, p + "ffn_down.weight");
                    break;
            }
        }
        return m;
    }

    if (hp.arch == arch_t::QWEN35) {
        // every block has attn_norm + a second pre-FFN norm (named "post_attention_norm.weight"
        // here, NOT "ffn_norm.weight" like every other supported arch -- confirmed via inspect)
        // + FFN (always separate gate/up, ChatML-family style). Only the mixer varies per layer:
        // delta-net blocks have no attn_output.weight at all (ssm_out is their own output
        // projection instead), attention blocks have no ssm_*/attn_gate tensors.
        for (int64_t il = 0; il < hp.n_layer; il++) {
            std::string p = "blk." + std::to_string(il) + ".";
            layer_weights & l = m.layers[il];
            l.attn_norm = must_get(wctx, p + "attn_norm.weight");
            l.ffn_norm  = must_get(wctx, p + "post_attention_norm.weight");
            l.ffn_gate  = must_get(wctx, p + "ffn_gate.weight");
            l.ffn_up    = must_get(wctx, p + "ffn_up.weight");
            l.ffn_down  = must_get(wctx, p + "ffn_down.weight");

            if (hp.is_deltanet_layer(il)) {
                l.attn_qkv     = must_get(wctx, p + "attn_qkv.weight");
                l.ssm_alpha    = must_get(wctx, p + "ssm_alpha.weight");
                l.ssm_beta     = must_get(wctx, p + "ssm_beta.weight");
                l.ssm_conv1d_w = must_get(wctx, p + "ssm_conv1d.weight"); // no bias tensor exists here
                l.ssm_dt_b     = must_get(wctx, p + "ssm_dt.bias");
                l.ssm_a        = must_get(wctx, p + "ssm_a");
                l.ssm_norm     = must_get(wctx, p + "ssm_norm.weight");
                l.ssm_out      = must_get(wctx, p + "ssm_out.weight");
                l.attn_gate    = must_get(wctx, p + "attn_gate.weight");
            } else {
                l.attn_q      = must_get(wctx, p + "attn_q.weight");
                l.attn_k      = must_get(wctx, p + "attn_k.weight");
                l.attn_v      = must_get(wctx, p + "attn_v.weight");
                l.attn_output = must_get(wctx, p + "attn_output.weight");
                l.attn_q_norm = must_get(wctx, p + "attn_q_norm.weight");
                l.attn_k_norm = must_get(wctx, p + "attn_k_norm.weight");
            }
        }
        return m;
    }

    for (int64_t il = 0; il < hp.n_layer; il++) {
        std::string p = "blk." + std::to_string(il) + ".";
        layer_weights & l = m.layers[il];
        l.attn_norm   = must_get(wctx, p + "attn_norm.weight");
        l.attn_output = must_get(wctx, p + "attn_output.weight");
        l.ffn_norm    = must_get(wctx, p + "ffn_norm.weight");
        l.ffn_down    = must_get(wctx, p + "ffn_down.weight");

        if (hp.qkv_fused) {
            l.attn_qkv = must_get(wctx, p + "attn_qkv.weight");
        } else {
            l.attn_q = must_get(wctx, p + "attn_q.weight");
            l.attn_k = must_get(wctx, p + "attn_k.weight");
            l.attn_v = must_get(wctx, p + "attn_v.weight");
            if (hp.has_qkv_bias) {
                l.attn_q_bias = must_get(wctx, p + "attn_q.bias");
                l.attn_k_bias = must_get(wctx, p + "attn_k.bias");
                l.attn_v_bias = must_get(wctx, p + "attn_v.bias");
            }
            if (hp.has_qk_norm) {
                l.attn_q_norm = must_get(wctx, p + "attn_q_norm.weight");
                l.attn_k_norm = must_get(wctx, p + "attn_k_norm.weight");
            }
        }

        if (hp.ffn_fused) {
            l.ffn_up_gate = must_get(wctx, p + "ffn_up.weight");
        } else {
            l.ffn_gate = must_get(wctx, p + "ffn_gate.weight");
            l.ffn_up   = must_get(wctx, p + "ffn_up.weight");
        }
    }
    return m;
}

weights_store load_weights_filtered(const char * path, ggml_backend_t backend,
                                     const std::vector<std::string> & prefixes,
                                     struct gguf_context ** out_gguf) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx || !meta_ctx) {
        fprintf(stderr, "error: failed to load '%s'\n", path);
        exit(1);
    }

    int64_t n_tensors = gguf_get_n_tensors(gctx);
    auto wanted = [&](const std::string & name) {
        if (prefixes.empty()) return true;
        for (const std::string & p : prefixes) {
            if (name.compare(0, p.size(), p) == 0) return true;
        }
        return false;
    };

    // meta_ctx (no_alloc) holds descriptors for every tensor in the file; ggml has no API to strip
    // unwanted ones out of it, and ggml_backend_alloc_ctx_tensors() always allocates for the whole
    // context. So build a second, smaller context containing only the tensors we actually want,
    // and allocate/read from that instead -- meta_ctx is freed right after, only used to discover
    // names/shapes and (via gctx) file offsets.
    int64_t n_wanted = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        if (wanted(gguf_get_tensor_name(gctx, i))) n_wanted++;
    }
    struct ggml_init_params ip = {
        /*.mem_size   =*/ n_wanted * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * wctx = ggml_init(ip);

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gctx, i);
        if (!wanted(name)) continue;
        struct ggml_tensor * src = ggml_get_tensor(meta_ctx, name);
        struct ggml_tensor * t = ggml_dup_tensor(wctx, src);
        ggml_set_name(t, name);
    }

    weights_store ws;
    ws.ctx = wctx;
    ws.buffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!ws.buffer) {
        fprintf(stderr, "error: failed to allocate backend memory for weights from '%s'\n", path);
        exit(1);
    }

    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: failed to reopen '%s' to read tensor data\n", path);
        exit(1);
    }
    size_t data_base = gguf_get_data_offset(gctx);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gctx, i);
        if (!wanted(name)) continue;
        struct ggml_tensor * t = ggml_get_tensor(wctx, name);
        size_t size = ggml_nbytes(t);
        size_t offset = data_base + gguf_get_tensor_offset(gctx, i);
        buf.resize(size);
        if (fseek(f, (long) offset, SEEK_SET) != 0 || fread(buf.data(), 1, size, f) != size) {
            fprintf(stderr, "error: failed to read tensor '%s' from '%s'\n", name, path);
            exit(1);
        }
        ggml_backend_tensor_set(t, buf.data(), 0, size);
    }
    fclose(f);

    *out_gguf = gctx;
    ggml_free(meta_ctx);
    return ws;
}

weights_store load_weights(const char * path, ggml_backend_t backend, struct gguf_context ** out_gguf) {
    return load_weights_filtered(path, backend, {}, out_gguf);
}

ggml_backend_t init_backend(const std::string & name, int n_threads) {
    if (name == "cpu") {
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, n_threads);
        return backend;
    }
    if (name == "vulkan") {
        if (ggml_backend_vk_get_device_count() == 0) {
            fprintf(stderr, "error: backend 'vulkan' requested but no Vulkan device is available\n");
            exit(1);
        }
        char desc[256];
        ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
        fprintf(stderr, "vulkan device 0: %s\n", desc);
        return ggml_backend_vk_init(0);
    }
    fprintf(stderr, "error: unknown backend '%s' (expected cpu or vulkan)\n", name.c_str());
    exit(1);
}

// ---- persistent KV cache ----

kv_cache init_kv_cache(const hparams & hp, ggml_backend_t backend) {
    kv_cache kv;
    const int64_t head_dim = hp.n_embd_head;

    size_t buf_size = ggml_tensor_overhead() * (size_t)(hp.n_layer * 2 + 8);
    struct ggml_init_params ip = { buf_size, nullptr, /*.no_alloc =*/ true };
    kv.ctx = ggml_init(ip);

    kv.k.resize(hp.n_layer);
    kv.v.resize(hp.n_layer);
    for (int64_t il = 0; il < hp.n_layer; il++) {
        // nemotron_h: most layers aren't attention at all -- size those slots minimally (1) rather
        // than 0 (ggml doesn't allow zero-sized dims) since they're simply never written/read.
        int64_t n_head_kv = hp.arch == arch_t::NEMOTRON_H
                                 ? std::max<int64_t>(1, hp.n_head_kv_layer[il])
                                 : hp.n_head_kv;
        kv.k[il] = ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F16, head_dim, n_head_kv, N_CTX_MAX);
        kv.v[il] = ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F16, head_dim, n_head_kv, N_CTX_MAX);
    }

    kv.buffer = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buffer) {
        fprintf(stderr, "error: failed to allocate KV cache (%lld positions x %lld layers)\n",
                (long long) N_CTX_MAX, (long long) hp.n_layer);
        exit(1);
    }
    return kv;
}

// ---- nemotron_h's Mamba2 recurrent state (conv window + SSM state), single-sequence only ----

ssm_state init_ssm_state(const hparams & hp, ggml_backend_t backend) {
    ssm_state ss;
    if (hp.arch != arch_t::NEMOTRON_H && hp.arch != arch_t::QWEN35) return ss; // no-op otherwise

    // per-layer conv-window width and state shape differ by arch/mixer: nemotron_h's Mamba2 state
    // is rectangular [d_state,head_dim,n_head] (see ggml_ssm_scan's own shape contract), qwen35's
    // Gated DeltaNet state is a square [head_dim,head_dim,n_head] outer-product accumulator (see
    // ggml_gated_delta_net's) -- computed once per layer below rather than assuming uniform sizing.
    auto layer_is_recurrent = [&](int64_t il) {
        if (hp.arch == arch_t::NEMOTRON_H) return hp.layer_kind(il) == layer_kind_t::SSM;
        return hp.is_deltanet_layer(il);
    };
    auto conv_width = [&]() {
        return hp.arch == arch_t::NEMOTRON_H
                   ? hp.ssm_d_inner + 2 * hp.ssm_n_group * hp.ssm_d_state // nemotron_h: xBC width
                   : 3 * hp.dn_n_head * hp.dn_head_dim;                  // qwen35: fused q/k/v width
    };
    auto conv_kernel = [&]() { return hp.arch == arch_t::NEMOTRON_H ? hp.ssm_d_conv : hp.dn_d_conv; };
    auto state_dims = [&](int64_t & d0, int64_t & d1, int64_t & d2) {
        if (hp.arch == arch_t::NEMOTRON_H) {
            d0 = hp.ssm_d_state; d1 = hp.ssm_d_inner / hp.ssm_n_head; d2 = hp.ssm_n_head;
        } else {
            d0 = hp.dn_head_dim; d1 = hp.dn_head_dim; d2 = hp.dn_n_head;
        }
    };

    const int64_t d_conv_w = conv_width();
    int64_t s0, s1, s2;
    state_dims(s0, s1, s2);

    size_t buf_size = ggml_tensor_overhead() * (size_t)(hp.n_layer * 2 + 8);
    struct ggml_init_params ip = { buf_size, nullptr, /*.no_alloc =*/ true };
    ss.ctx = ggml_init(ip);

    ss.ids = ggml_new_tensor_1d(ss.ctx, GGML_TYPE_I32, 1); // always {0}: single sequence, slot 0
    ss.conv.resize(hp.n_layer, nullptr);
    ss.state.resize(hp.n_layer, nullptr);
    for (int64_t il = 0; il < hp.n_layer; il++) {
        if (!layer_is_recurrent(il)) continue;
        ss.conv[il]  = ggml_new_tensor_3d(ss.ctx, GGML_TYPE_F32, conv_kernel() - 1, d_conv_w, 1);
        ss.state[il] = ggml_new_tensor_4d(ss.ctx, GGML_TYPE_F32, s0, s1, s2, 1);
    }

    ss.buffer = ggml_backend_alloc_ctx_tensors(ss.ctx, backend);
    if (!ss.buffer) {
        fprintf(stderr, "error: failed to allocate %s recurrent state\n", hp.arch_name.c_str());
        exit(1);
    }
    // conv/state start at zero (matches Mamba2/DeltaNet's own zero-initial-state convention); ids
    // is the constant {0} throughout this loader's lifetime (never batches multiple sequences).
    // conv and state tensors can have very different byte sizes -- size the zero buffer for the
    // larger of the two, since it's reused (via ggml_nbytes-bounded copies) for both.
    size_t conv_bytes  = (size_t)(conv_kernel() - 1) * d_conv_w * ggml_type_size(GGML_TYPE_F32);
    size_t state_bytes = (size_t) s0 * s1 * s2 * ggml_type_size(GGML_TYPE_F32);
    std::vector<uint8_t> zero(std::max(conv_bytes, state_bytes), 0);
    for (int64_t il = 0; il < hp.n_layer; il++) {
        if (ss.conv[il])  ggml_backend_tensor_set(ss.conv[il],  zero.data(), 0, ggml_nbytes(ss.conv[il]));
        if (ss.state[il]) ggml_backend_tensor_set(ss.state[il], zero.data(), 0, ggml_nbytes(ss.state[il]));
    }
    int32_t zero_id = 0;
    ggml_backend_tensor_set(ss.ids, &zero_id, 0, sizeof(zero_id));
    return ss;
}

void free_ssm_state(ssm_state & ss) {
    if (ss.buffer) ggml_backend_buffer_free(ss.buffer);
    if (ss.ctx)    ggml_free(ss.ctx);
    ss.buffer = nullptr;
    ss.ctx = nullptr;
}

// ---- graph construction: processes only the newest tokens, attending over the KV cache ----

struct built_graph {
    struct ggml_cgraph * gf;
    struct ggml_tensor * tokens;    // I32 [n_new] input
    struct ggml_tensor * positions; // I32 [n_new] input (absolute positions: n_past..n_past+n_new-1)
    struct ggml_tensor * mask;      // F32 [n_kv, n_new] input
    struct ggml_tensor * logits;    // F32 [n_vocab, n_new] output
};

static struct ggml_tensor * rms_norm_mul(struct ggml_context * ctx, struct ggml_tensor * x,
                                          struct ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Builds Q/K/V for one layer from the post-attn-norm input `x`, as [head_dim, n_head(_kv), n_new]
// tensors ready for RoPE. Branches on hparams::qkv_fused: phi3 has one fused attn_qkv weight
// (split via strided views, since mul_mat's single output can't be reshaped directly), qwen2 has
// separate attn_q/k/v (+ bias) weights (each a fresh contiguous mul_mat output, so reshape works).
// n_head_kv is passed explicitly rather than read from hp: nemotron_h's varies per layer (the
// scalar hp.n_head_kv is meaningless for that arch -- see kv_cache's own per-layer sizing).
static void build_qkv(struct ggml_context * ctx, const hparams & hp, const layer_weights & l,
                       struct ggml_tensor * x, int64_t n_new, int64_t head_dim, int64_t n_head_kv,
                       struct ggml_tensor ** out_Q, struct ggml_tensor ** out_K,
                       struct ggml_tensor ** out_V) {
    const size_t es = sizeof(float);
    if (hp.qkv_fused) {
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, l.attn_qkv, x); // [n_embd + 2*n_embd_kv, n_new]
        *out_Q = ggml_view_3d(ctx, qkv, head_dim, hp.n_head, n_new,
                               head_dim * es, qkv->nb[1], 0);
        *out_K = ggml_view_3d(ctx, qkv, head_dim, n_head_kv, n_new,
                               head_dim * es, qkv->nb[1], hp.n_head * head_dim * es);
        *out_V = ggml_view_3d(ctx, qkv, head_dim, n_head_kv, n_new,
                               head_dim * es, qkv->nb[1], (hp.n_head + n_head_kv) * head_dim * es);
    } else {
        struct ggml_tensor * q = ggml_mul_mat(ctx, l.attn_q, x); // [n_head*head_dim, n_new]
        struct ggml_tensor * k = ggml_mul_mat(ctx, l.attn_k, x); // [n_head_kv*head_dim, n_new]
        struct ggml_tensor * v = ggml_mul_mat(ctx, l.attn_v, x);
        if (hp.has_qkv_bias) {
            q = ggml_add(ctx, q, l.attn_q_bias);
            k = ggml_add(ctx, k, l.attn_k_bias);
            v = ggml_add(ctx, v, l.attn_v_bias);
        }
        *out_Q = ggml_reshape_3d(ctx, q, head_dim, hp.n_head, n_new);
        *out_K = ggml_reshape_3d(ctx, k, head_dim, n_head_kv, n_new);
        *out_V = ggml_reshape_3d(ctx, v, head_dim, n_head_kv, n_new);
    }
}

// SwiGLU FFN activation (silu(gate) * up), before the down projection. Branches on
// hparams::ffn_fused: phi3 packs gate+up into one ffn_up weight (split via a strided view),
// qwen2 has separate ffn_gate/ffn_up weights.
static struct ggml_tensor * build_ffn_act(struct ggml_context * ctx, const hparams & hp,
                                           const layer_weights & l, struct ggml_tensor * y,
                                           int64_t n_new) {
    struct ggml_tensor * gate;
    struct ggml_tensor * up;
    if (hp.ffn_fused) {
        struct ggml_tensor * gate_up = ggml_mul_mat(ctx, l.ffn_up_gate, y); // [2*n_ff, n_new]
        gate = ggml_view_2d(ctx, gate_up, hp.n_ff, n_new, gate_up->nb[1], 0);
        up   = ggml_view_2d(ctx, gate_up, hp.n_ff, n_new, gate_up->nb[1], hp.n_ff * sizeof(float));
    } else {
        gate = ggml_mul_mat(ctx, l.ffn_gate, y);
        up   = ggml_mul_mat(ctx, l.ffn_up, y);
    }
    return ggml_mul(ctx, ggml_silu(ctx, gate), up);
}

// nemotron_h's Mamba2 mixer. Recipe (op order, which tensor gets which transform) verified against
// llama.cpp's own build_mamba2_layer, not guessed -- see layer_weights::ssm_* field comments and
// the model.h ssm_state comment for what's cross-checked and why it matters (silent wrong-but-
// plausible output on a mistake here, same failure mode as the earlier llama RoPE-type bug).
static struct ggml_tensor * build_mamba2_layer(struct ggml_context * ctx, struct ggml_cgraph * gf,
                                                 const hparams & hp, const layer_weights & l,
                                                 const ssm_state & ss, int64_t il,
                                                 struct ggml_tensor * x, int64_t n_new) {
    const int64_t d_state  = hp.ssm_d_state;
    const int64_t d_conv   = hp.ssm_d_conv;
    const int64_t n_group  = hp.ssm_n_group;
    const int64_t d_inner  = hp.ssm_d_inner;
    const int64_t n_head   = hp.ssm_n_head;
    const int64_t head_dim = d_inner / n_head;
    const int64_t d_bc     = n_group * d_state;
    const int64_t d_xbc    = d_inner + 2 * d_bc;
    const size_t  es       = sizeof(float);

    struct ggml_tensor * proj = ggml_mul_mat(ctx, l.ssm_in, x); // [d_inner + d_xbc + n_head, n_new]

    // split order confirmed against upstream: z, then xBC, then dt (matches this checkpoint's own
    // ssm_in output width exactly: d_inner + d_xbc + n_head)
    struct ggml_tensor * z   = ggml_view_2d(ctx, proj, d_inner, n_new, proj->nb[1], 0);
    struct ggml_tensor * xBC = ggml_view_2d(ctx, proj, d_xbc, n_new, proj->nb[1], d_inner * es);
    struct ggml_tensor * dt  = ggml_view_2d(ctx, proj, n_head, n_new, proj->nb[1], (d_inner + d_xbc) * es);

    // causal conv1d: concat this layer's rolling conv-state (previous d_conv-1 steps, zero-init)
    // with the new tokens' xBC transposed to [time, channel], run ggml_ssm_conv, add bias, then
    // silu the WHOLE xBC block (not just the x portion -- confirmed against upstream: x/B/C all
    // get silu'd together before being split apart)
    struct ggml_tensor * xBC_t   = ggml_cont(ctx, ggml_transpose(ctx, xBC)); // [n_new, d_xbc]
    struct ggml_tensor * conv_in = ggml_concat(ctx, ss.conv[il], xBC_t, 0);  // [d_conv-1+n_new, d_xbc]
    conv_in = ggml_reshape_3d(ctx, conv_in, d_conv - 1 + n_new, d_xbc, 1);

    // the last (d_conv-1) columns of conv_in become the next step's rolling conv-state -- written
    // into the graph now, same pattern as the KV-cache K/V writes elsewhere in this file (ggml's
    // topological execution guarantees this cpy runs after everything that reads the OLD
    // ss.conv[il]/ss.state[il], since those reads are upstream dependencies of this cpy's own input)
    struct ggml_tensor * new_conv = ggml_view_3d(ctx, conv_in, d_conv - 1, d_xbc, 1,
                                                  conv_in->nb[1], conv_in->nb[2], (size_t) n_new * conv_in->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_conv, ss.conv[il]));

    struct ggml_tensor * xBC_conv = ggml_ssm_conv(ctx, conv_in, l.ssm_conv1d_w); // [d_xbc, n_new, 1]
    xBC_conv = ggml_add(ctx, xBC_conv, l.ssm_conv1d_b);
    xBC_conv = ggml_silu(ctx, xBC_conv);

    struct ggml_tensor * xs = ggml_view_2d(ctx, xBC_conv, d_inner, n_new, xBC_conv->nb[1], 0);
    struct ggml_tensor * B  = ggml_view_2d(ctx, xBC_conv, d_bc, n_new, xBC_conv->nb[1], d_inner * es);
    struct ggml_tensor * C  = ggml_view_2d(ctx, xBC_conv, d_bc, n_new, xBC_conv->nb[1], (d_inner + d_bc) * es);

    struct ggml_tensor * x4  = ggml_reshape_4d(ctx, ggml_cont(ctx, xs), head_dim, n_head, n_new, 1);
    struct ggml_tensor * B4  = ggml_reshape_4d(ctx, ggml_cont(ctx, B), d_state, n_group, n_new, 1);
    struct ggml_tensor * C4  = ggml_reshape_4d(ctx, ggml_cont(ctx, C), d_state, n_group, n_new, 1);
    // dt bias-add produces a fresh contiguous tensor regardless of the view's own strides
    struct ggml_tensor * dt3 = ggml_reshape_3d(ctx, ggml_add(ctx, dt, l.ssm_dt_b), n_head, n_new, 1);

    // A used raw (no exp/negate) -- confirmed against upstream: this gguf's ssm_a is already
    // stored in its final usable (negative, per-head decay) form
    struct ggml_tensor * result = ggml_ssm_scan(ctx, ss.state[il], x4, dt3, l.ssm_a, B4, C4, ss.ids, 1);

    // result packs y (nelements(x4)) followed by the new recurrent state
    struct ggml_tensor * y = ggml_view_4d(ctx, result, head_dim, n_head, n_new, 1,
                                           head_dim * es, head_dim * n_head * es,
                                           head_dim * n_head * n_new * es, 0);
    struct ggml_tensor * new_state = ggml_view_4d(ctx, result, d_state, head_dim, n_head, 1,
                                                   d_state * es, d_state * head_dim * es,
                                                   d_state * head_dim * n_head * es,
                                                   ggml_nelements(x4) * es);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ss.state[il]));

    // D-skip residual, then gate with z (silu(z)*y via a fused GLU op), then optional grouped
    // RMSNorm -- this exact order (D-skip -> gate -> norm) confirmed against upstream; gating
    // before the norm, not after, is the detail most likely to get silently swapped by guessing
    struct ggml_tensor * d4 = ggml_reshape_4d(ctx, l.ssm_d, 1, n_head, 1, 1);
    y = ggml_add(ctx, y, ggml_mul(ctx, x4, d4));
    struct ggml_tensor * z4 = ggml_reshape_4d(ctx, ggml_cont(ctx, z), head_dim, n_head, n_new, 1);
    y = ggml_swiglu_split(ctx, z4, y);

    if (l.ssm_norm) {
        struct ggml_tensor * y_grp = ggml_reshape_4d(ctx, y, d_inner / n_group, n_group, n_new, 1);
        y_grp = ggml_mul(ctx, ggml_rms_norm(ctx, y_grp, hp.rms_eps), l.ssm_norm);
        y = ggml_reshape_3d(ctx, y_grp, d_inner, n_new, 1);
    } else {
        y = ggml_reshape_3d(ctx, y, d_inner, n_new, 1);
    }

    return ggml_mul_mat(ctx, l.ssm_out, ggml_reshape_2d(ctx, y, d_inner, n_new)); // [n_embd, n_new]
}

// qwen35's Gated DeltaNet mixer (linear-attention layers, see hparams::is_deltanet_layer). Op
// sequence and the exact q/k/v/g/beta/state contract verified against llama.cpp upstream's
// qwen3next.cpp AND this project's own vendored ggml_gated_delta_net CPU kernel directly (its
// exact use of g/beta -- exp(g) applied INSIDE the kernel, beta used raw with no internal sigmoid
// -- isn't obvious from the op's declaration alone, and getting it backwards would silently
// produce wrong-but-plausible output, same risk class as the Mamba2 mixer above).
static struct ggml_tensor * build_gated_delta_net_layer(struct ggml_context * ctx, struct ggml_cgraph * gf,
                                                          const hparams & hp, const layer_weights & l,
                                                          const ssm_state & ss, int64_t il,
                                                          struct ggml_tensor * x, int64_t n_new) {
    const int64_t n_head   = hp.dn_n_head;
    const int64_t head_dim = hp.dn_head_dim;
    const int64_t d_conv   = hp.dn_d_conv;
    const int64_t d_qk     = n_head * head_dim;
    const int64_t d_qkv    = 3 * d_qk;
    const size_t  es       = sizeof(float);

    struct ggml_tensor * qkv = ggml_mul_mat(ctx, l.attn_qkv, x); // [d_qkv, n_new]

    // causal conv1d over the whole fused q/k/v block (no bias tensor exists for this arch), then
    // silu -- same rolling-window-carry pattern as nemotron_h's Mamba2 mixer, applied to q/k/v
    // jointly instead of x/B/C
    struct ggml_tensor * qkv_t   = ggml_cont(ctx, ggml_transpose(ctx, qkv)); // [n_new, d_qkv]
    struct ggml_tensor * conv_in = ggml_concat(ctx, ss.conv[il], qkv_t, 0);  // [d_conv-1+n_new, d_qkv]
    conv_in = ggml_reshape_3d(ctx, conv_in, d_conv - 1 + n_new, d_qkv, 1);

    struct ggml_tensor * new_conv = ggml_view_3d(ctx, conv_in, d_conv - 1, d_qkv, 1,
                                                  conv_in->nb[1], conv_in->nb[2], (size_t) n_new * conv_in->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_conv, ss.conv[il]));

    struct ggml_tensor * qkv_conv = ggml_ssm_conv(ctx, conv_in, l.ssm_conv1d_w); // [d_qkv, n_new, 1]
    qkv_conv = ggml_silu(ctx, qkv_conv);

    struct ggml_tensor * q = ggml_view_2d(ctx, qkv_conv, d_qk, n_new, qkv_conv->nb[1], 0);
    struct ggml_tensor * k = ggml_view_2d(ctx, qkv_conv, d_qk, n_new, qkv_conv->nb[1], (size_t) d_qk * es);
    struct ggml_tensor * v = ggml_view_2d(ctx, qkv_conv, d_qk, n_new, qkv_conv->nb[1], (size_t) 2 * d_qk * es);

    // L2-norm on q/k specifically, not v -- confirmed against upstream. Per-HEAD, not over the
    // whole concatenated d_qk vector -- reshape to [head_dim,n_head,...] BEFORE normalizing, so
    // ggml_l2_norm (which normalizes over ne0) sees head_dim, not all heads mixed together.
    struct ggml_tensor * q4 = ggml_l2_norm(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, q), head_dim, n_head, n_new, 1), 1e-6f);
    struct ggml_tensor * k4 = ggml_l2_norm(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, k), head_dim, n_head, n_new, 1), 1e-6f);
    struct ggml_tensor * v4 = ggml_reshape_4d(ctx, ggml_cont(ctx, v), head_dim, n_head, n_new, 1);

    // g (decay gate): softplus(ssm_alpha(x) + dt_bias) * ssm_a, left RAW -- ggml_gated_delta_net's
    // own CPU kernel does expf(g) internally (unlike beta below, which it uses un-transformed)
    struct ggml_tensor * g = ggml_mul_mat(ctx, l.ssm_alpha, x); // [n_head, n_new]
    g = ggml_softplus(ctx, ggml_add(ctx, g, l.ssm_dt_b));
    g = ggml_mul(ctx, g, ggml_reshape_2d(ctx, l.ssm_a, n_head, 1)); // broadcasts over n_new
    g = ggml_reshape_4d(ctx, g, 1, n_head, n_new, 1);

    // beta (mixing rate): sigmoid(ssm_beta(x)) -- MUST be sigmoid'd here; the kernel uses it raw
    struct ggml_tensor * beta = ggml_sigmoid(ctx, ggml_mul_mat(ctx, l.ssm_beta, x)); // [n_head, n_new]
    beta = ggml_reshape_4d(ctx, beta, 1, n_head, n_new, 1);

    struct ggml_tensor * result = ggml_gated_delta_net(ctx, q4, k4, v4, g, beta, ss.state[il], 1);

    struct ggml_tensor * y = ggml_view_4d(ctx, result, head_dim, n_head, n_new, 1,
                                           head_dim * es, head_dim * n_head * es,
                                           head_dim * n_head * n_new * es, 0);
    struct ggml_tensor * new_state = ggml_view_4d(ctx, result, head_dim, head_dim, n_head, 1,
                                                   head_dim * es, head_dim * head_dim * es,
                                                   head_dim * head_dim * n_head * es,
                                                   (size_t) n_new * d_qk * es);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ss.state[il]));

    // gated norm: RMSNorm(y) * silu(gate) -- norm BEFORE gating here, opposite of the Mamba2
    // mixer's D-skip-then-gate-then-norm order above; confirmed against upstream's own description
    // ("output_norm = RMSNorm(recurrent_output) * SiLU(z)") -- getting this order backwards is
    // exactly the kind of silent, fluent-looking-but-wrong mistake this whole recipe was
    // cross-checked against real sources to avoid.
    y = rms_norm_mul(ctx, y, l.ssm_norm, hp.rms_eps);
    struct ggml_tensor * gate = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, l.attn_gate, x), head_dim, n_head, n_new, 1);
    y = ggml_mul(ctx, y, ggml_silu(ctx, gate));

    return ggml_mul_mat(ctx, l.ssm_out, ggml_reshape_2d(ctx, y, d_qk, n_new)); // [n_embd, n_new]
}

static built_graph build_graph(struct ggml_context * ctx, const model & m, const hparams & hp,
                                kv_cache & kv, ssm_state * ss, int64_t n_past, int64_t n_new,
                                bool use_flash_attn) {
    built_graph bg{};
    bg.gf = ggml_new_graph(ctx);

    const int64_t head_dim = hp.n_embd_head;
    const int64_t n_kv     = n_past + n_new;
    const float   kq_scale = 1.0f / sqrtf((float) head_dim);

    bg.tokens    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new);
    bg.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new);
    bg.mask      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_new);

    struct ggml_tensor * cur = ggml_get_rows(ctx, m.token_embd, bg.tokens); // [n_embd, n_new]

    // ext_factor=1 (LongRoPE/YaRN blending) only for phi3; plain RoPE otherwise (ext_factor=0
    // disables the blend entirely, per ggml_rope_ext's documented defaults)
    const float rope_ext_factor  = hp.has_rope_scaling ? 1.0f : 0.0f;
    const float rope_beta_fast   = hp.has_rope_scaling ? 32.0f : 0.0f;
    const float rope_beta_slow   = hp.has_rope_scaling ? 1.0f : 0.0f;

    for (int64_t il = 0; il < hp.n_layer; il++) {
        const layer_weights & l = m.layers[il];
        struct ggml_tensor * residual = cur;

        struct ggml_tensor * x = rms_norm_mul(ctx, cur, l.attn_norm, hp.rms_eps);

        const layer_kind_t kind = hp.layer_kind(il);

        if (kind == layer_kind_t::SSM) {
            // nemotron_h: SSM-only block, no separate FFN in the same block (single mixer+residual)
            cur = ggml_add(ctx, build_mamba2_layer(ctx, bg.gf, hp, l, *ss, il, x, n_new), residual);
            continue;
        }
        if (kind == layer_kind_t::FFN) {
            // nemotron_h: plain FFN-only block, ReLU^2 (not SwiGLU -- no gate tensor exists here),
            // confirmed against upstream
            struct ggml_tensor * up = ggml_mul_mat(ctx, l.ffn_up, x);
            up = ggml_sqr(ctx, ggml_relu(ctx, up));
            cur = ggml_add(ctx, ggml_mul_mat(ctx, l.ffn_down, up), residual);
            continue;
        }

        if (hp.is_deltanet_layer(il)) {
            // qwen35: Gated DeltaNet linear-attention mixer instead of standard attention -- the
            // block's FFN half below still runs normally afterward (unlike nemotron_h's SSM
            // blocks, which have no FFN at all).
            cur = ggml_add(ctx, build_gated_delta_net_layer(ctx, bg.gf, hp, l, *ss, il, x, n_new), residual);
        } else {
            // kind == ATTN: shared by every architecture (nemotron_h's and qwen35's attn-kind
            // blocks included). n_head_kv comes from this layer's own KV-cache tensor shape rather
            // than hp.n_head_kv -- identical to the scalar for the uniform-transformer archs, but
            // nemotron_h's attention blocks vary per layer (see init_kv_cache), so the cache
            // tensor is the actual source of truth for both.
            const int64_t n_head_kv = kv.k[il]->ne[1];

            struct ggml_tensor * Qcur;
            struct ggml_tensor * Kcur;
            struct ggml_tensor * Vcur;
            struct ggml_tensor * out_gate = nullptr; // qwen35 only, see below
            if (hp.arch == arch_t::QWEN35) {
                // qwen35's attention layers fuse an *output* gate into the same projection as Q
                // (attn_q.weight is 2*n_head*head_dim wide here, confirmed via this checkpoint's
                // own tensor shape). Verified against upstream: the split is NOT a block split
                // (all-Q-then-all-gate) -- it's per-head interleaved, i.e. reshape to
                // [head_dim*2, n_head, n_new] FIRST, then Q is the first head_dim of each head's
                // 2*head_dim slice and gate is the second head_dim -- getting this as a naive
                // first-half/second-half split instead (an earlier, wrong version of this code)
                // silently scrambled every head past the first, exactly the kind of fluent-but-
                // wrong failure this whole recipe was being cross-checked against upstream to avoid.
                struct ggml_tensor * q_raw = ggml_mul_mat(ctx, l.attn_q, x); // [2*n_head*head_dim, n_new]
                q_raw = ggml_reshape_4d(ctx, q_raw, head_dim * 2, hp.n_head, n_new, 1);
                struct ggml_tensor * q_only = ggml_view_4d(ctx, q_raw, head_dim, hp.n_head, n_new, 1,
                                                            q_raw->nb[1], q_raw->nb[2], q_raw->nb[3], 0);
                out_gate = ggml_view_4d(ctx, q_raw, head_dim, hp.n_head, n_new, 1,
                                         q_raw->nb[1], q_raw->nb[2], q_raw->nb[3],
                                         (size_t) head_dim * sizeof(float));
                Qcur = ggml_reshape_3d(ctx, ggml_cont(ctx, q_only), head_dim, hp.n_head, n_new);
                out_gate = ggml_cont_2d(ctx, out_gate, head_dim * hp.n_head, n_new); // matches cur's shape at the gating point below
                Kcur = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.attn_k, x), head_dim, n_head_kv, n_new);
                Vcur = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.attn_v, x), head_dim, n_head_kv, n_new);
            } else {
                build_qkv(ctx, hp, l, x, n_new, head_dim, n_head_kv, &Qcur, &Kcur, &Vcur);
            }

            if (hp.has_qk_norm) {
                // per-head RMSNorm: rms_norm operates over ne0, which is head_dim here, so this
                // normalizes each (head, token) vector independently -- exactly qwen3's QK-Norm
                Qcur = rms_norm_mul(ctx, Qcur, l.attn_q_norm, hp.rms_eps);
                Kcur = rms_norm_mul(ctx, Kcur, l.attn_k_norm, hp.rms_eps);
            }

            if (!hp.no_rope) {
                // nemotron_h's attention blocks apply NO positional encoding at all (verified
                // against llama.cpp upstream: rope_type reports NONE for this arch family) --
                // skip entirely rather than applying rope with some default, which would silently
                // scramble position information exactly like the earlier llama rope_neox bug did
                // the other way around.
                const int rope_mode = hp.rope_neox ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;
                Qcur = ggml_rope_ext(ctx, Qcur, bg.positions, m.rope_factors,
                                      (int) hp.n_rot, rope_mode, (int) hp.n_ctx_orig,
                                      hp.rope_freq_base, 1.0f, rope_ext_factor, hp.rope_attn_factor,
                                      rope_beta_fast, rope_beta_slow);
                Kcur = ggml_rope_ext(ctx, Kcur, bg.positions, m.rope_factors,
                                      (int) hp.n_rot, rope_mode, (int) hp.n_ctx_orig,
                                      hp.rope_freq_base, 1.0f, rope_ext_factor, hp.rope_attn_factor,
                                      rope_beta_fast, rope_beta_slow);
            }

            // write the newest K/V (post-RoPE for K) into the persistent cache at [n_past, n_past+n_new)
            struct ggml_tensor * k_dst = ggml_view_3d(ctx, kv.k[il], head_dim, n_head_kv, n_new,
                                                       kv.k[il]->nb[1], kv.k[il]->nb[2],
                                                       n_past * kv.k[il]->nb[2]);
            struct ggml_tensor * v_dst = ggml_view_3d(ctx, kv.v[il], head_dim, n_head_kv, n_new,
                                                       kv.v[il]->nb[1], kv.v[il]->nb[2],
                                                       n_past * kv.v[il]->nb[2]);
            ggml_build_forward_expand(bg.gf, ggml_cpy(ctx, Kcur, k_dst));
            ggml_build_forward_expand(bg.gf, ggml_cpy(ctx, Vcur, v_dst));

            struct ggml_tensor * K_full = ggml_view_3d(ctx, kv.k[il], head_dim, n_head_kv, n_kv,
                                                         kv.k[il]->nb[1], kv.k[il]->nb[2], 0);
            struct ggml_tensor * V_full = ggml_view_3d(ctx, kv.v[il], head_dim, n_head_kv, n_kv,
                                                         kv.v[il]->nb[1], kv.v[il]->nb[2], 0);

            struct ggml_tensor * Q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);   // [head_dim, n_new, n_head]
            struct ggml_tensor * K = ggml_permute(ctx, K_full, 0, 2, 1, 3); // [head_dim, n_kv, n_head_kv]

            // n_head*head_dim is the attn_output projection's *input* width, which isn't always
            // n_embd (qwen3 decouples them -- see hparams::n_embd_head)
            if (use_flash_attn) {
                // ggml_flash_attn_ext wants: q=[head_dim,n_new,n_head] (have it), k=[head_dim,n_kv,
                // n_head_kv] (have it), v=[head_dim,n_kv,n_head_kv] *not* transposed (unlike the
                // manual path below), mask as F16. Result is already permuted to [head_dim,n_head,n_new].
                struct ggml_tensor * V_flash = ggml_permute(ctx, V_full, 0, 2, 1, 3); // [head_dim, n_kv, n_head_kv]
                struct ggml_tensor * mask_f16 = ggml_cast(ctx, bg.mask, GGML_TYPE_F16);
                struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, Q, K, V_flash, mask_f16,
                                                                     kq_scale, 0.0f, 0.0f);
                cur = ggml_cont_2d(ctx, attn_out, hp.n_head * head_dim, n_new);
            } else {
                struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q); // [n_kv, n_new, n_head] (GQA-broadcast)
                KQ = ggml_soft_max_ext(ctx, KQ, bg.mask, kq_scale, 0.0f);

                struct ggml_tensor * V_trans = ggml_cont_3d(ctx,
                    ggml_permute(ctx, V_full, 1, 2, 0, 3), n_kv, head_dim, n_head_kv); // [n_kv, head_dim, n_head_kv]

                struct ggml_tensor * KQV = ggml_mul_mat(ctx, V_trans, KQ); // [head_dim, n_new, n_head]
                struct ggml_tensor * merged = ggml_permute(ctx, KQV, 0, 2, 1, 3); // [head_dim, n_head, n_new]
                cur = ggml_cont_2d(ctx, merged, hp.n_head * head_dim, n_new);
            }

            if (out_gate) {
                // qwen35: gated_attn = attn_out * sigmoid(gate) -- confirmed against upstream
                // ("gate = sigmoid(gate); gated_attn = attn_out * gate")
                cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, out_gate));
            }
            cur = ggml_mul_mat(ctx, l.attn_output, cur);
            cur = ggml_add(ctx, cur, residual);
        }

        if (hp.arch == arch_t::NEMOTRON_H) {
            // attn-kind nemotron_h blocks have no separate FFN (single mixer per block) -- done.
            continue;
        }

        residual = cur;
        struct ggml_tensor * y = rms_norm_mul(ctx, cur, l.ffn_norm, hp.rms_eps);
        struct ggml_tensor * ffn_act = build_ffn_act(ctx, hp, l, y, n_new);

        cur = ggml_mul_mat(ctx, l.ffn_down, ffn_act);
        cur = ggml_add(ctx, cur, residual);
    }

    cur = rms_norm_mul(ctx, cur, m.output_norm, hp.rms_eps);
    bg.logits = ggml_mul_mat(ctx, m.output, cur); // [n_vocab, n_new]

    ggml_build_forward_expand(bg.gf, bg.logits);
    return bg;
}

decode_session init_decode_session(ggml_backend_t backend, bool use_flash_attn) {
    decode_session ds;
    size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    ds.graph_buf.resize(buf_size);
    ds.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ds.use_flash_attn = use_flash_attn;
    return ds;
}

void free_decode_session(decode_session & ds) {
    ggml_gallocr_free(ds.galloc);
    ds.galloc = nullptr;
}

std::vector<float> forward_step(const model & m, const hparams & hp, kv_cache & kv,
                                 decode_session & ds, ggml_backend_t backend, int64_t n_past,
                                 const std::vector<int32_t> & new_tokens, int64_t n_vocab,
                                 ssm_state * ss) {
    int64_t n_new = (int64_t) new_tokens.size();
    int64_t n_kv  = n_past + n_new;

    struct ggml_init_params gip = {
        /*.mem_size   =*/ ds.graph_buf.size(),
        /*.mem_buffer =*/ ds.graph_buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * gctx_build = ggml_init(gip);

    built_graph bg = build_graph(gctx_build, m, hp, kv, ss, n_past, n_new, ds.use_flash_attn);

    // reused across calls: replans the (possibly differently-shaped) activation layout each
    // time rather than malloc/free-ing a fresh allocator every step
    if (!ggml_gallocr_alloc_graph(ds.galloc, bg.gf)) {
        fprintf(stderr, "error: graph allocation failed\n");
        exit(1);
    }

    ggml_backend_tensor_set(bg.tokens, new_tokens.data(), 0, n_new * sizeof(int32_t));

    // bg.positions only ever feeds ggml_rope_ext calls -- for hp.no_rope (nemotron_h), the graph
    // never references it at all, so ggml_gallocr never allocates it a backend buffer, and setting
    // it would abort in ggml_backend_tensor_set (an unallocated-tensor write, not a bounds issue).
    if (!hp.no_rope) {
        std::vector<int32_t> positions(n_new);
        for (int64_t i = 0; i < n_new; i++) positions[i] = (int32_t)(n_past + i);
        ggml_backend_tensor_set(bg.positions, positions.data(), 0, n_new * sizeof(int32_t));
    }

    std::vector<float> mask((size_t)(n_kv * n_new));
    for (int64_t q = 0; q < n_new; q++) {
        int64_t abs_q = n_past + q;
        for (int64_t k = 0; k < n_kv; k++) {
            mask[q * n_kv + k] = (k <= abs_q) ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(bg.mask, mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend, bg.gf);

    std::vector<float> last_logits(n_vocab);
    size_t last_col_offset = (size_t)(n_new - 1) * n_vocab * sizeof(float);
    ggml_backend_tensor_get(bg.logits, last_logits.data(), last_col_offset, n_vocab * sizeof(float));

    ggml_free(gctx_build); // frees only the ggml_context bookkeeping; ds.graph_buf is ours to reuse

    return last_logits;
}

// Prefills `tokens` (splitting into multiple forward_step() calls if that's faster -- see below),
// returning the same thing a single forward_step() call would: logits for the last token only.
//
// Why this exists: ggml-vulkan picks between three very different mul_mat strategies based on
// the batch size (the number of new tokens, "n"): n<=8 uses a dedicated, per-n-specialized
// dequant+dot mat-vec pipeline; n>~32 uses its "medium"/"large" tiled GEMM shader; in between (the
// exact cutoffs are internal to ggml-vulkan, empirically ~9..32 here) it falls back to the "small"
// tiled GEMM shader, whose fixed per-dispatch costs (workgroup-tile loading of src0 etc.) don't
// amortize over so few columns. Measured on this project's actual target (Mali-G615, no matrix
// cores, so no coopmat2 -- just the int-dot-product path), that middle zone is not a mild
// slowdown but a 3-5x cliff below *both* neighboring zones (e.g. one real run: pp8 7.1 tok/s,
// pp16 2.3 tok/s, pp32 4.5 tok/s, pp64 12.0 tok/s) -- and ordinary chat turns land in exactly that
// zone constantly (a short user message is very often 9-30 tokens). Splitting such a prefill into
// <=8-token forward_step() calls (each individually landing in the fast mat-vec zone, sequentially
// appended to the same KV cache -- forward_step() already only returns the last token's logits
// regardless of batch size, so this changes nothing observable except speed) measured at ~7 tok/s
// sustained, i.e. the fast zone's own throughput, instead of the 2-3 tok/s the single bad-zone
// dispatch would have gotten. Left as a single call for n<=8 (already the fast path) and n>32
// (already the fast tiled-GEMM path) -- chunking those would only add dispatch overhead for no
// benefit, and for large n would actively regress (16x 8-token calls sustain ~7 tok/s, well below
// one big call's 10-12 tok/s).
//
// Vulkan-only: `use_chunking` should be false for the CPU backend, whose mul_mat threading scales
// close to linearly with batch size (no equivalent shader-tile cliff), so splitting would just add
// per-chunk overhead for nothing.
std::vector<float> prefill(const model & m, const hparams & hp, kv_cache & kv, decode_session & ds,
                            ggml_backend_t backend, int64_t n_past, const std::vector<int32_t> & tokens,
                            int64_t n_vocab, bool use_chunking, ssm_state * ss) {
    constexpr size_t kFastChunk = 8; // ggml-vulkan's mul_mat_vec_max_cols -- see block comment above
    constexpr size_t kBadZoneMax = 32; // beyond this, a single dispatch is already in the fast tiled-GEMM zone

    if (!use_chunking || tokens.size() <= kFastChunk || tokens.size() > kBadZoneMax) {
        return forward_step(m, hp, kv, ds, backend, n_past, tokens, n_vocab, ss);
    }

    std::vector<float> last_logits;
    for (size_t off = 0; off < tokens.size(); off += kFastChunk) {
        size_t n = std::min(kFastChunk, tokens.size() - off);
        std::vector<int32_t> chunk(tokens.begin() + off, tokens.begin() + off + n);
        last_logits = forward_step(m, hp, kv, ds, backend, n_past + (int64_t) off, chunk, n_vocab, ss);
    }
    return last_logits;
}

int32_t argmax(const std::vector<float> & logits) {
    int32_t best = 0;
    float best_v = logits[0];
    for (size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = (int32_t) i; }
    }
    return best;
}

sampler init_sampler(const sampler_params & params) {
    sampler smp;
    smp.params = params;
    smp.rng.seed(params.seed != 0 ? params.seed : std::random_device{}());
    return smp;
}

int32_t sample(sampler & smp, std::vector<float> & logits, const std::vector<int32_t> & recent_tokens) {
    const sampler_params & p = smp.params;

    if (p.repeat_penalty != 1.0f && !recent_tokens.empty()) {
        size_t start = recent_tokens.size() > (size_t) p.repeat_last_n
                            ? recent_tokens.size() - (size_t) p.repeat_last_n : 0;
        for (size_t i = start; i < recent_tokens.size(); i++) {
            int32_t id = recent_tokens[i];
            if (id < 0 || (size_t) id >= logits.size()) continue;
            float & l = logits[id];
            l = l > 0.0f ? l / p.repeat_penalty : l * p.repeat_penalty;
        }
    }

    if (p.temp <= 0.0f) {
        return argmax(logits);
    }

    std::vector<int32_t> ids(logits.size());
    for (size_t i = 0; i < logits.size(); i++) ids[i] = (int32_t) i;

    size_t k = (p.top_k > 0 && (size_t) p.top_k < ids.size()) ? (size_t) p.top_k : ids.size();
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                       [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    ids.resize(k);

    float max_logit = logits[ids[0]];
    std::vector<float> probs(ids.size());
    float sum = 0.0f;
    for (size_t i = 0; i < ids.size(); i++) {
        probs[i] = expf((logits[ids[i]] - max_logit) / p.temp);
        sum += probs[i];
    }
    for (float & pr : probs) pr /= sum;

    // ids/probs are sorted descending (from the partial_sort above), so a running cumulative
    // sum directly gives the nucleus cutoff
    if (p.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cutoff = probs.size();
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (cum >= p.top_p) { cutoff = i + 1; break; }
        }
        ids.resize(cutoff);
        probs.resize(cutoff);
        float s = 0.0f;
        for (float pr : probs) s += pr;
        for (float & pr : probs) pr /= s;
    }

    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return ids[dist(smp.rng)];
}
