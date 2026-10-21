// Standalone OpenAI CLIP BPE tokenizer (encode only -- SDXL only ever needs to tokenize prompts,
// never detokenize). This is a *different* tokenizer from the GPT-2 byte-level BPE in model.h: same
// byte<->unicode trick, but CLIP lowercases + whitespace-normalizes first, splits with its own
// regex (single digits, not digit runs; no GPT-2 "leading space attaches to next run" rule), and
// suffixes the last byte-symbol of each word with "</w>" before merging. The SDXL gguf carries zero
// tokenizer metadata (see sdxl_vae.h's file header), so the merge table can't be read from the
// model file -- it's the standard, public, MIT-licensed OpenAI CLIP vocab (49408 tokens: 256 byte
// tokens + 256 byte+"</w>" tokens + 48894 BPE merges + <|startoftext|>/<|endoftext|>), embedded in
// clip_merges_data.h and identical across every SD1.x/SDXL checkpoint.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct clip_vocab {
    std::unordered_map<std::string, int32_t> token_to_id;

    // BOS + up to 75 BPE-encoded tokens + EOS, then padded with EOS to exactly `n_ctx` (77, SDXL's
    // fixed context length) -- matches stable-diffusion.cpp's tokenize()+pad_tokens() behavior for
    // a single (non-chunked) prompt.
    std::vector<int32_t> tokenize(const std::string & text, int n_ctx = 77) const;
};

clip_vocab load_clip_vocab();
