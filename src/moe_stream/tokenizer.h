/*
 * tokenizer.h — byte-level vocab from the GGUF, decode + approximate encode.
 *
 * The GGUF stores GPT-2 style byte-level tokens ("Ġhello" where Ġ is U+0120
 * standing for 0x20). Decoding maps those codepoints back to raw bytes.
 *
 * Encoding here is greedy longest-match against the vocabulary, not true BPE.
 * For plain English/code prompts it produces a valid, sensible segmentation;
 * it is not bit-identical to the HF tokenizer. Use tok_encode_ids() with
 * precomputed ids when exactness matters.
 */
#ifndef MOE_TOKENIZER_H
#define MOE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char     *blob;       /* NUL-separated token strings */
    uint32_t *off;        /* n+1 offsets into blob */
    uint32_t  n;
    uint32_t *ht;         /* open addressing: id+1, 0 = empty */
    uint32_t  ht_cap;
    uint32_t  max_cp;     /* longest token, in codepoints */
    uint32_t  byte_cp[256]; /* byte -> unicode codepoint */
} tok_t;

int  tok_load(tok_t *t, const char *gguf_path);
void tok_free(tok_t *t);

uint32_t tok_count(const tok_t *t);
const char *tok_str(const tok_t *t, int id);

/* Decode one token into raw bytes. Returns bytes written, or -1. */
int tok_decode(const tok_t *t, int id, char *out, size_t cap);

/* Approximate encode. Returns token count, or -1 on error/overflow. */
int tok_encode(const tok_t *t, const char *text, int *ids, int max_ids);

#endif /* MOE_TOKENIZER_H */
