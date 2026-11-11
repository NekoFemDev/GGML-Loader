#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "tokenizer.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- gguf metadata scan ------------------------------------------------- */

static int rd_exact(FILE *f, void *buf, size_t n)
{
    return fread(buf, 1, n, f) == n ? 0 : -EIO;
}

static int rd_u32(FILE *f, uint32_t *v) { return rd_exact(f, v, 4); }
static int rd_u64(FILE *f, uint64_t *v) { return rd_exact(f, v, 8); }

static int rd_string(FILE *f, char *dst, size_t cap)
{
    uint64_t len = 0;
    if (rd_u64(f, &len) != 0 || len > (1ull << 24)) {
        return -EINVAL;
    }
    size_t keep = (size_t)len < cap - 1 ? (size_t)len : cap - 1;
    if (keep && rd_exact(f, dst, keep) != 0) {
        return -EIO;
    }
    dst[keep] = '\0';
    if ((uint64_t)keep < len && fseek(f, (long)(len - keep), SEEK_CUR) != 0) {
        return -EIO;
    }
    return 0;
}

static int skip_string(FILE *f)
{
    uint64_t len = 0;
    if (rd_u64(f, &len) != 0 || len > (1ull << 34)) {
        return -EINVAL;
    }
    return fseek(f, (long)len, SEEK_CUR) == 0 ? 0 : -EIO;
}

static size_t fixed_size(uint32_t t)
{
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}

static int skip_value(FILE *f, uint32_t type)
{
    size_t sz = fixed_size(type);
    if (sz) {
        return fseek(f, (long)sz, SEEK_CUR) == 0 ? 0 : -EIO;
    }
    if (type == 8) {
        return skip_string(f);
    }
    if (type == 9) {
        uint32_t et = 0;
        uint64_t n  = 0;
        if (rd_u32(f, &et) != 0 || rd_u64(f, &n) != 0) {
            return -EIO;
        }
        if (et == 8) {
            for (uint64_t i = 0; i < n; i++) {
                int rc = skip_string(f);
                if (rc != 0) {
                    return rc;
                }
            }
            return 0;
        }
        size_t esz = fixed_size(et);
        if (esz == 0 || n > (1ull << 40) / esz) {
            return -EINVAL;
        }
        return fseek(f, (long)(esz * n), SEEK_CUR) == 0 ? 0 : -EIO;
    }
    return -EINVAL;
}

/* Read array<str> of `count` strings into blob/off. */
static int read_str_array(FILE *f, uint64_t count, char **blob_out,
                          size_t *blob_len, uint32_t **off_out)
{
    if (count == 0 || count > (1ull << 22)) {
        return -EINVAL;
    }

    size_t   cap = 1u << 20;
    size_t   len = 0;
    char    *blob = (char *)malloc(cap);
    uint32_t *off = (uint32_t *)malloc((size_t)(count + 1) * sizeof(uint32_t));
    if (!blob || !off) {
        free(blob);
        free(off);
        return -ENOMEM;
    }
    off[0] = 0;

    for (uint64_t i = 0; i < count; i++) {
        uint64_t sl = 0;
        if (rd_u64(f, &sl) != 0 || sl > (1ull << 20)) {
            free(blob);
            free(off);
            return -EINVAL;
        }
        while (len + sl + 1 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(blob, cap);
            if (!nb) {
                free(blob);
                free(off);
                return -ENOMEM;
            }
            blob = nb;
        }
        if (sl && rd_exact(f, blob + len, (size_t)sl) != 0) {
            free(blob);
            free(off);
            return -EIO;
        }
        len += (size_t)sl;
        blob[len++] = '\0';
        off[i + 1] = (uint32_t)len;
    }

    *blob_out = blob;
    *blob_len = len;
    *off_out  = off;
    return 0;
}

/* ---- utf8 --------------------------------------------------------------- */

static int utf8_encode(uint32_t cp, char *out)
{
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

static int utf8_decode(const char *s, uint32_t *cp)
{
    const unsigned char *p = (const unsigned char *)s;
    if (p[0] < 0x80) {
        *cp = p[0];
        return 1;
    }
    if ((p[0] & 0xE0) == 0xC0) {
        *cp = ((uint32_t)(p[0] & 0x1F) << 6) | (p[1] & 0x3F);
        return 2;
    }
    if ((p[0] & 0xF0) == 0xE0) {
        *cp = ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) |
              (p[2] & 0x3F);
        return 3;
    }
    *cp = p[0];
    return 1;
}

/* ---- tokenizer ---------------------------------------------------------- */

static uint64_t fnv1a(const char *s, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static void ht_insert(tok_t *t, uint32_t id)
{
    const char *s = t->blob + t->off[id];
    size_t      n = t->off[id + 1] - t->off[id] - 1;
    size_t      i = (size_t)fnv1a(s, n) & (t->ht_cap - 1);

    while (t->ht[i] != 0) {
        i = (i + 1) & (t->ht_cap - 1);
    }
    t->ht[i] = id + 1;
}

static int tok_lookup(const tok_t *t, const char *s, size_t n)
{
    size_t i = (size_t)fnv1a(s, n) & (t->ht_cap - 1);

    for (;;) {
        uint32_t v = t->ht[i];
        if (v == 0) {
            return -1;
        }
        uint32_t id = v - 1;
        size_t   len = t->off[id + 1] - t->off[id] - 1;
        if (len == n && memcmp(t->blob + t->off[id], s, n) == 0) {
            return (int)id;
        }
        i = (i + 1) & (t->ht_cap - 1);
    }
}

int tok_load(tok_t *t, const char *gguf_path)
{
    memset(t, 0, sizeof(*t));

    FILE *f = fopen(gguf_path, "rb");
    if (!f) {
        return -errno;
    }

    int      rc = -EINVAL;
    char     magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    uint64_t tok_count = 0;
    char    *blob = NULL;
    uint32_t *off = NULL;

    if (rd_exact(f, magic, 4) != 0 || memcmp(magic, "GGUF", 4) != 0) {
        goto out;
    }
    if (rd_u32(f, &version) != 0 || version < 2 || version > 3) {
        goto out;
    }
    if (rd_u64(f, &n_tensors) != 0 || rd_u64(f, &n_kv) != 0) {
        goto out;
    }

    for (uint64_t i = 0; i < n_kv; i++) {
        char key[256];
        if (rd_string(f, key, sizeof(key)) != 0) {
            goto out;
        }
        uint32_t type = 0;
        if (rd_u32(f, &type) != 0) {
            goto out;
        }

        if (!strcmp(key, "tokenizer.ggml.tokens") && type == 9) {
            uint32_t et = 0;
            if (rd_u32(f, &et) != 0 || rd_u64(f, &tok_count) != 0 || et != 8) {
                goto out;
            }
            size_t blob_len = 0;
            if (read_str_array(f, tok_count, &blob, &blob_len, &off) != 0) {
                goto out;
            }
        } else if (skip_value(f, type) != 0) {
            goto out;
        }
    }

    if (!blob || tok_count == 0) {
        goto out;
    }

    t->blob = blob;
    t->off  = off;
    t->n    = (uint32_t)tok_count;

    uint32_t cap = 1u << 10;
    while (cap < t->n * 2u) {
        cap <<= 1;
    }
    t->ht_cap = cap;
    t->ht     = (uint32_t *)calloc(cap, sizeof(uint32_t));
    if (!t->ht) {
        goto out;
    }
    for (uint32_t i = 0; i < t->n; i++) {
        ht_insert(t, i);
    }

    /* byte <-> codepoint table (GPT-2 bytes_to_unicode) */
    int other = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            t->byte_cp[b] = (uint32_t)b;
        } else {
            t->byte_cp[b] = 256u + (uint32_t)other;
            other++;
        }
    }

    /* longest token in codepoints, capped: bounds the greedy search */
    uint32_t maxcp = 1;
    for (uint32_t i = 0; i < t->n; i++) {
        const char *s = t->blob + t->off[i];
        const char *e = t->blob + t->off[i + 1] - 1;
        uint32_t cps = 0;
        while (s < e) {
            uint32_t cp;
            s += utf8_decode(s, &cp);
            cps++;
        }
        if (cps > maxcp) {
            maxcp = cps;
        }
    }
    t->max_cp = maxcp > 24 ? 24 : maxcp;

    blob = NULL;
    off  = NULL;
    rc   = 0;

out:
    free(blob);
    free(off);
    fclose(f);
    if (rc != 0) {
        tok_free(t);
    }
    return rc;
}

void tok_free(tok_t *t)
{
    if (!t) {
        return;
    }
    free(t->blob);
    free(t->off);
    free(t->ht);
    memset(t, 0, sizeof(*t));
}

uint32_t tok_count(const tok_t *t) { return t ? t->n : 0; }

const char *tok_str(const tok_t *t, int id)
{
    if (!t || id < 0 || (uint32_t)id >= t->n) {
        return "";
    }
    return t->blob + t->off[id];
}

int tok_decode(const tok_t *t, int id, char *out, size_t cap)
{
    if (!t || id < 0 || (uint32_t)id >= t->n || cap == 0) {
        return -1;
    }

    const char *s = t->blob + t->off[id];
    const char *e = t->blob + t->off[id + 1] - 1;
    size_t      o = 0;

    while (s < e) {
        uint32_t cp;
        s += utf8_decode(s, &cp);

        int byte = -1;
        if (cp < 256) {
            /* printable ranges map to themselves */
            if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) ||
                (cp >= 174 && cp <= 255)) {
                byte = (int)cp;
            }
        }
        if (byte < 0 && cp >= 256) {
            int other = 0;
            for (int b = 0; b < 256; b++) {
                uint32_t c = 0;
                if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                    (b >= 174 && b <= 255)) {
                    continue;
                }
                c = 256u + (uint32_t)other;
                if (c == cp) {
                    byte = b;
                    break;
                }
                other++;
            }
        }
        if (byte >= 0 && o < cap) {
            out[o++] = (char)byte;
        }
    }
    return (int)o;
}

int tok_encode(const tok_t *t, const char *text, int *ids, int max_ids)
{
    if (!t || !text || !ids || max_ids <= 0) {
        return -1;
    }

    size_t   nbytes = strlen(text);
    /* worst case: one codepoint can be 3 bytes of UTF-8 */
    char    *enc = (char *)malloc(nbytes * 3 + 8);
    uint32_t *cpoff = (uint32_t *)malloc((nbytes + 2) * sizeof(uint32_t));
    if (!enc || !cpoff) {
        free(enc);
        free(cpoff);
        return -1;
    }

    size_t   elen = 0;
    uint32_t ncp  = 0;
    cpoff[0] = 0;
    for (size_t i = 0; i < nbytes; i++) {
        elen += (size_t)utf8_encode(t->byte_cp[(unsigned char)text[i]], enc + elen);
        cpoff[++ncp] = (uint32_t)elen;
    }

    int n = 0;
    uint32_t i = 0;
    while (i < ncp) {
        int best = -1;
        uint32_t best_len = 0;
        uint32_t maxl = ncp - i < t->max_cp ? ncp - i : t->max_cp;

        for (uint32_t L = maxl; L >= 1; L--) {
            const char *s = enc + cpoff[i];
            size_t      l = cpoff[i + L] - cpoff[i];
            int id = tok_lookup(t, s, l);
            if (id >= 0) {
                best = id;
                best_len = L;
                break;
            }
        }
        if (best < 0) {
            i++;
            continue;
        }
        if (n >= max_ids) {
            free(enc);
            free(cpoff);
            return -1;
        }
        ids[n++] = best;
        i += best_len;
    }

    free(enc);
    free(cpoff);
    return n;
}
