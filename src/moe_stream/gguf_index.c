#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _FILE_OFFSET_BITS 64

#include "gguf_index.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- ggml type geometry -------------------------------------------------
 * (elements per block, bytes per block). Anything not listed is rejected
 * rather than guessed: a wrong stride silently corrupts every expert read. */
static const struct {
    uint32_t blck;
    uint32_t bytes;
} k_type_geom[] = {
    [0]  = { 1,   4  },  /* F32     */
    [1]  = { 1,   2  },  /* F16     */
    [2]  = { 32,  18 },  /* Q4_0    */
    [3]  = { 32,  20 },  /* Q4_1    */
    [6]  = { 32,  22 },  /* Q5_0    */
    [7]  = { 32,  24 },  /* Q5_1    */
    [8]  = { 32,  34 },  /* Q8_0    */
    [9]  = { 32,  36 },  /* Q8_1    */
    [10] = { 256, 84 },  /* Q2_K    */
    [11] = { 256, 110 }, /* Q3_K    */
    [12] = { 256, 144 }, /* Q4_K    */
    [13] = { 256, 176 }, /* Q5_K    */
    [14] = { 256, 210 }, /* Q6_K    */
    [15] = { 256, 292 }, /* Q8_K    */
    [16] = { 256, 66 },  /* IQ2_XXS */
    [17] = { 256, 74 },  /* IQ2_XS  */
    [18] = { 256, 98 },  /* IQ3_XXS */
    [19] = { 256, 50 },  /* IQ1_S   */
    [20] = { 32,  18 },  /* IQ4_NL  */
    [21] = { 256, 110 }, /* IQ3_S   */
    [22] = { 256, 82 },  /* IQ2_S   */
    [23] = { 256, 136 }, /* IQ4_XS  */
    [24] = { 1,   1  },  /* I8      */
    [25] = { 1,   2  },  /* I16     */
    [26] = { 1,   4  },  /* I32     */
    [27] = { 1,   8  },  /* I64     */
    [28] = { 1,   8  },  /* F64     */
    [29] = { 256, 56 },  /* IQ1_M   */
    [30] = { 1,   2  },  /* BF16    */
    [34] = { 256, 54 },  /* TQ1_0   */
    [35] = { 256, 66 },  /* TQ2_0   */
    [39] = { 32,  17 },  /* MXFP4   */
};

uint64_t ggml_tensor_nbytes(uint32_t type, uint64_t nelements)
{
    if (type >= sizeof(k_type_geom) / sizeof(k_type_geom[0])) {
        return 0;
    }
    uint32_t blck = k_type_geom[type].blck;
    uint32_t bytes = k_type_geom[type].bytes;
    if (blck == 0 || bytes == 0) {
        return 0;
    }
    if (nelements % blck != 0) {
        return 0;
    }
    return (nelements / blck) * bytes;
}

/* ---- binary reader ------------------------------------------------------ */

static int rd_exact(FILE *f, void *buf, size_t n)
{
    if (n == 0) {
        return 0;
    }
    if (fread(buf, 1, n, f) != n) {
        return ferror(f) ? -errno : -EIO;
    }
    return 0;
}

static int rd_u32(FILE *f, uint32_t *v) { return rd_exact(f, v, sizeof(*v)); }
static int rd_u64(FILE *f, uint64_t *v) { return rd_exact(f, v, sizeof(*v)); }

static int rd_string(FILE *f, char *dst, size_t cap)
{
    uint64_t len = 0;
    int rc = rd_u64(f, &len);
    if (rc != 0) {
        return rc;
    }
    if (len > (1ull << 24)) {
        return -EINVAL;
    }
    if (dst && cap) {
        size_t keep = (size_t)len < cap - 1 ? (size_t)len : cap - 1;
        if (keep && rd_exact(f, dst, keep) != 0) {
            return -EIO;
        }
        dst[keep] = '\0';
        if ((uint64_t)keep < len && fseek(f, (long)(len - keep), SEEK_CUR) != 0) {
            return -EIO;
        }
    } else if (len && fseek(f, (long)len, SEEK_CUR) != 0) {
        return -EIO;
    }
    return 0;
}

static int skip_string(FILE *f)
{
    uint64_t len = 0;
    int rc = rd_u64(f, &len);
    if (rc != 0) {
        return rc;
    }
    if (len > (1ull << 34)) {
        return -EINVAL;
    }
    return fseek(f, (long)len, SEEK_CUR) == 0 ? 0 : -EIO;
}

static size_t fixed_value_size(uint32_t t)
{
    switch (t) {
        case 0: case 1: case 7: return 1;   /* u8, i8, bool */
        case 2: case 3:         return 2;   /* u16, i16 */
        case 4: case 5: case 6: return 4;   /* u32, i32, f32 */
        case 10: case 11: case 12: return 8;/* u64, i64, f64 */
        default: return 0;
    }
}

static int skip_value(FILE *f, uint32_t type)
{
    size_t sz = fixed_value_size(type);
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
        size_t esz = fixed_value_size(et);
        if (esz == 0 || n > (1ull << 40) / esz) {
            return -EINVAL;
        }
        return fseek(f, (long)(esz * n), SEEK_CUR) == 0 ? 0 : -EIO;
    }
    return -EINVAL;
}

static int read_value_u32(FILE *f, uint32_t type, uint32_t *out)
{
    if (type == 4) {
        return rd_u32(f, out);
    }
    if (type == 10) {
        uint64_t v = 0;
        int rc = rd_u64(f, &v);
        *out = (uint32_t)v;
        return rc;
    }
    if (type == 5) { /* i32 */
        int32_t v = 0;
        int rc = rd_exact(f, &v, sizeof(v));
        *out = (uint32_t)v;
        return rc;
    }
    return -EINVAL;
}

int gguf_index_load(gguf_index_t *idx, const char *path)
{
    if (!idx || !path) {
        return -EINVAL;
    }
    memset(idx, 0, sizeof(*idx));

    FILE *f = fopen(path, "rb");
    if (!f) {
        return -errno;
    }

    int rc = -EINVAL;
    char magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;

    if (rd_exact(f, magic, 4) != 0 || memcmp(magic, "GGUF", 4) != 0) {
        goto out;
    }
    if (rd_u32(f, &version) != 0 || version < 2 || version > 3) {
        goto out;
    }
    if (rd_u64(f, &n_tensors) != 0 || rd_u64(f, &n_kv) != 0) {
        goto out;
    }
    if (n_tensors > (1ull << 24) || n_kv > (1ull << 24)) {
        goto out;
    }

    idx->alignment = 32;

    for (uint64_t i = 0; i < n_kv; i++) {
        char key[GGUF_MAX_NAME];
        if (rd_string(f, key, sizeof(key)) != 0) {
            goto out;
        }
        uint32_t type = 0;
        if (rd_u32(f, &type) != 0) {
            goto out;
        }

        uint32_t as_u32  = 0;
        float    as_f32  = 0.0f;
        int      have_u32 = 0;
        int      have_f32 = 0;

        if (!strcmp(key, "general.architecture") && type == 8) {
            if (rd_string(f, idx->arch, sizeof(idx->arch)) != 0) {
                goto out;
            }
        } else if (type == 4 || type == 5 || type == 10) {
            if (read_value_u32(f, type, &as_u32) != 0) {
                goto out;
            }
            have_u32 = 1;
        } else if (type == 6) {
            if (rd_exact(f, &as_f32, sizeof(as_f32)) != 0) {
                goto out;
            }
            have_f32 = 1;
        } else if (skip_value(f, type) != 0) {
            goto out;
        }

        const char *dot   = strchr(key, '.');
        const char *field = dot ? dot + 1 : key;

        if (have_f32) {
            if (!strcmp(field, "rope.freq_base")) {
                idx->rope_theta = as_f32;
            } else if (!strcmp(field, "attention.layer_norm_rms_epsilon")) {
                idx->eps = as_f32;
            }
            continue;
        }
        if (!have_u32) {
            continue;
        }
        if (!strcmp(key, "general.alignment")) {
            idx->alignment = as_u32 ? as_u32 : 32;
            continue;
        }

        if (!strcmp(field, "block_count")) {
            idx->n_layer = as_u32;
        } else if (!strcmp(field, "embedding_length")) {
            idx->n_embd = as_u32;
        } else if (!strcmp(field, "expert_feed_forward_length")) {
            idx->n_ff_exp = as_u32;
        } else if (!strcmp(field, "expert_count")) {
            idx->n_expert = as_u32;
        } else if (!strcmp(field, "expert_used_count")) {
            idx->n_expert_used = as_u32;
        } else if (!strcmp(field, "attention.head_count")) {
            idx->n_head = as_u32;
        } else if (!strcmp(field, "attention.head_count_kv")) {
            idx->n_head_kv = as_u32;
        } else if (!strcmp(field, "attention.key_length")) {
            idx->head_dim = as_u32;
        }
    }

    idx->tensors = (gguf_tensor_t *)calloc((size_t)n_tensors, sizeof(*idx->tensors));
    if (!idx->tensors) {
        rc = -ENOMEM;
        goto out;
    }

    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_t *t = &idx->tensors[i];
        if (rd_string(f, t->name, sizeof(t->name)) != 0) {
            goto out;
        }
        if (rd_u32(f, &t->n_dims) != 0 || t->n_dims > 4) {
            goto out;
        }
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (rd_u64(f, &t->dims[d]) != 0) {
                goto out;
            }
        }
        if (rd_u32(f, &t->type) != 0 || rd_u64(f, &t->offset) != 0) {
            goto out;
        }
    }

    long pos = ftell(f);
    if (pos < 0) {
        goto out;
    }
    idx->count = (uint32_t)n_tensors;
    idx->data_start = ((uint64_t)pos + idx->alignment - 1) / idx->alignment * idx->alignment;

    for (uint32_t i = 0; i < idx->count; i++) {
        if (!strcmp(idx->tensors[i].name, "token_embd.weight") &&
            idx->tensors[i].n_dims >= 2) {
            idx->n_vocab = (uint32_t)idx->tensors[i].dims[1];
            break;
        }
    }
    if (idx->rope_theta <= 0.0f) {
        idx->rope_theta = 10000.0f;
    }
    if (idx->eps <= 0.0f) {
        idx->eps = 1e-6f;
    }

    rc = 0;

out:
    fclose(f);
    if (rc != 0) {
        gguf_index_free(idx);
    }
    return rc;
}

void gguf_index_free(gguf_index_t *idx)
{
    if (!idx) {
        return;
    }
    free(idx->tensors);
    idx->tensors = NULL;
    idx->count   = 0;
}

const gguf_tensor_t *gguf_find(const gguf_index_t *idx, const char *name)
{
    if (!idx || !idx->tensors || !name) {
        return NULL;
    }
    for (uint32_t i = 0; i < idx->count; i++) {
        if (!strcmp(idx->tensors[i].name, name)) {
            return &idx->tensors[i];
        }
    }
    return NULL;
}

int moe_tensor_span(const gguf_index_t *idx, const gguf_tensor_t *t,
                    uint32_t expert, uint64_t *abs_off, uint64_t *bytes)
{
    if (!idx || !t || !abs_off || !bytes) {
        return -EINVAL;
    }
    if (t->n_dims != 3) {
        return -EINVAL;
    }

    uint64_t n_expert = t->dims[2];
    if (n_expert == 0 || expert >= n_expert) {
        return -ERANGE;
    }

    uint64_t ne = t->dims[0] * t->dims[1] * t->dims[2];
    uint64_t nb = ggml_tensor_nbytes(t->type, ne);
    if (nb == 0 || nb % n_expert != 0) {
        return -ENOTSUP;
    }

    uint64_t stride = nb / n_expert;
    *bytes   = stride;
    *abs_off = idx->data_start + t->offset + (uint64_t)expert * stride;
    return 0;
}

int moe_plan_build(const gguf_index_t *idx, moe_plan_t *plan)
{
    if (!idx || !plan) {
        return -EINVAL;
    }
    memset(plan, 0, sizeof(*plan));
    plan->idx           = idx;
    plan->n_expert      = idx->n_expert;
    plan->n_expert_used = idx->n_expert_used;
    plan->n_embd        = idx->n_embd;
    plan->n_ff_exp      = idx->n_ff_exp;

    int max_layer = -1;
    for (uint32_t i = 0; i < idx->count; i++) {
        const char *n = idx->tensors[i].name;
        int layer = -1;
        if (sscanf(n, "blk.%d.ffn_gate_exps.weight", &layer) == 1 && layer > max_layer) {
            max_layer = layer;
        }
    }
    if (max_layer < 0 || max_layer + 1 > MOE_MAX_LAYERS) {
        return -ENOENT;
    }

    plan->n_layers = max_layer + 1;

    for (int l = 0; l < plan->n_layers; l++) {
        moe_layer_t *ml = &plan->layers[l];
        char name[GGUF_MAX_NAME];

        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
        const gguf_tensor_t *tg = gguf_find(idx, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
        const gguf_tensor_t *tu = gguf_find(idx, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
        const gguf_tensor_t *td = gguf_find(idx, name);

        if (!tg || !tu || !td) {
            return -ENOENT;
        }

        uint64_t off = 0, len = 0;
        if (moe_tensor_span(idx, tg, 0, &off, &len) != 0) {
            return -ENOTSUP;
        }
        ml->gate.base = off;
        ml->gate.stride = len;
        ml->gate.n_expert = (uint32_t)tg->dims[2];
        ml->gate.ne0 = (uint32_t)tg->dims[0];
        ml->gate.ne1 = (uint32_t)tg->dims[1];
        ml->gate.type = tg->type;
        ml->gate.present = 1;

        if (moe_tensor_span(idx, tu, 0, &off, &len) != 0) {
            return -ENOTSUP;
        }
        ml->up.base = off;
        ml->up.stride = len;
        ml->up.n_expert = (uint32_t)tu->dims[2];
        ml->up.ne0 = (uint32_t)tu->dims[0];
        ml->up.ne1 = (uint32_t)tu->dims[1];
        ml->up.type = tu->type;
        ml->up.present = 1;

        if (moe_tensor_span(idx, td, 0, &off, &len) != 0) {
            return -ENOTSUP;
        }
        ml->down.base = off;
        ml->down.stride = len;
        ml->down.n_expert = (uint32_t)td->dims[2];
        ml->down.ne0 = (uint32_t)td->dims[0];
        ml->down.ne1 = (uint32_t)td->dims[1];
        ml->down.type = td->type;
        ml->down.present = 1;

        ml->expert_bytes = ml->gate.stride + ml->up.stride + ml->down.stride;
    }

    if (plan->n_expert == 0) {
        plan->n_expert = plan->layers[0].gate.n_expert;
    }
    plan->total_expert_bytes = 0;
    plan->max_expert_bytes   = 0;
    for (int l = 0; l < plan->n_layers; l++) {
        uint64_t eb = plan->layers[l].expert_bytes;
        plan->total_expert_bytes += (uint64_t)plan->n_expert * eb;
        if (eb > plan->max_expert_bytes) {
            plan->max_expert_bytes = eb;
        }
    }
    return 0;
}

void moe_plan_print(const moe_plan_t *plan)
{
    if (!plan) {
        return;
    }
    printf("moe plan: %d layers x %u experts (top-%u), embd=%u, ff_exp=%u\n",
           plan->n_layers, plan->n_expert, plan->n_expert_used,
           plan->n_embd, plan->n_ff_exp);
    printf("  expert payload: gate=%" PRIu64 " up=%" PRIu64 " down=%" PRIu64
           " total=%" PRIu64 " B\n",
           plan->layers[0].gate.stride, plan->layers[0].up.stride,
           plan->layers[0].down.stride, plan->layers[0].expert_bytes);
    printf("  all experts on disk: %.2f GiB, largest layer payload %.2f MiB\n",
           (double)plan->total_expert_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)plan->max_expert_bytes / (1024.0 * 1024.0));
}
