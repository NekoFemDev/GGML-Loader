/*
 * dequant.c — ggml K-quant block decoders.
 *
 * Layouts (all little-endian, blocks of 256 elements unless noted):
 *
 *   q4_K  144 B : f16 d, f16 dmin, u8 scales[12], u8 qs[128]
 *   q5_K  176 B : f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]
 *   q6_K  210 B : u8 ql[128], u8 qh[64], i8 scales[16], f16 d
 *   q8_0   34 B : f16 d, i8 qs[32]                     (32 elements)
 *
 * These follow the reference ggml dequantisers; the 6-bit scale/min unpacking
 * in q4_K/q5_K is the part that is easy to get subtly wrong.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "dequant.h"

#include <string.h>

float f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x0400u)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t rd_u16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

size_t ggml_type_block_elems(uint32_t type)
{
    switch (type) {
        case GGML_T_F32:  return 1;
        case GGML_T_F16:  return 1;
        case GGML_T_Q4_K: return QK_K;
        case GGML_T_Q5_K: return QK_K;
        case GGML_T_Q6_K: return QK_K;
        case GGML_T_Q8_0: return 32;
        default:          return 0;
    }
}

size_t ggml_type_block_bytes(uint32_t type)
{
    switch (type) {
        case GGML_T_F32:  return 4;
        case GGML_T_F16:  return 2;
        case GGML_T_Q4_K: return 144;
        case GGML_T_Q5_K: return 176;
        case GGML_T_Q6_K: return 210;
        case GGML_T_Q8_0: return 34;
        default:          return 0;
    }
}

uint64_t ggml_type_row_bytes(uint32_t type, uint64_t ne)
{
    size_t be = ggml_type_block_elems(type);
    size_t bb = ggml_type_block_bytes(type);
    if (be == 0 || ne % be != 0) {
        return 0;
    }
    return (ne / be) * bb;
}

/* 6-bit scale/min unpacking shared by q4_K and q5_K. */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void dequant_q4_K(const uint8_t *src, float *y)
{
    const float    d    = f16_to_f32(rd_u16(src));
    const float    dmin = f16_to_f32(rd_u16(src + 2));
    const uint8_t *sc   = src + 4;
    const uint8_t *q    = src + 16;
    int            is   = 0;

    for (int j = 0; j < QK_K; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is + 0, sc, &s, &m);
        const float d1 = d * s;
        const float m1 = dmin * m;
        get_scale_min_k4(is + 1, sc, &s, &m);
        const float d2 = d * s;
        const float m2 = dmin * m;

        for (int l = 0; l < 32; ++l) {
            y[l] = d1 * (float)(q[l] & 0x0F) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            y[l + 32] = d2 * (float)(q[l] >> 4) - m2;
        }

        q += 32;
        is += 2;
        y += 64;
    }
}

static void dequant_q5_K(const uint8_t *src, float *y)
{
    const float    d    = f16_to_f32(rd_u16(src));
    const float    dmin = f16_to_f32(rd_u16(src + 2));
    const uint8_t *sc   = src + 4;
    const uint8_t *qh   = src + 16;
    const uint8_t *ql   = src + 48;
    int            is   = 0;
    uint8_t        u1   = 1;
    uint8_t        u2   = 2;

    for (int j = 0; j < QK_K; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is + 0, sc, &s, &m);
        const float d1 = d * s;
        const float m1 = dmin * m;
        get_scale_min_k4(is + 1, sc, &s, &m);
        const float d2 = d * s;
        const float m2 = dmin * m;

        for (int l = 0; l < 32; ++l) {
            const int hi = (qh[l] & u1) ? 16 : 0;
            y[l] = d1 * (float)((ql[l] & 0x0F) + hi) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            const int hi = (qh[l] & u2) ? 16 : 0;
            y[l + 32] = d2 * (float)((ql[l] >> 4) + hi) - m2;
        }

        ql += 32;
        is += 2;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
        y += 64;
    }
}

static void dequant_q6_K(const uint8_t *src, float *y)
{
    const uint8_t *ql = src;
    const uint8_t *qh = src + 128;
    const int8_t  *sc = (const int8_t *)(src + 192);
    const float    d  = f16_to_f32(rd_u16(src + 208));

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            y[l + 0]  = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static void dequant_q8_0(const uint8_t *src, float *y)
{
    const float  d = f16_to_f32(rd_u16(src));
    const int8_t *q = (const int8_t *)(src + 2);
    for (int i = 0; i < 32; i++) {
        y[i] = d * (float)q[i];
    }
}

void dequant_row(uint32_t type, const uint8_t *src, float *dst, uint64_t n)
{
    switch (type) {
        case GGML_T_F32:
            memcpy(dst, src, (size_t)n * sizeof(float));
            return;
        case GGML_T_F16:
            for (uint64_t i = 0; i < n; i++) {
                dst[i] = f16_to_f32(rd_u16(src + i * 2));
            }
            return;
        case GGML_T_Q4_K: {
            size_t bs = ggml_type_block_bytes(type);
            for (uint64_t b = 0; b < n / QK_K; b++) {
                dequant_q4_K(src + b * bs, dst + b * QK_K);
            }
            return;
        }
        case GGML_T_Q5_K: {
            size_t bs = ggml_type_block_bytes(type);
            for (uint64_t b = 0; b < n / QK_K; b++) {
                dequant_q5_K(src + b * bs, dst + b * QK_K);
            }
            return;
        }
        case GGML_T_Q6_K: {
            size_t bs = ggml_type_block_bytes(type);
            for (uint64_t b = 0; b < n / QK_K; b++) {
                dequant_q6_K(src + b * bs, dst + b * QK_K);
            }
            return;
        }
        case GGML_T_Q8_0: {
            size_t bs = ggml_type_block_bytes(type);
            for (uint64_t b = 0; b < n / 32; b++) {
                dequant_q8_0(src + b * bs, dst + b * 32);
            }
            return;
        }
        default:
            memset(dst, 0, (size_t)n * sizeof(float));
            return;
    }
}

/* Fused dequant + dot. Avoids writing 256 floats to memory for q4_K/q5_K/q6_K
 * when the caller only needs the scalar product. */
float vec_dot(uint32_t type, const uint8_t *row, const float *x, uint64_t n)
{
    if (type == GGML_T_F32) {
        float acc = 0.0f;
        const float *w = (const float *)row;
        for (uint64_t i = 0; i < n; i++) {
            acc += w[i] * x[i];
        }
        return acc;
    }
    if (type == GGML_T_F16) {
        float acc = 0.0f;
        for (uint64_t i = 0; i < n; i++) {
            acc += f16_to_f32(rd_u16(row + i * 2)) * x[i];
        }
        return acc;
    }

    const size_t bs = ggml_type_block_bytes(type);
    if (bs == 0) {
        return 0.0f;
    }

    float    acc = 0.0f;
    float    tmp[QK_K];
    uint64_t nblk = n / QK_K;

    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *src = row + b * bs;
        switch (type) {
            case GGML_T_Q4_K: dequant_q4_K(src, tmp); break;
            case GGML_T_Q5_K: dequant_q5_K(src, tmp); break;
            case GGML_T_Q6_K: dequant_q6_K(src, tmp); break;
            case GGML_T_Q8_0: dequant_q8_0(src, tmp); break;
            default:          memset(tmp, 0, sizeof(tmp)); break;
        }
        const float *xp = x + b * QK_K;
        for (int i = 0; i < QK_K; i++) {
            acc += tmp[i] * xp[i];
        }
    }
    return acc;
}
