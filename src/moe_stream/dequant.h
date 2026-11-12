/*
 * dequant.h — on-the-fly dequantization for the ggml types in this GGUF.
 *
 * The model never materialises a whole weight matrix in f32. Each matmul walks
 * the quantised rows and accumulates block-by-block, so peak extra memory is
 * one 256-float scratch.
 */
#ifndef MOE_DEQUANT_H
#define MOE_DEQUANT_H

#include <stddef.h>
#include <stdint.h>

#define QK_K 256

/* ggml type ids present in the Qwen3-Coder UD-Q4_K_XL file. */
enum {
    GGML_T_F32  = 0,
    GGML_T_F16  = 1,
    GGML_T_Q2_K = 10,
    GGML_T_Q3_K = 11,
    GGML_T_Q4_K = 12,
    GGML_T_Q5_K = 13,
    GGML_T_Q6_K = 14,
    GGML_T_Q8_0 = 8,
};

float    f16_to_f32(uint16_t h);
size_t   ggml_type_block_elems(uint32_t type);
size_t   ggml_type_block_bytes(uint32_t type);
uint64_t ggml_type_row_bytes(uint32_t type, uint64_t ne);

/* Dequantize `n` consecutive elements (n multiple of the block size for
 * quantised types) from src into dst. */
void dequant_row(uint32_t type, const uint8_t *src, float *dst, uint64_t n);

/* dot(row, x) without materialising the row. */
float vec_dot(uint32_t type, const uint8_t *row, const float *x, uint64_t n);

#endif /* MOE_DEQUANT_H */
