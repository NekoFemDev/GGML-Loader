#!/bin/sh
# Compile the compute shaders and regenerate the embedded SPIR-V header.
set -e
cd "$(dirname "$0")"

OUT=../build/shaders
mkdir -p "$OUT"

GLSLC=${GLSLC:-glslc}
"$GLSLC" --target-env=vulkan1.1 -O matmul_f32.comp   -o "$OUT/matmul_f32.spv"
"$GLSLC" --target-env=vulkan1.1 -O matmul_quant.comp -o "$OUT/matmul_quant.spv"
"$GLSLC" --target-env=vulkan1.1 -O matmul_gemm.comp  -o "$OUT/matmul_gemm.spv"

if command -v spirv-val >/dev/null 2>&1; then
    spirv-val --target-env vulkan1.1 "$OUT/matmul_f32.spv"
    spirv-val --target-env vulkan1.1 "$OUT/matmul_quant.spv"
    spirv-val --target-env vulkan1.1 "$OUT/matmul_gemm.spv"
fi

python3 spv2h.py "$OUT/matmul_f32.spv" "$OUT/matmul_quant.spv" "$OUT/matmul_gemm.spv" > ../vulkan_shaders.h
echo "wrote ../vulkan_shaders.h"
