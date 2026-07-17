#pragma once

#include "core/Tensor.cuh"

namespace ops {

// Multiplicacion de matrices por batch y por cabeza, para Multi-Head Attention.
//
// A: {batch, heads, M, K}
// B: {batch, heads, K, N}   (transpose_b = false)
//    {batch, heads, N, K}   (transpose_b = true) -> se calcula A @ B^T sobre
//    las ultimas 2 dimensiones de B, sin materializar la transposicion.
//
// Salida: {batch, heads, M, N}
//
// Implementacion host-side por ahora (sin kernel CUDA todavia); candidato a
// optimizar despues.
Tensor batched_matmul(const Tensor& a, const Tensor& b, bool transpose_b = false);

} // namespace ops
