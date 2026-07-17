#pragma once

#include "core/Tensor.cuh"

namespace ops {

// Multiplicacion de matrices por batch y por cabeza, para Multi-Head Attention.
//
// A: {batch, heads, M, K}   (transpose_a = false)
//    {batch, heads, K, M}   (transpose_a = true)  -> se usa A^T
// B: {batch, heads, K, N}   (transpose_b = false)
//    {batch, heads, N, K}   (transpose_b = true)  -> se usa B^T
//
// Ninguna transposicion se materializa: solo cambia como se leen los indices.
// Salida: {batch, heads, M, N}
//
// Implementacion host-side por ahora (sin kernel CUDA todavia); candidato a
// optimizar despues.
Tensor batched_matmul(const Tensor& a, const Tensor& b, bool transpose_b = false,
                     bool transpose_a = false);

} // namespace ops
