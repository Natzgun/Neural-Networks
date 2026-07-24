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
Tensor batched_matmul(const Tensor& a, const Tensor& b, bool transpose_b = false,
                     bool transpose_a = false);

// Intercambia los ejes 1 y 2 de un tensor 4D contiguo: {d0, d1, d2, d3} ->
// {d0, d2, d1, d3}. Es su propia inversa (aplicarla dos veces devuelve el
// tensor original), por eso split_heads y combine_heads en
// MultiHeadAttentionLayer llaman a la misma funcion.
Tensor swap_middle_axes(const Tensor& x);

} // namespace ops
