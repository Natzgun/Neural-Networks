#pragma once

#include "layers/Layer.hpp"
#include "layers/dense/DenseLayer.hpp"

// Self-attention multi-cabeza: cada token "mira" a todos los demas tokens de
// la secuencia y decide, con un puntaje de atencion, cuanta informacion tomar
// de cada uno.
//
// Flujo: X -> proyectar a Q,K,V -> separar en cabezas ->
//        softmax(Q K^T / sqrt(head_dim)) V -> combinar cabezas -> proyeccion
//        de salida.
//
// Entrada y salida: {batch, tokens, embed_dim} (ver docs/shapes.md)
class MultiHeadAttentionLayer : public Layer {
public:
  MultiHeadAttentionLayer(int embed_dim, int num_heads);

  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float lr) override;

private:
  // {batch, tokens, embed_dim} <-> {batch, heads, tokens, head_dim}
  Tensor split_heads(const Tensor& x, int batch, int tokens) const;
  Tensor combine_heads(const Tensor& x, int batch, int tokens) const;

  int embed_dim_;
  int num_heads_;
  int head_dim_;

  DenseLayer w_q_;
  DenseLayer w_k_;
  DenseLayer w_v_;
  DenseLayer w_o_;

  // Cache del forward, necesario para el backward.
  Tensor q_;
  Tensor k_;
  Tensor v_;
  Tensor attn_;
};
