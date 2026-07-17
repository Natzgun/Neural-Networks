#pragma once

#include "layers/GELULayer.hpp"
#include "layers/Layer.hpp"
#include "layers/LayerNormLayer.hpp"
#include "layers/MultiHeadAttentionLayer.hpp"
#include "layers/dense/DenseLayer.hpp"

// Un bloque Transformer estilo ViT (pre-norm):
//
//   X ------------------------------------+
//   |                                     |
//   LayerNorm -> MultiHeadAttention       |
//   |                                     |
//   + <------------------------------------
//   |
//   |------------------------------------+
//   |                                     |
//   LayerNorm -> Linear -> GELU -> Linear |
//   |                                     |
//   + <------------------------------------
//   |
//   Salida
//
// Las conexiones residuales (las "+") reparten el gradiente sin modificarlo
// hacia ambas ramas durante el backward.
// Entrada y salida: {batch, tokens, embed_dim} (ver docs/shapes.md)
class TransformerBlock : public Layer {
public:
  TransformerBlock(int embed_dim, int num_heads, int mlp_hidden_dim);

  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float lr) override;

private:
  int embed_dim_;

  LayerNormLayer ln1_;
  MultiHeadAttentionLayer attn_;
  LayerNormLayer ln2_;
  DenseLayer fc1_;
  GELULayer gelu_;
  DenseLayer fc2_;
};
