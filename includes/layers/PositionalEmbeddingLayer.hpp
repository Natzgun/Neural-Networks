#pragma once

#include "layers/Layer.hpp"

// Le suma a cada token un embedding de posicion que la red aprende durante
// el entrenamiento. Entrada y salida tienen shape {batch, tokens, embed_dim}
// (ver docs/shapes.md).
class PositionalEmbeddingLayer : public Layer {
public:
  PositionalEmbeddingLayer(int tokens, int embed_dim);

  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float lr) override;

private:
  int tokens_;
  int embed_dim_;
  Tensor pos_embed_;      // parametro aprendible, shape {1, tokens * embed_dim}
  Tensor grad_pos_embed_; // gradiente acumulado de pos_embed_, misma shape
};
