#pragma once

#include "layers/Layer.hpp"

// Normaliza cada token (ultimo eje) a media 0 / varianza 1, y aplica una
// escala y un desplazamiento aprendibles (gamma, beta) por canal.
// A diferencia de BatchNorm, la normalizacion es POR TOKEN, no por batch.
// Entrada y salida: {batch, tokens, embed_dim} (ver docs/shapes.md)
class LayerNormLayer : public Layer {
public:
  explicit LayerNormLayer(int embed_dim, float eps = 1e-5f);

  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float lr) override;

private:
  int embed_dim_;
  float eps_;

  Tensor gamma_;
  Tensor beta_;
  Tensor grad_gamma_;
  Tensor grad_beta_;

  // Cache del forward, necesario para el backward.
  Tensor x_norm_; // {rows, embed_dim}, rows = batch*tokens
  Tensor std_;    // {rows}, desviacion estandar por fila
};
