#pragma once

#include "layers/Layer.hpp"

// Antepone un token CLS aprendible a la secuencia de patches. Sirve como
// "resumen" que despues se usa para clasificar (ver docs/shapes.md).
// Entrada:  {batch, tokens, embed_dim}
// Salida:   {batch, tokens + 1, embed_dim}
class CLSTokenLayer : public Layer {
public:
  explicit CLSTokenLayer(int embed_dim);

  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float lr) override;

private:
  int embed_dim_;

  Tensor cls_token_;      // parametro aprendible, shape {embed_dim}
  Tensor grad_cls_token_; // gradiente acumulado, misma shape
};
