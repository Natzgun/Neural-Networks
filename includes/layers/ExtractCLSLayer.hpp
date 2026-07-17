#pragma once

#include "layers/Layer.hpp"

// Extrae el token CLS (posicion 0) de la secuencia para pasarlo a la cabeza
// de clasificacion. En el backward, todo el gradiente vuelve a la posicion 0
// y el resto queda en 0 (esas posiciones no participaron del forward).
// Entrada: {batch, tokens, embed_dim}
// Salida:  {batch, embed_dim}
class ExtractCLSLayer : public Layer {
public:
  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;

private:
  int tokens_ = 0; // cache, necesario para reconstruir la shape en backward
};
