#pragma once

#include "layers/Layer.hpp"

// A diferencia de ReLULayer/SigmoidLayer, cachea la entrada (no la salida):
// la derivada de GELU no se puede expresar solo en funcion de la salida.
class GELULayer : public Layer {
public:
  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;

private:
  Tensor input_;
};
