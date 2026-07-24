#include "layers/GELULayer.hpp"

#include "core/ops/activations.cuh"
#include "core/ops/linalg.cuh"

// gelu_derivative da dGELU/dx punto a punto; el hadamard con grad_output es
// lo que realmente aplica la regla de la cadena y produce dL/dx.
Tensor GELULayer::forward(const Tensor& input) {
  input_ = input;
  return ops::gelu_forward(input);
}

Tensor GELULayer::backward(const Tensor& grad_output) {
  Tensor derivative = ops::gelu_derivative(input_);
  return ops::hadamard(grad_output, derivative);
}
