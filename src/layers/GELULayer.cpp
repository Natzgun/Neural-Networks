#include "layers/GELULayer.hpp"

#include <cmath>

// Version manual en C++ (temporal): antes llamaba a ops::gelu_forward /
// ops::gelu_derivative (CUDA, en core/ops/activations.cuh). Misma formula
// exacta de GELU, solo que host-side.
Tensor GELULayer::forward(const Tensor& input) {
  input_ = input;
  input_.download();

  Tensor out(input_.sizes());
  int n = input_.numel();
  for (int i = 0; i < n; ++i) {
    float x = input_.flat(i);
    out.flat(i) = 0.5f * x * (1.0f + std::erf(x * 0.70710678f));
  }

  out.upload();
  return out;
}

Tensor GELULayer::backward(const Tensor& grad_output) {
  grad_output.download();
  input_.download();

  Tensor dx(input_.sizes());
  int n = input_.numel();
  for (int i = 0; i < n; ++i) {
    float x = input_.flat(i);
    float cdf = 0.5f * (1.0f + std::erf(x * 0.70710678f));
    float pdf = std::exp(-0.5f * x * x) * 0.3989422804f;
    dx.flat(i) = grad_output.flat(i) * (cdf + x * pdf);
  }

  dx.upload();
  return dx;
}
