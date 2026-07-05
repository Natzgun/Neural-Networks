#include "layers/SoftmaxLayer.hpp"

#include "core/ops/activations.cuh"

Tensor SoftmaxLayer::forward(const Tensor& input) {
  output_ = ops::softmax_forward(input);
  return output_;
}

Tensor SoftmaxLayer::backward(const Tensor& grad_output) {
  return grad_output;
}
