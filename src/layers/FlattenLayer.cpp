#include "layers/conv/FlattenLayer.hpp"

Tensor FlattenLayer::forward(const Tensor& input) {
  input_shape_ = input.sizes();

  int batch = input.dim(0);
  int rest = 1;
  for (size_t i = 1; i < input_shape_.size(); ++i)
    rest *= input_shape_[i];

  return input.reshape({batch, rest});
}

Tensor FlattenLayer::backward(const Tensor& grad_output) {
  return grad_output.reshape(input_shape_);
}
