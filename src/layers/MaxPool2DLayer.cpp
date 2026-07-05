#include "layers/conv/MaxPool2DLayer.hpp"

#include "core/ops/pool_ops.cuh"

MaxPool2DLayer::MaxPool2DLayer(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride == -1 ? kernel_size : stride) {}

MaxPool2DLayer::~MaxPool2DLayer() {
  mask_.free();
}

Tensor MaxPool2DLayer::forward(const Tensor& input) {
  inputs_ = input;
  return ops::maxpool2d_forward(input, kernel_size_, stride_, mask_);
}

Tensor MaxPool2DLayer::backward(const Tensor& grad_output) {
  return ops::maxpool2d_backward(grad_output, mask_, inputs_.sizes());
}
