#include "layers/ExtractCLSLayer.hpp"

#include "core/ops/vit_ops.cuh"

Tensor ExtractCLSLayer::forward(const Tensor& input) {
  int batch = input.dim(0);
  tokens_ = input.dim(1);
  int embed_dim = input.dim(2);
  return ops::extract_cls_forward(input, batch, tokens_, embed_dim);
}

Tensor ExtractCLSLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  int embed_dim = grad_output.dim(1);
  return ops::extract_cls_backward(grad_output, batch, tokens_, embed_dim);
}
