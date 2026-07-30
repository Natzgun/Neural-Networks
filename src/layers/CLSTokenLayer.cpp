#include "layers/CLSTokenLayer.hpp"

#include "core/ops/linalg.cuh"
#include "core/ops/vit_ops.cuh"

CLSTokenLayer::CLSTokenLayer(int embed_dim) : embed_dim_(embed_dim), grad_cls_token_({embed_dim}) {
  cls_token_ = Tensor::random_normal({embed_dim}, 0.0f, 0.02f);
  cls_token_.upload();
  grad_cls_token_.upload();
}

Tensor CLSTokenLayer::forward(const Tensor& input) {
  int batch = input.dim(0);
  int tokens = input.dim(1);
  return ops::prepend_cls_token(input, cls_token_, batch, tokens, embed_dim_);
}

Tensor CLSTokenLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  int tokens = grad_output.dim(1) - 1;
  return ops::cls_token_backward(grad_output, batch, tokens, embed_dim_, grad_cls_token_);
}

void CLSTokenLayer::update(float lr) {
  ops::sub_inplace(cls_token_, ops::scalar_mul(grad_cls_token_, lr));
}
