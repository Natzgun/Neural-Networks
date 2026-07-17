#include "layers/CLSTokenLayer.hpp"

#include "core/ops/linalg.cuh"

CLSTokenLayer::CLSTokenLayer(int embed_dim) : embed_dim_(embed_dim), grad_cls_token_({embed_dim}) {
  cls_token_ = Tensor::random_normal({embed_dim}, 0.0f, 0.02f);
  cls_token_.upload();
  grad_cls_token_.upload();
}

Tensor CLSTokenLayer::forward(const Tensor& input) {
  int batch = input.dim(0);
  int tokens = input.dim(1);

  input.download();
  cls_token_.download();

  Tensor out({batch, tokens + 1, embed_dim_});
  for (int b = 0; b < batch; ++b) {
    for (int e = 0; e < embed_dim_; ++e)
      out.at({b, 0, e}) = cls_token_.at({e});

    for (int n = 0; n < tokens; ++n)
      for (int e = 0; e < embed_dim_; ++e)
        out.at({b, n + 1, e}) = input.at({b, n, e});
  }

  out.upload();
  return out;
}

Tensor CLSTokenLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  int tokens = grad_output.dim(1) - 1;

  grad_output.download();

  Tensor grad_cls({embed_dim_});
  Tensor grad_input({batch, tokens, embed_dim_});

  for (int b = 0; b < batch; ++b) {
    for (int e = 0; e < embed_dim_; ++e)
      grad_cls.at({e}) += grad_output.at({b, 0, e});

    for (int n = 0; n < tokens; ++n)
      for (int e = 0; e < embed_dim_; ++e)
        grad_input.at({b, n, e}) = grad_output.at({b, n + 1, e});
  }

  grad_cls_token_ = grad_cls;
  grad_cls_token_.upload();

  grad_input.upload();
  return grad_input;
}

void CLSTokenLayer::update(float lr) {
  ops::sub_inplace(cls_token_, ops::scalar_mul(grad_cls_token_, lr));
}
