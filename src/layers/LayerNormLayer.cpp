#include "layers/LayerNormLayer.hpp"

#include "core/ops/linalg.cuh"
#include "core/ops/vit_ops.cuh"

LayerNormLayer::LayerNormLayer(int embed_dim, float eps)
    : embed_dim_(embed_dim), eps_(eps), grad_gamma_({embed_dim}), grad_beta_({embed_dim}) {
  // gamma arranca en 1 y beta en 0 para que LayerNorm empiece siendo casi la
  // identidad sobre la entrada ya normalizada.
  gamma_ = Tensor::ones({embed_dim});
  beta_ = Tensor::zeros({embed_dim});

  gamma_.upload();
  beta_.upload();
  grad_gamma_.upload();
  grad_beta_.upload();
}

Tensor LayerNormLayer::forward(const Tensor& input) {
  int batch = input.dim(0);
  int tokens = input.dim(1);
  int rows = batch * tokens;

  Tensor flat = input.reshape({rows, embed_dim_});
  Tensor out_flat = ops::layer_norm_forward(flat, gamma_, beta_, eps_, x_norm_, std_);
  return out_flat.reshape({batch, tokens, embed_dim_});
}

Tensor LayerNormLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  int tokens = grad_output.dim(1);
  int rows = batch * tokens;

  Tensor flat_grad = grad_output.reshape({rows, embed_dim_});
  Tensor dx_flat =
      ops::layer_norm_backward(flat_grad, gamma_, x_norm_, std_, grad_gamma_, grad_beta_);
  return dx_flat.reshape({batch, tokens, embed_dim_});
}

void LayerNormLayer::update(float lr) {
  ops::sub_inplace(gamma_, ops::scalar_mul(grad_gamma_, lr));
  ops::sub_inplace(beta_, ops::scalar_mul(grad_beta_, lr));
}
