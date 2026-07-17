#include "layers/LayerNormLayer.hpp"

#include <cmath>

#include "core/ops/linalg.cuh"

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
  flat.download();
  gamma_.download();
  beta_.download();

  x_norm_ = Tensor({rows, embed_dim_});
  std_.assign(rows, 0.0f);
  Tensor out_flat({rows, embed_dim_});

  for (int r = 0; r < rows; ++r) {
    float mean = 0.0f;
    for (int e = 0; e < embed_dim_; ++e)
      mean += flat.at({r, e});
    mean /= embed_dim_;

    float var = 0.0f;
    for (int e = 0; e < embed_dim_; ++e) {
      float diff = flat.at({r, e}) - mean;
      var += diff * diff;
    }
    var /= embed_dim_;

    float std = std::sqrt(var + eps_);
    std_[r] = std;

    for (int e = 0; e < embed_dim_; ++e) {
      float xn = (flat.at({r, e}) - mean) / std;
      x_norm_.at({r, e}) = xn;
      out_flat.at({r, e}) = gamma_.at({e}) * xn + beta_.at({e});
    }
  }

  Tensor out = out_flat.reshape({batch, tokens, embed_dim_});
  out.upload();
  return out;
}

Tensor LayerNormLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  int tokens = grad_output.dim(1);
  int rows = batch * tokens;

  Tensor flat_grad = grad_output.reshape({rows, embed_dim_});
  flat_grad.download();
  gamma_.download();

  Tensor dgamma({embed_dim_});
  Tensor dbeta({embed_dim_});
  Tensor dx_flat({rows, embed_dim_});

  for (int r = 0; r < rows; ++r) {
    std::vector<float> dxn(embed_dim_);
    float mean_dxn = 0.0f;
    float mean_dxn_xn = 0.0f;

    for (int e = 0; e < embed_dim_; ++e) {
      float grad = flat_grad.at({r, e});
      float xn = x_norm_.at({r, e});

      dxn[e] = grad * gamma_.at({e});
      mean_dxn += dxn[e];
      mean_dxn_xn += dxn[e] * xn;

      dgamma.at({e}) += grad * xn;
      dbeta.at({e}) += grad;
    }
    mean_dxn /= embed_dim_;
    mean_dxn_xn /= embed_dim_;

    for (int e = 0; e < embed_dim_; ++e) {
      float xn = x_norm_.at({r, e});
      dx_flat.at({r, e}) = (dxn[e] - mean_dxn - xn * mean_dxn_xn) / std_[r];
    }
  }

  grad_gamma_ = dgamma;
  grad_beta_ = dbeta;
  grad_gamma_.upload();
  grad_beta_.upload();

  Tensor dx = dx_flat.reshape({batch, tokens, embed_dim_});
  dx.upload();
  return dx;
}

void LayerNormLayer::update(float lr) {
  ops::sub_inplace(gamma_, ops::scalar_mul(grad_gamma_, lr));
  ops::sub_inplace(beta_, ops::scalar_mul(grad_beta_, lr));
}
