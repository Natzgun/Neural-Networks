#include "layers/ExtractCLSLayer.hpp"

Tensor ExtractCLSLayer::forward(const Tensor& input) {
  int batch = input.dim(0);
  tokens_ = input.dim(1);
  int embed_dim = input.dim(2);

  input.download();

  Tensor out({batch, embed_dim});
  for (int b = 0; b < batch; ++b)
    for (int e = 0; e < embed_dim; ++e)
      out.at({b, e}) = input.at({b, 0, e});

  out.upload();
  return out;
}

Tensor ExtractCLSLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  int embed_dim = grad_output.dim(1);

  grad_output.download();

  Tensor grad_input({batch, tokens_, embed_dim}); // fill=0.0f por defecto
  for (int b = 0; b < batch; ++b)
    for (int e = 0; e < embed_dim; ++e)
      grad_input.at({b, 0, e}) = grad_output.at({b, e});

  grad_input.upload();
  return grad_input;
}
