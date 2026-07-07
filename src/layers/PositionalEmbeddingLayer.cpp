#include "layers/PositionalEmbeddingLayer.hpp"

#include "core/ops/linalg.cuh"

PositionalEmbeddingLayer::PositionalEmbeddingLayer(int tokens, int embed_dim)
    : tokens_(tokens), embed_dim_(embed_dim), grad_pos_embed_({1, tokens * embed_dim}) {
  pos_embed_ = Tensor::random_normal({1, tokens * embed_dim}, 0.0f, 0.02f);
  pos_embed_.upload();
  grad_pos_embed_.upload();
}

Tensor PositionalEmbeddingLayer::forward(const Tensor& input) {
  int batch = input.dim(0);
  // Aplanamos tokens*embed_dim en una sola dimension para poder usar
  // add_bias tal cual esta: al repetir pos_embed_ cada "cols" elementos,
  // termina sumando el mismo embedding de posicion a cada muestra del batch.
  Tensor flat_input = input.reshape({batch, tokens_ * embed_dim_});

  Tensor out = ops::add_bias(flat_input, pos_embed_);

  return out.reshape({batch, tokens_, embed_dim_});
}

Tensor PositionalEmbeddingLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  Tensor flat_grad = grad_output.reshape({batch, tokens_ * embed_dim_});

  // Como pos_embed_ se sumo igual en cada muestra del batch, su gradiente
  // es simplemente la suma de los gradientes de todas esas muestras.
  grad_pos_embed_ = ops::sum_rows(flat_grad);

  // Es una suma, asi que el gradiente pasa intacto hacia la entrada.
  return flat_grad.reshape({batch, tokens_, embed_dim_});
}

void PositionalEmbeddingLayer::update(float lr) {
  ops::sub_inplace(pos_embed_, ops::scalar_mul(grad_pos_embed_, lr));
}
