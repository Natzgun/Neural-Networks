#include "layers/PatchEmbeddingLayer.hpp"

PatchEmbeddingLayer::PatchEmbeddingLayer(int image_h, int image_w, int patch_size,
                                         int in_channels, int embed_dim)
    : patch_size_(patch_size), embed_dim_(embed_dim), grid_h_(image_h / patch_size),
      grid_w_(image_w / patch_size), num_patches_(grid_h_ * grid_w_),
      conv_(in_channels, embed_dim, patch_size, patch_size, 0) {
}

Tensor PatchEmbeddingLayer::forward(const Tensor& input) {
  // TODO: reshape+permute la salida de conv_ de {batch, embed_dim, grid_h,
  // grid_w} a {batch, num_patches, embed_dim}.
  return conv_.forward(input);
}

Tensor PatchEmbeddingLayer::backward(const Tensor& grad_output) {
  // TODO: deshacer el permute+reshape antes de pasarle el gradiente a conv_.
  return conv_.backward(grad_output);
}

void PatchEmbeddingLayer::update(float lr) {
  conv_.update(lr);
}
