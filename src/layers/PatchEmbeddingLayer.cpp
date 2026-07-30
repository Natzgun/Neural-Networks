#include "layers/PatchEmbeddingLayer.hpp"

#include "core/ops/vit_ops.cuh"

PatchEmbeddingLayer::PatchEmbeddingLayer(int image_h, int image_w, int patch_size,
                                         int in_channels, int embed_dim)
    : patch_size_(patch_size), embed_dim_(embed_dim), grid_h_(image_h / patch_size),
      grid_w_(image_w / patch_size), num_patches_(grid_h_ * grid_w_),
      conv_(in_channels, embed_dim, patch_size, patch_size, 0) {
}

Tensor PatchEmbeddingLayer::forward(const Tensor& input) {
  Tensor conv_out = conv_.forward(input); // {batch, embed_dim, grid_h, grid_w}
  int batch = conv_out.dim(0);

  // Colapsa grid_h*grid_w en num_patches (contiguo, no mueve memoria).
  Tensor flat = conv_out.reshape({batch, embed_dim_, num_patches_});

  // Permute {batch, embed_dim, num_patches} -> {batch, num_patches, embed_dim}.
  return ops::patch_tokens_forward(flat, batch, embed_dim_, num_patches_);
}

Tensor PatchEmbeddingLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);

  // Inverso del forward: {batch, num_patches, embed_dim} -> {batch, embed_dim, num_patches}.
  Tensor flat_grad = ops::patch_tokens_backward(grad_output, batch, embed_dim_, num_patches_);

  Tensor conv_grad = flat_grad.reshape({batch, embed_dim_, grid_h_, grid_w_});
  return conv_.backward(conv_grad);
}

void PatchEmbeddingLayer::update(float lr) {
  conv_.update(lr);
}
