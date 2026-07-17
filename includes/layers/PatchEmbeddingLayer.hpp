#pragma once

#include "layers/Layer.hpp"
#include "layers/conv/ConvLayer.hpp"

// Convierte una imagen en una secuencia de patches embebidos, reusando
// Conv2DLayer con kernel_size = stride = patch_size (sin solapamiento) para
// cortar+proyectar cada patch en un solo paso.
// Entrada:  {batch, channels, height, width}
// Salida:   {batch, num_patches, embed_dim} (ver docs/shapes.md)
class PatchEmbeddingLayer : public Layer {
public:
  PatchEmbeddingLayer(int image_h, int image_w, int patch_size, int in_channels, int embed_dim);

  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float lr) override;

private:
  int patch_size_;
  int embed_dim_;
  int grid_h_;
  int grid_w_;
  int num_patches_;

  Conv2DLayer conv_;
};
