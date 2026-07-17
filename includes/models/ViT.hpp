#pragma once

#include "layers/CLSTokenLayer.hpp"
#include "layers/ExtractCLSLayer.hpp"
#include "layers/LayerNormLayer.hpp"
#include "layers/PatchEmbeddingLayer.hpp"
#include "layers/PositionalEmbeddingLayer.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/TransformerBlock.hpp"
#include "layers/dense/DenseLayer.hpp"
#include "network/Network.hpp"

inline Network make_vit(int image_h, int image_w, int patch_size, int in_channels, int embed_dim,
                        int num_heads, int mlp_hidden_dim, int num_layers, int n_classes) {
  int grid_h = image_h / patch_size;
  int grid_w = image_w / patch_size;
  int num_patches = grid_h * grid_w;

  Network net;
  net.add<PatchEmbeddingLayer>(image_h, image_w, patch_size, in_channels, embed_dim);
  net.add<CLSTokenLayer>(embed_dim);
  net.add<PositionalEmbeddingLayer>(num_patches + 1, embed_dim);

  for (int i = 0; i < num_layers; ++i)
    net.add<TransformerBlock>(embed_dim, num_heads, mlp_hidden_dim);

  net.add<LayerNormLayer>(embed_dim);
  net.add<ExtractCLSLayer>();
  net.add<DenseLayer>(embed_dim, n_classes);
  net.add<SoftmaxLayer>();

  return net;
}
