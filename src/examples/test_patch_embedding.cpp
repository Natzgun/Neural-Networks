#include "layers/PatchEmbeddingLayer.hpp"

#include <cstdlib>
#include <iostream>

namespace {
void expect_shape(const Tensor& t, const std::vector<int>& expected, const std::string& label) {
  if (t.sizes() != expected) {
    std::cerr << label << ": shape inesperada" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << label << ": OK ";
  t.print();
}
} // namespace

int main() {
  int batch = 2;
  int in_channels = 1;
  int image_h = 28;
  int image_w = 28;
  int patch_size = 7;
  int embed_dim = 32;

  int grid_h = image_h / patch_size;
  int grid_w = image_w / patch_size;
  int num_patches = grid_h * grid_w;

  PatchEmbeddingLayer layer(image_h, image_w, patch_size, in_channels, embed_dim);

  Tensor input = Tensor::random_uniform({batch, in_channels, image_h, image_w}, -1.0f, 1.0f);
  input.upload();

  Tensor tokens = layer.forward(input);
  expect_shape(tokens, {batch, num_patches, embed_dim}, "forward output");

  Tensor grad_output = Tensor::random_uniform({batch, num_patches, embed_dim}, -1.0f, 1.0f);
  grad_output.upload();

  Tensor grad_input = layer.backward(grad_output);
  expect_shape(grad_input, {batch, in_channels, image_h, image_w}, "backward output (grad_input)");

  std::cout << "PatchEmbeddingLayer: shapes correctas." << std::endl;
  return 0;
}
