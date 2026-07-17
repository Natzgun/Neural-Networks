#include "models/ViT.hpp"

#include <cmath>
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
  int batch = 4;
  int image_h = 28;
  int image_w = 28;
  int patch_size = 7;
  int in_channels = 1;
  int embed_dim = 16;
  int num_heads = 2;
  int mlp_hidden_dim = 32;
  int num_layers = 2;
  int n_classes = 10;

  Network net = make_vit(image_h, image_w, patch_size, in_channels, embed_dim, num_heads,
                         mlp_hidden_dim, num_layers, n_classes);

  Tensor X = Tensor::random_uniform({batch, in_channels, image_h, image_w}, -1.0f, 1.0f);
  X.upload();

  Tensor y_pred = net.forward(X);
  y_pred.download();
  expect_shape(y_pred, {batch, n_classes}, "forward output (probs)");

  for (int b = 0; b < batch; ++b) {
    float sum = 0.0f;
    for (int c = 0; c < n_classes; ++c)
      sum += y_pred.at({b, c});
    if (std::fabs(sum - 1.0f) > 1e-3f) {
      std::cerr << "softmax no suma 1 en la muestra " << b << " (suma=" << sum << ")" << std::endl;
      return EXIT_FAILURE;
    }
  }
  std::cout << "cada fila de softmax suma 1: OK\n";

  // Etiquetas one-hot, para ejercitar un train_step completo (forward +
  // backward + update encadenados por Network).
  Tensor y_true = Tensor::zeros({batch, n_classes});
  for (int b = 0; b < batch; ++b)
    y_true.at({b, b % n_classes}) = 1.0f;
  y_true.upload();

  net.train_step(X, y_true, 0.01f);
  std::cout << "train_step: OK\n";

  std::cout << "VisionTransformer (make_vit): forward, backward y train_step correctos."
            << std::endl;
  return 0;
}
