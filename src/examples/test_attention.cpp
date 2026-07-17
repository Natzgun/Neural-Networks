#include "layers/MultiHeadAttentionLayer.hpp"

#include <algorithm>
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

// "perdida" ficticia sum(Y * dY), igual que en la referencia Python, para
// hacer gradient checking numerico sobre dX.
float loss_fn(MultiHeadAttentionLayer& layer, const Tensor& x, const Tensor& dy) {
  Tensor y = layer.forward(x);
  y.download();

  float loss = 0.0f;
  for (int i = 0; i < y.numel(); ++i)
    loss += y.flat(i) * dy.flat(i);
  return loss;
}
} // namespace

int main() {
  int batch = 2;
  int tokens = 3;
  int embed_dim = 6;
  int heads = 2;

  MultiHeadAttentionLayer layer(embed_dim, heads);

  Tensor X = Tensor::random_normal({batch, tokens, embed_dim}, 0.0f, 1.0f);
  X.upload();

  Tensor Y = layer.forward(X);
  Y.download();
  expect_shape(Y, {batch, tokens, embed_dim}, "forward output");

  Tensor dY = Tensor::random_normal({batch, tokens, embed_dim}, 0.0f, 1.0f);
  dY.upload();
  dY.download(); // dY.flat(i) se usa en host dentro de loss_fn

  Tensor dX = layer.backward(dY);
  dX.download();
  expect_shape(dX, {batch, tokens, embed_dim}, "backward output (dX)");

  // Gradient checking numerico: confirma que Q/K/V -> softmax -> combine ->
  // W_o esta bien encadenado de punta a punta.
  X.download();

  float eps = 1e-3f;
  float max_diff = 0.0f;
  int checks = 0;

  for (int b = 0; b < batch; ++b) {
    for (int n = 0; n < tokens; ++n) {
      for (int e = 0; e < embed_dim; ++e) {
        Tensor x_plus = X;
        Tensor x_minus = X;
        x_plus.at({b, n, e}) += eps;
        x_minus.at({b, n, e}) -= eps;
        x_plus.upload();
        x_minus.upload();

        float loss_plus = loss_fn(layer, x_plus, dY);
        float loss_minus = loss_fn(layer, x_minus, dY);
        float numeric = (loss_plus - loss_minus) / (2.0f * eps);
        float analytic = dX.at({b, n, e});

        max_diff = std::max(max_diff, std::fabs(numeric - analytic));
        ++checks;
      }
    }
  }

  std::cout << "gradient checking (" << checks << " elementos), max diff: " << max_diff
             << std::endl;

  if (max_diff > 5e-2f) {
    std::cerr << "gradient checking: diferencia demasiado grande" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "MultiHeadAttentionLayer: shapes y gradient checking correctos." << std::endl;
  return 0;
}
