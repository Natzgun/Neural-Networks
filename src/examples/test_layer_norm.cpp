#include "layers/LayerNormLayer.hpp"

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

void expect_close(float value, float target, float tol, const std::string& label) {
  if (std::fabs(value - target) > tol) {
    std::cerr << label << ": valor inesperado (" << value << ", esperado ~" << target << ")"
               << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
} // namespace

int main() {
  int batch = 2;
  int tokens = 5;
  int embed_dim = 8;

  LayerNormLayer layer(embed_dim);

  // Valores lejos de media 0 / var 1 para forzar a que la normalizacion haga
  // trabajo real.
  Tensor input = Tensor::random_normal({batch, tokens, embed_dim}, 10.0f, 5.0f);
  input.upload();

  Tensor out = layer.forward(input);
  expect_shape(out, {batch, tokens, embed_dim}, "forward output");

  // gamma=1, beta=0 al inicio => cada fila de la salida deberia tener
  // media ~0 y desviacion estandar ~1.
  out.download();
  for (int b = 0; b < batch; ++b) {
    for (int n = 0; n < tokens; ++n) {
      float mean = 0.0f;
      for (int e = 0; e < embed_dim; ++e)
        mean += out.at({b, n, e});
      mean /= embed_dim;

      float var = 0.0f;
      for (int e = 0; e < embed_dim; ++e) {
        float diff = out.at({b, n, e}) - mean;
        var += diff * diff;
      }
      var /= embed_dim;

      expect_close(mean, 0.0f, 1e-3f, "media por token");
      expect_close(std::sqrt(var), 1.0f, 1e-2f, "desviacion por token");
    }
  }
  std::cout << "forward normaliza cada token: OK" << std::endl;

  Tensor grad_output = Tensor::random_uniform({batch, tokens, embed_dim}, -1.0f, 1.0f);
  grad_output.upload();

  Tensor grad_input = layer.backward(grad_output);
  expect_shape(grad_input, {batch, tokens, embed_dim}, "backward output (grad_input)");

  std::cout << "LayerNormLayer: shapes y normalizacion correctas." << std::endl;
  return 0;
}
