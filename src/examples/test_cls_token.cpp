#include "layers/CLSTokenLayer.hpp"

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
    std::cerr << label << ": valor inesperado (" << value << ", esperado " << target << ")"
               << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
} // namespace

int main() {
  int batch = 2;
  int tokens = 4;
  int embed_dim = 5;

  CLSTokenLayer layer(embed_dim);

  Tensor input = Tensor::random_uniform({batch, tokens, embed_dim}, -1.0f, 1.0f);
  input.upload();
  input.download(); // valores originales en host, para comparar despues

  Tensor out = layer.forward(input);
  expect_shape(out, {batch, tokens + 1, embed_dim}, "forward output");
  out.download();

  // Mismo token CLS en la posicion 0 para cada muestra del batch.
  for (int e = 0; e < embed_dim; ++e) {
    float cls_b0 = out.at({0, 0, e});
    for (int b = 1; b < batch; ++b)
      expect_close(out.at({b, 0, e}), cls_b0, 1e-6f, "cls token identico entre muestras");
  }

  // Los tokens originales quedan corridos una posicion, sin cambiar.
  for (int b = 0; b < batch; ++b)
    for (int n = 0; n < tokens; ++n)
      for (int e = 0; e < embed_dim; ++e)
        expect_close(out.at({b, n + 1, e}), input.at({b, n, e}), 1e-6f, "token original corrido");

  std::cout << "forward antepone el CLS y corre el resto: OK\n";

  Tensor grad_output = Tensor::random_uniform({batch, tokens + 1, embed_dim}, -1.0f, 1.0f);
  grad_output.upload();
  grad_output.download();

  Tensor grad_input = layer.backward(grad_output);
  expect_shape(grad_input, {batch, tokens, embed_dim}, "backward output (grad_input)");
  grad_input.download();

  // grad_input debe ser exactamente grad_output sin la primera posicion.
  for (int b = 0; b < batch; ++b)
    for (int n = 0; n < tokens; ++n)
      for (int e = 0; e < embed_dim; ++e)
        expect_close(grad_input.at({b, n, e}), grad_output.at({b, n + 1, e}), 1e-6f,
                     "grad_input == grad_output desplazado");

  std::cout << "backward recorta el gradiente del CLS y propaga el resto: OK\n";
  std::cout << "CLSTokenLayer: shapes y valores correctos." << std::endl;
  return 0;
}
