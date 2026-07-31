#include "core/ops/attention_ops.cuh"
#include "core/ops/linalg.cuh"

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

void test_matmul_non_multiple_dimensions() {
  constexpr int M = 3;
  constexpr int K = 17;
  constexpr int N = 5;

  Tensor A({M, K});
  Tensor B({K, N});
  for (int i = 0; i < A.numel(); ++i)
    A.flat(i) = static_cast<float>((i % 7) - 3) * 0.25f;
  for (int i = 0; i < B.numel(); ++i)
    B.flat(i) = static_cast<float>((i % 5) - 2) * 0.2f;
  A.upload();
  B.upload();

  Tensor out = ops::matmul(A, B);
  out.download();
  expect_shape(out, {M, N}, "matmul tiled (dimensiones no multiplos)");

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float expected = 0.0f;
      for (int k = 0; k < K; ++k)
        expected += A.at({row, k}) * B.at({k, col});
      expect_close(out.at({row, col}), expected, 1e-4f, "matmul tiled vs CPU");
    }
  }
  std::cout << "matmul tiled coincide con referencia CPU: OK\n";
}

// Extrae la sub-matriz 2D correspondiente a (b, h) de un tensor 4D
// {batch, heads, rows, cols}, para comparar contra ops::matmul (ya probado).
Tensor extract_slice(const Tensor& t, int b, int h) {
  t.download();
  int rows = t.dim(2);
  int cols = t.dim(3);
  Tensor slice({rows, cols});
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      slice.at({i, j}) = t.at({b, h, i, j});
  slice.upload();
  return slice;
}
} // namespace

int main() {
  test_matmul_non_multiple_dimensions();

  int batch = 2, heads = 2, M = 3, K = 4, N = 5;

  Tensor A = Tensor::random_uniform({batch, heads, M, K}, -1.0f, 1.0f);
  Tensor B = Tensor::random_uniform({batch, heads, K, N}, -1.0f, 1.0f);
  A.upload();
  B.upload();

  // --- Caso normal: A @ B ---
  Tensor out = ops::batched_matmul(A, B);
  expect_shape(out, {batch, heads, M, N}, "batched_matmul (sin transpose)");
  out.download();

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      Tensor a_slice = extract_slice(A, b, h);
      Tensor b_slice = extract_slice(B, b, h);
      Tensor expected = ops::matmul(a_slice, b_slice);
      expected.download();

      for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
          expect_close(out.at({b, h, i, j}), expected.at({i, j}), 1e-4f,
                       "batched_matmul vs ops::matmul");
    }
  }
  std::cout << "batched_matmul (sin transpose) coincide con ops::matmul por slice: OK\n";

  // --- Caso transpose_b=true: simula Q @ K^T de la atencion ---
  Tensor Bt = Tensor::random_uniform({batch, heads, N, K}, -1.0f, 1.0f);
  Bt.upload();

  Tensor out_t = ops::batched_matmul(A, Bt, /*transpose_b=*/true);
  expect_shape(out_t, {batch, heads, M, N}, "batched_matmul (transpose_b)");
  out_t.download();

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      Tensor a_slice = extract_slice(A, b, h);
      Tensor bt_slice = extract_slice(Bt, b, h);
      Tensor expected = ops::matmul(a_slice, ops::transpose(bt_slice));
      expected.download();

      for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
          expect_close(out_t.at({b, h, i, j}), expected.at({i, j}), 1e-4f,
                       "batched_matmul(transpose_b) vs ops::matmul+transpose");
    }
  }
  std::cout << "batched_matmul (transpose_b) coincide con ops::matmul+transpose: OK\n";

  std::cout << "batched_matmul: OK" << std::endl;
  return 0;
}
