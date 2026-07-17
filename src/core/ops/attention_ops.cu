#include "core/ops/attention_ops.cuh"

#include <cstdlib>
#include <iostream>

static inline void ensure_device(const Tensor& t) {
  if (!t.on_device()) {
    std::cerr << "batched_matmul requires tensors on device" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

namespace ops {

Tensor batched_matmul(const Tensor& a, const Tensor& b, bool transpose_b) {
  ensure_device(a);
  ensure_device(b);

  if (a.ndim() != 4 || b.ndim() != 4) {
    std::cerr << "batched_matmul: se esperan tensores 4D {batch, heads, rows, cols}" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int batch = a.dim(0);
  int heads = a.dim(1);
  int M = a.dim(2);
  int K = a.dim(3);

  if (b.dim(0) != batch || b.dim(1) != heads) {
    std::cerr << "batched_matmul: batch/heads no coinciden" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int N = transpose_b ? b.dim(2) : b.dim(3);
  int Kb = transpose_b ? b.dim(3) : b.dim(2);
  if (Kb != K) {
    std::cerr << "batched_matmul: dimension interna no coincide" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  a.download();
  b.download();

  Tensor out({batch, heads, M, N});

  for (int bi = 0; bi < batch; ++bi) {
    for (int h = 0; h < heads; ++h) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k) {
            float a_val = a.at({bi, h, i, k});
            float b_val = transpose_b ? b.at({bi, h, j, k}) : b.at({bi, h, k, j});
            sum += a_val * b_val;
          }
          out.at({bi, h, i, j}) = sum;
        }
      }
    }
  }

  out.upload();
  return out;
}

} // namespace ops
