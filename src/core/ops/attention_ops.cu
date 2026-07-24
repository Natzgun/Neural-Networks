#include "core/ops/attention_ops.cuh"
#include "cuda_utils.cuh"

#include <cstdlib>
#include <iostream>

static inline void ensure_device(const Tensor& t) {
  if (!t.on_device()) {
    std::cerr << "batched_matmul requires tensors on device" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// Una matriz (batch, head) por cada valor de blockIdx.z; dentro de esa
// matriz, cada thread resuelve un C[i][j] con el mismo patron que
// matmul_kernel (linalg.cu). transpose_a/transpose_b no mueven datos: solo
// cambian que (fila, col) fisica se lee para el mismo (i, j, k) logico,
// asi que A y B pueden pasarse tal cual estan en memoria.
__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int bh_total,
                                       int M, int N, int K, int a_dim2, int a_dim3, int b_dim2,
                                       int b_dim3, bool transpose_a, bool transpose_b) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int bh = blockIdx.z;

  if (i >= M || j >= N || bh >= bh_total)
    return;

  const float* A_mat = A + (size_t)bh * a_dim2 * a_dim3;
  const float* B_mat = B + (size_t)bh * b_dim2 * b_dim3;
  float* C_mat = C + (size_t)bh * M * N;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    int a_row = transpose_a ? k : i;
    int a_col = transpose_a ? i : k;
    int b_row = transpose_b ? j : k;
    int b_col = transpose_b ? k : j;
    sum += A_mat[a_row * a_dim3 + a_col] * B_mat[b_row * b_dim3 + b_col];
  }
  C_mat[i * N + j] = sum;
}

// idx recorre el tensor de salida {d0, d2, d1, d3}; se descompone en (a, c,
// b, e) y se lee del tensor de entrada en la posicion (a, b, c, e), que es
// donde vive ese mismo valor en {d0, d1, d2, d3}.
__global__ void swap_middle_axes_kernel(const float* in, float* out, int d0, int d1, int d2,
                                        int d3) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = d0 * d1 * d2 * d3;
  if (idx >= total)
    return;

  int e = idx % d3;
  int tmp = idx / d3;
  int b = tmp % d1;
  tmp /= d1;
  int c = tmp % d2;
  int a = tmp / d2;

  int in_idx = ((a * d1 + b) * d2 + c) * d3 + e;
  out[idx] = in[in_idx];
}

namespace ops {

Tensor batched_matmul(const Tensor& a, const Tensor& b, bool transpose_b, bool transpose_a) {
  ensure_device(a);
  ensure_device(b);

  if (a.ndim() != 4 || b.ndim() != 4) {
    std::cerr << "batched_matmul: se esperan tensores 4D {batch, heads, rows, cols}" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int batch = a.dim(0);
  int heads = a.dim(1);
  int M = transpose_a ? a.dim(3) : a.dim(2);
  int K = transpose_a ? a.dim(2) : a.dim(3);

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

  Tensor out({batch, heads, M, N});
  out.upload();

  int bh_total = batch * heads;
  dim3 block(16, 16);
  dim3 grid(div_ceil(N, 16), div_ceil(M, 16), bh_total);
  batched_matmul_kernel<<<grid, block>>>(a.device_ptr(), b.device_ptr(), out.device_ptr(),
                                         bh_total, M, N, K, a.dim(2), a.dim(3), b.dim(2), b.dim(3),
                                         transpose_a, transpose_b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

Tensor swap_middle_axes(const Tensor& x) {
  ensure_device(x);
  if (x.ndim() != 4) {
    std::cerr << "swap_middle_axes: se espera un tensor 4D" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int d0 = x.dim(0), d1 = x.dim(1), d2 = x.dim(2), d3 = x.dim(3);
  Tensor out({d0, d2, d1, d3});
  out.upload();

  int n = x.numel();
  swap_middle_axes_kernel<<<div_ceil(n, 256), 256>>>(x.device_ptr(), out.device_ptr(), d0, d1, d2,
                                                     d3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

} // namespace ops
