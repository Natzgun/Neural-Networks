#include "core/ops/linalg.cuh"
#include "cuda_utils.cuh"

#include <cfloat>
#include <iostream>

static inline void ensure_device(const Tensor& t) {
  if (!t.on_device()) {
    std::cerr << "linalg op requires tensor on device" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

static inline void ensure_2d(const Tensor& t) {
  if (t.ndim() != 2) {
    std::cerr << "linalg op requires 2D tensor" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__global__ void fill_kernel(float* out, float val, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = val;
}

__global__ void add_kernel(const float* A, const float* B, float* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] + B[idx];
}

__global__ void sub_kernel(const float* A, const float* B, float* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] - B[idx];
}

__global__ void sub_inplace_kernel(float* A, const float* B, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    A[idx] -= B[idx];
}

__global__ void hadamard_kernel(const float* A, const float* B, float* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] * B[idx];
}

__global__ void scalar_mul_kernel(const float* A, float s, float* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] * s;
}

__global__ void scalar_div_kernel(const float* A, float s, float* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] / s;
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
      sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

__global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows && c < cols)
    out[c * rows + r] = in[r * cols + c];
}

__global__ void add_bias_kernel(const float* A, const float* bias, float* C, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols)
    C[idx] = A[idx] + bias[idx % cols];
}

__global__ void sum_rows_kernel(const float* A, float* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < cols) {
    float sum = 0.0f;
    for (int i = 0; i < rows; i++)
      sum += A[i * cols + col];
    out[col] = sum;
  }
}

namespace ops {

void fill(Tensor& a, float val) {
  if (!a.on_device())
    a.upload();
  int n = a.numel();
  fill_kernel<<<div_ceil(n, 256), 256>>>(a.device_ptr(), val, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

Tensor add(const Tensor& a, const Tensor& b) {
  ensure_device(a);
  ensure_device(b);
  Tensor r(a.sizes());
  r.upload();
  int n = a.numel();
  add_kernel<<<div_ceil(n, 256), 256>>>(a.device_ptr(), b.device_ptr(), r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor sub(const Tensor& a, const Tensor& b) {
  ensure_device(a);
  ensure_device(b);
  Tensor r(a.sizes());
  r.upload();
  int n = a.numel();
  sub_kernel<<<div_ceil(n, 256), 256>>>(a.device_ptr(), b.device_ptr(), r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor hadamard(const Tensor& a, const Tensor& b) {
  ensure_device(a);
  ensure_device(b);
  Tensor r(a.sizes());
  r.upload();
  int n = a.numel();
  hadamard_kernel<<<div_ceil(n, 256), 256>>>(a.device_ptr(), b.device_ptr(), r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor scalar_mul(const Tensor& a, float s) {
  ensure_device(a);
  Tensor r(a.sizes());
  r.upload();
  int n = a.numel();
  scalar_mul_kernel<<<div_ceil(n, 256), 256>>>(a.device_ptr(), s, r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor scalar_div(const Tensor& a, float s) {
  ensure_device(a);
  Tensor r(a.sizes());
  r.upload();
  int n = a.numel();
  scalar_div_kernel<<<div_ceil(n, 256), 256>>>(a.device_ptr(), s, r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

void sub_inplace(Tensor& a, const Tensor& b) {
  ensure_device(a);
  ensure_device(b);
  int n = a.numel();
  sub_inplace_kernel<<<div_ceil(n, 256), 256>>>(a.device_ptr(), b.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  ensure_device(a);
  ensure_device(b);
  ensure_2d(a);
  ensure_2d(b);

  int M = a.dim(0);
  int K = a.dim(1);
  int N = b.dim(1);

  if (b.dim(0) != K) {
    std::cerr << "matmul shape mismatch" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Tensor r({M, N});
  r.upload();

  dim3 block(16, 16);
  dim3 grid(div_ceil(N, 16), div_ceil(M, 16));
  matmul_kernel<<<grid, block>>>(a.device_ptr(), b.device_ptr(), r.device_ptr(), M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor transpose(const Tensor& a) {
  ensure_device(a);
  ensure_2d(a);

  int rows = a.dim(0);
  int cols = a.dim(1);
  Tensor r({cols, rows});
  r.upload();

  dim3 block(16, 16);
  dim3 grid(div_ceil(cols, 16), div_ceil(rows, 16));
  transpose_kernel<<<grid, block>>>(a.device_ptr(), r.device_ptr(), rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor add_bias(const Tensor& a, const Tensor& bias) {
  ensure_device(a);
  ensure_device(bias);
  ensure_2d(a);

  int rows = a.dim(0);
  int cols = a.dim(1);

  Tensor r(a.sizes());
  r.upload();
  int n = a.numel();
  add_bias_kernel<<<div_ceil(n, 256), 256>>>(a.device_ptr(), bias.device_ptr(), r.device_ptr(),
                                             rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor sum_rows(const Tensor& a) {
  ensure_device(a);
  ensure_2d(a);

  int rows = a.dim(0);
  int cols = a.dim(1);
  Tensor r({1, cols});
  r.upload();

  sum_rows_kernel<<<div_ceil(cols, 256), 256>>>(a.device_ptr(), r.device_ptr(), rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

} // namespace ops
