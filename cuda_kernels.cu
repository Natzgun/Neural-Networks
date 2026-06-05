#include <cfloat>

static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void fill_kernel(float *out, float val, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = val;
}

void cuda_fill(float *d_ptr, float val, int n) {
  int block = 256;
  fill_kernel<<<div_ceil(n, block), block>>>(d_ptr, val, n);
  cudaDeviceSynchronize();
}

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
      sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

void cuda_matmul(const float *A, const float *B, float *C, int M, int N,
                 int K) {
  dim3 block(16, 16);
  dim3 grid(div_ceil(N, 16), div_ceil(M, 16));
  matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

__global__ void transpose_kernel(const float *in, float *out, int rows,
                                 int cols) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows && c < cols)
    out[c * rows + r] = in[r * cols + c];
}

void cuda_transpose(const float *in, float *out, int rows, int cols) {
  dim3 block(16, 16);
  dim3 grid(div_ceil(cols, 16), div_ceil(rows, 16));
  transpose_kernel<<<grid, block>>>(in, out, rows, cols);
  cudaDeviceSynchronize();
}

__global__ void add_kernel(const float *A, const float *B, float *C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] + B[idx];
}

void cuda_add(const float *A, const float *B, float *C, int n) {
  int block = 256;
  add_kernel<<<div_ceil(n, block), block>>>(A, B, C, n);
  cudaDeviceSynchronize();
}

__global__ void sub_kernel(const float *A, const float *B, float *C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] - B[idx];
}

void cuda_sub(const float *A, const float *B, float *C, int n) {
  int block = 256;
  sub_kernel<<<div_ceil(n, block), block>>>(A, B, C, n);
  cudaDeviceSynchronize();
}

__global__ void sub_inplace_kernel(float *A, const float *B, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    A[idx] -= B[idx];
}

void cuda_sub_inplace(float *A, const float *B, int n) {
  int block = 256;
  sub_inplace_kernel<<<div_ceil(n, block), block>>>(A, B, n);
  cudaDeviceSynchronize();
}

__global__ void hadamard_kernel(const float *A, const float *B, float *C,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] * B[idx];
}

void cuda_hadamard(const float *A, const float *B, float *C, int n) {
  int block = 256;
  hadamard_kernel<<<div_ceil(n, block), block>>>(A, B, C, n);
  cudaDeviceSynchronize();
}

__global__ void scalar_mul_kernel(const float *A, float s, float *C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] * s;
}

void cuda_scalar_mul(const float *A, float s, float *C, int n) {
  int block = 256;
  scalar_mul_kernel<<<div_ceil(n, block), block>>>(A, s, C, n);
  cudaDeviceSynchronize();
}

__global__ void scalar_div_kernel(const float *A, float s, float *C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    C[idx] = A[idx] / s;
}

void cuda_scalar_div(const float *A, float s, float *C, int n) {
  int block = 256;
  scalar_div_kernel<<<div_ceil(n, block), block>>>(A, s, C, n);
  cudaDeviceSynchronize();
}

// add bias
__global__ void add_bias_kernel(const float *A, const float *bias, float *C,
                                int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols)
    C[idx] = A[idx] + bias[idx % cols];
}

void cuda_add_bias(const float *A, const float *bias, float *C, int rows,
                   int cols) {
  int n = rows * cols;
  int block = 256;
  add_bias_kernel<<<div_ceil(n, block), block>>>(A, bias, C, rows, cols);
  cudaDeviceSynchronize();
}

// sum rows result[0][j] = sum_i A[i][j]

__global__ void sum_rows_kernel(const float *A, float *out, int rows,
                                int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < cols) {
    float sum = 0.0f;
    for (int i = 0; i < rows; i++)
      sum += A[i * cols + col];
    out[col] = sum;
  }
}

void cuda_sum_rows(const float *A, float *out, int rows, int cols) {
  int block = 256;
  sum_rows_kernel<<<div_ceil(cols, block), block>>>(A, out, rows, cols);
  cudaDeviceSynchronize();
}

__global__ void relu_forward_kernel(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = fmaxf(0.0f, in[idx]);
}

void cuda_relu_forward(const float *in, float *out, int n) {
  int block = 256;
  relu_forward_kernel<<<div_ceil(n, block), block>>>(in, out, n);
  cudaDeviceSynchronize();
}

__global__ void relu_derivative_kernel(const float *output, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = output[idx] > 0.0f ? 1.0f : 0.0f;
}

void cuda_relu_derivative(const float *output, float *out, int n) {
  int block = 256;
  relu_derivative_kernel<<<div_ceil(n, block), block>>>(output, out, n);
  cudaDeviceSynchronize();
}

// ── Sigmoid ─────────────────────────────────────────────────────────────────

__global__ void sigmoid_forward_kernel(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = fmaxf(-250.0f, fminf(250.0f, in[idx]));
    out[idx] = 1.0f / (1.0f + expf(-val));
  }
}

void cuda_sigmoid_forward(const float *in, float *out, int n) {
  int block = 256;
  sigmoid_forward_kernel<<<div_ceil(n, block), block>>>(in, out, n);
  cudaDeviceSynchronize();
}

__global__ void sigmoid_derivative_kernel(const float *output, float *out,
                                          int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = output[idx] * (1.0f - output[idx]);
}

void cuda_sigmoid_derivative(const float *output, float *out, int n) {
  int block = 256;
  sigmoid_derivative_kernel<<<div_ceil(n, block), block>>>(output, out, n);
  cudaDeviceSynchronize();
}

// ── Softmax

__global__ void softmax_kernel(const float *in, float *out, int rows,
                               int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const float *row_in = in + row * cols;
  float *row_out = out + row * cols;

  // Find max for numerical stability
  float max_val = -FLT_MAX;
  for (int j = 0; j < cols; j++)
    max_val = fmaxf(max_val, row_in[j]);

  // Exp and sum
  float sum = 0.0f;
  for (int j = 0; j < cols; j++) {
    row_out[j] = expf(row_in[j] - max_val);
    sum += row_out[j];
  }

  // Normalize
  for (int j = 0; j < cols; j++)
    row_out[j] /= sum;
}

void cuda_softmax_forward(const float *in, float *out, int rows, int cols) {
  // one block per row single thread per block
  // needs optimization
  softmax_kernel<<<rows, 1>>>(in, out, rows, cols);
  cudaDeviceSynchronize();
}
