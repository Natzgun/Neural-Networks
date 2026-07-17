#include "core/ops/activations.cuh"
#include "cuda_utils.cuh"

#include <cfloat>
#include <iostream>

static inline void ensure_device(const Tensor& t) {
  if (!t.on_device()) {
    std::cerr << "activation op requires tensor on device" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__global__ void relu_forward_kernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = fmaxf(0.0f, in[idx]);
}

__global__ void relu_derivative_kernel(const float* output, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = output[idx] > 0.0f ? 1.0f : 0.0f;
}

__global__ void sigmoid_forward_kernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = fmaxf(-250.0f, fminf(250.0f, in[idx]));
    out[idx] = 1.0f / (1.0f + expf(-val));
  }
}

__global__ void sigmoid_derivative_kernel(const float* output, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = output[idx] * (1.0f - output[idx]);
}

__global__ void softmax_kernel(const float* in, float* out, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const float* row_in = in + row * cols;
  float* row_out = out + row * cols;

  float max_val = -FLT_MAX;
  for (int j = 0; j < cols; j++)
    max_val = fmaxf(max_val, row_in[j]);

  float sum = 0.0f;
  for (int j = 0; j < cols; j++) {
    row_out[j] = expf(row_in[j] - max_val);
    sum += row_out[j];
  }

  for (int j = 0; j < cols; j++)
    row_out[j] /= sum;
}

namespace ops {

Tensor relu_forward(const Tensor& x) {
  ensure_device(x);
  Tensor r(x.sizes());
  r.upload();
  int n = x.numel();
  relu_forward_kernel<<<div_ceil(n, 256), 256>>>(x.device_ptr(), r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor relu_derivative(const Tensor& output) {
  ensure_device(output);
  Tensor r(output.sizes());
  r.upload();
  int n = output.numel();
  relu_derivative_kernel<<<div_ceil(n, 256), 256>>>(output.device_ptr(), r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor sigmoid_forward(const Tensor& x) {
  ensure_device(x);
  Tensor r(x.sizes());
  r.upload();
  int n = x.numel();
  sigmoid_forward_kernel<<<div_ceil(n, 256), 256>>>(x.device_ptr(), r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor sigmoid_derivative(const Tensor& output) {
  ensure_device(output);
  Tensor r(output.sizes());
  r.upload();
  int n = output.numel();
  sigmoid_derivative_kernel<<<div_ceil(n, 256), 256>>>(output.device_ptr(), r.device_ptr(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor softmax_forward(const Tensor& x) {
  ensure_device(x);
  if (x.ndim() != 2) {
    std::cerr << "softmax requires 2D tensor" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Tensor r(x.sizes());
  r.upload();

  int rows = x.dim(0);
  int cols = x.dim(1);
  softmax_kernel<<<rows, 1>>>(x.device_ptr(), r.device_ptr(), rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return r;
}

Tensor softmax_backward(const Tensor& grad_output, const Tensor& softmax_output) {
  ensure_device(grad_output);
  ensure_device(softmax_output);
  if (grad_output.ndim() != 2 || softmax_output.ndim() != 2) {
    std::cerr << "softmax_backward requires 2D tensors" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int rows = grad_output.dim(0);
  int cols = grad_output.dim(1);

  // Host-side por ahora (sin kernel CUDA todavia).
  grad_output.download();
  softmax_output.download();

  Tensor dx({rows, cols});
  for (int r = 0; r < rows; ++r) {
    float dot = 0.0f;
    for (int c = 0; c < cols; ++c)
      dot += grad_output.at({r, c}) * softmax_output.at({r, c});

    for (int c = 0; c < cols; ++c) {
      float g = grad_output.at({r, c});
      float s = softmax_output.at({r, c});
      dx.at({r, c}) = s * (g - dot);
    }
  }

  dx.upload();
  return dx;
}

} // namespace ops
