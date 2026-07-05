#include "core/ops/pool_ops.cuh"
#include "core/ops/linalg.cuh"

#include <cfloat>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

__global__ void maxpool2d_forward_kernel(const float* in, float* out, int* mask, int N, int C,
                                         int H, int W, int k, int stride, int outH, int outW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * outH * outW;
  if (idx >= total)
    return;

  int ow = idx % outW;
  int tmp = idx / outW;
  int oh = tmp % outH;
  tmp /= outH;
  int c = tmp % C;
  int n = tmp / C;

  int h_start = oh * stride;
  int w_start = ow * stride;

  float max_val = -FLT_MAX;
  int max_idx = -1;
  for (int kh = 0; kh < k; ++kh) {
    for (int kw = 0; kw < k; ++kw) {
      int ih = h_start + kh;
      int iw = w_start + kw;
      if (ih < H && iw < W) {
        int in_idx = ((n * C + c) * H + ih) * W + iw;
        if (in[in_idx] > max_val) {
          max_val = in[in_idx];
          max_idx = in_idx;
        }
      }
    }
  }
  out[idx] = max_val;
  mask[idx] = max_idx;
}

__global__ void maxpool2d_backward_kernel(const float* dout, const int* mask, float* din, int N,
                                          int C, int outH, int outW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * outH * outW;
  if (idx >= total)
    return;

  int in_idx = mask[idx];
  if (in_idx >= 0) {
    atomicAdd(&din[in_idx], dout[idx]);
  }
}

void ops::PoolMask::free() {
  if (data) {
    cudaFree(data);
    data = nullptr;
    size = 0;
  }
}

namespace ops {

Tensor maxpool2d_forward(const Tensor& input, int kernel_size, int stride, PoolMask& mask) {
  int N = input.dim(0);
  int C = input.dim(1);
  int H = input.dim(2);
  int W = input.dim(3);

  int outH = (H - kernel_size) / stride + 1;
  int outW = (W - kernel_size) / stride + 1;

  Tensor output({N, C, outH, outW});
  output.upload();

  int new_size = N * C * outH * outW;
  if (new_size > mask.size) {
    mask.free();
    CUDA_CHECK(cudaMalloc(&mask.data, new_size * sizeof(int)));
    mask.size = new_size;
  }

  int total = new_size;
  maxpool2d_forward_kernel<<<div_ceil(total, 256), 256>>>(input.device_ptr(), output.device_ptr(),
                                                          mask.data, N, C, H, W, kernel_size,
                                                          stride, outH, outW);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return output;
}

Tensor maxpool2d_backward(const Tensor& grad_output, const PoolMask& mask,
                          const std::vector<int>& input_shape) {
  int N = input_shape[0];
  int C = input_shape[1];
  int H = input_shape[2];
  int W = input_shape[3];

  int outH = grad_output.dim(2);
  int outW = grad_output.dim(3);

  Tensor grad_input(input_shape);
  ops::fill(grad_input, 0.0f);

  int total = N * C * outH * outW;
  maxpool2d_backward_kernel<<<div_ceil(total, 256), 256>>>(
      grad_output.device_ptr(), mask.data, grad_input.device_ptr(), N, C, outH, outW);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return grad_input;
}

} // namespace ops
