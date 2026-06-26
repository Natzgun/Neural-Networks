#include "core/ops/conv_ops.cuh"
#include "cuda_utils.cuh"

#include <iostream>

__global__ void conv2d_forward_kernel(const float* in, const float* w,
                                      const float* b, float* out,
                                      int N, int C, int H, int W,
                                      int K, int kH, int kW,
                                      int outH, int outW,
                                      int stride, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int tmp = idx / outW;
    int oh = tmp % outH;
    tmp /= outH;
    int k = tmp % K;
    int n = tmp / K;

    float sum = b[k];
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int ih = oh * stride - pad + kh;
                int iw = ow * stride - pad + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx = ((n * C + c) * H + ih) * W + iw;
                    int w_idx = ((k * C + c) * kH + kh) * kW + kw;
                    sum += in[in_idx] * w[w_idx];
                }
            }
        }
    }
    int out_idx = ((n * K + k) * outH + oh) * outW + ow;
    out[out_idx] = sum;
}

__global__ void conv2d_backward_input_kernel(const float* dout, const float* w,
                                             float* din,
                                             int N, int C, int H, int W,
                                             int K, int kH, int kW,
                                             int outH, int outW,
                                             int stride, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int iw = idx % W;
    int tmp = idx / W;
    int ih = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int n = tmp / C;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int top = ih + pad - kh;
                int left = iw + pad - kw;
                if (top % stride != 0 || left % stride != 0) continue;
                int oh = top / stride;
                int ow = left / stride;
                if (oh < 0 || oh >= outH || ow < 0 || ow >= outW) continue;

                int dout_idx = ((n * K + k) * outH + oh) * outW + ow;
                int w_idx = ((k * C + c) * kH + kh) * kW + kw;
                sum += dout[dout_idx] * w[w_idx];
            }
        }
    }
    din[idx] = sum;
}

__global__ void conv2d_backward_weights_kernel(const float* in, const float* dout,
                                               float* dw,
                                               int N, int C, int H, int W,
                                               int K, int kH, int kW,
                                               int outH, int outW,
                                               int stride, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * C * kH * kW;
    if (idx >= total) return;

    int kw = idx % kW;
    int tmp = idx / kW;
    int kh = tmp % kH;
    tmp /= kH;
    int c = tmp % C;
    int k = tmp / C;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                int ih = oh * stride - pad + kh;
                int iw = ow * stride - pad + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx = ((n * C + c) * H + ih) * W + iw;
                    int dout_idx = ((n * K + k) * outH + oh) * outW + ow;
                    sum += in[in_idx] * dout[dout_idx];
                }
            }
        }
    }
    dw[idx] = sum;
}

__global__ void conv2d_backward_bias_kernel(const float* dout, float* db,
                                            int N, int K, int outH, int outW) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                int idx = ((n * K + k) * outH + oh) * outW + ow;
                sum += dout[idx];
            }
        }
    }
    db[k] = sum;
}

namespace ops {

Tensor conv2d_forward(const Tensor& input, const Tensor& weights,
                      const Tensor& bias, int stride, int padding) {
    if (input.ndim() != 4 || weights.ndim() != 4 || bias.ndim() != 1) {
        std::cerr << "conv2d_forward: invalid dimensions" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int N = input.dim(0);
    int C = input.dim(1);
    int H = input.dim(2);
    int W = input.dim(3);

    int K = weights.dim(0);
    int kH = weights.dim(2);
    int kW = weights.dim(3);

    if (weights.dim(1) != C) {
        std::cerr << "conv2d_forward: channel mismatch" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int outH = (H + 2 * padding - kH) / stride + 1;
    int outW = (W + 2 * padding - kW) / stride + 1;

    Tensor output({N, K, outH, outW});
    output.upload();

    int total = N * K * outH * outW;
    conv2d_forward_kernel<<<div_ceil(total, 256), 256>>>(
        input.device_ptr(), weights.device_ptr(), bias.device_ptr(),
        output.device_ptr(),
        N, C, H, W, K, kH, kW, outH, outW, stride, padding);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

Tensor conv2d_backward_input(const Tensor& grad_output, const Tensor& weights,
                             int stride, int padding,
                             const std::vector<int>& input_shape) {
    int N = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];

    int K = weights.dim(0);
    int kH = weights.dim(2);
    int kW = weights.dim(3);

    int outH = grad_output.dim(2);
    int outW = grad_output.dim(3);

    Tensor grad_input({N, C, H, W});
    grad_input.upload();

    int total = N * C * H * W;
    conv2d_backward_input_kernel<<<div_ceil(total, 256), 256>>>(
        grad_output.device_ptr(), weights.device_ptr(), grad_input.device_ptr(),
        N, C, H, W, K, kH, kW, outH, outW, stride, padding);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

Tensor conv2d_backward_weights(const Tensor& input, const Tensor& grad_output,
                               int stride, int padding, int kernel_size,
                               int out_channels) {
    int N = input.dim(0);
    int C = input.dim(1);
    int H = input.dim(2);
    int W = input.dim(3);

    int K = out_channels;
    int kH = kernel_size;
    int kW = kernel_size;

    int outH = grad_output.dim(2);
    int outW = grad_output.dim(3);

    Tensor grad_weights({K, C, kH, kW});
    grad_weights.upload();

    int total = K * C * kH * kW;
    conv2d_backward_weights_kernel<<<div_ceil(total, 256), 256>>>(
        input.device_ptr(), grad_output.device_ptr(), grad_weights.device_ptr(),
        N, C, H, W, K, kH, kW, outH, outW, stride, padding);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_weights;
}

Tensor conv2d_backward_bias(const Tensor& grad_output) {
    int N = grad_output.dim(0);
    int K = grad_output.dim(1);
    int outH = grad_output.dim(2);
    int outW = grad_output.dim(3);

    Tensor grad_bias({K});
    grad_bias.upload();

    conv2d_backward_bias_kernel<<<div_ceil(K, 256), 256>>>(
        grad_output.device_ptr(), grad_bias.device_ptr(),
        N, K, outH, outW);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_bias;
}

}  // namespace ops
