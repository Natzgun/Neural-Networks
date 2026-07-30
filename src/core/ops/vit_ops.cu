#include "core/ops/vit_ops.cuh"
#include "cuda_utils.cuh"

#include <iostream>

static inline void ensure_device(const Tensor& t) {
  if (!t.on_device()) {
    std::cerr << "vit op requires tensor on device" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// ── LayerNorm ────────────────────────────────────────────────────────────

__global__ void layer_norm_forward_kernel(const float* x, const float* gamma, const float* beta,
                                          float* out, float* x_norm, float* std_out, int rows,
                                          int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const float* row_x = x + (size_t)row * cols;
  float* row_norm = x_norm + (size_t)row * cols;
  float* row_out = out + (size_t)row * cols;

  float mean = 0.0f;
  for (int c = 0; c < cols; ++c)
    mean += row_x[c];
  mean /= cols;

  float var = 0.0f;
  for (int c = 0; c < cols; ++c) {
    float diff = row_x[c] - mean;
    var += diff * diff;
  }
  var /= cols;

  float std_val = sqrtf(var + eps);
  std_out[row] = std_val;

  for (int c = 0; c < cols; ++c) {
    float xn = (row_x[c] - mean) / std_val;
    row_norm[c] = xn;
    row_out[c] = gamma[c] * xn + beta[c];
  }
}

__global__ void layer_norm_backward_kernel(const float* grad_output, const float* gamma,
                                           const float* x_norm, const float* std_in, float* dx,
                                           float* dgamma, float* dbeta, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const float* g = grad_output + (size_t)row * cols;
  const float* xn = x_norm + (size_t)row * cols;
  float std_val = std_in[row];

  float mean_dxn = 0.0f;
  float mean_dxn_xn = 0.0f;
  for (int c = 0; c < cols; ++c) {
    float dxn_c = g[c] * gamma[c];
    mean_dxn += dxn_c;
    mean_dxn_xn += dxn_c * xn[c];
  }
  mean_dxn /= cols;
  mean_dxn_xn /= cols;

  float* row_dx = dx + (size_t)row * cols;
  for (int c = 0; c < cols; ++c) {
    float dxn_c = g[c] * gamma[c];
    row_dx[c] = (dxn_c - mean_dxn - xn[c] * mean_dxn_xn) / std_val;
    atomicAdd(&dgamma[c], g[c] * xn[c]);
    atomicAdd(&dbeta[c], g[c]);
  }
}

// ── Permute BCP <-> BPC (PatchEmbedding) ────────────────────────────────

__global__ void bcp_to_bpc_kernel(const float* in, float* out, int batch, int channels,
                                  int patches) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * patches * channels;
  if (idx >= total)
    return;

  int c = idx % channels;
  int tmp = idx / channels;
  int p = tmp % patches;
  int b = tmp / patches;

  out[idx] = in[((size_t)b * channels + c) * patches + p];
}

__global__ void bpc_to_bcp_kernel(const float* in, float* out, int batch, int channels,
                                  int patches) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * channels * patches;
  if (idx >= total)
    return;

  int p = idx % patches;
  int tmp = idx / patches;
  int c = tmp % channels;
  int b = tmp / channels;

  out[idx] = in[((size_t)b * patches + p) * channels + c];
}

// ── CLS token prepend / extract ─────────────────────────────────────────

__global__ void prepend_cls_kernel(const float* input, const float* cls, float* out, int batch,
                                   int tokens, int embed_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_tokens = tokens + 1;
  int total = batch * out_tokens * embed_dim;
  if (idx >= total)
    return;

  int e = idx % embed_dim;
  int tmp = idx / embed_dim;
  int n = tmp % out_tokens;
  int b = tmp / out_tokens;

  if (n == 0) {
    out[idx] = cls[e];
  } else {
    out[idx] = input[((size_t)b * tokens + (n - 1)) * embed_dim + e];
  }
}

__global__ void cls_token_backward_kernel(const float* grad_output, float* grad_input,
                                          float* grad_cls, int batch, int tokens, int embed_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_tokens = tokens + 1;
  int total = batch * out_tokens * embed_dim;
  if (idx >= total)
    return;

  int e = idx % embed_dim;
  int tmp = idx / embed_dim;
  int n = tmp % out_tokens;
  int b = tmp / out_tokens;

  float g = grad_output[idx];
  if (n == 0) {
    atomicAdd(&grad_cls[e], g);
  } else {
    grad_input[((size_t)b * tokens + (n - 1)) * embed_dim + e] = g;
  }
}

__global__ void extract_cls_forward_kernel(const float* input, float* out, int batch, int tokens,
                                           int embed_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * embed_dim;
  if (idx >= total)
    return;

  int e = idx % embed_dim;
  int b = idx / embed_dim;
  out[idx] = input[(size_t)b * tokens * embed_dim + e];
}

__global__ void extract_cls_backward_kernel(const float* grad_output, float* grad_input,
                                            int batch, int tokens, int embed_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * embed_dim;
  if (idx >= total)
    return;

  int e = idx % embed_dim;
  int b = idx / embed_dim;
  grad_input[(size_t)b * tokens * embed_dim + e] = grad_output[idx];
}

namespace ops {

Tensor layer_norm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps,
                          Tensor& x_norm_out, Tensor& std_out) {
  ensure_device(x);
  ensure_device(gamma);
  ensure_device(beta);

  int rows = x.dim(0);
  int cols = x.dim(1);

  Tensor out({rows, cols});
  out.upload();
  x_norm_out = Tensor({rows, cols});
  x_norm_out.upload();
  std_out = Tensor({rows});
  std_out.upload();

  layer_norm_forward_kernel<<<rows, 1>>>(x.device_ptr(), gamma.device_ptr(), beta.device_ptr(),
                                         out.device_ptr(), x_norm_out.device_ptr(),
                                         std_out.device_ptr(), rows, cols, eps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

Tensor layer_norm_backward(const Tensor& grad_output, const Tensor& gamma, const Tensor& x_norm,
                           const Tensor& std, Tensor& dgamma_out, Tensor& dbeta_out) {
  ensure_device(grad_output);
  ensure_device(gamma);
  ensure_device(x_norm);
  ensure_device(std);

  int rows = grad_output.dim(0);
  int cols = grad_output.dim(1);

  Tensor dx({rows, cols});
  dx.upload();
  dgamma_out = Tensor::zeros({cols});
  dgamma_out.upload();
  dbeta_out = Tensor::zeros({cols});
  dbeta_out.upload();

  layer_norm_backward_kernel<<<rows, 1>>>(grad_output.device_ptr(), gamma.device_ptr(),
                                          x_norm.device_ptr(), std.device_ptr(), dx.device_ptr(),
                                          dgamma_out.device_ptr(), dbeta_out.device_ptr(), rows,
                                          cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return dx;
}

Tensor patch_tokens_forward(const Tensor& x, int batch, int channels, int patches) {
  ensure_device(x);
  Tensor out({batch, patches, channels});
  out.upload();
  int n = out.numel();
  bcp_to_bpc_kernel<<<div_ceil(n, 256), 256>>>(x.device_ptr(), out.device_ptr(), batch, channels,
                                               patches);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

Tensor patch_tokens_backward(const Tensor& grad_output, int batch, int channels, int patches) {
  ensure_device(grad_output);
  Tensor out({batch, channels, patches});
  out.upload();
  int n = out.numel();
  bpc_to_bcp_kernel<<<div_ceil(n, 256), 256>>>(grad_output.device_ptr(), out.device_ptr(), batch,
                                               channels, patches);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

Tensor prepend_cls_token(const Tensor& input, const Tensor& cls_token, int batch, int tokens,
                         int embed_dim) {
  ensure_device(input);
  ensure_device(cls_token);
  Tensor out({batch, tokens + 1, embed_dim});
  out.upload();
  int n = out.numel();
  prepend_cls_kernel<<<div_ceil(n, 256), 256>>>(input.device_ptr(), cls_token.device_ptr(),
                                                out.device_ptr(), batch, tokens, embed_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

Tensor cls_token_backward(const Tensor& grad_output, int batch, int tokens, int embed_dim,
                          Tensor& grad_cls_out) {
  ensure_device(grad_output);
  Tensor grad_input({batch, tokens, embed_dim});
  grad_input.upload();
  grad_cls_out = Tensor::zeros({embed_dim});
  grad_cls_out.upload();

  int n = grad_output.numel();
  cls_token_backward_kernel<<<div_ceil(n, 256), 256>>>(
      grad_output.device_ptr(), grad_input.device_ptr(), grad_cls_out.device_ptr(), batch, tokens,
      embed_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return grad_input;
}

Tensor extract_cls_forward(const Tensor& input, int batch, int tokens, int embed_dim) {
  ensure_device(input);
  Tensor out({batch, embed_dim});
  out.upload();
  int n = out.numel();
  extract_cls_forward_kernel<<<div_ceil(n, 256), 256>>>(input.device_ptr(), out.device_ptr(),
                                                        batch, tokens, embed_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

Tensor extract_cls_backward(const Tensor& grad_output, int batch, int tokens, int embed_dim) {
  ensure_device(grad_output);
  Tensor grad_input = Tensor::zeros({batch, tokens, embed_dim});
  grad_input.upload();
  int n = grad_output.numel();
  extract_cls_backward_kernel<<<div_ceil(n, 256), 256>>>(grad_output.device_ptr(),
                                                         grad_input.device_ptr(), batch, tokens,
                                                         embed_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return grad_input;
}

} // namespace ops
