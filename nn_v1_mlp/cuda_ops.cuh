#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " — "    \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

void cuda_fill(float *d_ptr, float val, int n);

void cuda_matmul(const float *A, const float *B, float *C, int M, int N, int K);

void cuda_transpose(const float *in, float *out, int rows, int cols);

void cuda_add(const float *A, const float *B, float *C, int n);

void cuda_sub(const float *A, const float *B, float *C, int n);

void cuda_sub_inplace(float *A, const float *B, int n);

void cuda_hadamard(const float *A, const float *B, float *C, int n);

void cuda_scalar_mul(const float *A, float s, float *C, int n);

void cuda_scalar_div(const float *A, float s, float *C, int n);

void cuda_add_bias(const float *A, const float *bias, float *C, int rows,
                   int cols);

void cuda_sum_rows(const float *A, float *out, int rows, int cols);

void cuda_relu_forward(const float *in, float *out, int n);

void cuda_relu_derivative(const float *output, float *out, int n);

void cuda_sigmoid_forward(const float *in, float *out, int n);

void cuda_sigmoid_derivative(const float *output, float *out, int n);

void cuda_softmax_forward(const float *in, float *out, int rows, int cols);

// ── im2col / col2im for convolution ────────────────────────────────────────

void cuda_im2col(const float *im, float *col, int N, int C, int H, int W, int k,
                 int s, int p, int out_h, int out_w);

void cuda_col2im(const float *col, float *im, int N, int C, int H, int W, int k,
                 int s, int p, int out_h, int out_w);
