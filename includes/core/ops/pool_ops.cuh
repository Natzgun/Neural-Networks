#pragma once

#include "core/Tensor.cuh"

namespace ops {

struct PoolMask {
  int* data = nullptr;
  int size = 0;
  void free();
};

Tensor maxpool2d_forward(const Tensor& input, int kernel_size, int stride, PoolMask& mask);

Tensor maxpool2d_backward(const Tensor& grad_output, const PoolMask& mask,
                          const std::vector<int>& input_shape);

} // namespace ops
