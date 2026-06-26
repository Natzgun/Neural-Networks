#pragma once

#include "core/Tensor.cuh"

namespace ops {

Tensor conv2d_forward(const Tensor& input, const Tensor& weights,
                      const Tensor& bias, int stride, int padding);

Tensor conv2d_backward_input(const Tensor& grad_output, const Tensor& weights,
                             int stride, int padding,
                             const std::vector<int>& input_shape);

Tensor conv2d_backward_weights(const Tensor& input, const Tensor& grad_output,
                               int stride, int padding, int kernel_size,
                               int out_channels);

Tensor conv2d_backward_bias(const Tensor& grad_output);

}  // namespace ops
