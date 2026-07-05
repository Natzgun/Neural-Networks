#pragma once

#include "core/Tensor.cuh"

namespace ops {

Tensor relu_forward(const Tensor& x);
Tensor relu_derivative(const Tensor& output);

Tensor sigmoid_forward(const Tensor& x);
Tensor sigmoid_derivative(const Tensor& output);

Tensor softmax_forward(const Tensor& x);

} // namespace ops
