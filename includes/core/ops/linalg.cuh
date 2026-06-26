#pragma once

#include "core/Tensor.cuh"

namespace ops {

// Element-wise / scalar operations on Tensors. All assume device memory is
// allocated and valid; result is allocated on device automatically.
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor hadamard(const Tensor& a, const Tensor& b);
Tensor scalar_mul(const Tensor& a, float s);
Tensor scalar_div(const Tensor& a, float s);

// In-place: a = a - b (device memory).
void sub_inplace(Tensor& a, const Tensor& b);

// Matrix multiplication. A shape (M, K), B shape (K, N) -> result (M, N).
Tensor matmul(const Tensor& a, const Tensor& b);

// Transpose. Input (rows, cols) -> output (cols, rows).
Tensor transpose(const Tensor& a);

// Add broadcasted bias: A (rows, cols) + bias (1, cols).
Tensor add_bias(const Tensor& a, const Tensor& bias);

// Sum over rows: A (rows, cols) -> output (1, cols).
Tensor sum_rows(const Tensor& a);

// Fill a tensor with a scalar value on device.
void fill(Tensor& a, float val);

}  // namespace ops
