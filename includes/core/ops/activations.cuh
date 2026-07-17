#pragma once

#include "core/Tensor.cuh"

namespace ops {

Tensor relu_forward(const Tensor& x);
Tensor relu_derivative(const Tensor& output);

Tensor sigmoid_forward(const Tensor& x);
Tensor sigmoid_derivative(const Tensor& output);

Tensor softmax_forward(const Tensor& x);

// Backward real de softmax (vector-Jacobiano), por fila:
//   dX[i] = softmax_output[i] * (grad_output[i] - sum_j(grad_output[j] * softmax_output[j]))
// Distinto del passthrough de SoftmaxLayer::backward, que solo es valido
// cuando el softmax esta pegado a una loss de cross-entropy al final de la
// red. Necesario para los scores de atencion, que no estan pegados a ninguna
// loss.
Tensor softmax_backward(const Tensor& grad_output, const Tensor& softmax_output);

} // namespace ops
