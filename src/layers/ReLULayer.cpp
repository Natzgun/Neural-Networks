#include "layers/ReLULayer.hpp"

#include "core/ops/activations.cuh"
#include "core/ops/linalg.cuh"

Tensor ReLULayer::forward(const Tensor& input) {
    output_ = ops::relu_forward(input);
    return output_;
}

Tensor ReLULayer::backward(const Tensor& grad_output) {
    Tensor derivative = ops::relu_derivative(output_);
    return ops::hadamard(grad_output, derivative);
}
