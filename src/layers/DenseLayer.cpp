#include "layers/dense/DenseLayer.hpp"

#include <cmath>

#include "core/ops/linalg.cuh"

DenseLayer::DenseLayer(int in_features, int out_features)
    : grad_weights_({in_features, out_features}),
      grad_biases_({1, out_features}) {
    float stddev = std::sqrt(2.0f / in_features);
    weights_ = Tensor::random_normal({in_features, out_features}, 0.0f, stddev);
    biases_ = Tensor::zeros({1, out_features});

    weights_.upload();
    biases_.upload();
    grad_weights_.upload();
    grad_biases_.upload();
}

Tensor DenseLayer::forward(const Tensor& input) {
    inputs_ = input;
    Tensor z = ops::add_bias(ops::matmul(input, weights_), biases_);
    outputs_ = z;
    return outputs_;
}

Tensor DenseLayer::backward(const Tensor& grad_output) {
    int batch_size = inputs_.dim(0);

    Tensor weights_t = ops::transpose(weights_);
    Tensor grad_input = ops::matmul(grad_output, weights_t);

    Tensor inputs_t = ops::transpose(inputs_);
    grad_weights_ = ops::matmul(inputs_t, grad_output);
    grad_biases_ = ops::sum_rows(grad_output);

    return grad_input;
}

void DenseLayer::update(float lr) {
    ops::sub_inplace(weights_, ops::scalar_mul(grad_weights_, lr));
    ops::sub_inplace(biases_, ops::scalar_mul(grad_biases_, lr));
}
