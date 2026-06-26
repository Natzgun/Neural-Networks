#include "layers/conv/ConvLayer.hpp"

#include <cmath>

#include "core/ops/conv_ops.cuh"
#include "core/ops/linalg.cuh"

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                         int stride, int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      grad_weights_({out_channels, in_channels, kernel_size, kernel_size}),
      grad_biases_({out_channels}) {
    float fan_in = in_channels * kernel_size * kernel_size;
    float stddev = std::sqrt(2.0f / fan_in);

    weights_ = Tensor::random_normal(
        {out_channels, in_channels, kernel_size, kernel_size}, 0.0f, stddev);
    biases_ = Tensor::zeros({out_channels});

    weights_.upload();
    biases_.upload();
    grad_weights_.upload();
    grad_biases_.upload();
}

Tensor Conv2DLayer::forward(const Tensor& input) {
    inputs_ = input;
    outputs_ = ops::conv2d_forward(input, weights_, biases_, stride_, padding_);
    return outputs_;
}

Tensor Conv2DLayer::backward(const Tensor& grad_output) {
    grad_weights_ = ops::conv2d_backward_weights(
        inputs_, grad_output, stride_, padding_, kernel_size_, out_channels_);
    grad_biases_ = ops::conv2d_backward_bias(grad_output);

    Tensor grad_input = ops::conv2d_backward_input(
        grad_output, weights_, stride_, padding_, inputs_.sizes());

    return grad_input;
}

void Conv2DLayer::update(float lr) {
    ops::sub_inplace(weights_, ops::scalar_mul(grad_weights_, lr));
    ops::sub_inplace(biases_, ops::scalar_mul(grad_biases_, lr));
}
