#pragma once

#include "layers/Layer.hpp"

class Conv2DLayer : public Layer {
public:
    Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                int stride = 1, int padding = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update(float lr) override;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;

    Tensor weights_;
    Tensor biases_;
    Tensor inputs_;
    Tensor outputs_;
    Tensor grad_weights_;
    Tensor grad_biases_;
};
