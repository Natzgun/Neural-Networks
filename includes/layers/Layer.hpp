#pragma once

#include "core/Tensor.cuh"

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update(float lr) {}
};
