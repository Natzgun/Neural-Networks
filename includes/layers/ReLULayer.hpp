#pragma once

#include "layers/Layer.hpp"

class ReLULayer : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    Tensor output_;
};
