#pragma once

#include "layers/Layer.hpp"

class FlattenLayer : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    std::vector<int> input_shape_;
};
