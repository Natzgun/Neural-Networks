#pragma once

#include "layers/Layer.hpp"

class DenseLayer : public Layer {
public:
  DenseLayer(int in_features, int out_features);

  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float lr) override;

private:
  Tensor weights_;
  Tensor biases_;
  Tensor inputs_;
  Tensor outputs_;
  Tensor grad_weights_;
  Tensor grad_biases_;
};
