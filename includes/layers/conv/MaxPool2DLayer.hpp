#pragma once

#include "core/ops/pool_ops.cuh"
#include "layers/Layer.hpp"

class MaxPool2DLayer : public Layer {
public:
  MaxPool2DLayer(int kernel_size, int stride = -1);
  ~MaxPool2DLayer();

  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float lr) override {}

private:
  int kernel_size_;
  int stride_;
  Tensor inputs_;
  ops::PoolMask mask_;
};
