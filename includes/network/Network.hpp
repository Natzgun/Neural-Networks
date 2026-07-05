#pragma once

#include <memory>
#include <vector>

#include "layers/Layer.hpp"

class Network {
public:
  template <typename LayerType, typename... Args> void add(Args&&... args) {
    layers_.emplace_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
  }

  Tensor forward(const Tensor& input);
  void backward(const Tensor& grad_output);
  void update(float lr);

  void train_step(const Tensor& X, const Tensor& Y, float lr);

  const std::vector<std::unique_ptr<Layer>>& layers() const {
    return layers_;
  }

private:
  std::vector<std::unique_ptr<Layer>> layers_;
};
