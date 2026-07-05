#include "network/Network.hpp"

#include "core/ops/linalg.cuh"

Tensor Network::forward(const Tensor& input) {
  Tensor current = input;
  for (auto& layer : layers_) {
    current = layer->forward(current);
  }
  return current;
}

void Network::backward(const Tensor& grad_output) {
  Tensor grad = grad_output;
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    grad = layers_[i]->backward(grad);
  }
}

void Network::update(float lr) {
  for (auto& layer : layers_) {
    layer->update(lr);
  }
}

void Network::train_step(const Tensor& X, const Tensor& Y, float lr) {
  Tensor y_pred = forward(X);

  int batch_size = X.dim(0);

  Tensor grad = ops::sub(y_pred, Y);
  grad = ops::scalar_div(grad, static_cast<float>(batch_size));

  backward(grad);
  update(lr);
}
