#pragma once

#include <algorithm>
#include <vector>
#include <random>
#include "Layer.hpp"

using std::vector;


class MLP {
public:
  void add_layer(int num_inputs, int num_neurons, ActivationFunction *act) {
    layers.emplace_back(num_inputs, num_neurons, act);
  }

  Matrix forward(const Matrix &input) {
    Matrix current = input;
    for (auto& layer : layers)
      current = layer.forward(current);
    return current;
  }

  void train(const Matrix &X, const Matrix &Y, int epochs, float lr, int batch_size) {
    int n_samples = X.rows;
    vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(std::random_device{}());

    for (size_t e = 0; e < epochs; e++) {
      std::shuffle(indices.begin(), indices.end(), gen);

      for (size_t start = 0; start < n_samples; start += batch_size) {
        int end = std::min((int)start + batch_size, n_samples);
        int current_batch = end - start;

        Matrix batch_X(current_batch, X.cols);
        Matrix batch_Y(current_batch, Y.cols);

        for (size_t i = 0; i < current_batch; i++) {
          int idx = indices[start + i];
          for (size_t j = 0; j < X.cols; j++)
            batch_X.at(i, j) = X.at(idx, j);
          for (size_t j = 0; j < Y.cols; j++)
            batch_Y.at(i, j) = Y.at(idx, j);
        }

      }

    }
  }

private:
  std::vector<Layer> layers;
};
