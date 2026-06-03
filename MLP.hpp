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

        Matrix y_pred = forward(batch_X);

        // Cross entropy derivative
        Matrix gradient = (y_pred - batch_Y) / (float)current_batch;

        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
          gradient = layers[i].backward(gradient, lr);
      }

      Matrix y_pred_all = forward(X);
      std::cout << "Epoch " << e + 1 << "/" << epochs << ": ";
      evaluate_sm(n_samples, Y, y_pred_all);

    }
  }


  void evaluate_sm(int n_samples, const Matrix &Y, const Matrix &y_pred_all ){
    // cross- entropy loss
    float loss = 0.0f;
    for (size_t i = 0; i < n_samples; i++)
      for (size_t j = 0; j < Y.cols; j++)
        loss -= Y.at(i, j) * log(y_pred_all.at(i, j) + 1e-9f);

    loss /= n_samples;

    // Accuracy
    int correct = 0;
    for (size_t i = 0; i < n_samples; i++) {
      int pred_class = 0, true_class = 0;
      for (size_t j = 1; j < Y.cols; j++) {
        if (y_pred_all.at(i, j) > y_pred_all.at(i, pred_class))
          pred_class = j;
        if (Y.at(i, j) > Y.at(i, true_class))
          true_class = j;
      }
      if (pred_class == true_class)
        correct++;
    }

    float accuracy = (float)correct / n_samples;

    std::cout << "Loss: " << loss << ", Accuracy: " << accuracy  << "%\n";
  }

private:
  std::vector<Layer> layers;
};
