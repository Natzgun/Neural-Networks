#pragma once

#include "Matrix.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

class ActivationFunction {
public:
  virtual Matrix forward(const Matrix& input) = 0;
  virtual Matrix derivative(const Matrix& output) = 0;
  virtual ~ActivationFunction() = default;
};

class Softmax : public ActivationFunction {
public:
  Matrix forward(const Matrix &x) override {
    Matrix result(x.rows, x.cols);

    for (size_t i = 0; i < x.rows; i++) {

      float max_val = x.at(i, 0);
      for (size_t j = 0; j < x.cols; j++)
        max_val = std::max(max_val, x.at(i, j));

      float sum = 0.0f;
      for (size_t j = 0; j < x.cols; j++) {
        result.at(i, j) = std::exp(x.at(i, j) - max_val);
        sum += result.at(i, j);
      }

      for (size_t j = 0; j < x.cols; j++)
        result.at(i, j) /= sum;
    }

    return result;
  }

  // Cross entropy, the gradient is already calculed
  Matrix derivative(const Matrix &output) override {
    return Matrix(output.rows, output.cols, 1.0f);
  }

};

class ReLU : public ActivationFunction {
public:
  Matrix forward(const Matrix &x) override {
    Matrix result(x.rows, x.cols);
    for (size_t i = 0; i< x.rows * x.cols; i++)
      result.data[i] = std::max(0.0f, x.data[i]);
    return result;
  }

  Matrix derivative(const Matrix &output) override {
    Matrix result(output.rows, output.cols);
    for (size_t i = 0; i< output.rows * output.cols; i++)
      result.data[i] = output.data[i] > 0.0f ? 1.0f : 0.0f;
    return result;
  }
};


class Sigmoid : public ActivationFunction {
public:
  Matrix forward(const Matrix &x) override {
    Matrix result(x.rows, x.cols);
    for (size_t i = 0; i< x.rows * x.cols; i++) {
      float val = std::max(-250.0f, std::min(250.0f, x.data[i]));
      result.data[i] = 1.0f / (1.0f + std::exp(-val));
    }
    return result;
  }

  Matrix derivative(const Matrix &output) override {
    Matrix result(output.rows, output.cols);
    for (size_t i = 0; i< output.rows * output.cols; i++)
      result.data[i] = output.data[i] * (1.0f - output.data[i]);
    return result;
  }
};
