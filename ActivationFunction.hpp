#pragma once

// #include "Matrix.hpp"
#include "DeviceMatrix.cuh"
using Matrix = DeviceMatrix;

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
    cuda_softmax_forward(x.d_data, result.d_data, x.rows, x.cols);
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
    cuda_relu_forward(x.d_data, result.d_data, x.rows * x.cols);
    return result;
  }

  Matrix derivative(const Matrix &output) override {
    Matrix result(output.rows, output.cols);
    cuda_relu_derivative(output.d_data, result.d_data, output.rows * output.cols);
    return result;
  }
};


class Sigmoid : public ActivationFunction {
public:
  Matrix forward(const Matrix &x) override {
    Matrix result(x.rows, x.cols);
    cuda_sigmoid_forward(x.d_data, result.d_data, x.rows * x.cols);
    return result;
  }

  Matrix derivative(const Matrix &output) override {
    Matrix result(output.rows, output.cols);
    cuda_sigmoid_derivative(output.d_data, result.d_data, output.rows * output.cols);
    return result;
  }
};
