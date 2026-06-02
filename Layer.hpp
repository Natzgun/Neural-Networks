#pragma once

#include "Matrix.hpp"
#include "ActivationFunction.hpp"

class Layer {
public:
  Layer(int nums_inputs, int num_neurons, ActivationFunction *act) : activation(act) {
    this->weights = Matrix::random_uniform(nums_inputs, num_neurons);
    this->biases = Matrix(1, num_neurons, 0.0f);
  }

  Matrix forward(const Matrix &inputs_) {
    this->inputs = inputs_;
    Matrix z =  inputs.matmul(this->weights.T());
    outputs = activation->forward(z);
    return outputs;
  }
private:
  Matrix weights;
  Matrix biases;
  Matrix inputs;
  Matrix outputs;
  ActivationFunction *activation;
};
