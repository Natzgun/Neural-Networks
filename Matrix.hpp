#pragma once

#include <cstddef>
#include <vector>
#include <cassert>
#include <random>
#include <iostream>

struct Matrix {
  std::vector<float> data;
  int rows, cols;

  Matrix() : rows(0), cols(0) {}

  Matrix(int rows, int cols, float fill = 0.0f)
      : rows(rows), cols(cols), data(rows * cols, fill) {}

  float &at(int r, int c) { return data[r * cols + c]; }
  float at(int r, int c) const { return data[r * cols + c]; }

  Matrix matmul(const Matrix &other) const {
    assert(cols == other.rows && "matmul: shapes incompatibles");
    Matrix result(rows, other.cols, 0.0f);

    for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < other.cols; j++)
        // dot product
        for (size_t k = 0; k < cols; k++)
          result.at(i, j) += at(i, k) * other.at(k, j);
    return result;
  }

  Matrix T() const {
    Matrix result(cols, rows);

    for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++)
        result.at(j, i) = at(i, j);

    return result;
  }

  Matrix operator+(const Matrix &o) const {
    assert(rows == o.rows && cols == o.cols);
    Matrix r(rows, cols);
    for (size_t i = 0; i < rows * cols; i++)
      r.data[i] = data[i] + o.data[i];
    return r;
  }

  Matrix operator-(const Matrix &o) const {
    assert(rows == o.rows && cols == o.cols);
    Matrix r(rows, cols);
    for (size_t i = 0; i < rows * cols; i++)
      r.data[i] = data[i] - o.data[i];
    return r;
  }

  // Hadamard
  Matrix operator*(const Matrix &o) const {
    assert(rows == o.rows && cols == o.cols);
    Matrix r(rows, cols);
    for (size_t i = 0; i < rows * cols; i++)
      r.data[i] = data[i] * o.data[i];
    return r;
  }

  // Matrix-scalar product
  Matrix operator*(float s) const {
    Matrix r(rows, cols);
    for (size_t i = 0; i < rows * cols; i++)
      r.data[i] = data[i] * s;
    return r;
  }

  Matrix operator/(float s) const {
    Matrix r(rows, cols);
    for (size_t i = 0; i < rows * cols; i++)
      r.data[i] = data[i] / s;
    return r;
  }

  Matrix& operator-=(const Matrix &o) {
    for (size_t i = 0; i < rows * cols; i++)
      data[i] -= o.data[i];
    return *this;
  }

  Matrix add_bias(const Matrix &bias) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++)
        result.at(i, j) = at(i, j) + bias.at(0, j);
    return result;
  }

  Matrix sum_rows() const {
    Matrix result(1, cols, 0.0f);
    for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++)
        result.at(0, j) += at(i, j);
    return result;
  }

  static Matrix random_uniform(int rows, int cols,
                               float low = -1.0f, float high = 1.0f) {
    Matrix m(rows, cols);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(low, high);
    for (auto& x : m.data) x = dis(gen);
    return m;
  }


  void print(const std::string& name = "") const {
    if (!name.empty())
      std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++)
        std::cout << at(i,j) << "\t";
      std::cout << "\n";
    }
  }

};
