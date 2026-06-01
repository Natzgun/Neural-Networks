#pragma once

#include <cstddef>
#include <vector>
#include <cassert>

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

};
