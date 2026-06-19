#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include "cuda_ops.cuh"

class DeviceMatrix {
public:
  std::vector<float> data; // host mirror
  float *d_data = nullptr; // device pointer
  int rows, cols;
  bool on_device = false;

  DeviceMatrix() : rows(0), cols(0) {}

  DeviceMatrix(int rows, int cols, float fill = 0.0f)
      : rows(rows), cols(cols), data(rows * cols, fill) {
    alloc_device();
    if (fill == 0.0f) {
      CUDA_CHECK(cudaMemset(d_data, 0, size_bytes()));
    } else {
      cuda_fill(d_data, fill, rows * cols);
    }
    on_device = true;
  }


  ~DeviceMatrix() { free_device(); }

  // Copy constructor
  DeviceMatrix(const DeviceMatrix &o)
      : data(o.data), rows(o.rows), cols(o.cols), on_device(o.on_device) {
    if (o.d_data && o.on_device) {
      alloc_device();
      CUDA_CHECK(
          cudaMemcpy(d_data, o.d_data, size_bytes(), cudaMemcpyDeviceToDevice));
    }
  }

  // Copy assignment
  DeviceMatrix &operator=(const DeviceMatrix &o) {
    if (this == &o)
      return *this;
    free_device();
    data = o.data;
    rows = o.rows;
    cols = o.cols;
    on_device = o.on_device;
    if (o.d_data && o.on_device) {
      alloc_device();
      CUDA_CHECK(
          cudaMemcpy(d_data, o.d_data, size_bytes(), cudaMemcpyDeviceToDevice));
    }
    return *this;
  }

  // Move constructor
  DeviceMatrix(DeviceMatrix &&o) noexcept
      : data(std::move(o.data)), d_data(o.d_data), rows(o.rows), cols(o.cols),
        on_device(o.on_device) {
    o.d_data = nullptr;
    o.rows = o.cols = 0;
    o.on_device = false;
  }

  // Move assignment
  DeviceMatrix &operator=(DeviceMatrix &&o) noexcept {
    if (this == &o)
      return *this;
    free_device();
    data = std::move(o.data);
    d_data = o.d_data;
    rows = o.rows;
    cols = o.cols;
    on_device = o.on_device;
    o.d_data = nullptr;
    o.rows = o.cols = 0;
    o.on_device = false;
    return *this;
  }


  float &at(int r, int c) { return data[r * cols + c]; }
  float at(int r, int c) const { return data[r * cols + c]; }

  // Host <-> Device transfers

  void upload() {
    if (!d_data)
      alloc_device();
    CUDA_CHECK(
        cudaMemcpy(d_data, data.data(), size_bytes(), cudaMemcpyHostToDevice));
    on_device = true;
  }

  void download() {
    assert(d_data && "download: no device memory allocated");
    data.resize(rows * cols);
    CUDA_CHECK(
        cudaMemcpy(data.data(), d_data, size_bytes(), cudaMemcpyDeviceToHost));
  }

  // Matrix operations (all on GPU)

  DeviceMatrix matmul(const DeviceMatrix &other) const {
    assert(cols == other.rows && "matmul: shapes incompatibles");
    DeviceMatrix result(rows, other.cols, 0.0f);
    cuda_matmul(d_data, other.d_data, result.d_data, rows, other.cols, cols);
    return result;
  }

  DeviceMatrix T() const {
    DeviceMatrix result(cols, rows);
    cuda_transpose(d_data, result.d_data, rows, cols);
    return result;
  }

  DeviceMatrix operator+(const DeviceMatrix &o) const {
    assert(rows == o.rows && cols == o.cols);
    DeviceMatrix r(rows, cols);
    cuda_add(d_data, o.d_data, r.d_data, rows * cols);
    return r;
  }

  DeviceMatrix operator-(const DeviceMatrix &o) const {
    assert(rows == o.rows && cols == o.cols);
    DeviceMatrix r(rows, cols);
    cuda_sub(d_data, o.d_data, r.d_data, rows * cols);
    return r;
  }

  // Hadamard (element-wise)
  DeviceMatrix operator*(const DeviceMatrix &o) const {
    assert(rows == o.rows && cols == o.cols);
    DeviceMatrix r(rows, cols);
    cuda_hadamard(d_data, o.d_data, r.d_data, rows * cols);
    return r;
  }

  // Matrix-scalar product
  DeviceMatrix operator*(float s) const {
    DeviceMatrix r(rows, cols);
    cuda_scalar_mul(d_data, s, r.d_data, rows * cols);
    return r;
  }

  DeviceMatrix operator/(float s) const {
    DeviceMatrix r(rows, cols);
    cuda_scalar_div(d_data, s, r.d_data, rows * cols);
    return r;
  }

  DeviceMatrix &operator-=(const DeviceMatrix &o) {
    cuda_sub_inplace(d_data, o.d_data, rows * cols);
    return *this;
  }

  DeviceMatrix add_bias(const DeviceMatrix &bias) const {
    DeviceMatrix result(rows, cols);
    cuda_add_bias(d_data, bias.d_data, result.d_data, rows, cols);
    return result;
  }

  DeviceMatrix sum_rows() const {
    DeviceMatrix result(1, cols, 0.0f);
    cuda_sum_rows(d_data, result.d_data, rows, cols);
    return result;
  }

  // Initialization weightds

  static DeviceMatrix random_uniform(int rows, int cols, float low = -1.0f,
                                     float high = 1.0f) {
    DeviceMatrix m(rows, cols);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(low, high);
    for (auto &x : m.data)
      x = dis(gen);
    m.upload();
    return m;
  }

  static DeviceMatrix random_he(int rows, int cols) {
    DeviceMatrix m(rows, cols);
    std::mt19937 gen(std::random_device{}());
    float stddev = std::sqrt(2.0f / cols);
    std::normal_distribution<float> dis(0.0f, stddev);
    for (auto &x : m.data)
      x = dis(gen);
    m.upload();
    return m;
  }

  void print(const std::string &name = "") const {
    // Need to work with host data
    const_cast<DeviceMatrix *>(this)->download();
    if (!name.empty())
      std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++)
        std::cout << data[i * cols + j] << "\t";
      std::cout << "\n";
    }
  }

private:
  size_t size_bytes() const { return rows * cols * sizeof(float); }

  void alloc_device() {
    if (!d_data && rows > 0 && cols > 0)
      CUDA_CHECK(cudaMalloc(&d_data, size_bytes()));
  }

  void free_device() {
    if (d_data) {
      cudaFree(d_data);
      d_data = nullptr;
    }
  }
};
