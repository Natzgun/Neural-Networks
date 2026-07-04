#pragma once

#include <cassert>
#include <cstddef>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>

struct Tensor3D {
  std::vector<float> data;
  int channels, height, width;

  Tensor3D() : channels(0), height(0), width(0) {}

  Tensor3D(int c, int h, int w, float fill = 0.0f)
      : channels(c), height(h), width(w), data(c * h * w, fill) {}

  float &at(int c, int y, int x) {
    return data[c * (height * width) + y * width + x];
  }

  float at(int c, int y, int x) const {
    return data[c * (height * width) + y * width + x];
  }

  int spatial_size() const { return height * width; }
  int total_size() const { return channels * height * width; }

  void print_shape(const std::string &name = "") const {
    if (!name.empty())
      std::cout << name << " ";
    std::cout << "[" << channels << " x " << height << " x " << width << "]\n";
  }

  void print(const std::string &name = "") const {
    print_shape(name);
    for (int c = 0; c < channels; c++) {
      std::cout << "  channel " << c << ":\n";
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
          std::cout << at(c, y, x) << "\t";
        std::cout << "\n";
      }
    }
  }
};
