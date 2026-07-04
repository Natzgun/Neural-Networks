#pragma once

#include "Tensor3D.hpp"
#include <algorithm>
#include <cmath>

struct Conv2D {
  int in_channels, out_channels, kernel_size;
  std::vector<float> kernels;
  std::vector<float> biases;

  Conv2D(int in_c, int out_c, int k_size = 3)
      : in_channels(in_c), out_channels(out_c), kernel_size(k_size),
        kernels(out_c * in_c * k_size * k_size, 0.0f),
        biases(out_c, 0.0f) {
    init_he();
  }

  void init_he() {
    std::mt19937 gen(std::random_device{}());
    float stddev = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<float> dis(0.0f, stddev);
    for (auto &v : kernels)
      v = dis(gen);
  }

  int k_idx(int oc, int ic, int ky, int kx) const {
    return oc * (in_channels * kernel_size * kernel_size)
         + ic * (kernel_size * kernel_size)
         + ky * kernel_size
         + kx;
  }

  Tensor3D forward(const Tensor3D &input) {
    assert(input.channels == in_channels);
    int out_h = input.height - kernel_size + 1;
    int out_w = input.width - kernel_size + 1;

    Tensor3D output(out_channels, out_h, out_w, 0.0f);

    for (int oc = 0; oc < out_channels; oc++) {
      for (int ic = 0; ic < in_channels; ic++) {
        for (int oy = 0; oy < out_h; oy++) {
          for (int ox = 0; ox < out_w; ox++) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernel_size; ky++) {
              for (int kx = 0; kx < kernel_size; kx++) {
                sum += input.at(ic, oy + ky, ox + kx)
                     * kernels[k_idx(oc, ic, ky, kx)];
              }
            }
            output.at(oc, oy, ox) += sum;
          }
        }
      }
      for (int oy = 0; oy < out_h; oy++)
        for (int ox = 0; ox < out_w; ox++)
          output.at(oc, oy, ox) += biases[oc];
    }

    return output;
  }
};
