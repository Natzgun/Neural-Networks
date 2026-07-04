#pragma once

#include "Tensor3D.hpp"
#include <algorithm>
#include <cassert>
#include <limits>

struct MaxPool2D {
  int pool_size;
  int stride;

  MaxPool2D(int p_size = 2, int str = 2)
      : pool_size(p_size), stride(str) {}

  Tensor3D forward(const Tensor3D &input) {
    int out_h = (input.height - pool_size) / stride + 1;
    int out_w = (input.width - pool_size) / stride + 1;

    Tensor3D output(input.channels, out_h, out_w, 0.0f);

    for (int c = 0; c < input.channels; c++) {
      for (int oy = 0; oy < out_h; oy++) {
        for (int ox = 0; ox < out_w; ox++) {
          float max_val = -std::numeric_limits<float>::infinity();
          for (int py = 0; py < pool_size; py++) {
            for (int px = 0; px < pool_size; px++) {
              float val = input.at(c, oy * stride + py, ox * stride + px);
              if (val > max_val) max_val = val;
            }
          }
          output.at(c, oy, ox) = max_val;
        }
      }
    }

    return output;
  }
};
