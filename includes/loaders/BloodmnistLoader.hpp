#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "core/Tensor.cuh"
#include "data/Dataset.hpp"
#include "vendored/libnpy/npy.hpp"

inline Dataset load_bloodmnist(const std::string& base_path, const std::string& split) {
  std::string images_path = base_path + "/" + split + "_images.npy";
  std::string labels_path = base_path + "/" + split + "_labels.npy";

  std::vector<unsigned long> img_shape;
  std::vector<uint8_t> img_raw;
  npy::LoadArrayFromNumpy<uint8_t>(images_path, img_shape, img_raw);

  std::vector<unsigned long> lbl_shape;
  std::vector<uint8_t> lbl_raw;
  npy::LoadArrayFromNumpy<uint8_t>(labels_path, lbl_shape, lbl_raw);

  if (img_shape.size() != 4 || img_shape[3] != 3) {
    std::cerr << "BloodMNIST images must be NHWC with 3 channels" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int n = static_cast<int>(img_shape[0]);
  int h = static_cast<int>(img_shape[1]);
  int w = static_cast<int>(img_shape[2]);
  int c = static_cast<int>(img_shape[3]);
  int n_classes = 8;

  std::cout << "Loading BloodMNIST " << split << ": " << n << " samples, " << h << "x" << w << "x"
            << c << ", " << n_classes << " classes\n";

  Tensor X({n, c, h, w});
  Tensor Y({n, n_classes}, 0.0f);

  for (int i = 0; i < n; ++i) {
    for (int ch = 0; ch < c; ++ch) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          int nhwc_idx = ((i * h + y) * w + x) * c + ch;
          int nchw_idx = ((i * c + ch) * h + y) * w + x;
          X.flat(nchw_idx) = static_cast<float>(img_raw[nhwc_idx]) / 255.0f;
        }
      }
    }
    Y.flat(i * n_classes + lbl_raw[i]) = 1.0f;
  }

  X.upload();
  Y.upload();

  Dataset ds;
  ds.X = std::move(X);
  ds.Y = std::move(Y);
  ds.n_samples = n;
  return ds;
}
