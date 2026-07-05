#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "vendored/stb_image.h"

#include "core/Tensor.cuh"
#include "data/Dataset.hpp"

inline std::vector<std::string> simpsons_class_names() {
  return {"bart_simpson",        "charles_montgomery_burns",
          "homer_simpson",       "krusty_the_clown",
          "lisa_simpson",        "marge_simpson",
          "milhouse_van_houten", "moe_szyslak",
          "ned_flanders",        "principal_skinner"};
}

inline std::string simpsons_to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

inline std::vector<std::string> list_simpsons_image_files(const std::string& dir_path) {
  std::vector<std::string> files;
  std::error_code ec;
  std::filesystem::directory_iterator it(dir_path, ec);
  if (ec) {
    return files;
  }
  for (const auto& entry : it) {
    if (!entry.is_regular_file())
      continue;
    std::string ext = simpsons_to_lower(entry.path().extension().string());
    if (ext == ".jpg" || ext == ".jpeg") {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

inline Dataset load_simpsons(const std::string& base_path) {
  auto classes = simpsons_class_names();
  int n_classes = static_cast<int>(classes.size());

  std::vector<std::vector<float>> all_samples;
  std::vector<int> all_labels;

  for (int label = 0; label < n_classes; ++label) {
    std::string dir_path = base_path + "/" + classes[label];
    std::vector<std::string> files = list_simpsons_image_files(dir_path);

    std::cout << "Loading " << files.size() << " images from " << classes[label] << "...\n";

    for (const auto& f : files) {
      int w = 0, h = 0, channels = 0;
      unsigned char* img = stbi_load(f.c_str(), &w, &h, &channels, 1);
      if (!img) {
        std::cerr << "Warning: could not read " << f << " (" << stbi_failure_reason() << ")\n";
        continue;
      }

      std::vector<float> sample;
      sample.reserve(static_cast<size_t>(h) * w);
      const size_t total = static_cast<size_t>(h) * w;
      for (size_t i = 0; i < total; ++i) {
        sample.push_back(static_cast<float>(img[i]) / 255.0f);
      }
      stbi_image_free(img);

      all_samples.push_back(std::move(sample));
      all_labels.push_back(label);
    }
  }

  int n = static_cast<int>(all_samples.size());
  int features = n > 0 ? static_cast<int>(all_samples[0].size()) : 0;

  Tensor X({n, features});
  Tensor Y({n, n_classes}, 0.0f);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < features; ++j) {
      X.flat(i * features + j) = all_samples[i][j];
    }
    Y.flat(i * n_classes + all_labels[i]) = 1.0f;
  }

  std::cout << "Dataset loaded: " << n << " samples, " << features << " features, " << n_classes
            << " classes.\n";

  X.upload();
  Y.upload();

  Dataset ds;
  ds.X = std::move(X);
  ds.Y = std::move(Y);
  ds.n_samples = n;
  return ds;
}
