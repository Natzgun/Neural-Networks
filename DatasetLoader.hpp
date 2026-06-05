#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "Matrix.hpp"
#include "MLP.hpp"

struct SimpsonDataset {
  Matrix X; // (n_samples, 784)
  Matrix Y; // (n_samples, 10) one-hot
  int n_samples;
};

inline SimpsonDataset load_simpson_dataset(const std::string& base_path) {
  const std::vector<std::string> class_names = {
    "bart_simpson",
    "charles_montgomery_burns",
    "homer_simpson",
    "krusty_the_clown",
    "lisa_simpson",
    "marge_simpson",
    "milhouse_van_houten",
    "moe_szyslak",
    "ned_flanders",
    "principal_skinner"
  };

  const int n_classes = static_cast<int>(class_names.size());
  std::vector<std::vector<float>> all_samples;
  std::vector<int> all_labels;

  for (int label = 0; label < n_classes; ++label) {
    std::string pattern = base_path + "/" + class_names[label] + "/*.jpg";
    std::vector<cv::String> files;
    cv::glob(pattern, files, false);

    std::cout << "Loading " << files.size() << " images from " << class_names[label] << "...\n";

    for (const auto& f : files) {
      cv::Mat img = cv::imread(f, cv::IMREAD_GRAYSCALE);
      if (img.empty()) {
        std::cerr << "Warning: could not read " << f << "\n";
        continue;
      }

      // Flatten and normalize [0,255] -> [0,1]
      std::vector<float> sample;
      sample.reserve(img.rows * img.cols);
      for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
          sample.push_back(static_cast<float>(img.at<uchar>(i, j)) / 255.0f);
        }
      }

      all_samples.push_back(std::move(sample));
      all_labels.push_back(label);
    }
  }

  int n = static_cast<int>(all_samples.size());
  int features = n > 0 ? static_cast<int>(all_samples[0].size()) : 0;

  Matrix X(n, features);
  Matrix Y(n, n_classes, 0.0f);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < features; ++j) {
      X.at(i, j) = all_samples[i][j];
    }
    Y.at(i, all_labels[i]) = 1.0f;
  }

  std::cout << "Dataset loaded: " << n << " samples, " << features << " features, " << n_classes << " classes.\n";

  SimpsonDataset ds;
  ds.X = std::move(X);
  ds.Y = std::move(Y);
  ds.n_samples = n;
  return ds;
}
