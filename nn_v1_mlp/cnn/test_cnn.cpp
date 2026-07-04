#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Tensor3D.hpp"
#include "Convolution.hpp"
#include "MaxPool.hpp"

Tensor3D image_to_tensor(const cv::Mat &img) {
  Tensor3D t(1, img.rows, img.cols);
  for (int y = 0; y < img.rows; y++)
    for (int x = 0; x < img.cols; x++)
      t.at(0, y, x) = static_cast<float>(img.at<uchar>(y, x)) / 255.0f;
  return t;
}

void relu_inplace(Tensor3D &t) {
  for (auto &v : t.data)
    if (v < 0.0f) v = 0.0f;
}

int main(int argc, char *argv[]) {
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

  const std::string base = "grayscale_train/train";

  const int n_filters = 6;
  const int k_size = 3;
  const int pool_size = 2;

  std::cout << "Pipeline: Conv2D(" << n_filters << " filters, "
            << k_size << "x" << k_size << ") -> ReLU -> MaxPool2D("
            << pool_size << "x" << pool_size << ")\n\n";

  Conv2D conv(1, n_filters, k_size);
  MaxPool2D pool(pool_size, pool_size);

  int total_pixels = 0;
  int n_images = 0;

  for (const auto &cls : class_names) {
    std::string pattern = base + "/" + cls + "/*.jpg";
    std::vector<cv::String> files;
    cv::glob(pattern, files, false);

    int n_show = std::min(3, (int)files.size());

    for (int i = 0; i < n_show; i++) {
      cv::Mat img = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
      if (img.empty()) continue;

      Tensor3D input = image_to_tensor(img);
      std::cout << cls << " #" << (i + 1) << ":\n";
      input.print_shape("  Input          ");

      Tensor3D conv_out = conv.forward(input);
      conv_out.print_shape("  Conv2D output  ");

      relu_inplace(conv_out);

      Tensor3D pool_out = pool.forward(conv_out);
      pool_out.print_shape("  MaxPool output ");

      int pooled_pixels = pool_out.total_size();
      total_pixels += pooled_pixels;
      n_images++;

      std::cout << "  Sample conv[0] (first 6 values): ";
      for (int c = 0; c < std::min(2, conv_out.channels); c++) {
        std::cout << "[c" << c << "] ";
        for (int x = 0; x < 3; x++)
          std::cout << std::fixed << std::setprecision(2)
                    << conv_out.at(c, 0, x) << " ";
      }
      std::cout << "\n  Sample pool[0] (first 6 values): ";
      for (int c = 0; c < std::min(2, pool_out.channels); c++) {
        std::cout << "[c" << c << "] ";
        for (int x = 0; x < 3; x++)
          std::cout << std::fixed << std::setprecision(2)
                    << pool_out.at(c, 0, x) << " ";
      }
      std::cout << "\n\n";
    }
  }

  std::cout << "------------------------------------------\n";
  std::cout << "Total images processed: " << n_images << "\n";
  std::cout << "Average pooled pixels per image: "
            << (n_images > 0 ? total_pixels / n_images : 0) << "\n";
  std::cout << "All convolution & maxpool operations completed successfully.\n";

  return 0;
}
