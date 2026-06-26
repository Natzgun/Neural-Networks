#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "core/Tensor.cuh"
#include "data/Dataset.hpp"

static std::vector<std::string> class_names() {
    return {
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
}

Dataset load_simpsons(const std::string& base_path) {
    auto classes = class_names();
    int n_classes = static_cast<int>(classes.size());

    std::vector<std::vector<float>> all_samples;
    std::vector<int> all_labels;

    for (int label = 0; label < n_classes; ++label) {
        std::string pattern = base_path + "/" + classes[label] + "/*.jpg";
        std::vector<cv::String> files;
        cv::glob(pattern, files, false);

        std::cout << "Loading " << files.size() << " images from "
                  << classes[label] << "...\n";

        for (const auto& f : files) {
            cv::Mat img = cv::imread(f, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Warning: could not read " << f << "\n";
                continue;
            }

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

    Tensor X({n, features});
    Tensor Y({n, n_classes}, 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < features; ++j) {
            X.flat(i * features + j) = all_samples[i][j];
        }
        Y.flat(i * n_classes + all_labels[i]) = 1.0f;
    }

    std::cout << "Dataset loaded: " << n << " samples, " << features
              << " features, " << n_classes << " classes.\n";

    X.upload();
    Y.upload();

    Dataset ds;
    ds.X = std::move(X);
    ds.Y = std::move(Y);
    ds.n_samples = n;
    return ds;
}
