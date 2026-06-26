#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "core/Tensor.cuh"
#include "data/Dataset.hpp"

static int32_t read_int32_be(std::ifstream& file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return static_cast<int32_t>((bytes[0] << 24) | (bytes[1] << 16) |
                                (bytes[2] << 8) | bytes[3]);
}

Dataset load_mnist(const std::string& images_path,
                   const std::string& labels_path) {
    std::ifstream img_file(images_path, std::ios::binary);
    std::ifstream lbl_file(labels_path, std::ios::binary);

    if (!img_file || !lbl_file) {
        std::cerr << "Failed to open MNIST files" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int32_t img_magic = read_int32_be(img_file);
    int32_t n_images = read_int32_be(img_file);
    int32_t n_rows = read_int32_be(img_file);
    int32_t n_cols = read_int32_be(img_file);

    int32_t lbl_magic = read_int32_be(lbl_file);
    int32_t n_labels = read_int32_be(lbl_file);

    if (n_images != n_labels) {
        std::cerr << "MNIST image/label count mismatch" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int features = n_rows * n_cols;
    int n_classes = 10;

    Tensor X({n_images, features});
    Tensor Y({n_images, n_classes}, 0.0f);

    for (int i = 0; i < n_images; ++i) {
        for (int j = 0; j < features; ++j) {
            uint8_t pixel;
            img_file.read(reinterpret_cast<char*>(&pixel), 1);
            X.flat(i * features + j) = static_cast<float>(pixel) / 255.0f;
        }

        uint8_t label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        Y.flat(i * n_classes + label) = 1.0f;
    }

    X.upload();
    Y.upload();

    Dataset ds;
    ds.X = std::move(X);
    ds.Y = std::move(Y);
    ds.n_samples = n_images;
    return ds;
}
