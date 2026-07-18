#include <cstdlib>
#include <iostream>
#include <string>

#include "core/Tensor.cuh"
#include "loaders/FashionMnistLoader.hpp"
#include "loaders/MnistLoader.hpp"
#include "models/ViT.hpp"
#include "utils/ExampleRunners.hpp"
#include "utils/Training.hpp"

static int parse_positive_int_arg(int argc, char* argv[], int index, int default_value,
                                  const char* name) {
  if (argc <= index)
    return default_value;

  int value = std::atoi(argv[index]);
  if (value <= 0) {
    std::cerr << name << " must be positive\n";
    std::exit(EXIT_FAILURE);
  }
  return value;
}

int run_vit_training(int argc, char* argv[]) {
  std::string dataset = argc > 1 ? argv[1] : "fashionmnist";

  Dataset train, test;
  if (dataset == "mnist") {
    train = load_mnist("datasets/mnist/train-images.idx3-ubyte",
                       "datasets/mnist/train-labels.idx1-ubyte");
    test = load_mnist("datasets/mnist/t10k-images.idx3-ubyte",
                      "datasets/mnist/t10k-labels.idx1-ubyte");
  } else if (dataset == "fashionmnist") {
    train = load_fashionmnist("datasets/fashionmnist/train-images-idx3-ubyte",
                              "datasets/fashionmnist/train-labels-idx1-ubyte");
    test = load_fashionmnist("datasets/fashionmnist/t10k-images-idx3-ubyte",
                             "datasets/fashionmnist/t10k-labels-idx1-ubyte");
  } else {
    std::cerr << "Unknown ViT dataset: " << dataset << "\n";
    std::cerr << "Usage: neural_networks vit [fashionmnist|mnist] [epochs] [batch_size]\n";
    return 1;
  }

  int epochs = parse_positive_int_arg(argc, argv, 2, 1, "epochs");
  int batch_size = parse_positive_int_arg(argc, argv, 3, 8, "batch_size");
  float lr = 0.01f;

  int image_h = 28;
  int image_w = 28;
  int patch_size = 7;
  int in_channels = 1;
  int embed_dim = 32;
  int num_heads = 4;
  int mlp_hidden_dim = 64;
  int num_layers = 1;
  int n_classes = train.Y.dim(1);

  train.X = train.X.reshape({train.n_samples, in_channels, image_h, image_w});
  test.X = test.X.reshape({test.n_samples, in_channels, image_h, image_w});

  std::cout << "ViT training on " << dataset << "\n";
  std::cout << "Train: " << train.n_samples << ", Test: " << test.n_samples
            << ", classes: " << n_classes << "\n";
  std::cout << "Config: patch_size=" << patch_size << ", embed_dim=" << embed_dim
            << ", heads=" << num_heads << ", layers=" << num_layers
            << ", mlp_hidden=" << mlp_hidden_dim << "\n";

  Network net = make_vit(image_h, image_w, patch_size, in_channels, embed_dim, num_heads,
                         mlp_hidden_dim, num_layers, n_classes);

  train_epochs(net, train, test, epochs, batch_size, lr);
  return 0;
}
