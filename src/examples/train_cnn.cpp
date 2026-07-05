#include <iostream>
#include <string>

#include "core/Tensor.cuh"
#include "loaders/BloodmnistLoader.hpp"
#include "loaders/MnistLoader.hpp"
#include "loaders/SimpsonsLoader.hpp"
#include "models/CNN.hpp"
#include "utils/ExampleRunners.hpp"
#include "utils/Training.hpp"

int run_cnn_training(int argc, char* argv[]) {
    Dataset train, test;
    std::string dataset = argc > 1 ? argv[1] : "mnist";

    if (dataset == "simpsons") {
        train = load_simpsons("datasets/simpsons_mnist/grayscale_train/train");
        test  = load_simpsons("datasets/simpsons_mnist/grayscale_test/test");
    } else if (dataset == "bloodmnist") {
        train = load_bloodmnist("datasets/bloodmnist", "train");
        test  = load_bloodmnist("datasets/bloodmnist", "test");
    } else {
        train = load_mnist(
            "datasets/mnist/train-images.idx3-ubyte",
            "datasets/mnist/train-labels.idx1-ubyte");
        test = load_mnist(
            "datasets/mnist/t10k-images.idx3-ubyte",
            "datasets/mnist/t10k-labels.idx1-ubyte");
    }

    int channels = train.X.ndim() == 4 ? train.X.dim(1) : 1;
    int n_classes = train.Y.dim(1);

    train.X = train.X.reshape({train.n_samples, channels, 28, 28});
    test.X = test.X.reshape({test.n_samples, channels, 28, 28});

    std::cout << "Train: " << train.n_samples << ", Test: " << test.n_samples
              << ", channels: " << channels << ", classes: " << n_classes << "\n";

    Network net = make_cnn(channels, n_classes);

    int epochs = 5;
    int batch_size = 8;
    float lr = 0.1f;

    train_epochs(net, train, test, epochs, batch_size, lr);

    return 0;
}
