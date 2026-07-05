#include "utils/ExampleRunners.hpp"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  std::string example = argc > 1 ? argv[1] : "cnn";

  if (example == "cnn") {
    return run_cnn_training(argc - 1, argv + 1);
  }

  if (example == "mnist" || example == "simpsons" || example == "bloodmnist") {
    return run_cnn_training(argc, argv);
  }

  std::cerr << "Unknown example: " << example << "\n";
  return 1;
}
