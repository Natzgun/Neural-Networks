#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "MLP.hpp"
#include "ActivationFunction.hpp"
#include "Matrix.hpp"
#include "DatasetLoader.hpp"


const std::vector<std::string> LABEL_NAMES = {
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

struct EvalResult {
  float accuracy;
  std::vector<int> pred_classes;
  std::vector<int> true_classes;
  std::vector<float> confidences;
};

EvalResult evaluate(MLP& network,
                    const Matrix& X,
                    const Matrix& Y,
                    const std::string& split_name = "Test",
                    int n_show = 10) {

  Matrix y_pred = network.forward(X);
  int n         = X.rows;
  int n_classes = Y.cols;

  EvalResult result;
  result.pred_classes.resize(n);
  result.true_classes.resize(n);
  result.confidences.resize(n);

  int correct = 0;
  for (int i = 0; i < n; i++) {
    int pred_class = 0, true_class = 0;
    for (int j = 1; j < n_classes; j++) {
      if (y_pred.at(i,j) > y_pred.at(i,pred_class)) pred_class = j;
      if (Y.at(i,j)      > Y.at(i,true_class))      true_class = j;
    }
    result.pred_classes[i]  = pred_class;
    result.true_classes[i]  = true_class;
    result.confidences[i]   = y_pred.at(i, pred_class);
    if (pred_class == true_class) correct++;
  }
  result.accuracy = (float)(correct) / n;

  std::cout << "\n── Predictions " << split_name << " (firsts "
    << n_show << ") ──────────────────\n";
  for (int i = 0; i < std::min(n_show, n); i++) {
    std::string mark = (result.pred_classes[i] == result.true_classes[i]) ? "YES" : "X";
    std::cout << mark
      // << "  Real Label: [" << result.true_classes[i] << "]"
      // << " -> Prediction: [" << result.pred_classes[i]  << "]"
      << "  Real: [" << LABEL_NAMES[result.true_classes[i]] << "]"
      << " -> Pred: [" << LABEL_NAMES[result.pred_classes[i]]  << "]"
      << " (Accuracy: "
      << std::fixed << std::setprecision(4)
      << result.confidences[i] << ")\n";
  }

  std::cout << "\n" << split_name << " Accuracy: "
    << std::fixed << std::setprecision(4)
    << result.accuracy << "  ("
    << correct << "/" << n << ")\n";

  return result;
}

int main (int argc, char *argv[]) {
  // MLP network;
  //
  // network.add_layer(2, 8, new ReLU());
  // network.add_layer(8, 2, new Softmax());
  //
  // Matrix X(4, 2);
  // X.data = {
  //   0, 0,
  //   0, 1,
  //   1, 0,
  //   1, 1
  // };
  //
  // // One hot 4 muestras, 2 clases (0 o 1)
  // Matrix Y(4, 2);
  // Y.data = {
  //   1, 0,
  //   0, 1,
  //   0, 1,
  //   1, 0
  // };
  //
  // std::cout << "Entrenando XOR...\n";
  // network.train(X, Y, 10, 0.5f, 4);
  //
  // EvalResult eval_result = evaluate(network, X, Y, "XOR Test", 4);
  SimpsonDataset train_ds = load_simpson_dataset("grayscale_train/train");
  SimpsonDataset test_ds  = load_simpson_dataset("grayscale_test/test");

  int input_size = train_ds.X.cols;
  int n_classes  = train_ds.Y.cols;

  MLP network;
  network.add_layer(input_size, 256, new ReLU());
  network.add_layer(256, 128, new ReLU());
  network.add_layer(128, n_classes, new Softmax());

  network.train(train_ds.X, train_ds.Y, 100, 0.1f, 64);

  evaluate(network, train_ds.X, train_ds.Y, "Train", 10);
  evaluate(network, test_ds.X, test_ds.Y, "Test", 10);


  return 0;
}
