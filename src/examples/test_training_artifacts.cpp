#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/Tensor.cuh"
#include "data/Dataset.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/conv/FlattenLayer.hpp"
#include "layers/dense/DenseLayer.hpp"
#include "network/Network.hpp"
#include "utils/Training.hpp"

namespace {
std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ','))
    fields.push_back(field);
  return fields;
}

std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(input, line))
    if (!line.empty())
      rows.push_back(split_csv_line(line));
  return rows;
}

bool validate_metrics(const std::filesystem::path& path) {
  auto rows = read_csv(path);
  const std::vector<std::string> expected_header{
      "epoch", "seed", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "time_ms"};
  if (rows.size() != 2 || rows[0] != expected_header || rows[1].size() != expected_header.size())
    return false;

  return std::stoi(rows[1][0]) == 1 && std::stoi(rows[1][1]) == 42 &&
         std::isfinite(std::stof(rows[1][2])) && std::isfinite(std::stof(rows[1][3])) &&
         std::isfinite(std::stof(rows[1][4])) && std::isfinite(std::stof(rows[1][5])) &&
         std::stoll(rows[1][6]) >= 0;
}

bool validate_predictions(const std::filesystem::path& path) {
  auto rows = read_csv(path);
  const std::vector<std::string> expected_header{"index",      "true_label", "predicted_label",
                                                 "confidence", "prob_0",     "prob_1"};
  if (rows.size() != 3 || rows[0] != expected_header)
    return false;

  for (int row_index = 1; row_index < static_cast<int>(rows.size()); ++row_index) {
    const auto& row = rows[row_index];
    if (row.size() != expected_header.size())
      return false;

    int index = std::stoi(row[0]);
    int true_label = std::stoi(row[1]);
    int predicted_label = std::stoi(row[2]);
    float confidence = std::stof(row[3]);
    float prob_0 = std::stof(row[4]);
    float prob_1 = std::stof(row[5]);
    float expected_confidence = predicted_label == 0 ? prob_0 : prob_1;

    if (index != row_index - 1 || true_label != row_index - 1 || predicted_label < 0 ||
        predicted_label > 1 || !std::isfinite(confidence) || prob_0 < 0.0f || prob_1 < 0.0f ||
        std::abs(prob_0 + prob_1 - 1.0f) > 1e-4f ||
        std::abs(confidence - expected_confidence) > 1e-5f)
      return false;
  }
  return true;
}
} // namespace

int main() {
  Tensor::set_random_seed(42);

  Tensor images({2, 28 * 28}, 0.0f);
  images.flat(0) = 1.0f;
  images.flat(28 * 28 + 1) = 1.0f;
  images.upload();

  Tensor labels({2, 2}, 0.0f);
  labels.at({0, 0}) = 1.0f;
  labels.at({1, 1}) = 1.0f;
  labels.upload();

  Dataset dataset{std::move(images), std::move(labels), 2};
  Network network;
  network.add<FlattenLayer>();
  network.add<DenseLayer>(28 * 28, 2);
  network.add<SoftmaxLayer>();

  std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "neural_network_training_artifacts_test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  std::filesystem::path metrics_path = directory / "metrics.csv";
  std::filesystem::path predictions_path = directory / "predictions.csv";

  train_epochs(network, dataset, dataset, 1, 2, 0.01f, 42, metrics_path.string());
  export_predictions(network, dataset, predictions_path.string());

  bool metrics_ok = validate_metrics(metrics_path);
  bool predictions_ok = validate_predictions(predictions_path);
  std::filesystem::remove_all(directory);

  if (!metrics_ok || !predictions_ok) {
    std::cerr << "Training artifact CSV contract failed\n";
    return 1;
  }

  std::cout << "Training artifact CSV contract: OK\n";
  return 0;
}
