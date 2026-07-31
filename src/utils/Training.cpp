#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/Tensor.cuh"
#include "data/Dataset.hpp"
#include "network/Network.hpp"
#include "utils/Training.hpp"

namespace {
constexpr int kImageHeight = 28;
constexpr int kImageWidth = 28;
constexpr int kEvaluationBatchSize = 1000;

int argmax_row(const Tensor& tensor, int row, int columns) {
  int result = 0;
  for (int column = 1; column < columns; ++column)
    if (tensor.at({row, column}) > tensor.at({row, result}))
      result = column;
  return result;
}

template <typename Consumer>
void for_each_prediction_batch(Network& net, const Dataset& ds, Consumer consume) {
  int n = ds.n_samples;
  int features = ds.X.numel() / n;
  int channels = features / (kImageHeight * kImageWidth);

  for (int start = 0; start < n; start += kEvaluationBatchSize) {
    int end = std::min(start + kEvaluationBatchSize, n);
    int current = end - start;

    Tensor x({current, features});
    for (int i = 0; i < current; ++i)
      for (int j = 0; j < features; ++j)
        x.flat(i * features + j) = ds.X.flat((start + i) * features + j);
    x.upload();

    Tensor pred = net.forward(x.reshape({current, channels, kImageHeight, kImageWidth}));
    pred.download();
    consume(start, current, pred);
  }
}
} // namespace

Dataset build_batch(const Dataset& ds, const std::vector<int>& indices, int start, int batch_size) {
  int n = ds.n_samples;
  int features = ds.X.numel() / n;
  int n_classes = ds.Y.dim(1);

  int end = std::min(start + batch_size, n);
  int current = end - start;

  Tensor X({current, features});
  Tensor Y({current, n_classes}, 0.0f);

  for (int i = 0; i < current; ++i) {
    int idx = indices[start + i];
    for (int j = 0; j < features; ++j)
      X.flat(i * features + j) = ds.X.flat(idx * features + j);
    for (int j = 0; j < n_classes; ++j)
      Y.flat(i * n_classes + j) = ds.Y.flat(idx * n_classes + j);
  }

  X.upload();
  Y.upload();

  int channels = features / (kImageHeight * kImageWidth);

  Dataset batch;
  batch.X = X.reshape({current, channels, kImageHeight, kImageWidth});
  batch.Y = std::move(Y);
  batch.n_samples = current;
  return batch;
}

Metrics evaluate(Network& net, const Dataset& ds) {
  int n = ds.n_samples;
  int n_classes = ds.Y.dim(1);
  const float eps = 1e-9f;

  Tensor labels = ds.Y;
  labels.download();

  int correct = 0;
  float total_loss = 0.0f;
  for_each_prediction_batch(net, ds, [&](int start, int current, const Tensor& pred) {
    for (int i = 0; i < current; ++i) {
      int pred_class = argmax_row(pred, i, n_classes);
      int true_class = argmax_row(labels, start + i, n_classes);
      if (pred_class == true_class)
        ++correct;

      total_loss -= std::log(pred.at({i, true_class}) + eps);
    }
  });

  Metrics metrics;
  metrics.accuracy = static_cast<float>(correct) / n;
  metrics.loss = total_loss / n;
  return metrics;
}

void train_epochs(Network& net, Dataset& train, Dataset& test, int epochs, int batch_size, float lr,
                  std::uint32_t seed, const std::string& metrics_path) {
  int n = train.n_samples;
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 generator(seed);

  std::ofstream metrics_file;
  if (!metrics_path.empty()) {
    metrics_file.open(metrics_path);
    if (!metrics_file)
      throw std::runtime_error("Failed to open metrics file: " + metrics_path);
    metrics_file << "epoch,seed,train_loss,train_accuracy,test_loss,test_accuracy,time_ms\n";
    metrics_file << std::fixed << std::setprecision(6);
  }

  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto start_time = std::chrono::steady_clock::now();
    std::shuffle(indices.begin(), indices.end(), generator);

    for (int start = 0; start < n; start += batch_size) {
      Dataset batch = build_batch(train, indices, start, batch_size);
      net.train_step(batch.X, batch.Y, lr);
    }

    Metrics train_metrics = evaluate(net, train);
    Metrics test_metrics = evaluate(net, test);

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " | train loss: " << train_metrics.loss
              << " | train acc: " << train_metrics.accuracy << " | test loss: " << test_metrics.loss
              << " | test acc: " << test_metrics.accuracy << " | time: " << elapsed_ms << "ms\n";

    if (metrics_file) {
      metrics_file << epoch + 1 << ',' << seed << ',' << train_metrics.loss << ','
                   << train_metrics.accuracy << ',' << test_metrics.loss << ','
                   << test_metrics.accuracy << ',' << elapsed_ms << '\n';
      metrics_file.flush();
      if (!metrics_file)
        throw std::runtime_error("Failed to write metrics file: " + metrics_path);
    }
  }
}

void export_predictions(Network& net, const Dataset& ds, const std::string& output_path) {
  std::ofstream output(output_path);
  if (!output)
    throw std::runtime_error("Failed to open predictions file: " + output_path);

  int n_classes = ds.Y.dim(1);
  output << "index,true_label,predicted_label,confidence";
  for (int class_index = 0; class_index < n_classes; ++class_index)
    output << ",prob_" << class_index;
  output << '\n' << std::fixed << std::setprecision(6);

  Tensor labels = ds.Y;
  labels.download();

  for_each_prediction_batch(net, ds, [&](int start, int current, const Tensor& pred) {
    for (int i = 0; i < current; ++i) {
      int predicted_label = argmax_row(pred, i, n_classes);
      int true_label = argmax_row(labels, start + i, n_classes);

      output << start + i << ',' << true_label << ',' << predicted_label << ','
             << pred.at({i, predicted_label});
      for (int class_index = 0; class_index < n_classes; ++class_index)
        output << ',' << pred.at({i, class_index});
      output << '\n';
    }
  });

  output.flush();
  if (!output)
    throw std::runtime_error("Failed to write predictions file: " + output_path);
}
