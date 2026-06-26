#include <algorithm>
#include <cmath>
#include <vector>

#include "core/Tensor.cuh"
#include "data/Dataset.hpp"
#include "network/Network.hpp"

struct Metrics {
    float accuracy;
    float loss;
};

Dataset build_batch(const Dataset& ds, const std::vector<int>& indices,
                    int start, int batch_size) {
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

    int channels = features / (28 * 28);

    Dataset batch;
    batch.X = X.reshape({current, channels, 28, 28});
    batch.Y = std::move(Y);
    batch.n_samples = current;
    return batch;
}

Metrics evaluate(Network& net, const Dataset& ds) {
    int n = ds.n_samples;
    int n_classes = ds.Y.dim(1);
    int eval_batch = 1000;
    int features = ds.X.numel() / n;
    int channels = features / (28 * 28);
    const float eps = 1e-9f;

    Tensor labels = ds.Y;
    labels.download();

    int correct = 0;
    float total_loss = 0.0f;
    for (int start = 0; start < n; start += eval_batch) {
        int end = std::min(start + eval_batch, n);
        int current = end - start;

        Tensor x({current, features});
        for (int i = 0; i < current; ++i)
            for (int j = 0; j < features; ++j)
                x.flat(i * features + j) = ds.X.flat((start + i) * features + j);
        x.upload();

        Tensor pred = net.forward(x.reshape({current, channels, 28, 28}));
        pred.download();

        for (int i = 0; i < current; ++i) {
            int pred_class = 0;
            int true_class = 0;
            for (int j = 1; j < n_classes; ++j) {
                if (pred.at({i, j}) > pred.at({i, pred_class})) pred_class = j;
                if (labels.at({start + i, j}) > labels.at({start + i, true_class}))
                    true_class = j;
            }
            if (pred_class == true_class) ++correct;

            for (int j = 0; j < n_classes; ++j) {
                if (labels.at({start + i, j}) > 0.5f)
                    total_loss -= std::log(pred.at({i, j}) + eps);
            }
        }
    }

    Metrics m;
    m.accuracy = static_cast<float>(correct) / n;
    m.loss = total_loss / n;
    return m;
}
