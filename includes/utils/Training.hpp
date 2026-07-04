#pragma once

#include <vector>

#include "data/Dataset.hpp"

class Network;

struct Metrics {
    float accuracy;
    float loss;
};

Dataset build_batch(const Dataset& ds, const std::vector<int>& indices,
                    int start, int batch_size);

Metrics evaluate(Network& net, const Dataset& ds);
