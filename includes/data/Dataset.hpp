#pragma once

#include "core/Tensor.cuh"

struct Dataset {
  Tensor X;
  Tensor Y;
  int n_samples = 0;
};
