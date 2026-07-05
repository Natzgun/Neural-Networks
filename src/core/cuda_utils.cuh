#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err = (call);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " — "                        \
                << cudaGetErrorString(err) << std::endl;                                           \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

static inline int div_ceil(int a, int b) {
  return (a + b - 1) / b;
}
