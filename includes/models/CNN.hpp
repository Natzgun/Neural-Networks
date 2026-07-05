#pragma once

#include "layers/ReLULayer.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/conv/ConvLayer.hpp"
#include "layers/conv/FlattenLayer.hpp"
#include "layers/conv/MaxPool2DLayer.hpp"
#include "layers/dense/DenseLayer.hpp"
#include "network/Network.hpp"

inline Network make_cnn(int channels, int n_classes) {
  Network net;
  net.add<Conv2DLayer>(channels, 16, 3, 1, 1);
  net.add<ReLULayer>();
  net.add<MaxPool2DLayer>(2, 2);
  net.add<Conv2DLayer>(16, 32, 3, 1, 1);
  net.add<ReLULayer>();
  net.add<MaxPool2DLayer>(2, 2);
  net.add<FlattenLayer>();
  net.add<DenseLayer>(1568, 128);
  net.add<ReLULayer>();
  net.add<DenseLayer>(128, n_classes);
  net.add<SoftmaxLayer>();
  return net;
}
