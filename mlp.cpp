#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using std::vector;
using std::cin;
using std::cout;
using std::endl;

class ActivationFunction {
public:
  virtual float forward(float x) = 0;
  virtual float derivative(float x) = 0;
  
};

class Sigmoid : public ActivationFunction {
public:
  float forward(float x) {
    return 1.0f / (1.0f + std::exp(-x));
  }

  float derivative(float x) {
    return x * (1.0f - x);
  }
  
};

vector<vector<float>> random_weights(int num_inputs, int num_neurons) {
  vector<vector<float>> weights(num_neurons, vector<float>(num_inputs));
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (int i = 0; i < num_neurons; ++i) {
    for (int j = 0; j < num_inputs; ++j) {
      weights[i][j] = dis(gen);
    }
  }
  return weights;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); };
float sigmoid_derivative(float s) {
  return s * (1.0f - s);
};

class Layer {
public:
  Layer(int num_inputs, int num_neurons) {
    inputs.resize(num_inputs);
    outputs.resize(num_neurons);

    biases = vector<float>(num_neurons, 0.0f);
    weights = random_weights(num_inputs, num_neurons);
  }

  vector<float> forward(const vector<float> &inputs_) {
    this->inputs = inputs_;

    int num_neurons = weights.size();
    int num_inputs = inputs.size();

    for (size_t j = 0; j < num_neurons; j++) {
      float z = 1 * biases[j];
      for (size_t i = 0; i < num_inputs; i++) {
        z += inputs[i] * weights[j][i];
      }
      outputs[j] = sigmoid(z);
    }
    return this->outputs;
  }

  vector<float> backward(const vector<float> &output_gradient, float learning_rate) {
    vector<float> input_gradient(inputs.size(), 0.0f);

    for (size_t j = 0; j < outputs.size(); j++) {
      float d_activation = sigmoid_derivative(outputs[j]);
      float delta = output_gradient[j] * d_activation;

      for (size_t i = 0; i < inputs.size(); i++) {
        input_gradient[i] += delta * weights[j][i];
        weights[j][i] -= learning_rate * delta * inputs[i];
      }
      biases[j] -= learning_rate * delta;
    }
    return input_gradient;
  }

private:
  vector<float> inputs;
  vector<float> outputs;

  vector<vector<float>> weights;
  vector<float> biases;
};

class MLP {
public:
  MLP() {};
  void add_layer(int num_inputs, int num_neurons) {
    layers.emplace_back(num_inputs, num_neurons);
  }

  vector<float> forward(const vector<float> &inputs) {
    vector<float> current_inputs = inputs;
    for (auto &layer : layers) {
      current_inputs = layer.forward(current_inputs);
    }
    return current_inputs;
  }
  void train(const vector<vector<float>> &training_inputs,
             const vector<float> &y_expected,
             int epochs,
             float learning_rate) {

    for (size_t e = 0; e < epochs; e++) {

      for (size_t s = 0; s < training_inputs.size(); s++) {
        vector<float> y = forward(training_inputs[s]);
        vector<float> gradient(y.size());

        for (size_t i = 0; i < y.size(); i++) {
          float error = y[i] - y_expected[s];
          gradient[i] = 2.0f * error / y.size();
        }
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
          gradient = it->backward(gradient, learning_rate);
        };
      }

      if ((e + 1) % 10 == 0) {
        float mse_error = 0.0f;
        for (size_t s = 0; s < training_inputs.size(); s++) {
          vector<float> y_pred = forward(training_inputs[s]);
          float error = y_expected[s] - y_pred[0];
          mse_error += error * error;
        }
        mse_error /= training_inputs.size();

        cout << "Epoch " << e + 1 << "/" << epochs << ", Loss: " << mse_error << "\n";
      }
    }
  }

private:
  vector<Layer> layers;

};

vector<float> image_to_vector(const cv::Mat &image) {
  int r = image.rows;
  int c = image.cols;
  vector<float> out;

  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < c; j++) {
      float value = static_cast<float>(image.at<char>(i, j));
      out.push_back(value);
    }
  }

  return out;
}

int main(int argc, char *argv[]) { 
  vector<cv::String> bart;
  vector<cv::String> burns;
  vector<cv::String> homer;
  vector<cv::String> krusty;
  vector<cv::String> lisa;
  vector<cv::String> marge;
  vector<cv::String> milhouse;
  vector<cv::String> moe;
  vector<cv::String> flanders;
  vector<cv::String> skinner;
  
  cv::glob("grayscale_train/train/bart_simpson/*.jpg", bart, false);
  cv::glob("grayscale_train/train/charles_montgomery_burns/*.jpg", burns, false);
  cv::glob("grayscale_train/train/homer_simpson/*.jpg", homer, false);
  cv::glob("grayscale_train/train/krusty_the_clown/*.jpg", krusty, false  );
  cv::glob("grayscale_train/train/lisa_simpson/*.jpg", lisa, false);
  cv::glob("grayscale_train/train/marge_simpson/*.jpg", marge, false);
  cv::glob("grayscale_train/train/milhouse_van_houten/*.jpg", milhouse, false);
  cv::glob("grayscale_train/train/moe_szyslak/*.jpg", moe, false);
  cv::glob("grayscale_train/train/ned_flanders/*.jpg", flanders, false);
  cv::glob("grayscale_train/train/principal_skinner/*.jpg", skinner, false);

  size_t max_per_char = std::min({bart.size(), burns.size(), homer.size(), krusty.size(),
                                   lisa.size(), marge.size(), milhouse.size(), moe.size(),
                                   flanders.size(), skinner.size()});

  vector<vector<float>> input_imgs;
  for (size_t i = 0; i < max_per_char; i++) {
    cv::Mat _0 = cv::imread(bart[i]);
    cv::Mat _1 = cv::imread(burns[i]);
    cv::Mat _2 = cv::imread(homer[i]);
    cv::Mat _3 = cv::imread(krusty[i]);
    cv::Mat _4 = cv::imread(lisa[i]);
    cv::Mat _5 = cv::imread(marge[i]);
    cv::Mat _6 = cv::imread(milhouse[i]);
    cv::Mat _7 = cv::imread(moe[i]);
    cv::Mat _8 = cv::imread(flanders[i]);
    cv::Mat _9 = cv::imread(skinner[i]);
    input_imgs.push_back(image_to_vector(_0));
    input_imgs.push_back(image_to_vector(_1));
    input_imgs.push_back(image_to_vector(_2));
    input_imgs.push_back(image_to_vector(_3));
    input_imgs.push_back(image_to_vector(_4));
    input_imgs.push_back(image_to_vector(_5));
    input_imgs.push_back(image_to_vector(_6));
    input_imgs.push_back(image_to_vector(_7));
    input_imgs.push_back(image_to_vector(_8));
    input_imgs.push_back(image_to_vector(_9));
  }

  cout << "Loaded " << input_imgs.size() << " images" << endl;

  MLP* neural_network = new MLP();
  neural_network->add_layer(input_imgs[0].size(), 128);
  neural_network->add_layer(128, 64);
  neural_network->add_layer(64, 10);

  vector<float> y_expected(input_imgs.size(), 0.0f);
  for (size_t i = 0; i < input_imgs.size(); i++) {
    y_expected[i] = static_cast<float>(i / 10);
  }

  neural_network->train(input_imgs, y_expected, 5, 0.001f);

  cv::Mat test = cv::imread(bart[0]);
  vector<float> test_img = image_to_vector(test);
  vector<float> prediction = neural_network->forward(test_img);
  cout << "Test prediction (Bart): " << prediction[0] << endl;
  
  return 0;
}
