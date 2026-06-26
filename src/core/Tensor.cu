#include "core/Tensor.cuh"
#include "cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <random>

Tensor::Tensor() { compute_strides(); }

Tensor::Tensor(const std::vector<int>& shape, float fill) : shape_(shape) {
    compute_strides();
    data_.assign(numel(), fill);
}

Tensor::~Tensor() {
    if (d_data_) {
        cudaFree(d_data_);
        d_data_ = nullptr;
    }
}

Tensor::Tensor(const Tensor& other)
    : data_(other.data_), shape_(other.shape_), strides_(other.strides_) {
    if (other.d_data_) {
        CUDA_CHECK(cudaMalloc(&d_data_, numel() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data_, other.d_data_, numel() * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    if (d_data_) {
        cudaFree(d_data_);
        d_data_ = nullptr;
    }

    data_   = other.data_;
    shape_  = other.shape_;
    strides_ = other.strides_;

    if (other.d_data_) {
        CUDA_CHECK(cudaMalloc(&d_data_, numel() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data_, other.d_data_, numel() * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)),
      d_data_(other.d_data_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)) {
    other.d_data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    if (d_data_) cudaFree(d_data_);

    data_    = std::move(other.data_);
    d_data_  = other.d_data_;
    shape_   = std::move(other.shape_);
    strides_ = std::move(other.strides_);

    other.d_data_ = nullptr;
    return *this;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    int stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

int Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<int>());
}

int Tensor::ndim() const { return static_cast<int>(shape_.size()); }
int Tensor::dim(int i) const { return shape_[i]; }
const std::vector<int>& Tensor::sizes() const { return shape_; }

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                    std::multiplies<int>());
    if (new_numel != numel()) {
        std::cerr << "reshape: shape mismatch (" << new_numel << " vs "
                  << numel() << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    Tensor out(new_shape);
    out.data_ = data_;
    if (d_data_) {
        CUDA_CHECK(cudaMalloc(&out.d_data_, numel() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(out.d_data_, d_data_, numel() * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    return out;
}

void Tensor::upload() {
    if (numel() == 0) return;
    if (!d_data_) {
        CUDA_CHECK(cudaMalloc(&d_data_, numel() * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_data_, data_.data(), numel() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Tensor::download() const {
    if (!d_data_) return;
    data_.resize(numel());
    CUDA_CHECK(cudaMemcpy(data_.data(), d_data_, numel() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

namespace {
inline void assert_same_shape(const Tensor& a, const Tensor& b) {
    if (a.sizes() != b.sizes()) {
        std::cerr << "Tensor shape mismatch in element-wise op" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
}  // namespace

Tensor Tensor::operator+(const Tensor& o) const {
    assert_same_shape(*this, o);
    Tensor r(shape_);
    for (int i = 0; i < numel(); ++i) r.data_[i] = data_[i] + o.data_[i];
    return r;
}

Tensor Tensor::operator-(const Tensor& o) const {
    assert_same_shape(*this, o);
    Tensor r(shape_);
    for (int i = 0; i < numel(); ++i) r.data_[i] = data_[i] - o.data_[i];
    return r;
}

Tensor Tensor::operator*(const Tensor& o) const {
    assert_same_shape(*this, o);
    Tensor r(shape_);
    for (int i = 0; i < numel(); ++i) r.data_[i] = data_[i] * o.data_[i];
    return r;
}

Tensor Tensor::operator*(float s) const {
    Tensor r(shape_);
    for (int i = 0; i < numel(); ++i) r.data_[i] = data_[i] * s;
    return r;
}

Tensor Tensor::operator/(float s) const {
    Tensor r(shape_);
    for (int i = 0; i < numel(); ++i) r.data_[i] = data_[i] / s;
    return r;
}

Tensor& Tensor::operator-=(const Tensor& o) {
    assert_same_shape(*this, o);
    for (int i = 0; i < numel(); ++i) data_[i] -= o.data_[i];
    return *this;
}

float& Tensor::flat(int i) { return data_[i]; }
float  Tensor::flat(int i) const { return data_[i]; }

float& Tensor::at(std::initializer_list<int> idx) {
    int flat_idx = 0;
    int i = 0;
    for (int id : idx) flat_idx += id * strides_[i++];
    return data_[flat_idx];
}

float Tensor::at(std::initializer_list<int> idx) const {
    int flat_idx = 0;
    int i = 0;
    for (int id : idx) flat_idx += id * strides_[i++];
    return data_[flat_idx];
}

float*       Tensor::device_ptr()       { return d_data_; }
const float* Tensor::device_ptr() const { return d_data_; }
bool         Tensor::on_device() const  { return d_data_ != nullptr; }

Tensor Tensor::zeros(const std::vector<int>& shape) { return Tensor(shape, 0.0f); }
Tensor Tensor::ones(const std::vector<int>& shape)  { return Tensor(shape, 1.0f); }

Tensor Tensor::random_uniform(const std::vector<int>& shape, float lo, float hi) {
    Tensor t(shape);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(lo, hi);
    for (auto& x : t.data_) x = dis(gen);
    return t;
}

Tensor Tensor::random_normal(const std::vector<int>& shape, float mean, float std) {
    Tensor t(shape);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dis(mean, std);
    for (auto& x : t.data_) x = dis(gen);
    return t;
}

void Tensor::print(const std::string& name) const {
    if (!name.empty()) std::cout << name << " ";
    std::cout << "shape [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ",";
    }
    std::cout << "]\n";

    int n = numel();
    if (n <= 100) {
        for (int i = 0; i < n; ++i) {
            std::cout << data_[i];
            if (i + 1 < n) std::cout << " ";
        }
        std::cout << "\n";
    } else {
        std::cout << "... (" << n << " elements)\n";
    }
}
