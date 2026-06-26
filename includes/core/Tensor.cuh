#pragma once

#include <initializer_list>
#include <string>
#include <vector>

class Tensor {
public:
    Tensor();
    explicit Tensor(const std::vector<int>& shape, float fill = 0.0f);

    ~Tensor();
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    int numel() const;
    int ndim() const;
    int dim(int i) const;
    const std::vector<int>& sizes() const;

    // Returns a new Tensor that shares no memory — just reinterpreted shape.
    // FlattenLayer uses this: {N, C, H, W} → {N, C*H*W}
    Tensor reshape(const std::vector<int>& new_shape) const;

    // ── Host ↔ Device
    void upload();          // host → device
    void download() const;  // device → host (const: data_ is mutable)

    // ── Element-wise ops (any shape)
    Tensor  operator+(const Tensor& o) const;
    Tensor  operator-(const Tensor& o) const;
    Tensor  operator*(const Tensor& o) const;  // Hadamard product
    Tensor  operator*(float s) const;
    Tensor  operator/(float s) const;
    Tensor& operator-=(const Tensor& o);

    // ── Host indexing
    float& flat(int i);        // raw flat index — fastest host access
    float  flat(int i) const;

    // Multi-dimensional index: tensor.at({n, c, h, w}) or tensor.at({r, c})
    // Computed via strides_ — no shape knowledge required at call site
    float& at(std::initializer_list<int> idx);
    float  at(std::initializer_list<int> idx) const;

    // ── Raw access (free functions / CUDA kernels)
    float*       device_ptr();
    const float* device_ptr() const;
    bool         on_device() const;

    // ── Factory
    static Tensor zeros(const std::vector<int>& shape);
    static Tensor ones(const std::vector<int>& shape);
    static Tensor random_uniform(const std::vector<int>& shape, float lo, float hi);
    static Tensor random_normal(const std::vector<int>& shape, float mean, float std);

    void print(const std::string& name = "") const;

private:
    mutable std::vector<float> data_;  // mutable → download() can be const
    float*           d_data_  = nullptr;
    std::vector<int> shape_;
    std::vector<int> strides_;  // row-major strides — precomputed for at()

    // Called in every constructor and reshape() to keep strides_ in sync
    void compute_strides();
};
