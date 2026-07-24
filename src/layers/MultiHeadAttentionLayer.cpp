#include "layers/MultiHeadAttentionLayer.hpp"

#include <cmath>

#include "core/ops/activations.cuh"
#include "core/ops/attention_ops.cuh"
#include "core/ops/linalg.cuh"

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int embed_dim, int num_heads)
    : embed_dim_(embed_dim), num_heads_(num_heads), head_dim_(embed_dim / num_heads),
      w_q_(embed_dim, embed_dim), w_k_(embed_dim, embed_dim), w_v_(embed_dim, embed_dim),
      w_o_(embed_dim, embed_dim) {
}

Tensor MultiHeadAttentionLayer::split_heads(const Tensor& x, int batch, int tokens) const {
  // {batch, tokens, embed_dim} -> {batch, tokens, heads, head_dim} (reshape,
  // gratis) -> {batch, heads, tokens, head_dim} (swap de ejes, en GPU).
  Tensor reshaped = x.reshape({batch, tokens, num_heads_, head_dim_});
  return ops::swap_middle_axes(reshaped);
}

Tensor MultiHeadAttentionLayer::combine_heads(const Tensor& x, int batch, int tokens) const {
  // Inverso de split_heads: mismo swap (es su propia inversa).
  Tensor merged = ops::swap_middle_axes(x);
  return merged.reshape({batch, tokens, embed_dim_});
}

Tensor MultiHeadAttentionLayer::forward(const Tensor& input) {
  int batch = input.dim(0);
  int tokens = input.dim(1);

  // DenseLayer solo acepta 2D: colapsamos batch*tokens para reusarla tal
  // cual, igual que en LayerNormLayer/PositionalEmbeddingLayer.
  Tensor flat_input = input.reshape({batch * tokens, embed_dim_});

  Tensor q_flat = w_q_.forward(flat_input);
  Tensor k_flat = w_k_.forward(flat_input);
  Tensor v_flat = w_v_.forward(flat_input);

  q_ = split_heads(q_flat.reshape({batch, tokens, embed_dim_}), batch, tokens);
  k_ = split_heads(k_flat.reshape({batch, tokens, embed_dim_}), batch, tokens);
  v_ = split_heads(v_flat.reshape({batch, tokens, embed_dim_}), batch, tokens);

  Tensor scores = ops::batched_matmul(q_, k_, /*transpose_b=*/true);
  scores = ops::scalar_mul(scores, 1.0f / std::sqrt(static_cast<float>(head_dim_)));

  Tensor scores_flat = scores.reshape({batch * num_heads_ * tokens, tokens});
  Tensor attn_flat = ops::softmax_forward(scores_flat);
  attn_ = attn_flat.reshape({batch, num_heads_, tokens, tokens});

  Tensor out_heads = ops::batched_matmul(attn_, v_);
  Tensor combined = combine_heads(out_heads, batch, tokens);

  Tensor combined_flat = combined.reshape({batch * tokens, embed_dim_});
  Tensor y_flat = w_o_.forward(combined_flat);

  return y_flat.reshape({batch, tokens, embed_dim_});
}

Tensor MultiHeadAttentionLayer::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  int tokens = grad_output.dim(1);

  Tensor grad_flat = grad_output.reshape({batch * tokens, embed_dim_});
  Tensor d_combined_flat = w_o_.backward(grad_flat);
  Tensor d_out_heads =
      split_heads(d_combined_flat.reshape({batch, tokens, embed_dim_}), batch, tokens);

  // d_attn = d_out @ V^T ; dV = attn^T @ d_out
  Tensor d_attn = ops::batched_matmul(d_out_heads, v_, /*transpose_b=*/true);
  Tensor d_v = ops::batched_matmul(attn_, d_out_heads, /*transpose_b=*/false, /*transpose_a=*/true);

  // Backward real de softmax (no el passthrough de SoftmaxLayer): por fila,
  // d_scores = attn * (d_attn - sum(d_attn * attn)).
  Tensor d_attn_flat = d_attn.reshape({batch * num_heads_ * tokens, tokens});
  Tensor attn_flat = attn_.reshape({batch * num_heads_ * tokens, tokens});
  Tensor d_scores_flat = ops::softmax_backward(d_attn_flat, attn_flat);

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  d_scores_flat = ops::scalar_mul(d_scores_flat, scale);
  Tensor d_scores = d_scores_flat.reshape({batch, num_heads_, tokens, tokens});

  // dQ = d_scores @ K ; dK = d_scores^T @ Q
  Tensor d_q = ops::batched_matmul(d_scores, k_);
  Tensor d_k = ops::batched_matmul(d_scores, q_, /*transpose_b=*/false, /*transpose_a=*/true);

  Tensor dx_q =
      w_q_.backward(combine_heads(d_q, batch, tokens).reshape({batch * tokens, embed_dim_}));
  Tensor dx_k =
      w_k_.backward(combine_heads(d_k, batch, tokens).reshape({batch * tokens, embed_dim_}));
  Tensor dx_v =
      w_v_.backward(combine_heads(d_v, batch, tokens).reshape({batch * tokens, embed_dim_}));

  Tensor dx = ops::add(dx_q, ops::add(dx_k, dx_v));

  return dx.reshape({batch, tokens, embed_dim_});
}

void MultiHeadAttentionLayer::update(float lr) {
  w_q_.update(lr);
  w_k_.update(lr);
  w_v_.update(lr);
  w_o_.update(lr);
}
