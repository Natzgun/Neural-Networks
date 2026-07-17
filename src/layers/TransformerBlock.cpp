#include "layers/TransformerBlock.hpp"

#include "core/ops/linalg.cuh"

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, int mlp_hidden_dim)
    : embed_dim_(embed_dim), ln1_(embed_dim), attn_(embed_dim, num_heads), ln2_(embed_dim),
      fc1_(embed_dim, mlp_hidden_dim), fc2_(mlp_hidden_dim, embed_dim) {
}

Tensor TransformerBlock::forward(const Tensor& input) {
  // Rama 1: atencion, con su propio residual.
  Tensor attn_out = attn_.forward(ln1_.forward(input));
  Tensor x1 = ops::add(input, attn_out);

  // Rama 2: MLP, con su propio residual. DenseLayer/GELULayer solo aceptan
  // 2D, asi que colapsamos batch*tokens igual que en MultiHeadAttentionLayer.
  int batch = x1.dim(0);
  int tokens = x1.dim(1);

  Tensor normed2_flat = ln2_.forward(x1).reshape({batch * tokens, embed_dim_});
  Tensor fc1_out_flat = fc1_.forward(normed2_flat);
  Tensor gelu_out_flat = gelu_.forward(fc1_out_flat);
  Tensor mlp_out_flat = fc2_.forward(gelu_out_flat);
  Tensor mlp_out = mlp_out_flat.reshape({batch, tokens, embed_dim_});

  return ops::add(x1, mlp_out);
}

Tensor TransformerBlock::backward(const Tensor& grad_output) {
  int batch = grad_output.dim(0);
  int tokens = grad_output.dim(1);

  // Rama 2 (MLP): x2 = x1 + fc2(gelu(fc1(ln2(x1)))).
  Tensor d_mlp_out_flat = grad_output.reshape({batch * tokens, embed_dim_});
  Tensor d_gelu_out_flat = fc2_.backward(d_mlp_out_flat);
  Tensor d_fc1_out_flat = gelu_.backward(d_gelu_out_flat);
  Tensor d_normed2_flat = fc1_.backward(d_fc1_out_flat);
  Tensor d_x1_from_mlp = ln2_.backward(d_normed2_flat.reshape({batch, tokens, embed_dim_}));

  // La suma residual reparte el gradiente sin modificarlo hacia ambas ramas.
  Tensor d_x1 = ops::add(grad_output, d_x1_from_mlp);

  // Rama 1 (Attention): x1 = input + attn(ln1(input)).
  Tensor d_normed1 = attn_.backward(d_x1);
  Tensor d_input_from_attn = ln1_.backward(d_normed1);

  Tensor d_input = ops::add(d_x1, d_input_from_attn);

  return d_input;
}

void TransformerBlock::update(float lr) {
  ln1_.update(lr);
  attn_.update(lr);
  ln2_.update(lr);
  fc1_.update(lr);
  gelu_.update(lr);
  fc2_.update(lr);
}
