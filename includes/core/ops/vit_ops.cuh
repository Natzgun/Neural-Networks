#pragma once

#include "core/Tensor.cuh"

namespace ops {

// Normaliza cada fila (token) a media 0 / varianza 1, luego aplica gamma/beta
// por canal. x: {rows, embed_dim} (rows = batch*tokens).
// x_norm_out/std_out son salidas: la capa los necesita cachear para el backward.
Tensor layer_norm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps,
                          Tensor& x_norm_out, Tensor& std_out);

Tensor layer_norm_backward(const Tensor& grad_output, const Tensor& gamma, const Tensor& x_norm,
                           const Tensor& std, Tensor& dgamma_out, Tensor& dbeta_out);

// Permutes para PatchEmbeddingLayer, sin paso por host.
// patch_tokens_forward:  {batch, channels, patches} -> {batch, patches, channels}
// patch_tokens_backward: {batch, patches, channels} -> {batch, channels, patches}
Tensor patch_tokens_forward(const Tensor& x, int batch, int channels, int patches);
Tensor patch_tokens_backward(const Tensor& grad_output, int batch, int channels, int patches);

// Antepone el token CLS: {batch, tokens, embed_dim} -> {batch, tokens + 1, embed_dim}.
// cls_token: {embed_dim}.
Tensor prepend_cls_token(const Tensor& input, const Tensor& cls_token, int batch, int tokens,
                         int embed_dim);

// Inverso: separa el gradiente de la posicion 0 (-> grad_cls_out, acumulado
// sobre el batch) del resto (-> grad_input, {batch, tokens, embed_dim}).
Tensor cls_token_backward(const Tensor& grad_output, int batch, int tokens, int embed_dim,
                          Tensor& grad_cls_out);

// Extrae la posicion 0 de la secuencia: {batch, tokens, embed_dim} -> {batch, embed_dim}.
Tensor extract_cls_forward(const Tensor& input, int batch, int tokens, int embed_dim);

// Inverso: el gradiente vuelve a la posicion 0, el resto queda en 0.
Tensor extract_cls_backward(const Tensor& grad_output, int batch, int tokens, int embed_dim);

} // namespace ops
