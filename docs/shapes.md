# Guia de Shapes en Tensores

Los shapes describen las dimensiones de un tensor. En este proyecto, revisar
shapes es la forma mas rapida de saber si un modelo es matematicamente valido
antes de depurar kernels CUDA o backpropagation.

## Reglas Rapidas

- La entrada de un MLP normalmente usa `[batch, features]`.
- La entrada de una CNN normalmente usa `[batch, channels, height, width]`.
- La entrada de un Transformer normalmente usa `[batch, tokens, embed_dim]`.
- La multiplicacion de matrices sigue `[M, K] x [K, N] = [M, N]`.
- La primera dimension suele ser `batch`: la cantidad de muestras procesadas juntas.

## Shapes En MLP

Un MLP recibe cada muestra como un vector plano.

| Tensor                | Shape                         | Significado                      |
| --------------------- | ----------------------------- | -------------------------------- |
| Batch de entrada      | `[batch, features]`           | Un batch de muestras aplanadas   |
| Pesos dense           | `[in_features, out_features]` | Matriz aprendible de proyeccion  |
| Salida dense          | `[batch, out_features]`       | Nueva representacion por muestra |
| Logits/probabilidades | `[batch, classes]`            | Una fila de scores por muestra   |

Ejemplo con MNIST:

```txt
Imagen de entrada:     [28, 28]
Imagen aplanada:       [784]
Batch de entrada:      [8, 784]
Pesos dense:           [784, 128]
Salida dense:          [8, 128]
Prediccion final:      [8, 10]
```

La multiplicacion importante es:

```txt
[8, 784] x [784, 128] = [8, 128]
```

## Shapes En CNN

Una CNN conserva la estructura espacial de la imagen en vez de aplanarla desde
el inicio.

| Tensor           | Shape                                             | Significado                  |
| ---------------- | ------------------------------------------------- | ---------------------------- |
| Batch de entrada | `[batch, channels, height, width]`                | Imagenes en formato NCHW     |
| Pesos conv       | `[out_channels, in_channels, kernel_h, kernel_w]` | Filtros aprendibles          |
| Salida conv      | `[batch, out_channels, out_h, out_w]`             | Mapas de caracteristicas     |
| Salida flatten   | `[batch, channels * height * width]`              | Vector antes de capas densas |
| Prediccion final | `[batch, classes]`                                | Probabilidades por clase     |

Ejemplo de la CNN actual:

```txt
Entrada:           [8, channels, 28, 28]
Salida conv 1:     [8, 16, 28, 28]
Salida maxpool 1:  [8, 16, 14, 14]
Salida conv 2:     [8, 32, 14, 14]
Salida maxpool 2:  [8, 32, 7, 7]
Salida flatten:    [8, 1568]
Salida dense:      [8, 128]
Prediccion:        [8, classes]
```

Por que `1568`?

```txt
32 * 7 * 7 = 1568
```

## Shapes En Transformer Y ViT

Un Vision Transformer convierte una imagen en una secuencia de patches. Cada
patch se vuelve un token, y cada token tiene un vector de embedding.

| Tensor              | Shape                              | Significado                          |
| ------------------- | ---------------------------------- | ------------------------------------ |
| Batch de imagenes   | `[batch, channels, height, width]` | Imagenes de entrada                  |
| Tokens de patches   | `[batch, tokens, patch_dim]`       | Patches aplanados                    |
| Tokens embebidos    | `[batch, tokens, embed_dim]`       | Vectores usados por el Transformer   |
| Attention Q/K/V     | `[batch, heads, tokens, head_dim]` | Proyecciones de multi-head attention |
| Scores de attention | `[batch, heads, tokens, tokens]`   | Matriz de atencion token-a-token     |
| Salida Transformer  | `[batch, tokens, embed_dim]`       | Representaciones actualizadas        |
| Prediccion final    | `[batch, classes]`                 | Salida final del clasificador        |

Ejemplo con imagenes de 28x28 y patches de 7x7:

```txt
Imagen:               [8, 1, 28, 28]
Patch size:           7 x 7
Cantidad de patches:  (28 / 7) * (28 / 7) = 16
Patch dim:            1 * 7 * 7 = 49
Tokens de patches:    [8, 16, 49]
Tokens embebidos:     [8, 16, embed_dim]
```
