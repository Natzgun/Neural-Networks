# Vision Transformer en C++ y CUDA

Este repositorio implementa redes neuronales desde cero con C++20 y CUDA. Su
componente principal es un **Vision Transformer (ViT)** para clasificar imágenes
de `28 x 28` píxeles de MNIST y FashionMNIST. El proyecto no utiliza PyTorch,
TensorFlow ni un motor de diferenciación automática: cada capa define
explícitamente su propagación hacia delante, retropropagación y actualización de
parámetros.

## Contenido

1. [Compilación y ejecución](#compilación-y-ejecución)
2. [Arquitectura del proyecto](#arquitectura-del-proyecto)
3. [Hardware de referencia](#hardware-de-referencia)
4. [Configuración actual del ViT](#configuración-actual-del-vit)
5. [Recorrido completo de una imagen](#recorrido-completo-de-una-imagen)
6. [Archivos que componen el ViT](#archivos-que-componen-el-vit)
7. [Entrenamiento y retropropagación](#entrenamiento-y-retropropagación)
8. [Operaciones CUDA](#operaciones-cuda)
9. [DenseLayer dentro del ViT](#denselayer-dentro-del-vit)
10. [CNN y su relación con el ViT](#cnn-y-su-relación-con-el-vit)
11. [Datasets](#datasets)
12. [Pruebas disponibles](#pruebas-disponibles)
13. [Diferencias respecto al paper](#diferencias-respecto-al-paper)
14. [Limitaciones y oportunidades de optimización](#limitaciones-y-oportunidades-de-optimización)
15. [Orden recomendado de estudio](#orden-recomendado-de-estudio)

## Compilación y ejecución

El proyecto requiere CMake 3.24 o superior, un compilador compatible con C++20
y el toolkit de CUDA. CMake compila para la arquitectura de la GPU instalada
mediante `CMAKE_CUDA_ARCHITECTURES native`.

```bash
cmake -B build -S .
cmake --build build -j"$(nproc)"
```

Entrenamiento del ViT:

Los datasets no se distribuyen en el repositorio. Antes de ejecutar estos
comandos, los archivos IDX deben estar ubicados según la estructura descrita en
la sección [Datasets](#datasets).

```bash
# FashionMNIST: 1 época y batch d
./build/neural_networks vit fashionmnist

# FashionMNIST: 4 épocas y batch de 64
./build/neural_networks vit fashionmnist 4 64

# MNIST: 8 épocas y batch de 128
./build/neural_networks vit mnist 8 128

```

La forma general del comando es:

```text
./build/neural_networks vit [fashionmnist|mnist] [epochs] [batch_size]
```

El despacho se realiza en [`src/main.cpp`](src/main.cpp). Los argumentos
posteriores a `vit` se entregan a `run_vit_training()`.

## Arquitectura del proyecto

El proyecto está organizado en capas de responsabilidad. Los loaders preparan
los datos, `Network` coordina las capas de los modelos y las operaciones
numéricas se ejecutan mediante kernels CUDA propios.

```mermaid
flowchart LR
    A[Datasets] --> B[Loaders]
    B --> C[Training]
    C --> D[Network]
    D --> E[ViT / CNN]
    E --> F[Layers]
    F --> G[CUDA ops]
    G --> H[Tensor CPU/GPU]
```

| Área             | Rutas principales                                                                                  | Responsabilidad                                                            |
| ---------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Punto de entrada | [`src/main.cpp`](src/main.cpp), [`src/examples`](src/examples)                                     | Seleccionar CNN o ViT e iniciar el entrenamiento.                          |
| Modelos          | [`includes/models`](includes/models)                                                               | Definir el orden de las capas de CNN y ViT.                                |
| Red secuencial   | [`includes/network`](includes/network), [`src/network`](src/network)                               | Ejecutar `forward`, `backward` y `update` sobre todas las capas.           |
| Capas            | [`includes/layers`](includes/layers), [`src/layers`](src/layers)                                   | Implementar convolución, capas densas y componentes del Transformer.       |
| Tensores         | [`includes/core/Tensor.cuh`](includes/core/Tensor.cuh), [`src/core/Tensor.cu`](src/core/Tensor.cu) | Administrar shapes, memoria host, memoria CUDA y transferencias.           |
| Operaciones CUDA | [`includes/core/ops`](includes/core/ops), [`src/core/ops`](src/core/ops)                           | Implementar álgebra lineal, activaciones, convolución, pooling y atención. |
| Datos            | [`includes/data`](includes/data), [`includes/loaders`](includes/loaders)                           | Leer datasets, normalizar imágenes y construir etiquetas one-hot.          |
| Entrenamiento    | [`includes/utils`](includes/utils), [`src/utils`](src/utils)                                       | Construir batches, entrenar por épocas y calcular métricas.                |
| Pruebas          | [`src/examples/test_*.cpp`](src/examples)                                                          | Verificar shapes, valores y algunos gradientes numéricos.                  |

La abstracción común es [`Layer`](includes/layers/Layer.hpp). Toda capa debe
implementar `forward()` y `backward()`; las capas con parámetros también
sobrescriben `update()`:

```cpp
class Layer {
public:
  virtual Tensor forward(const Tensor& input) = 0;
  virtual Tensor backward(const Tensor& grad_output) = 0;
  virtual void update(float lr) {}
  virtual ~Layer() = default;
};
```

No existe autograd. Cada clase guarda los valores que necesita para calcular
manualmente sus gradientes durante el backward.

## Hardware de referencia

La siguiente es la configuración de GPU proporcionada como referencia para el
desarrollo y la ejecución del proyecto:

| Propiedad                        |                              Valor |
| -------------------------------- | ---------------------------------: |
| Nombre                           | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute Capability               |                                8.9 |
| Memoria global                   |                            7.65 GB |
| Multiprocesadores SM             |                                 24 |
| Frecuencia máxima del núcleo     |                        2250.00 MHz |
| Frecuencia de memoria            |                        8001.00 MHz |
| Memoria compartida por bloque    |                           48.00 KB |
| Memoria constante                |                           64.00 KB |
| Caché L2                         |                           32.00 MB |
| Registros por bloque             |                              65536 |
| Tamaño del warp                  |                           32 hilos |
| Máximo de hilos por bloque       |                               1024 |
| Máximo de hilos por SM           |                               1536 |
| Dimensión máxima del bloque      |                   1024 x 1024 x 64 |
| Dimensión máxima del grid        |         2147483647 x 65535 x 65535 |
| Ancho del bus de memoria         |                           128 bits |
| Ejecución concurrente de kernels |                                 Sí |
| Memoria unificada                |                                 Sí |
| Procesadores asíncronos          |                                  2 |

## Configuración actual del ViT

Los hiperparámetros se encuentran en
[`src/examples/train_vit.cpp`](src/examples/train_vit.cpp):

```cpp
float lr = 0.01f;

int image_h = 28;
int image_w = 28;
int patch_size = 7;
int in_channels = 1;
int embed_dim = 32;
int num_heads = 4;
int mlp_hidden_dim = 64;
int num_layers = 1;
int n_classes = train.Y.dim(1);
```

| Hiperparámetro       |      Valor | Efecto                                                             |
| -------------------- | ---------: | ------------------------------------------------------------------ |
| `image_h`, `image_w` | `28`, `28` | Resolución de MNIST y FashionMNIST.                                |
| `patch_size`         |        `7` | Divide la imagen en una cuadrícula de `4 x 4`.                     |
| `in_channels`        |        `1` | Las imágenes están en escala de grises.                            |
| `embed_dim`          |       `32` | Dimensión de cada token durante todo el Transformer.               |
| `num_heads`          |        `4` | Divide la atención en cuatro cabezas de ocho dimensiones.          |
| `mlp_hidden_dim`     |       `64` | Expande temporalmente cada token de 32 a 64 valores.               |
| `num_layers`         |        `1` | Utiliza un bloque Transformer.                                     |
| `n_classes`          |       `10` | Número de categorías del dataset.                                  |
| `lr`                 |     `0.01` | Tamaño del paso de SGD.                                            |
| `batch_size`         |        `8` | Cantidad de imágenes por actualización, salvo argumento diferente. |
| `epochs`             |        `1` | Recorridos completos del dataset, salvo argumento diferente.       |

De estos valores se derivan tres cantidades esenciales:

```text
num_patches = (28 / 7) * (28 / 7) = 16
num_tokens  = num_patches + token_CLS = 17
head_dim    = embed_dim / num_heads = 32 / 4 = 8
```

La configuración necesita cumplir tres condiciones matemáticas que todavía no
se validan explícitamente en el código:

```text
image_h % patch_size == 0
image_w % patch_size == 0
embed_dim % num_heads == 0
```

## Recorrido completo de una imagen

La arquitectura del modelo se construye en
[`includes/models/ViT.hpp`](includes/models/ViT.hpp). En lugar de representar
el recorrido como una cadena abstracta, la siguiente tabla muestra qué hace
cada etapa con un batch real.

| Etapa | Operación                  | Entrada          | Salida        | Archivo responsable                                                       |
| ----: | -------------------------- | ---------------- | ------------- | ------------------------------------------------------------------------- |
|     1 | Proyectar patches          | `[B, 1, 28, 28]` | `[B, 16, 32]` | [`PatchEmbeddingLayer.cpp`](src/layers/PatchEmbeddingLayer.cpp)           |
|     2 | Anteponer CLS              | `[B, 16, 32]`    | `[B, 17, 32]` | [`CLSTokenLayer.cpp`](src/layers/CLSTokenLayer.cpp)                       |
|     3 | Sumar posición             | `[B, 17, 32]`    | `[B, 17, 32]` | [`PositionalEmbeddingLayer.cpp`](src/layers/PositionalEmbeddingLayer.cpp) |
|     4 | Aplicar atención y MLP     | `[B, 17, 32]`    | `[B, 17, 32]` | [`TransformerBlock.cpp`](src/layers/TransformerBlock.cpp)                 |
|     5 | Normalizar salida          | `[B, 17, 32]`    | `[B, 17, 32]` | [`LayerNormLayer.cpp`](src/layers/LayerNormLayer.cpp)                     |
|     6 | Seleccionar CLS            | `[B, 17, 32]`    | `[B, 32]`     | [`ExtractCLSLayer.cpp`](src/layers/ExtractCLSLayer.cpp)                   |
|     7 | Clasificar                 | `[B, 32]`        | `[B, 10]`     | [`DenseLayer.cpp`](src/layers/DenseLayer.cpp)                             |
|     8 | Convertir a probabilidades | `[B, 10]`        | `[B, 10]`     | [`SoftmaxLayer.cpp`](src/layers/SoftmaxLayer.cpp)                         |

La dimensión `32` se conserva desde el patch embedding hasta la extracción de
CLS. Esto permite sumar conexiones residuales, porque ambas ramas siempre
producen tensores con la misma forma.

Con la configuración actual, el modelo contiene aproximadamente **11 114
parámetros aprendibles**:

| Componente               |                           Cálculo | Parámetros |
| ------------------------ | --------------------------------: | ---------: |
| Patch embedding          |             `32 * 1 * 7 * 7 + 32` |      1 600 |
| Token CLS                |                              `32` |         32 |
| Positional embedding     |                         `17 * 32` |        544 |
| LayerNorm del bloque     |                   `2 * (32 + 32)` |        128 |
| Proyecciones Q, K, V y O |              `4 * (32 * 32 + 32)` |      4 224 |
| MLP del bloque           | `(32 * 64 + 64) + (64 * 32 + 32)` |      4 192 |
| LayerNorm final          |                         `32 + 32` |         64 |
| Clasificador             |                    `32 * 10 + 10` |        330 |
| **Total**                |                                   | **11 114** |

## Archivos que componen el ViT

### `src/examples/train_vit.cpp`: configuración y punto de inicio

[`train_vit.cpp`](src/examples/train_vit.cpp) decide qué datos utilizar, valida
los argumentos básicos, define los hiperparámetros y conecta el modelo con el
bucle de entrenamiento.

#### `parse_positive_int_arg()`

```cpp
static int parse_positive_int_arg(int argc, char* argv[], int index,
                                  int default_value, const char* name) {
  if (argc <= index)
    return default_value;

  int value = std::atoi(argv[index]);
  if (value <= 0) {
    std::cerr << name << " must be positive\n";
    std::exit(EXIT_FAILURE);
  }
  return value;
}
```

Lee `epochs` y `batch_size`. Si el argumento no existe, devuelve el valor por
defecto; si es cero o negativo, detiene el programa. `std::atoi()` no distingue
entre texto inválido y cero, por lo que ambos casos terminan en el mismo error.

#### `run_vit_training()`

Esta función realiza cuatro tareas:

1. Carga MNIST o FashionMNIST.
2. Define la configuración del ViT.
3. Cambia las imágenes de `[N, 784]` a `[N, 1, 28, 28]`.
4. Construye la red y llama a `train_epochs()`.

```cpp
Network net = make_vit(image_h, image_w, patch_size, in_channels,
                       embed_dim, num_heads, mlp_hidden_dim,
                       num_layers, n_classes);

train_epochs(net, train, test, epochs, batch_size, lr);
```

Este archivo es el mejor punto de partida para experimentar, porque concentra
las decisiones del modelo sin mezclar sus kernels internos.

### `includes/models/ViT.hpp`: ensamblaje del modelo

[`ViT.hpp`](includes/models/ViT.hpp) actúa como la descripción ejecutable de la
arquitectura:

```cpp
inline Network make_vit(int image_h, int image_w, int patch_size,
                        int in_channels, int embed_dim, int num_heads,
                        int mlp_hidden_dim, int num_layers, int n_classes) {
  int grid_h = image_h / patch_size;
  int grid_w = image_w / patch_size;
  int num_patches = grid_h * grid_w;

  Network net;
  net.add<PatchEmbeddingLayer>(image_h, image_w, patch_size,
                               in_channels, embed_dim);
  net.add<CLSTokenLayer>(embed_dim);
  net.add<PositionalEmbeddingLayer>(num_patches + 1, embed_dim);

  for (int i = 0; i < num_layers; ++i)
    net.add<TransformerBlock>(embed_dim, num_heads, mlp_hidden_dim);

  net.add<LayerNormLayer>(embed_dim);
  net.add<ExtractCLSLayer>();
  net.add<DenseLayer>(embed_dim, n_classes);
  net.add<SoftmaxLayer>();
  return net;
}
```

`make_vit()` no ejecuta cálculos sobre imágenes. Su responsabilidad es crear
las capas en el orden correcto y devolver un `Network` listo para entrenar.

### `PatchEmbeddingLayer`: de imagen a secuencia

Archivos:

- Interfaz: [`includes/layers/PatchEmbeddingLayer.hpp`](includes/layers/PatchEmbeddingLayer.hpp)
- Implementación: [`src/layers/PatchEmbeddingLayer.cpp`](src/layers/PatchEmbeddingLayer.cpp)

Un Transformer procesa secuencias, no imágenes NCHW. `PatchEmbeddingLayer`
divide cada imagen en regiones no superpuestas y convierte cada región en un
token de dimensión `embed_dim`.

La proyección de un patch aplanado se puede expresar como:

```text
z_p = x_p E + b
```

`x_p` contiene `patch_size * patch_size * in_channels` píxeles y `E` es una
matriz aprendible. El proyecto implementa la misma operación mediante una
convolución cuyo kernel y stride son iguales al tamaño del patch:

```cpp
PatchEmbeddingLayer::PatchEmbeddingLayer(int image_h, int image_w,
                                         int patch_size, int in_channels,
                                         int embed_dim)
    : patch_size_(patch_size), embed_dim_(embed_dim),
      grid_h_(image_h / patch_size), grid_w_(image_w / patch_size),
      num_patches_(grid_h_ * grid_w_),
      conv_(in_channels, embed_dim, patch_size, patch_size, 0) {
}
```

Para la configuración actual, la convolución utiliza 32 filtros de `7 x 7`,
stride 7 y padding 0. Cada filtro produce una característica por patch.

#### `forward()`

```cpp
Tensor PatchEmbeddingLayer::forward(const Tensor& input) {
  Tensor conv_out = conv_.forward(input);
  int batch = conv_out.dim(0);

  Tensor flat = conv_out.reshape({batch, embed_dim_, num_patches_});
  return ops::patch_tokens_forward(flat, batch, embed_dim_, num_patches_);
}
```

| Operación                  | Shape resultante |
| -------------------------- | ---------------- |
| Entrada NCHW               | `[B, 1, 28, 28]` |
| Convolución de patches     | `[B, 32, 4, 4]`  |
| Agrupar cuadrícula         | `[B, 32, 16]`    |
| Permutar canales y patches | `[B, 16, 32]`    |

#### `backward()` y `update()`

`backward()` invierte la permutación, restaura la cuadrícula `4 x 4` y delega
el cálculo de gradientes a `Conv2DLayer`. `update()` delega la actualización de
pesos y bias a esa misma convolución.

```cpp
Tensor flat_grad = ops::patch_tokens_backward(
    grad_output, batch, embed_dim_, num_patches_);
Tensor conv_grad = flat_grad.reshape(
    {batch, embed_dim_, grid_h_, grid_w_});
return conv_.backward(conv_grad);
```

### `CLSTokenLayer`: token aprendible de clasificación

Archivos:

- Interfaz: [`includes/layers/CLSTokenLayer.hpp`](includes/layers/CLSTokenLayer.hpp)
- Implementación: [`src/layers/CLSTokenLayer.cpp`](src/layers/CLSTokenLayer.cpp)

El token CLS es un vector aprendible de 32 valores compartido por todas las
imágenes. Se inserta en la posición cero antes de los tokens de patches:

```text
Z = [x_cls; z_patch_1; z_patch_2; ...; z_patch_16]
```

El constructor lo inicializa con una distribución normal de desviación `0.02`:

```cpp
cls_token_ = Tensor::random_normal({embed_dim_}, 0.0f, 0.02f);
```

`forward()` llama a `ops::prepend_cls_token()` y transforma `[B, 16, 32]` en
`[B, 17, 32]`. Durante el backward, el gradiente de la posición cero se acumula
sobre todo el batch para actualizar el único vector `cls_token_`; los gradientes
de las otras posiciones vuelven a `PatchEmbeddingLayer`.

El CLS no contiene inicialmente un resumen de la imagen. Aprende a reunir
información porque participa como Query, Key y Value en cada bloque de
self-attention.

### `PositionalEmbeddingLayer`: posición de cada token

Archivos:

- Interfaz: [`includes/layers/PositionalEmbeddingLayer.hpp`](includes/layers/PositionalEmbeddingLayer.hpp)
- Implementación: [`src/layers/PositionalEmbeddingLayer.cpp`](src/layers/PositionalEmbeddingLayer.cpp)

La self-attention por sí sola no conoce el orden espacial de los patches. Por
eso se suma un vector aprendible diferente a cada posición:

```text
Z_0 = [x_cls; X_p^1 E; X_p^2 E; ...; X_p^N E] + E_pos
```

En esta implementación, `E_pos` se almacena aplanado como `[1, tokens *
embed_dim]`. El forward aplana `[B, T, D]`, reutiliza `ops::add_bias()` para
sumar el embedding a cada muestra y recupera la forma original:

```cpp
Tensor flat = input.reshape({batch, num_tokens_ * embed_dim_});
Tensor out = ops::add_bias(flat, pos_embedding_);
return out.reshape({batch, num_tokens_, embed_dim_});
```

El backward deja pasar el gradiente hacia los tokens y usa `sum_rows()` para
sumar el gradiente posicional de todas las imágenes del batch.

### `LayerNormLayer`: normalización por token

Archivos:

- Interfaz: [`includes/layers/LayerNormLayer.hpp`](includes/layers/LayerNormLayer.hpp)
- Implementación: [`src/layers/LayerNormLayer.cpp`](src/layers/LayerNormLayer.cpp)

LayerNorm normaliza cada token sobre sus `D` características:

```text
mu       = (1 / D) * sum(x_j)
variance = (1 / D) * sum((x_j - mu)^2)
x_norm   = (x - mu) / sqrt(variance + epsilon)
y        = gamma * x_norm + beta
```

`gamma` y `beta` son parámetros aprendibles de tamaño `D`. Se inicializan con
unos y ceros, respectivamente, por lo que inicialmente la capa solo normaliza.

El forward convierte temporalmente `[B, T, D]` en `[B * T, D]`: cada una de
esas filas corresponde a un token independiente. Guarda `x_norm` y la
desviación estándar para que `backward()` pueda calcular `dX`, `dGamma` y
`dBeta` mediante los kernels de [`vit_ops.cu`](src/core/ops/vit_ops.cu).

### `MultiHeadAttentionLayer`: self-attention multi-cabeza

Archivos:

- Interfaz: [`includes/layers/MultiHeadAttentionLayer.hpp`](includes/layers/MultiHeadAttentionLayer.hpp)
- Implementación: [`src/layers/MultiHeadAttentionLayer.cpp`](src/layers/MultiHeadAttentionLayer.cpp)
- Operaciones CUDA: [`src/core/ops/attention_ops.cu`](src/core/ops/attention_ops.cu)

Esta capa permite que cada token combine información de cualquier otro token de
la imagen. Como `Q`, `K` y `V` se obtienen de la misma entrada, se trata de
**self-attention**.

La clase contiene cuatro capas densas:

```cpp
w_q_(embed_dim, embed_dim),
w_k_(embed_dim, embed_dim),
w_v_(embed_dim, embed_dim),
w_o_(embed_dim, embed_dim)
```

Las tres primeras producen Query, Key y Value; `w_o_` mezcla el resultado de
las cabezas.

#### Proyecciones y separación de cabezas

```cpp
Tensor flat_input = input.reshape({batch * tokens, embed_dim_});

Tensor q_flat = w_q_.forward(flat_input);
Tensor k_flat = w_k_.forward(flat_input);
Tensor v_flat = w_v_.forward(flat_input);
```

Con `B` imágenes, 17 tokens, dimensión 32 y cuatro cabezas:

| Tensor | Antes de separar | Después de separar |
| ------ | ---------------- | ------------------ |
| `Q`    | `[B, 17, 32]`    | `[B, 4, 17, 8]`    |
| `K`    | `[B, 17, 32]`    | `[B, 4, 17, 8]`    |
| `V`    | `[B, 17, 32]`    | `[B, 4, 17, 8]`    |

`split_heads()` primero realiza un reshape a `[B, T, H, Dh]` y después
intercambia los ejes centrales mediante `ops::swap_middle_axes()`.

#### Scaled dot-product attention

Cada cabeza calcula:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(Dh)) V
```

El código correspondiente es:

```cpp
Tensor scores = ops::batched_matmul(q_, k_, /*transpose_b=*/true);
scores = ops::scalar_mul(
    scores, 1.0f / std::sqrt(static_cast<float>(head_dim_)));

Tensor scores_flat = scores.reshape(
    {batch * num_heads_ * tokens, tokens});
Tensor attn_flat = ops::softmax_forward(scores_flat);
attn_ = attn_flat.reshape(
    {batch, num_heads_, tokens, tokens});

Tensor out_heads = ops::batched_matmul(attn_, v_);
```

`QK^T` produce `[B, 4, 17, 17]`. Cada fila contiene cuánto debe atender un
token a los 17 tokens disponibles. La división por `sqrt(8)` evita productos
punto demasiado grandes que saturarían el softmax.

Después de multiplicar la atención por `V`, las cabezas se combinan y pasan por
`w_o_`:

```cpp
Tensor combined = combine_heads(out_heads, batch, tokens);
Tensor y_flat = w_o_.forward(
    combined.reshape({batch * tokens, embed_dim_}));
return y_flat.reshape({batch, tokens, embed_dim_});
```

#### Backward de atención

El backward recorre exactamente las operaciones en orden inverso:

```text
dV      = Attention^T * dOut
dAttn   = dOut * V^T
dScores = softmax_backward(dAttn) / sqrt(Dh)
dQ      = dScores * K
dK      = dScores^T * Q
dX      = dX_Q + dX_K + dX_V
```

A diferencia de la capa Softmax final, la atención sí utiliza el backward real
de softmax mediante `ops::softmax_backward()`.

### `TransformerBlock`: atención, MLP y residuales

Archivos:

- Interfaz: [`includes/layers/TransformerBlock.hpp`](includes/layers/TransformerBlock.hpp)
- Implementación: [`src/layers/TransformerBlock.cpp`](src/layers/TransformerBlock.cpp)

El bloque implementa la variante **pre-norm** utilizada por ViT:

```text
x1  = x + MHA(LN1(x))
out = x1 + MLP(LN2(x1))
```

Su composición interna es:

| Rama     | Componentes                                            | Propósito                                      |
| -------- | ------------------------------------------------------ | ---------------------------------------------- |
| Atención | `LayerNorm -> MultiHeadAttention -> suma residual`     | Intercambiar información entre tokens.         |
| MLP      | `LayerNorm -> Dense -> GELU -> Dense -> suma residual` | Transformar las características de cada token. |

El forward refleja directamente esas ecuaciones:

```cpp
Tensor attn_out = attn_.forward(ln1_.forward(input));
Tensor x1 = ops::add(input, attn_out);

Tensor normed2_flat = ln2_.forward(x1).reshape(
    {batch * tokens, embed_dim_});
Tensor fc1_out_flat = fc1_.forward(normed2_flat);
Tensor gelu_out_flat = gelu_.forward(fc1_out_flat);
Tensor mlp_out_flat = fc2_.forward(gelu_out_flat);
Tensor mlp_out = mlp_out_flat.reshape(
    {batch, tokens, embed_dim_});

return ops::add(x1, mlp_out);
```

El MLP trabaja con cada token por separado:

```text
[B * 17, 32] -> Dense -> [B * 17, 64]
              -> GELU  -> [B * 17, 64]
              -> Dense -> [B * 17, 32]
```

Las conexiones residuales conservan una ruta directa para los valores y los
gradientes. En el backward, el gradiente de cada suma se envía tanto por la
rama de identidad como por la rama transformada.

### `GELULayer`: activación del MLP

Archivos:

- Interfaz: [`includes/layers/GELULayer.hpp`](includes/layers/GELULayer.hpp)
- Implementación: [`src/layers/GELULayer.cpp`](src/layers/GELULayer.cpp)
- Kernel: [`src/core/ops/activations.cu`](src/core/ops/activations.cu)

GELU pondera cada valor utilizando la función de distribución normal:

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

El forward guarda la entrada y llama al kernel CUDA. El backward multiplica el
gradiente recibido por la derivada de GELU elemento a elemento. La capa no
tiene parámetros, por lo que no necesita `update()`.

### `ExtractCLSLayer`: representación final de la imagen

Archivos:

- Interfaz: [`includes/layers/ExtractCLSLayer.hpp`](includes/layers/ExtractCLSLayer.hpp)
- Implementación: [`src/layers/ExtractCLSLayer.cpp`](src/layers/ExtractCLSLayer.cpp)

Después del último bloque y de la LayerNorm final, solo se conserva el token de
la posición cero:

```cpp
Tensor ExtractCLSLayer::forward(const Tensor& input) {
  tokens_ = input.dim(1);
  return ops::extract_cls_forward(
      input, input.dim(0), tokens_, input.dim(2));
}
```

Esto reduce `[B, 17, 32]` a `[B, 32]`. El backward realiza la operación inversa:
crea un tensor `[B, 17, 32]`, coloca el gradiente en la posición CLS y deja en
cero las otras posiciones. Esos ceros no significan que los patches no
aprendan: dentro del Transformer, el CLS ya intercambió información con ellos
mediante atención.

### Clasificador y `SoftmaxLayer`

La salida CLS pasa por `DenseLayer(32, 10)`, que produce diez logits. Luego
[`SoftmaxLayer.cpp`](src/layers/SoftmaxLayer.cpp) convierte cada fila en una
distribución de probabilidades:

```text
p_i = exp(logit_i - max(logits)) / sum_j exp(logit_j - max(logits))
```

Restar el máximo mejora la estabilidad numérica sin cambiar el resultado.

```cpp
Tensor SoftmaxLayer::forward(const Tensor& input) {
  outputs_ = ops::softmax_forward(input);
  return outputs_;
}

Tensor SoftmaxLayer::backward(const Tensor& grad_output) {
  return grad_output;
}
```

El backward es intencionalmente un passthrough porque `Network::train_step()`
construye directamente el gradiente combinado de softmax con entropía cruzada:

```text
dLogits = (probabilidades - etiquetas_one_hot) / batch_size
```

Por tanto, `SoftmaxLayer` no implementa un backward genérico que pueda usarse
con cualquier función de pérdida.

## Entrenamiento y retropropagación

### `Network`: ejecución secuencial

Archivos:

- Interfaz: [`includes/network/Network.hpp`](includes/network/Network.hpp)
- Implementación: [`src/network/Network.cpp`](src/network/Network.cpp)

`Network` almacena las capas en un `std::vector<std::unique_ptr<Layer>>`.
`forward()` las recorre en orden y `backward()` en orden inverso:

```cpp
Tensor Network::forward(const Tensor& input) {
  Tensor current = input;
  for (auto& layer : layers_)
    current = layer->forward(current);
  return current;
}

void Network::backward(const Tensor& grad_output) {
  Tensor grad = grad_output;
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
    grad = layers_[i]->backward(grad);
}
```

`train_step()` combina predicción, gradiente y actualización:

```cpp
void Network::train_step(const Tensor& X, const Tensor& Y, float lr) {
  Tensor y_pred = forward(X);
  int batch_size = X.dim(0);

  Tensor grad = ops::sub(y_pred, Y);
  grad = ops::scalar_div(grad, static_cast<float>(batch_size));

  backward(grad);
  update(lr);
}
```

Cada capa con parámetros aplica SGD localmente:

```text
parametro = parametro - learning_rate * gradiente
```

No existe una clase `Optimizer`; tampoco hay momentum, Adam, AdamW, weight
decay, scheduler, clipping de gradientes o acumulación entre batches.

### `Training.cpp`: batches, épocas y métricas

[`src/utils/Training.cpp`](src/utils/Training.cpp) contiene tres funciones:

| Función          | Responsabilidad                                                              |
| ---------------- | ---------------------------------------------------------------------------- |
| `build_batch()`  | Copiar las muestras seleccionadas, subirlas a GPU y formar `[B, C, 28, 28]`. |
| `evaluate()`     | Ejecutar inferencia y calcular accuracy y entropía cruzada.                  |
| `train_epochs()` | Barajar índices, entrenar batches y evaluar al final de cada época.          |

El flujo de una época es:

1. Barajar los índices de entrenamiento.
2. Construir un batch en CPU.
3. Subir imágenes y etiquetas a GPU.
4. Ejecutar `Network::train_step()`.
5. Repetir hasta consumir las muestras.
6. Evaluar train y test completos.
7. Imprimir loss, accuracy y tiempo.

La entropía cruzada usada para evaluación es:

```text
loss = -log(probabilidad_de_la_clase_correcta + 1e-9)
```

El pipeline fuerza actualmente la resolución `28 x 28` dentro de
`build_batch()` y `evaluate()`. Por eso `make_vit()` parece configurable para
otras resoluciones, pero el entrenamiento compartido todavía no lo es.

## Operaciones CUDA

Las capas expresan la lógica de la red y delegan el cálculo numérico a las
operaciones de [`src/core/ops`](src/core/ops).

| Archivo                                             | Operaciones principales                     | Uso en el ViT                            |
| --------------------------------------------------- | ------------------------------------------- | ---------------------------------------- |
| [`linalg.cu`](src/core/ops/linalg.cu)               | Matmul, transpose, suma, bias y reducciones | Dense, residuales, embeddings y SGD.     |
| [`activations.cu`](src/core/ops/activations.cu)     | GELU, ReLU y softmax                        | MLP, atención y salida final.            |
| [`conv_ops.cu`](src/core/ops/conv_ops.cu)           | Forward y backward de Conv2D                | Proyección de patches.                   |
| [`attention_ops.cu`](src/core/ops/attention_ops.cu) | Batched matmul y permutación de ejes        | `QK^T`, `Attention * V` y cabezas.       |
| [`vit_ops.cu`](src/core/ops/vit_ops.cu)             | LayerNorm, patches y CLS                    | Operaciones específicas del Transformer. |

### `Tensor`: memoria host y device

[`Tensor.cuh`](includes/core/Tensor.cuh) y [`Tensor.cu`](src/core/Tensor.cu)
administran:

- Shape y strides row-major.
- Datos host mediante `std::vector<float>`.
- Datos device mediante `float*` y `cudaMalloc`.
- Transferencias con `upload()` y `download()`.
- Copias host-to-device, device-to-host y device-to-device.
- Inicialización de tensores.

Un detalle importante es que `Tensor::reshape()` no crea una vista barata. La
implementación actual crea otro tensor y, cuando los datos están en GPU,
realiza una copia device-to-device. Esto afecta al ViT porque atención y MLP
usan varios reshapes para adaptar tensores tridimensionales a `DenseLayer`.

Los wrappers CUDA también llaman frecuentemente a `cudaDeviceSynchronize()`
después de cada kernel. La sincronización ayuda a detectar fallos cerca de su
origen, pero introduce overhead e impide solapar trabajo asíncrono.

## DenseLayer dentro del ViT

Archivos:

- Interfaz: [`includes/layers/dense/DenseLayer.hpp`](includes/layers/dense/DenseLayer.hpp)
- Implementación: [`src/layers/DenseLayer.cpp`](src/layers/DenseLayer.cpp)

`DenseLayer` implementa una transformación afín:

```text
Y = XW + b
```

Su forward reutiliza las operaciones CUDA de álgebra lineal:

```cpp
Tensor DenseLayer::forward(const Tensor& input) {
  inputs_ = input;
  Tensor z = ops::add_bias(
      ops::matmul(input, weights_), biases_);
  outputs_ = z;
  return outputs_;
}
```

El backward calcula los tres gradientes necesarios:

```text
dX = dY W^T
dW = X^T dY
db = sum_rows(dY)
```

```cpp
Tensor grad_input = ops::matmul(
    grad_output, ops::transpose(weights_));
grad_weights_ = ops::matmul(
    ops::transpose(inputs_), grad_output);
grad_biases_ = ops::sum_rows(grad_output);
```

La capa aparece siete veces en el ViT actual:

| Ubicación               | Cantidad | Función                                          |
| ----------------------- | -------: | ------------------------------------------------ |
| Multi-head attention    |        4 | Proyectar `Q`, `K`, `V` y la salida combinada.   |
| MLP del bloque          |        2 | Expandir de 32 a 64 y contraer de 64 a 32.       |
| Cabeza de clasificación |        1 | Convertir CLS de 32 características a 10 logits. |

`DenseLayer` acepta matrices 2D. Por esta razón, las capas Transformer cambian
temporalmente `[B, T, D]` a `[B * T, D]` antes de utilizarla.

## CNN y su relación con el ViT

La CNN completa se construye en [`includes/models/CNN.hpp`](includes/models/CNN.hpp):

| Etapa                       | Salida para entrada `28 x 28` |
| --------------------------- | ----------------------------- |
| Conv2D `C -> 16`, kernel 3  | `[B, 16, 28, 28]`             |
| ReLU + MaxPool              | `[B, 16, 14, 14]`             |
| Conv2D `16 -> 32`, kernel 3 | `[B, 32, 14, 14]`             |
| ReLU + MaxPool              | `[B, 32, 7, 7]`               |
| Flatten                     | `[B, 1568]`                   |
| Dense                       | `[B, 128]`                    |
| Clasificador                | `[B, classes]`                |

La CNN completa **no forma parte del ViT**. La relación entre ambos modelos es
la reutilización de [`Conv2DLayer`](src/layers/ConvLayer.cpp) dentro de
`PatchEmbeddingLayer`.

| CNN                                                               | Patch embedding del ViT                                |
| ----------------------------------------------------------------- | ------------------------------------------------------ |
| Usa kernels pequeños y normalmente superpuestos.                  | Usa kernel y stride iguales al patch.                  |
| Produce mapas de características para otra convolución o pooling. | Produce tokens para el Transformer.                    |
| Introduce localidad mediante varias capas.                        | Entrega los tokens a atención global.                  |
| La CNN actual utiliza ReLU y MaxPool.                             | La proyección de patches no usa activación ni pooling. |

`Conv2DLayer` implementa forward, backward de entrada, backward de pesos,
backward de bias y actualización SGD. Dentro del ViT, sus filtros de `7 x 7`
aprenden la proyección inicial de cada patch.

## Datasets

El contenedor común está en [`includes/data/Dataset.hpp`](includes/data/Dataset.hpp):

```cpp
struct Dataset {
  Tensor X;
  Tensor Y;
  int n_samples;
};
```

El entrenamiento del ViT admite dos loaders:

| Dataset      | Loader                                                              | Entrada                               | Clases |
| ------------ | ------------------------------------------------------------------- | ------------------------------------- | -----: |
| MNIST        | [`MnistLoader.hpp`](includes/loaders/MnistLoader.hpp)               | Dígitos `28 x 28` en escala de grises |     10 |
| FashionMNIST | [`FashionMnistLoader.hpp`](includes/loaders/FashionMnistLoader.hpp) | Prendas `28 x 28` en escala de grises |     10 |

Los archivos descargados y descomprimidos deben respetar estas rutas y nombres:

```text
datasets/
├── mnist/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
└── fashionmnist/
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

El directorio `datasets/` está ignorado por Git. Un checkout nuevo necesita que
estos archivos se descarguen por separado antes del entrenamiento.

(Pudes buscar esto en internet es facil :D)

Ambos loaders leen archivos IDX big-endian, dividen cada píxel por `255.0` y
crean etiquetas one-hot. La normalización utilizada es únicamente `[0, 255] ->
[0, 1]`; no se aplica estandarización por media y desviación ni data
augmentation.

Los loaders de BloodMNIST y Simpsons pertenecen a la ruta de CNN y no se usan
en `run_vit_training()`.

## Pruebas disponibles

Las pruebas son ejecutables independientes definidos en [`CMakeLists.txt`](CMakeLists.txt).
No están registradas con CTest, por lo que deben ejecutarse manualmente:

```bash
./build/test_patch_embedding
./build/test_layer_norm
./build/test_batched_matmul
./build/test_attention
./build/test_transformer_block
./build/test_cls_token
./build/test_vit
```

| Ejecutable               | Evidencia que proporciona                                                  |
| ------------------------ | -------------------------------------------------------------------------- |
| `test_patch_embedding`   | Shapes del forward y backward.                                             |
| `test_layer_norm`        | Shape, media cercana a cero y desviación cercana a uno.                    |
| `test_batched_matmul`    | Comparaciones de multiplicación matricial y variantes transpuestas.        |
| `test_attention`         | Shapes y gradient checking numérico de la entrada.                         |
| `test_transformer_block` | Shapes, gradiente de entrada y ejecución de update.                        |
| `test_cls_token`         | Inserción de CLS y propagación del gradiente de entrada.                   |
| `test_vit`               | Forward completo, suma de softmax y ejecución de un paso de entrenamiento. |

Estas pruebas no demuestran todavía la corrección numérica de todos los
gradientes de parámetros. Tampoco verifican que la loss disminuya durante
varios pasos ni están integradas en una suite automatizada.

## Diferencias respecto al paper

El paper _An Image is Worth 16x16 Words_ presenta la misma idea arquitectónica,
pero estudia modelos y datasets de otra escala.

| Elemento         | Paper de ViT                                   | Implementación del proyecto                   |
| ---------------- | ---------------------------------------------- | --------------------------------------------- |
| Patch projection | Proyección lineal de patches                   | Conv2D equivalente con kernel y stride 7.     |
| Token CLS        | Aprendible                                     | Aprendible, inicializado con desviación 0.02. |
| Posición         | Embedding aprendible                           | Embedding aprendible.                         |
| Encoder          | Pre-norm, MHA y MLP con GELU                   | Misma estructura fundamental.                 |
| Profundidad      | 12 o más bloques según variante                | 1 bloque.                                     |
| Embedding        | Centenas de dimensiones                        | 32 dimensiones.                               |
| MLP              | Expansión generalmente 4x                      | Expansión 2x, de 32 a 64.                     |
| Regularización   | Dropout y otras técnicas según experimento     | Sin dropout ni stochastic depth.              |
| Optimización     | Adam y recetas de entrenamiento especializadas | SGD simple con learning rate 0.01.            |
| Datos            | Preentrenamiento a gran escala                 | Entrenamiento desde cero con 60 000 imágenes. |

Por tanto, el proyecto implementa un ViT real, pero no pretende reproducir la
escala ni la receta experimental del paper. Es una versión reducida que permite
estudiar manualmente todas las operaciones.

## Limitaciones y oportunidades de optimización

**"A tí N años en el futuro"**
Antes de cambiar hiperparámetros conviene separar optimización del modelo,
optimización del entrenamiento y optimización de CUDA.

### Modelo

| Observación actual                         | Experimento controlado                                    |
| ------------------------------------------ | --------------------------------------------------------- |
| Patches de `7 x 7` generan solo 16 tokens. | Comparar patches de 7 y 4 manteniendo lo demás constante. |
| Un solo bloque limita la profundidad.      | Probar 2 bloques y medir accuracy, tiempo y memoria.      |
| MLP con expansión 2x.                      | Comparar `mlp_hidden_dim=64` contra `128`.                |
| Sin dropout.                               | Añadirlo solo si se observa sobreajuste.                  |

Reducir el patch de 7 a 4 incrementa los patches de 16 a 49. La matriz de
atención pasa de `17 x 17` a `50 x 50`, por lo que el coste no crece linealmente.

### Entrenamiento

| Limitación                    | Consecuencia                                                        |
| ----------------------------- | ------------------------------------------------------------------- |
| SGD sin momentum              | Puede converger más lentamente que AdamW.                           |
| Learning rate fijo            | No existe warmup ni reducción gradual.                              |
| Sin semilla configurable      | Las ejecuciones no son reproducibles.                               |
| Evaluación de test cada época | Test puede influir indirectamente en decisiones de hiperparámetros. |
| Sin validation set            | No existe una separación correcta para seleccionar configuración.   |
| Sin checkpoints               | No se puede reanudar ni conservar el mejor modelo.                  |

## Contribute

Si quieres usar este repo con fines de aprendizaje, necesitas estudiar la teoria de Transformer y Vision Transformer.
Este repo fue realizado con fines educativos para el curso de intelgencia artifical asi que siente libre de explorarlo y analizarlo.
Para comprender el código antes de modificarlo:

1. Leer [`src/examples/train_vit.cpp`](src/examples/train_vit.cpp) para conocer la configuración.
2. Leer [`includes/models/ViT.hpp`](includes/models/ViT.hpp) para ver el ensamblaje.
3. Seguir `PatchEmbeddingLayer`, CLS y positional embedding.
4. Estudiar las dos ecuaciones de `TransformerBlock`.
5. Descomponer `MultiHeadAttentionLayer` usando sus shapes.
6. Revisar `Network.cpp` y `Training.cpp` para entender el backward global.
7. Entrar a los kernels CUDA solo después de dominar las operaciones matemáticas.
