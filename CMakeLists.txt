cmake_minimum_required(VERSION 3.24)
project(neural_networks LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

set(CORE_SOURCES
    src/core/Tensor.cu
    src/core/ops/linalg.cu
    src/core/ops/activations.cu
    src/core/ops/conv_ops.cu
    src/core/ops/pool_ops.cu
    src/core/ops/attention_ops.cu
)

set(LAYER_SOURCES
    src/layers/DenseLayer.cpp
    src/layers/ConvLayer.cpp
    src/layers/MaxPool2DLayer.cpp
    src/layers/ReLULayer.cpp
    src/layers/SoftmaxLayer.cpp
    src/layers/FlattenLayer.cpp
    src/layers/PositionalEmbeddingLayer.cpp
    src/layers/PatchEmbeddingLayer.cpp
    src/layers/LayerNormLayer.cpp
    src/layers/MultiHeadAttentionLayer.cpp
    src/layers/GELULayer.cpp
    src/layers/TransformerBlock.cpp
    src/layers/CLSTokenLayer.cpp
    src/layers/ExtractCLSLayer.cpp
)

set(NETWORK_SOURCES
    src/network/Network.cpp
)

set(UTIL_SOURCES
    src/utils/Training.cpp
)

set(EXAMPLE_SOURCES
    src/examples/train_cnn.cpp
)

add_executable(neural_networks
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${NETWORK_SOURCES}
    ${UTIL_SOURCES}
    ${EXAMPLE_SOURCES}
    src/main.cpp
)

target_include_directories(neural_networks PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)

# Pruebas sueltas para verificar shapes/valores de capas nuevas (no van al PR final).
add_executable(test_patch_embedding
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${NETWORK_SOURCES}
    src/examples/test_patch_embedding.cpp
)

target_include_directories(test_patch_embedding PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)

add_executable(test_layer_norm
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${NETWORK_SOURCES}
    src/examples/test_layer_norm.cpp
)

target_include_directories(test_layer_norm PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)

add_executable(test_batched_matmul
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${NETWORK_SOURCES}
    src/examples/test_batched_matmul.cpp
)

target_include_directories(test_batched_matmul PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)

add_executable(test_attention
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${NETWORK_SOURCES}
    src/examples/test_attention.cpp
)

target_include_directories(test_attention PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)

add_executable(test_transformer_block
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${NETWORK_SOURCES}
    src/examples/test_transformer_block.cpp
)

target_include_directories(test_transformer_block PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)

add_executable(test_cls_token
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${NETWORK_SOURCES}
    src/examples/test_cls_token.cpp
)

target_include_directories(test_cls_token PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)
