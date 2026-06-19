cmake_minimum_required(VERSION 3.18)
project(neural_networks LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4060 — compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenCV REQUIRED)

set(SOURCES
    src/main.cpp
    src/Tensor.cu
    src/nn_ops.cu
    src/cuda_kernels.cu
)

set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(neural_networks ${SOURCES})

target_include_directories(neural_networks PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(neural_networks PRIVATE
    ${OpenCV_LIBS}
)

add_custom_command(
    TARGET neural_networks POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/grayscale_train
            $<TARGET_FILE_DIR:neural_networks>/grayscale_train
)

add_custom_command(
    TARGET neural_networks POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/grayscale_test
            $<TARGET_FILE_DIR:neural_networks>/grayscale_test
)