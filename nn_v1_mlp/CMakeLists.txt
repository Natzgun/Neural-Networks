cmake_minimum_required(VERSION 3.18)
project(neural_networks LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4060 compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(mlp
    main.cpp
    cuda_kernels.cu
)

target_sources(mlp PRIVATE
    # mlp.cpp
    cuda_ops.cuh
    DeviceMatrix.cuh
    ActivationFunction.hpp
    Layer.hpp
    MLP.hpp
    DatasetLoader.hpp
)

add_custom_command(
    TARGET mlp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/grayscale_train
            $<TARGET_FILE_DIR:mlp>/grayscale_train
)

add_custom_command(
    TARGET mlp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/grayscale_test
            $<TARGET_FILE_DIR:mlp>/grayscale_test
)

target_link_libraries(mlp ${OpenCV_LIBS})


add_executable(test_cnn
    cnn/test_cnn.cpp
)

target_include_directories(test_cnn PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set_target_properties(test_cnn PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_custom_command(
    TARGET test_cnn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/grayscale_train
            $<TARGET_FILE_DIR:test_cnn>/grayscale_train
)

add_custom_command(
    TARGET test_cnn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/grayscale_test
            $<TARGET_FILE_DIR:test_cnn>/grayscale_test
)

target_link_libraries(test_cnn ${OpenCV_LIBS})
