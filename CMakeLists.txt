cmake_minimum_required(VERSION 3.15)
project(neural_networks)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(mlp mlp.cpp)

add_custom_command(
    TARGET mlp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/grayscale_train
            $<TARGET_FILE_DIR:mlp>/grayscale_train
)

target_link_libraries(mlp ${OpenCV_LIBS})
