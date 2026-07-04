cmake_minimum_required(VERSION 3.18)
project(cnn_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(test_cnn test_cnn.cpp)

target_include_directories(test_cnn PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_cnn ${OpenCV_LIBS})
