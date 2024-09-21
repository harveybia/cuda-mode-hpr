cmake_minimum_required(VERSION 3.10)
project(cuda_mode_hpr LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)

# Include CUDA CMake utilities
include(FindCUDA)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# CUDA library
cuda_add_library(cuda_preprocess
    include/preprocess.cu
)

cuda_add_library(cuda_convex_hull
    include/cuda_convex_hull.cu
)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math)

# Set CUDA architectures (adjust as needed)
set_target_properties(cuda_convex_hull PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80"
)
set_target_properties(cuda_preprocess PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80"
)

# Main executable
cuda_add_executable(cuda_mode_hpr
    src/entrypoint.cpp
    include/hidden_point_removal_impl.hpp
    include/hidden_point_removal.hpp
    include/cuda_convex_hull.hpp
    include/preprocess.hpp
)

# Link libraries
target_link_libraries(cuda_mode_hpr
    cuda_convex_hull
    cuda_preprocess
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${CUDA_LIBRARIES}
    Eigen3::Eigen
)

# Enable CUDA separate compilation
set_property(TARGET cuda_mode_hpr PROPERTY CUDA_SEPARABLE_COMPILATION ON)