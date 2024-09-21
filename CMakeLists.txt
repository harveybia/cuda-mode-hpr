cmake_minimum_required(VERSION 3.10)
project(cuda_mode_hpr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
# find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    # ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add executable
add_executable(cuda_mode_hpr
    include/hidden_point_removal_impl.hpp
    include/hidden_point_removal.hpp
    src/entrypoint.cpp
)

# Link libraries
target_link_libraries(cuda_mode_hpr
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    # ${CUDA_LIBRARIES}
    Eigen3::Eigen
)

# Set CUDA architectures (adjust as needed)
set_target_properties(cuda_mode_hpr PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80"
)

# Enable CUDA separate compilation
set_property(TARGET cuda_mode_hpr PROPERTY CUDA_SEPARABLE_COMPILATION ON)