cmake_minimum_required(VERSION 3.0)

project(MultMatrix)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Handle Warnings and C++11 Support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-Wno-deprecated-gpu-targets")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_library(mult_matrix_cuda mult_matrix_cuda.cu)

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
add_executable(mult_matrix mult_matrix_mpi.cpp)
target_link_libraries(mult_matrix ${MPI_CXX_LIBRARIES} mult_matrix_cuda)

