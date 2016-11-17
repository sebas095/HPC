cmake_minimum_required(VERSION 3.0)

project(MultMatrix)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

cuda_add_library(mult_matrix_cuda mult_matrix_cuda.cu)

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
add_executable(mult_matrix mult_matrix_mpi.cpp)
target_link_libraries(mult_matrix ${MPI_CXX_LIBRARIES} mult_matrix_cuda)
