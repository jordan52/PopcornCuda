cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(PopcornCuda LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


add_executable(PopcornCuda popcorn_cuda.cu)
target_link_libraries(PopcornCuda CUDA::cudart)
target_include_directories(PopcornCuda PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )                        
set_target_properties(PopcornCuda PROPERTIES CUDA_ARCHITECTURES "86")
