cmake_minimum_required(VERSION 3.1)
project(SGEMM-learning)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(CUDA 10 REQUIRED)
enable_language(CUDA)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

include_directories(.)

set(headers
    "SGEMM-my/db_gemm.cuh"
    "SGEMM-my/smem_gemm.cuh"
    "SGEMM-my/warp_op_gemm.cuh"
    )

set(sources
    "SGEMM-my/startup.cpp"
    "SGEMM-my/smem_gemm.cu"
    "SGEMM-my/warp_op_gemm.cu"
    "SGEMM-my/warp_op_noconflict.cu"
    "SGEMM-my/fixed_smem_gemm.cu"
    "SGEMM-my/db_gemm.cu"
    "SGEMM-my/db_gemm_exp.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})

target_link_libraries(${CMAKE_PROJECT_NAME})