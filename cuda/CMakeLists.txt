# -----------------------------------------------------------------------------
# Build for the CUDA port of the ADI time loop (executable: adi_cuda).
#
# Uses CUDA 12.8 by default; pass -DCMAKE_CUDA_COMPILER=... to choose another.
# Options (all OFF): ADI_EXACT, ADI_RECOMPUTE_FACTORS, ADI_DEBUG_SYNC.
# ADI_PREFETCH_DEPTH (default 8): elements the Thomas sweeps load ahead.
#
# Project: ADI finite-difference methods for American options under Heston
#          Math/CS 714, UW–Madison, Fall 2026
# Author:  Ganesh Arivoli <arivoli@wisc.edu>
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.24)

# These must be set before project(), which otherwise picks the nvcc on PATH and its default architecture.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda-12.8/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3070 Ti
endif()

project(adi_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # -O3 -DNDEBUG
endif()

option(ADI_EXACT "No FMA and Python's exact operations: outputs byte-identical to the fixtures" OFF)
option(ADI_RECOMPUTE_FACTORS "Recompute the Thomas factors every step instead of once at setup" OFF)
option(ADI_DEBUG_SYNC "Synchronize and check for errors after every kernel launch" OFF)
set(ADI_PREFETCH_DEPTH 8 CACHE STRING "Elements the Thomas sweeps load ahead of their arithmetic (1 to 32)")

# Each .cu file launches only its own kernels through plain host functions, so no separable compilation.
add_executable(adi_cuda main.cu host/io.cpp kernels/explicit.cu kernels/tridiag.cu solver/solver.cu
                        solver/schemes.cu)
target_include_directories(adi_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adi_cuda PRIVATE ${CMAKE_DL_LIBS})  # the header-only NVTX library loads its tool with dlopen
target_compile_options(adi_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-Wall,-Wextra,-ffp-contract=off>)

target_compile_definitions(adi_cuda PRIVATE ADI_PREFETCH_DEPTH=${ADI_PREFETCH_DEPTH})
if(ADI_EXACT)
  target_compile_definitions(adi_cuda PRIVATE ADI_EXACT=1)
  target_compile_options(adi_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
endif()
if(ADI_RECOMPUTE_FACTORS)
  target_compile_definitions(adi_cuda PRIVATE ADI_RECOMPUTE_FACTORS=1)
endif()
if(ADI_DEBUG_SYNC)
  target_compile_definitions(adi_cuda PRIVATE ADI_DEBUG_SYNC=1)
endif()

message(STATUS "adi_cuda: nvcc ${CMAKE_CUDA_COMPILER_VERSION}, sm_${CMAKE_CUDA_ARCHITECTURES}, ${CMAKE_BUILD_TYPE}, "
               "ADI_EXACT=${ADI_EXACT} ADI_RECOMPUTE_FACTORS=${ADI_RECOMPUTE_FACTORS} ADI_DEBUG_SYNC=${ADI_DEBUG_SYNC} "
               "ADI_PREFETCH_DEPTH=${ADI_PREFETCH_DEPTH}")
