cmake_minimum_required(VERSION 3.20)
project(adi_port LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # -O3 -DNDEBUG
endif()

find_package(Threads REQUIRED)

add_executable(adi_port src/main.cpp src/io.cpp src/kernels.cpp src/solver.cpp)
# -ffp-contract=off: no fused multiply-add, so the arithmetic matches NumPy's and is
# identical for every thread count.
target_compile_options(adi_port PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
target_link_libraries(adi_port PRIVATE Threads::Threads)
