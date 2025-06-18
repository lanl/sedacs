cmake_minimum_required(VERSION 3.18)
project(libnvda LANGUAGES CXX)

# ------------------------------------------------------
# Set NVIDIA compiler and flags from environment
# ------------------------------------------------------
#set(CMAKE_CXX_COMPILER $ENV{CXX})
#set(GPU_ARCH "cc80" CACHE STRING "Target GPU architecture")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gpu=${GPU_ARCH}")


# ------------------------------------------------------
# Configure MPI using find_package with custom path
# ------------------------------------------------------
find_package(MPI REQUIRED)

message(STATUS "MPI Include Path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")

# ------------------------------------------------------
# Include directories
# ------------------------------------------------------
include_directories(
    include
    ${MPI_INCLUDE_PATH}
    $ENV{NVHPC_ROOT}/compilers/include
    $ENV{NVHPC_ROOT}/math_libs/include
    $ENV{NVHPC_ROOT}/cuda/include
    $ENV{LAPACK_ROOT}/include
)

# ------------------------------------------------------
# Library search paths (if needed)
# ------------------------------------------------------
#link_directories(
#    $ENV{NVHPC_ROOT}/compilers/lib64
#    $ENV{NVHPC_ROOT}/math_libs/lib64
#    $ENV{NVHPC_ROOT}/cuda/lib64
#    $ENV{LAPACK_ROOT}/lib
#)

# ------------------------------------------------------
# Source files (treat .cu as C++ with nvc++)
# ------------------------------------------------------
file(GLOB_RECURSE ALL_SRCS
    src/diag/*.cu
    src/dnnsp2/*.cu
    src/dnnprt/*.cu
    src/goldensp2/*.cu
    src/movingmusp2/*.cu
    src/fastcheby/*.cu
    src/mlsp2/*.cu
    src/lib/*.cu
)

# Explicitly mark .cu as C++ files
foreach(file ${ALL_SRCS})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CXX)
endforeach()

# ------------------------------------------------------
# Shared library target
# ------------------------------------------------------
add_library(libnvda SHARED ${ALL_SRCS})
set_target_properties(libnvda PROPERTIES OUTPUT_NAME "nvda")

target_link_libraries(libnvda
    ${MPI_CXX_LIBRARIES}
    cudanvhpc
    cublas
    cusolver
    cudart
    nvToolsExt
    lapack
)

