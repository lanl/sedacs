#!/bin/bash

module load rocm/6.1.2
module load openmpi

# Make sure all the paths are correct
rm -r build
rm -r install

make clean

MY_PATH=$(pwd)

export CXX=hipcc
export GPU_ARCH=${GPU_ARCH:=cc80}  # use cc70 for V100, cc80 for A100 and cc90 for H100
export CXX_FLAGS=${CXX_FLAGS:=" -O3"}
echo "Your CXX compiler is: " ${CXX}

make
