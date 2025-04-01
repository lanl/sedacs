#!/bin/bash

# set env vars
export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3
export LD_LIBRARY_PATH=$NVHPC_ROOT/math_libs/lib64:$NVHPC_ROOT/cuda/lib64:$LD_LIBRARY_PATH

export OMPI_ROOT=/projects/shared/spack/opt/spack/linux-ubuntu22.04-zen4/gcc-12.3.0/openmpi-4.1.6-iqs52vvok2k7s6nb7vh3rxqedsfinihz  #/projects/shared/spack/opt/spack/linux-ubuntu22.04-zen4/nvhpc-24.3/openmpi-5.0.3-jw7heth5kztakv3khq3nqoyiecskyatk

# Make sure all the paths are correct
rm -r build
rm -r install

make clean

MY_PATH=$(pwd)

export CXX=nvc++ #${NVHPC_ROOT}/compilers/bin/nvc++
export GPU_ARCH=${GPU_ARCH:=cc89}  # use cc70 for V100, cc80 for A100 and cc90 for H100
export CXX_FLAGS=${CXX_FLAGS:=" -O3 -cuda -gpu=${GPU_ARCH} -acc=gpu -Minfo=accel"}
echo "Your CXX compiler is: " ${CXX}

make
