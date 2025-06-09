#!/bin/bash

# set env vars
export NVHPC_ROOT=/projects/shared/spack/opt/spack/linux-zen4/nvhpc-25.5-2z35ztggfsmvkpb74a2rouofaymnkyj6/Linux_x86_64/25.5/
export LD_LIBRARY_PATH=$NVHPC_ROOT/math_libs/lib64:$NVHPC_ROOT/cuda/lib64:$LD_LIBRARY_PATH
export MPI_ROOT=/projects/shared/spack/opt/spack/linux-zen4/openmpi-5.0.7-6dehamwfocnigimmbozkxhujq4b3epqf  #/projects/shared/spack/opt/spack/linux-ubuntu22.04-zen4/gcc-12.3.0/openmpi-4.1.6-iqs52vvok2k7s6nb7vh3rxqedsfinihz

# Make sure all the paths are correct
rm -r build
rm -r install

make clean

MY_PATH=$(pwd)

export CXX=nvc++                     # c++ compiler, needs to be able to compile with fpic
export GPU_ARCH=${GPU_ARCH:="89"}  # use 70 for V100, 80 for A100 and 90 for H100
export CXX_FLAGS=${CXX_FLAGS:="-O3"} # add compilation flags
export EXTRA_CXX_FLAGS=${EXTRA_CXX_FLAGS:="--diag_suppress=declared_but_not_referenced --diag_suppress=cuda_compile --diag_suppress=bad_macro_redef"}

echo "Your CXX compiler is: " ${CXX}

make
