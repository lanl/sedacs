#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

make clean

MY_PATH=$(pwd)

export CXX=nvc++
export GPU_ARCH=${GPU_ARCH:=sm_80}  # use sm_70 for V100, sm_80 for A100 and sm_90 for H100
export CXX_FLAGS=${CXX_FLAGS:=" -O3 -cuda -gpu=${GPU_ARCH}"}
echo ${CXX}

make

#pushd build
#make -j8
#make install
#popd
