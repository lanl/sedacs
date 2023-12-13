#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

make clean

MY_PATH=$(pwd)

export CXX=nvc++
export GPU_ARCH=${GPU_ARCH:=sm_80}
export CXX_FLAGS=${CXX_FLAGS:=" -O3 -cuda -gpu=${GPU_ARCH}"}
echo ${CXX}

make

#pushd build
#make -j8
#make install
#popd
