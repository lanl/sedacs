#!/bin/bash

# Make sure all the paths are correct


MY_PATH=$(pwd)
export CC=${CC:=gcc}
export FC=${FC:=gfortran}
export CXX=${CXX:=g++}
export MAGMA_ROOT=${MAGMA_ROOT:="${HOME}/magma"}
export BLAS_VENDOR=${BLAS_VENDOR:=OpenBLAS}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_CUDA=${BML_CUDA:=no}
export BML_MAGMA=${BML_MAGMA:=no}
export BML_CUSOLVER=${BML_CUSOLVER:=no}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/bml/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-g -O0 -lm -fPIC"}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-lm -fPIC "}

cd bml; ./build.sh configure; cd build; make -j; make install cd $MY_PATH
