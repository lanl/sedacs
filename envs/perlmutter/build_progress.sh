#!/bin/bash

# Make sure all the paths are correct

MY_PATH=`pwd`

BML_LIB="$MY_PATH/bml/install"

cd qmd-progress

export CC=${CC:=cc}
export FC=${FC:=ftn}
export CXX=${CXX:=CC}
export PROGRESS_OPENMP=${PROGRESS_OPENMP:=yes}
export INSTALL_DIR="${MY_PATH}/qmd-progress/install"
export PROGRESS_GRAPHLIB=${PROGRESS_GRAPHLIB:=no}
export PROGRESS_TESTING=${PROGRESS_TESTING:=yes}
export PROGRESS_MPI=${PROGRESS_MPI:=no}
export PROGRESS_BENCHMARKS=${PROGRESS_BENCHMARKS:=no}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export PROGRESS_EXAMPLES=${PROGRESS_EXAMPLES:=no}
export PKG_CONFIG_PATH="$BML_LIB/lib/pkgconfig:$BML_LIB/lib64/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:=$BML_LIB}
export BML_PREFIX_PATH=${BML_PREFIX_PATH:=$BML_LIB}
export EXTRA_FCFLAGS=${EXTRA_FCFLAGS:="-g -O0 -lm  -fPIC"}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-lm -fPIC"}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:=yes}
./build.sh configure ; cd build ; make -j16; make install; cd $MY_PATH


