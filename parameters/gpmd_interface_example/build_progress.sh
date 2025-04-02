#!/bin/bash

# Make sure all the paths are correct


rm -r build
rm -r install

# Set METIS and BML Library locations
METIS_LIB="$HOME/metis-5.1.0/build/Linux-x86_64/libmetis"
METIS_INC="$HOME/metis-5.1.0/build/Linux-x86_64/include"

MY_PATH=`pwd`

BML_LIB="$MY_PATH/bml/install"

cd qmd-progress

export CC=${CC:=gcc}
export FC=${FC:=gfortran}
#export FC=${FC:=/usr/bin/mpif90}
export CXX=${CXX:=g++}
export PROGRESS_OPENMP=${PROGRESS_OPENMP:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/qmd-progress/install"}
export PROGRESS_GRAPHLIB=${PROGRESS_GRAPHLIB:=no}
export PROGRESS_TESTING=${PROGRESS_TESTING:=yes}
export PROGRESS_MPI=${PROGRESS_MPI:=no}
export PROGRESS_BENCHMARKS=${PROGRESS_BENCHMARKS:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export PROGRESS_EXAMPLES=${PROGRESS_EXAMPLES:=no}
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:=$BML_LIB}
export BML_PREFIX_PATH=${BML_PREFIX_PATH:=$BML_LIB}
export EXTRA_FCFLAGS=${EXTRA_FCFLAGS:="-g -O0 -lm -fPIC"}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-lm -fPIC"}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:=yes}
./build.sh configure ; cd build ; make install


