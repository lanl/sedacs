#/bin/bash
# Make sure all the paths are correct

rm -r build
rm -r install

make clean

module load nvhpc-hpcx-cuda12/24.7

MY_PATH=$(pwd)

export CXX=nvc++
export GPU_ARCH=${GPU_ARCH:=cc90}  # use cc70 for V100, cc80 for A100 and cc90 for H100
export CXX_FLAGS=${CXX_FLAGS:=" -O3 -cuda -gpu=${GPU_ARCH} -acc=gpu -Minfo=accel"}
echo ${CXX}

make
