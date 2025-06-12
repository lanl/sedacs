# clear the build dir
rm -rf build;
mkdir build;
cd build;

cmake .. \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_CXX_FLAGS="-O3 -cuda -gpu=cc89 -acc=gpu -Minfo=accel -fPIC \
                     --diag_suppress=bad_macro_redef \
                     --diag_suppress=cuda_compile \
                     --diag_suppress=declared_but_not_referenced " \
#-DNVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3 \
#-DOMPI_ROOT=/projects/shared/spack/opt/spack/linux-zen4/openmpi-5.0.7-6dehamwfocnigimmbozkxhujq4b3epqf \
#-DLAPACK_ROOT=/path/to/lapack \
#-DCMAKE_PREFIX_PATH="/projects/shared/spack/opt/spack/linux-zen4/openmpi-5.0.7-6dehamwfocnigimmbozkxhujq4b3epqf"


make -j 
