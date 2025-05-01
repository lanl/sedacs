# Remove the build directory and create a new one
rm -rf build
mkdir build
# Move into the build directory
cd build
METIS_LIB="$HOME/metis-5.1.0/build/Linux-x86_64/libmetis"
METIS_INC="$HOME/metis-5.1.0/build/Linux-x86_64/include"
PROGRESS_LIB="$HOME/qmd-progress/install/"
PROGRESS_INC="$HOME/qmd-progress/install/include"
BML_LIB="$HOME/bml/install/"
BML_INC="$HOME/bml/install/include"


cmake -DPROGRESS="yes" -DCMAKE_Fortran_COMPILER="gfortran" -DPROGRESS_MPI="no" \
  -DEXTRA_FCFLAGS="-Wall -Wunused -fopenmp -lm -fPIC -g -O0 -fcheck=all --check=bounds -lmetis -lm -fPIC" \
  -DLIB="yes" -DLATTE_PREC="DOUBLE" \
  -DCMAKE_PREFIX_PATH="$PROGRESS_LIB;$PROGRESS_INC;$BML_LIB;$BML_INC;$METIS_LIB;$METIS_INC"  ../
make

cp libproxya_fortran.so proxya_fortran.so 




