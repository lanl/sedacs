THIS_PATH=`pwd`
METIS_LIB="$HOME/metis-5.1.0/build/Linux-x86_64/libmetis"
METIS_INC="$HOME/metis-5.1.0/build/Linux-x86_64/include"
PROGRESS_LIB="$THIS_PATH/qmd-progress/install/"
PROGRESS_INC="$THIS_PATH/qmd-progress/install/include"
BML_LIB="$THIS_PATH/bml/install/"
BML_INC="$THIS_PATH/bml/install/include"
cd ./qmd-progress/examples/gpmdk/
rm -rf build
mkdir build
cd build

# cmake using specified flags:
# Update the -L flag in "-DEXTRA_FCFLAGS" to point to the location of metis and GKlib libraries
# Update the "-DCMAKE_PREFIX_PATH" locations to point to the "install" directories of bml and qmd-progress as well as the general location of metis and GKlib
#cmake  -DCMAKE_Fortran_COMPILER="gfortran" -DPROGRESS_MPI="no" \
#  -DEXTRA_FCFLAGS="-Wall -Wunused -fopenmp -lm -fPIC -g -O0 -fcheck=all --check=bounds -lbml -lbml_fortran -L$BML_LIB/lib64 -L$METIS_LIB -lprogress -L$PROGRESS_LIB/lib64 -lmetis -I$METIS_INC -I$PROGRESS_INC -I$BML_INC -lm -fPIC" \
#  -DLIB="yes" -DLATTE="no" -DLATTE_PREC="DOUBLE" -DLATTE_DIR="/home/cnegre/LATTE/" \
#  -DCMAKE_PREFIX_PATH="$PROGRESS_LIB;$PROGRESS_INC;$BML_LIB;$BML_INC;$METIS_LIB;$METIS_INC"  ../src/
#make

cmake  -DCMAKE_Fortran_COMPILER="gfortran" -DPROGRESS_MPI="no" \
  -DEXTRA_FCFLAGS="-Wall -Wunused -fopenmp -lm -fPIC -g -O0 -fcheck=all --check=bounds -lmetis -lm -fPIC" \
  -DLIB="yes" -DLATTE="no" -DLATTE_PREC="DOUBLE" -DLATTE_DIR="" \
  -DCMAKE_PREFIX_PATH="$PROGRESS_LIB;$PROGRESS_INC;$BML_LIB;$BML_INC;$METIS_LIB;$METIS_INC"  ../src/
make




