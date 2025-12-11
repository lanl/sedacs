BML_PATH="$(spack location -i bml)/lib/"
gfortran -c -fPIC -shared -I$BML_PATH/include -L$BML_PATH/lib -lbml -lbml_fortran prg_graph_mod.F90
gfortran -c -fPIC -shared gpmdcov_neighbor.F90 
gfortran -c -fPIC -shared sedacs_part_lib.F90 
gfortran -fPIC -shared -I$BML_PATH/include -L$BML_PATH/lib -lbml -lbml_fortran -o sedacs_part_lib.so gpmdcov_neighbor.o sedacs_part_lib.o prg_graph_mod.o 
