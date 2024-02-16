#gfortran proxy_a_mod.F90 proxy_a.F90 -o proxy_a -llapack -lblas
gfortran -shared -fPIC proxy_a_lib.F90 proxy_a_mod.F90 -llapack -lblas  -o proxya_fortran.so 
