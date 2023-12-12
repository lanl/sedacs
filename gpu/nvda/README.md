## To build

Be sure to load the nvhpc module in your cluster environment, or have the nvhpc compilers installed, specifically nvc++. This compiler is needed to build 
the shared objects required by the ctypes python module. The makefile assumes the following dependencies

Dependencies:
nvc++, compiler
cuda, gpu library
cuBlas, gpu math library
cuSolver, gpu math library
openmpi, mpi communication library



make 
