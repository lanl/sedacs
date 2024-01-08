## To build

Be sure to load the nvhpc module in your cluster environment, or have the nvhpc compilers installed, specifically nvc++. This compiler is needed to build 
the shared objects required by the ctypes python module. Also load openmpi. The makefile assumes the following dependencies

Dependencies:
nvc++, compiler 
cuda, gpu library  
cuBlas, gpu math library  
cuSolver, gpu math library  
openmpi, mpi communication library 
cudanvhpc, gpu library
cudart, run-time cuda library


The makefile assumes several env variables:
NVHPC_ROOT - root directory of NVHPC 
MPI_ROOT - root directory of mpi 


To build simply execute the example build script, type './example_build' in the command line from this directory. 

The makefile will build the 'libnvda.so' shared object file that can be used by python.

