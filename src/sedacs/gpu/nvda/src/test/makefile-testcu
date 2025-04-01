# -fpic means Position Independent Code. It is necessary when compiling shared libraries
# -shared is used for creating a shared library

libpymodule.so: cu_code.cu
	nvc++ -I${NVHPC_ROOT} -fpic -Wall -shared -O3 -gpu=cc80 cu_code.cu -o libpymodule.so -lcudart -lcuda -L${NVHPC_ROOT}/math_libs/lib64/ -lcublas
