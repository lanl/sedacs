import numpy as np
import ctypes 

def funct(in1,in2,N):

    nocc=ctypes.c_int(2)
    array_type = ctypes.c_float * N  # equiv. to C double[3] type
    arr1 = array_type(*in1)        # equiv. to double arr[3] = {...} instance
    arr2 = array_type(*in2)        # equiv. to double arr[3] = {...} instance
    lib.main(arr1,arr2,nocc)  # pointer to array passed to function and modified
    return list(arr2)        # extract Python floats from ctypes-wrapped array


def berga(ham,dm,N):

    z=funct(ham,dm,N)

    return z


if(__name__ == '__main__'):

    # local path .so file
    pwd="/vast/home/finkeljo/ctypes/simplest/generalcodes/py-to-cpp/"
    lib = ctypes.CDLL(str(pwd)+"libpymodule.so")
    lib.berga.argtypes = ctypes.POINTER(ctypes.c_float),

    #init arrays
    N=16384
    ham=np.ones(N)
    dm=np.zeros(N)

    #call C++
    z=berga(ham,dm,N)
    print(z)
