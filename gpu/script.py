import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct

# Import module
mymodule = npct.load_library('libpymodule', '.')

def call_cpp(arr_in):
    ''' Convenience function for converting the arguments from numpy 
        array pointers to ctypes arrays. '''

    c_floatp = ct.POINTER(ct.c_float)                  # ctypes float pointer
    c_uintp = ct.POINTER(ct.c_uint)                    # ctypes unsigned integer pointer 

    # Call function
    mymodule.main(arr_in.ctypes.data_as(c_floatp), # Cast numpy array to ctypes integer pointer
                  arr_out.ctypes.data_as(c_floatp), 
                  shape.ctypes.data_as(c_uintp))
                    
    return arr_out

# Generate some 2D numpy array
N = 16384;
arr_in = np.arange(N**2, dtype=np.float32).reshape(N, N)

# Allocate the output array in memory, and get the shape of the array
arr_out = np.zeros_like(arr_in)
shape = np.array(arr_in.shape, dtype=np.uint32)

# Call function
arr_out = call_cpp(arr_in)

print(arr_out[0][:])

