#!/usr/bin/env python3

import numpy as np
import time

# to call cuda/hip
from ctypes import *
import ctypes

## gpuLib API call to neighborlist build
# This interface function will accept two numpy arrays along with three integers
# and call a gpuLib function and pass the arrays' C-pointers along with the integers.
# @param in1 Array you want to read from. 
# @param in2 Array you want to write to. 
# @param K Integer K, converted to a C-int type
# @param M Integer M, converted to a C-int type
# @param N Integer N, converted to a C-int type
# @return arr2 Numpy array you wrote to
#
def nlist(x,y,z,nlist,num_atoms,lib):
    size = 32*num_atoms
    array_type1 = ctypes.c_double*num_atoms     
    array_type2 = ctypes.c_int*size 
    x_c = array_type1(*x)                       
    y_c = array_type1(*y)                       
    z_c = array_type1(*z)                       
    nlist_c = array_type2(*nlist)               
    rcut = ctypes.c_float(2.0)
    nats = ctypes.c_int(num_atoms)
    dev = ctypes.c_int(0)
    print("dev=",dev) 

    tic = time.perf_counter()
    
    lib.nlist(x_c,     \
              y_c,     \
              z_c,     \
              nlist_c, \
              rcut,nats,0)
                 
    toc = time.perf_counter()
    print(f"Time = {toc - tic:0.4f} seconds")
    return list(nlist)                          



def dmDiag(ham,dm,matSize,nocc,lib):

    ## convert to flattened array
    matSize_sq = matSize*matSize
    ham_temp = np.zeros((matSize_sq,1))
    dm_temp = np.zeros((matSize_sq,1))

    for i in range(0,matSize):
        for j in range(0,matSize):
            ham_temp[i*matSize+j] = ham[i,j]
    
    ## time call
    tic = time.perf_counter()
    array_type1 = ctypes.c_double*matSize_sq

    ## copies the data to C data structures
    ham_c = array_type1(*ham_temp)                       
    dm_c = array_type1(*dm_temp)               
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)

    # end timer
    toc = time.perf_counter()
    print(f"Time to convert types = {toc - tic:0.4f} seconds")

    ## time call
    tic = time.perf_counter()
   
    # set C function arg types
    lib.dm_diag.argtypes = [POINTER(c_double), POINTER(c_double), c_double, c_int, c_int]
    
    kbt = 0.1;
    ## call diag from .so lib
    lib.dm_diag(ham_c,       \
                dm_c,        \
                kbt,                \
                matSize_c,          \
                nocc_c)   
     
    # end timer
    toc = time.perf_counter()

    for i in range(0,matSize):
        for j in range(0,matSize):
            dm[i,j] = dm_c[i*+matSize + j]


    print(f"Time from gpuLibInterface = {toc - tic:0.4f} seconds")
    return list(dm)                          


"""

## gpuLib API call to denisty matrix solver using diagonalization.
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with two integers, matSize and nocc. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using the DNN-SP2 method. For use with T=0 density
# matrix calculations.
# @param ham Hamiltonian matrix.
# @param dm Density matrix. 
# @param matSize Matrix sizes.
# @param nocc Occupation number.
# @return dm Density matrix that was constructed.
#
def dmDiag(ham,dm,matSize,nocc,arch,lib):

    ## convert to C data types
    
    ## time call
    tic = time.perf_counter()
    matSize_sq = matSize*matSize
    array_type1 = ctypes.c_double*matSize_sq
    ## copies the data
    ham_c = array_type1(*ham)                       
    dm_c = array_type1(*dm)               
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)

    # end timer
    toc = time.perf_counter()
    print(f"Time to convert types = {toc - tic:0.4f} seconds")

    ## time call
    tic = time.perf_counter()

    ## call diag from .so lib
    lib.dm_diag(ham_c,       \
                dm_c,        \
                matSize_c,   \
                nocc_c)   
     
    # end timer
    toc = time.perf_counter()
    print(f"Time from gpuLibInterface = {toc - tic:0.4f} seconds")
    return list(dm)                          

"""

## gpuLib API call to DNN-SP2 denisty matrix solver.
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with two integers, matSize and nocc. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using the DNN-SP2 method. For use with T=0 density
# matrix calculations.
# @param ham Hamiltonian matrix.
# @param dm Density matrix. 
# @param matSize Matrix sizes.
# @param nocc Occupation number.
# @return dm Density matrix that was constructed.
#
def dmDNNSP2(ham,dm,matSize,nocc,lib):

    ## convert to C data types
    array_type1 = ctypes.c_double*matSize
    ham_c = array_type1(*ham)                       
    dm_c = array_type1(*dm)               
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)

    ## time call
    tic = time.perf_counter()
  
    ## call dnn-sp2 from .so lib
    lib.dm_dnnsp2(ham_c,      \
                  dm_c,       \
                  matSize_c,  \
                  nocc_c)   
     
    # end timer
    toc = time.perf_counter()
    print(f"Time = {toc - tic:0.4f} seconds")
    return list(dm)                          


## gpuLib API call to Chebyshev denisty matrix solver.
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with two integers, matSize and expOrder. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using a fast Chebyshev expansion of order expOrder. 
# @param ham Hamiltonian matrix.
# @param dm Density matrix. 
# @param matSize Matrix sizes.
# @param expOrder Expansion order (largest poly. degree).
# @return dm Density matrix
#
def dmCheby(ham,dm,matSize,N,lib):

    ## convert to C data types
    array_type1 = ctypes.c_double*matSize
    ham_c = array_type1(*ham)                       
    dm_c = array_type1(*dm)               
    matSize_c = ctypes.c_int(matSize)
    expOrder_c = ctypes.c_int(expOrder)

    ## time call
    tic = time.perf_counter()
  
    ## call cheby from .so lib
    lib.dm_cheby(ham_c,      \
                 dm_c,       \
                 matSize_c,  \
                 expOrder_c)   
    # end timer
    toc = time.perf_counter()
    print(f"Time = {toc - tic:0.4f} seconds")
    return list(dm)                          




## Test gpuLib API call 
# This interface function will accept two numpy arrays along with three integers
# and call a gpuLib function and pass the arrays' C-pointers along with the integers.
# @param in1 Array you want to read from. 
# @param in2 Array you want to write to. 
# @param K Integer K, converted to a C-int type
# @param M Integer M, converted to a C-int type
# @param N Integer N, converted to a C-int type
# @return arr2 Numpy array you wrote to
#

def test_interface(in1,in2,K,M,N,arch):
    C_K=ctypes.c_int(K)
    C_M=ctypes.c_int(M)
    C_N=ctypes.c_int(N)
    array_type = ctypes.c_double*N              # equiv. to C double[N] type
    arr1 = array_type(*in1)                     # equiv. to double arr1[N] = {...} instance
    arr2 = array_type(*in2)                     # equiv. to double arr2[N] = {...} instance
    libnvda.test(arr1,arr2,C_K,C_M,600)         # pointer to array passed to function and modified
    return list(arr2)                           # extract Python floats from ctypes-wrapped array





