"""
gpuLibInterface.py
====================================
python interface to GPU/AI-hardware library written in C++/CUDA

"""

import ctypes
import time

import numpy as np
#from juliacall import Main as jl
from numpy.ctypeslib import ndpointer

__all__ = ["dev_alloc", "host_alloc_pinned", "get_device", "set_device", \
           "memcpyasyncHtoD", "memcpyasyncDtoH","memcpyHtoD", "memcpyDtoH", \
           "set_stream", "cublasInit", "dev_free", "dmDiag", "dmMLSP2", \
           "dmMovingMuSP2", "dmGoldenSP2", "dmDNNSP2", "dmDNNPRT", "dmCheby"]


def dev_alloc(size, lib):
    """
    This function wraps cudaMalloc to allocate 
    device memory.

    Parameters
    ----------
    size : int
        Number of bytes to allocate on device
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    C pointer
        Void pointer to allocated device memory

    """

    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.dev_alloc.argtypes = [ctypes.c_size_t]
    lib.dev_alloc.restype = ctypes.POINTER(ctypes.c_void_p)

    ptr = lib.dev_alloc(size_c)
    return ptr

def host_alloc_pinned(size, lib):
    """
    This function wraps cudaMallocHost to allocate 
    pinned host memory.

    Parameters
    ----------
    size : int
        Number of bytes to allocate on device
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    C pointer
        Void pointer to allocated pinned host memory

    """
    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.host_alloc_pinned.argtypes = [ctypes.c_size_t]
    lib.host_alloc_pinned.restype = ctypes.POINTER(ctypes.c_void_p)

    ptr = lib.host_alloc_pinned(size_c)
    return ptr

def get_device(lib):
    """
    This function wraps cudaGetDevice and returns
    id of the device.

    Parameters
    ----------
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    int
        Device id number

    """

    ## set C function arg types
    lib.set_device.restypes = [ctypes.c_int]

    device = lib.get_device()

    return device

def set_device(device, lib):
    """
    This function wraps cudaSetDevice to set
    which device to use. Nothing returned.

    Parameters
    ----------
    size : int
        Device id number to use
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    NULL

    """

    ## copies scalar data to C data structures
    device_c = ctypes.c_int(device)

    ## set C function arg types
    lib.set_device.argtypes = [ctypes.c_int]

    lib.set_device(device_c)

def memcpyasyncHtoD(dest_ptr, source_ptr, size, lib):
    """
    This function wraps cudaMemcpyAsync to copy
    data from host to device asynchronously.

    Parameters
    ----------
    dest_ptr : void C pointer
        Destination device pointer 
    source_ptr : numpy double pointer
        Source host pointer
    size : int
        Number of bytes to copy
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    NULL

    """
    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.memcpyasyncHtoD.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ctypes.c_size_t,
    ]
    
    lib.memcpyasyncHtoD(dest_ptr, source_ptr, size_c)

def memcpyasyncDtoH(dest_ptr, source_ptr, size, lib):
    """
    This function wraps cudaMemcpyAsync to copy
    data from device to host asynchronously.

    Parameters
    ----------
    dest_ptr : numpy double pointer
        Destination host pointer 
    source_ptr : void C pointer
        Source device pointer
    size : int
        Number of bytes to copy
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    NULL

    """
    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.memcpyasyncDtoH.argtypes = [
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]

    lib.memcpyasyncDtoH(dest_ptr, source_ptr, size_c)


def memcpyHtoD(dest_ptr, source_ptr, size, lib):
    """
    This function wraps cudaMemcpy to copy
    data from host to device.

    Parameters
    ----------
    dest_ptr : void C pointer
        Destination device pointer 
    source_ptr : numpy double pointer
        Source host pointer
    size : int
        Number of bytes to copy
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    NULL

    """
    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.memcpyHtoD.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ctypes.c_size_t,
    ]

    lib.memcpyHtoD(dest_ptr, source_ptr, size_c)

def memcpyDtoH(dest_ptr, source_ptr, size, lib):
    """
    This function wraps cudaMemcpy to copy
    data from device to host.

    Parameters
    ----------
    dest_ptr   : C pointer
        Destination host pointer 
    source_ptr : void C pointer
        Source device pointer
    size       : int
        Number of bytes to copy
    lib        : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    NULL

    """
    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.memcpyDtoH.argtypes = [
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]

    lib.memcpyDtoH(dest_ptr, source_ptr, size_c)

def memcpyHtoH(dest_ptr, source_ptr, size, lib):
    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.memcpyHtoH.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ctypes.c_size_t,
    ]

    lib.memcpyHtoH(dest_ptr, source_ptr, size_c)

def set_stream(lib):
    """
    This function wraps cudaStreamCreate to 
    set the cuda stream.

    Parameters
    ----------
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    C pointer
        Void pointer to device stream

    """
    ## set C function return types
    lib.set_stream.restype = ctypes.POINTER(ctypes.c_void_p)

    stream_ptr = lib.set_stream()

    return stream_ptr


def cublasInit(lib):
    """
    This function wraps cublasHandleCreate to 
    initialize the cublas library.

    Parameters
    ----------
    lib  : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    C pointer
        Void pointer to cublas handle

    """

    ## set C function arg types
    lib.cublasInit.restype = ctypes.POINTER(ctypes.c_void_p)

    ## time call
    ptr = lib.cublasInit()

    return ptr

def dev_free(devptr, lib):
    """
    This function wraps cudaFree to free 
    previously allocated memory on device.

    Parameters
    ----------
    devptr : C pointer
        Device pointer to memory to free 
    lib    : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    NULL

    """
    lib.dev_free(devptr)


## gpuLib API call to construction of DM using diagonalization..
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with two integers, matSize and nocc. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using diagonalization.
#
# @param ham Hamiltonian matrix.
# @param dm Density matrix..
# @param matSize Matrix sizes.
# @param nocc Occupation of elec orbitals.
# @param kbt Electronic temperature.
# @return dm Desnity matrix.
#
def dmDiag(ham, dm, matSize, nocc, kbt, lib):
    """
    This function constructs the density matrix
    on the device using double precision 
    divide-and-conquer diagonalization.

    Parameters
    ----------
    ham : void C pointer
        Device pointer to Hamiltonian matrix
    dm : void C pointer
        Device pointer to density matrix
    matSize : int
        Size of Hamiltonian/density matrices
    nocc : int
        Number of occupied orbitals
    kbt : double
        Value of k_b*T
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    double
        Time to compute dm 

    """

    # copies scalar data to C data structures
    kbt_c = ctypes.c_double(kbt)
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)

    lib.dm_diag.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
    ]

    # time call
    tic = time.perf_counter()
    lib.dm_diag(ham, dm, kbt_c, matSize_c, nocc_c)
    toc = time.perf_counter()
    timer = toc - tic

    return timer


## gpuLib API call to ML-SP2 denisty matrix solver.
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with two integers, matSize and nocc. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using the ML-SP2 method. For use with finite T
# matrix calculations.
# @param ham Hamiltonian matrix.
# @param dm Density matrix.
# @param matSize Matrix sizes.
# @param nocc Occupation number.
# @return dm Density matrix that was constructed.
#
def dmMLSP2(ham, dm, matSize, nocc, lib):
    ## copies scalar data to C data structures
    matSize_c = ctypes.c_int(matSize)

    ## run Julia code to parametrize SP2
    '''jl.seval("using Pkg")
    jl.seval('Pkg.activate("./GeneralizedSP2")')
    jl.seval("""using GeneralizedSP2, LinearAlgebra
           β = 400
           μ = 0.5
           branches = determine_branches(μ, 16)
            𝐱 = sample_by_pdf(bell_distribution(μ, β), μ, (0, 1))
            𝐲 = forward_pass(branches, 𝐱)
           θ_fermi, θ_entropy = fit_model(𝐱, μ, β, 16)
            model = reshape(θ_fermi, 4, :)
             display(model)
             savemodel("model.npy", model)
           """)
    model = np.ascontiguousarray(np.load("model.npy"), dtype=np.float64)
    print(model)
    nlayers = model.shape[1]
    ## set C function arg types
    lib.dm_mlsp2.argtypes = [
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
    ]'''
    tic = time.perf_counter()
    #lib.dm_mlsp2(model, ham, dm, nlayers, matSize_c)
    toc = time.perf_counter()
    timer = toc - tic
    return timer


## gpuLib API call to moving mu SP2 denisty matrix solver.
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with integer matSize and double mu. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using the moving mu SP2 method. For use with T=0 density
# matrix calculations.
# @param ham Hamiltonian matrix.
# @param dm Density matrix.
# @param matSize Matrix sizes.
# @param mu Chemical potential.
# @return dm Density matrix that was constructed.
#
def dmMovingMuSP2(ham, dm, matSize, mu, handle, lib):
    """
    This function constructs the density matrix
    on the device using the moving-mu SP2 method
    with tensor core acceleration. In this version 
    of SP2, only mu is known a prioi, not nocc.

    Parameters
    ----------
    ham : void C pointer
        Device pointer to Hamiltonian matrix
    dm : void C pointer
        Device pointer to density matrix
    matSize : int
        Size of Hamiltonian/density matrices
    mu : double
        Chemical potential
    handle : void C pointer
        cublas Handle
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    double
        Time to compute dm 

    """

    ## copies scalar data to C data structures
    matSize_c = ctypes.c_int(matSize)
    mu_c = ctypes.c_double(mu)

    ## set C function arg types
    lib.dm_movingmusp2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    # time call
    tic = time.perf_counter()
    lib.dm_movingmusp2(ham, dm, matSize_c, mu_c, handle)
    toc = time.perf_counter()
    timer = toc - tic

    return timer


## gpuLib API call to Golden SP2 denisty matrix solver.
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with integer matSize and double mu. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using the Golden-SP2 method. For use with T=0 density
# matrix calculations.
# @param ham Hamiltonian matrix.
# @param dm Density matrix.
# @param matSize Matrix sizes.
# @param mu Chemical potential.
# @return dm Density matrix that was constructed.
#
def dmGoldenSP2(ham, dm, matSize, mu, handle, lib):
    """
    This function constructs the density matrix
    on the device using the golden SP2 method
    tensor core acceleration. In this version of SP2, 
    only mu is known a prioi, not nocc.

    Parameters
    ----------
    ham : void C pointer
        Device pointer to Hamiltonian matrix
    dm : void C pointer
        Device pointer to density matrix
    matSize : int
        Size of Hamiltonian/density matrices
    mu : double
        Chemical potential
    handle : void C pointer
        cublas Handle
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    double
        Time to compute dm 

    """

    ## copies scalar data to C data structures
    matSize_c = ctypes.c_int(matSize)
    mu_c = ctypes.c_double(mu)

    ## set C function arg types
    lib.dm_goldensp2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    ## time call
    tic = time.perf_counter()
    lib.dm_goldensp2(ham, dm, matSize_c, mu_c, handle)
    toc = time.perf_counter()
    timer = toc - tic
    return timer


## gpuLib API call to inverse overlap factorization algorithm.
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with two integers, matSize and nocc. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using the DNN-SP2 method. For use with T=0 density
# matrix calculations.
# @param overlap Orbital overlap matrix.
# @param guess Inital guess for inverse factor matrix.
# @param factor Pointer to inverse overlap matrix factor.
# @param matSize Matrix sizes.
# @return factor Computed factor of inverse overlap  matrix, Z^TZ = S^{-1/2}.
#
def invOverlapFactor(overlap, guess, factor, matSize, lib):
    """
    This function constructs an inverse overlap
    matrix factorization, Z, such that ZSZ^T=I
    using mixed precision and Tensor cores. 
    Method described in JCTC XXXXX.

    Parameters
    ----------
    overlap : void C pointer
        Device pointer to overlap matrix
    guess : void C pointer
        Device pointer to initial guess
    factor : void C pointer
        Device pointer to matrix factor Z
    matSize : int
        Size of Hamiltonian/density matrices
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    double
        Time to compute the factor Z

    """

    ## copies scalar data to C data structures
    matSize_c = ctypes.c_int(matSize)

    ## time call
    tic = time.perf_counter()

    ## call involap from .so lib
    lib.involap.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int
    ]

    # end timer
    toc = time.perf_counter()
    print(f"Time = {toc - tic:0.4f} seconds")
    return list(factor)


## gpuLib API call to DNN-SP2 denisty matrix solver.
# This interface function will accept two numpy arrays, the hamiltonian, and the density matrix
# along with two integers, matSize and nocc. Function will build the density matrix from
# the Hamiltonian, which has size matSize, using the DNN-SP2 method. For use with T=0 density
# matrix calculations.
# @param ham Hamiltonian matrix.
# @param dm Density matrix.
# @param id Identity matrix.
# @param matSize Matrix sizes.
# @param nocc Occupation number.
# @return dm Density matrix that was constructed.
#
def dmDNNSP2(dev_list, matSize, nocc, handle, stream, lib):
    """
    This function constructs the density matrix
    on the device using the the SP2 method with
    tensor core acceleration. 

    Parameters
    ----------
    ham : void C pointer
        Device pointer to Hamiltonian matrix
    dm : void C pointer
        Device pointer to density matrix
    matSize : int
        Size of Hamiltonian/density matrices
    nocc : int
        Occupation number
    handle : void C pointer
        cublas Handle
    stream : void C pointer
        Pointer to device stream
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    double
        Time to compute dm 

    """

    ## copies scalar data to C data structures
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)

    ## set C function arg types
    lib.dm_dnnsp2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p)
    ]

    ## list of pre-allocated dev vars
    ham    = dev_list[0]
    dm     = dev_list[1]
    t02    = dev_list[2]
    iden   = dev_list[3]
    s0     = dev_list[4]
    s02    = dev_list[5]
    sbuf1  = dev_list[6]
    sbuf2  = dev_list[7]
    hbuf1  = dev_list[8]
    hbuf2  = dev_list[9]

    ## time call
    tic = time.perf_counter()
    lib.dm_dnnsp2(ham, 
                  dm, 
                  t02, 
                  iden, 
                  s0, 
                  s02, 
                  sbuf1, 
                  sbuf2, 
                  hbuf1, 
                  hbuf2, 
                  matSize_c, 
                  nocc_c, 
                  handle, 
                  stream)
    toc = time.perf_counter()
    timer = toc - tic

    return timer


## gpuLib API call to DNN-PRT denisty matrix linear response solver.
# This interface function will accept five device pointers, the Hamiltonian, the first order
# Hamiltonian perturbation, the density matrix, the density matrix linear response and an
# observable matrix. In addition, it will take two integers, the matrix size and occupation
# number, matSize and nocc. Function will build the density matrix and its response from
# the Hamiltonian and a perturbation, which has size matSize, using the DNN-PRT method.
# For use with T=0 density matrix calculations.
# @param ham Hamiltonian matrix.
# @param prt First order perturbation matrix.
# @param dm Density matrix.
# @param rsp Linear response in density matrix.
# @param matSize Matrix size.
# @param nocc Occupation number.
# @return dm Density matrix that was constructed.
# @return dm Response in density matrix that was constructed.
#
def dmDNNPRT(ham, prt, dm, rsp, matSize, nocc, handle, lib):
    """
    This function constructs the density matrix
    and its first order response on the device 
    using the the SP2 method and its extension to
    density matrix perturbation theory using 
    tensor core acceleration. 

    Parameters
    ----------
    ham : void C pointer
        Device pointer to Hamiltonian matrix
    prt : void C pointer
        Device pointer to perturbation matrix
    dm : void C pointer
        Device pointer to density matrix
    dm : void C pointer
        Device pointer to response matrix
    matSize : int
        Size of Hamiltonian/density matrices
    nocc : int
        Occupation number
    handle : void C pointer
        cublas Handle
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    double
        Time to compute dm 

    """

    ## copies scalar data to C data structures
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)

    ## set C function arg types
    lib.dm_dnnprt.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    ## time call
    tic = time.perf_counter()
    lib.dm_dnnprt(ham, prt, dm, rsp, matSize_c, nocc_c, handle)
    toc = time.perf_counter()

    timer = toc - tic
    return timer


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
def dmCheby(ham, dm, matSize, nocc, kbt, lib):
    """
    This function constructs the density matrix
    at finite (non-zero) electronic temperature 
    using the Patterson-Stockmeyer form of a 
    Chebyshev expansion. Method described in
    JCP XXXX. 

    Parameters
    ----------
    ham : void C pointer
        Device pointer to Hamiltonian matrix
    dm : void C pointer
        Device pointer to density matrix
    matSize : int
        Size of Hamiltonian/density matrices
    nocc : int
        Occupation number
    kbt : double
        Value of k_b*T
    lib : ctypes library
        Imported gpu library .so file   
    
    Returns 
    -------
    double
        Time to compute dm 

    """
    ## convert to C data types
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)
    kbt_c = ctypes.c_double(kbt)

    ## set C function arg types
    lib.dm_pscheby.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
    ]

    ## time call
    tic = time.perf_counter()
    lib.dm_pscheby(ham, dm, matSize_c, nocc_c, kbt_c)
    toc = time.perf_counter()
    
    return timer

