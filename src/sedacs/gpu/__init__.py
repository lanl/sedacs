import ctypes
import time

import numpy as np
from juliacall import Main as jl
from numpy.ctypeslib import ndpointer

__all__ = ["invOverlapFactor", "dmDiag", "dmMLSP2", "dmDNNSP2", "dmDNNPRT", "dmCheby"]


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
    ## convert to C data types
    array_type1 = ctypes.c_double * matSize
    overlap_c = array_type1(*overlap)
    guess_c = array_type1(*guess)
    factor_c = array_type1(*factor)
    matSize_c = ctypes.c_int(matSize)

    ## time call
    tic = time.perf_counter()

    ## call involap from .so lib
    lib.involap(overlap_c, guess_c, factor_c, matSize_c)

    # end timer
    toc = time.perf_counter()
    print(f"Time = {toc - tic:0.4f} seconds")
    return list(factor)


def dev_alloc(size, lib):
    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.dev_alloc.argtypes = [ctypes.c_size_t]
    lib.dev_alloc.restype = ctypes.POINTER(ctypes.c_void_p)

    ## time call
    ptr = lib.dev_alloc(size_c)

    return ptr


def memcpyHtoD(dest_ptr, source_ptr, size, lib):
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
    ## copies scalar data to C data structures
    size_c = ctypes.c_size_t(size)

    ## set C function arg types
    lib.memcpyDtoH.argtypes = [
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]

    lib.memcpyDtoH(dest_ptr, source_ptr, size_c)


def cublasInit(lib):
    print("hello")
    ## set C function arg types
    lib.cublasInit.restype = ctypes.POINTER(ctypes.c_void_p)

    ## time call
    ptr = lib.cublasInit()

    return ptr


def dev_free(devptr, lib):
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
    ## copies scalar data to C data structures
    kbt_c = ctypes.c_double(kbt)
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)

    ## set C function arg types
    # lib.dm_diag.argtypes = [ndpointer(np.float64,flags='aligned, c_contiguous'), \
    #                        ndpointer(np.float64,flags='aligned, c_contiguous'), \
    #                        ctypes.c_double, ctypes.c_int, ctypes.c_int]

    lib.dm_diag.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
    ]
    ## time call
    tic = time.perf_counter()
    lib.dm_diag(ham, dm, kbt_c, matSize_c, nocc_c)
    toc = time.perf_counter()
    timer = toc - tic

    # print(f"Time for lib call = {toc - tic:0.4f} seconds")
    return timer
    # return list(dm),timer


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
    jl.seval("using Pkg")
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
    ]
    tic = time.perf_counter()
    lib.dm_mlsp2(model, ham, dm, nlayers, matSize_c)
    toc = time.perf_counter()
    timer = toc - tic
    return timer


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
def dmDNNSP2(ham, dm, matSize, nocc, handle, lib):
    ## copies scalar data to C data structures
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)

    ## set C function arg types
    lib.dm_dnnsp2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    # lib.dm_dnnsp2.argtypes = [ndpointer(np.float64,flags='aligned, c_contiguous'), \
    #                          ndpointer(np.float64,flags='aligned, c_contiguous'), \
    #                          ctypes.c_int, ctypes.c_int]

    ## time call
    tic = time.perf_counter()
    lib.dm_dnnsp2(ham, dm, matSize_c, nocc_c, handle)
    toc = time.perf_counter()

    # print(f"Time for lib call = {toc - tic:0.10f} seconds")
    timer = toc - tic
    return timer
    # return list(dm), timer


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
    ## convert to C data types
    matSize_c = ctypes.c_int(matSize)
    nocc_c = ctypes.c_int(nocc)
    kbt_c = ctypes.c_double(kbt)

    ## set C function arg types
    lib.dm_pscheby.argtypes = [
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ndpointer(np.float64, flags="aligned, c_contiguous"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
    ]
    ## time call
    tic = time.perf_counter()

    ## call cheby from .so lib
    lib.dm_pscheby(ham, dm, matSize_c, nocc_c, kbt_c)
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


def test_interface(in1, in2, K, M, N, arch):
    C_K = ctypes.c_int(K)
    C_M = ctypes.c_int(M)
    C_N = ctypes.c_int(N)
    array_type = ctypes.c_double * N  # equiv. to C double[N] type
    arr1 = array_type(*in1)  # equiv. to double arr1[N] = {...} instance
    arr2 = array_type(*in2)  # equiv. to double arr2[N] = {...} instance
    libnvda.test(arr1, arr2, C_K, C_M, 600)  # pointer to array passed to function and modified
    return list(arr2)  # extract Python floats from ctypes-wrapped array
