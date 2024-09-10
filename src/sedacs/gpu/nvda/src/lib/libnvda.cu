#include <iostream>
#include <stdio.h>
#include <lib.h>
#include <structs.h>
#include <diag.cuh>
#include <dnnsp2.cuh>
#include <pscheby.cuh>
#include <error_check.cuh>

void dm_dnnsp2(double *ham, double *dm, int norb, int nocc, void *handle)
{

    // printf("DNN SP2\n");

    precision_t u = fp16_fp32;

    refine_t r = yes;

    dnnsp2(ham, dm, norb, nocc, u, r, handle);
}

void dm_mlsp2(double *ham, double *dm, int norb, int nocc)
{

    printf("ML SP2\n");

    precision_t u = fp16_fp32;

    refine_t r = yes;

    // dnnsp2(ham, dm, norb, nocc, u, r);
}

void dm_pscheby(double *ham, double *dm, int norb, int nocc, double kbt)
{

    double bndfil = 0.666666;

    precision_t u = fp64;

    refine_t r = yes;

    int K = 32;
    int M = 32;

    pscheby(ham, dm, K, M, norb, nocc, kbt);
}

void involap(double *overlap, double *guess, double *factor, int norb)
{

    // precision_t u = fp64;

    // refine_t r = yes;

    // invOlapFactorize(ham, dm, kbt, bndfil, u, norb, nocc);
}

void dm_diag(double *ham, double *dm, double kbt, int norb, int nocc)
{

    double bndfil = 0.666666;

    precision_t u = fp64;

    refine_t r = yes;

    diagonalize(ham, dm, kbt, bndfil, u, norb, nocc);
}

/*
    Wrap cudaMalloc with python
*/
void *dev_alloc(size_t size)
{

    double *devptr;
    cudaMalloc(&devptr, size);

    std::cout << devptr << std::endl;

    return (void *)devptr;
}

/*
    Wrap cudaMemcpy with python
*/
void memcpyDtoH(void *dest, void *source, size_t size)
{

    cudaMemcpy(dest, source, size, cudaMemcpyDeviceToHost);

    std::cout << "Memcpy" << std::endl;
}

/*
    Wrap cudaMemcpy with python
*/
void memcpyHtoD(void *dest, void *source, size_t size)
{

    cudaMemcpy(dest, source, size, cudaMemcpyHostToDevice);

    std::cout << "Memcpy" << std::endl;
}

/*
    Wrap cudaFree with python
*/
void dev_free(void *devptr)
{

    std::cout << devptr << std::endl;
    cudaFree(devptr);
    std::cout << "Free" << std::endl;
}

/*
    Initalize cublas handle with python
*/
void *cublasInit()
{

    cublasHandle_t handle;
    cublasCreate(&handle);
    // Set math mode
    CUBLAS_CHECK_ERR(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    std::cout << "cublas handle intialized" << std::endl;

    return (void *)handle;
}
