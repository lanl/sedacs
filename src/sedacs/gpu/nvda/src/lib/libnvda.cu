#include <iostream>
#include <vector>
#include <stdio.h>
#include <lib.h>
#include <structs.h>
#include <diag.cuh>
#include <dnnsp2.cuh>
#include <dnnprt.cuh>
#include <goldensp2.cuh>
#include <movingmusp2.cuh>
#include <pscheby.cuh>
#include <mlsp2.cuh>
#include <error_check.cuh>
#include <cuda_fp16.h>

void dm_dnnsp2(double *ham,
               double *dm,
               double *t02,
               float  *id,
               float  *s0,
               float  *s02,
               float  *sbuf1,
               float  *sbuf2,
               void   *vbuf1,
               void   *vbuf2,
               int norb, int nocc, void *handle, cudaStream_t *stream)
{

    precision_t u = fp16_fp32;
    refine_t r = yes;
    dnnsp2(ham, dm, t02, id, s0, s02, sbuf1, sbuf2, vbuf1, vbuf2, norb, nocc, u, r, handle, stream);
}

/*void dm_finitetsp2(double *ham,
                  double *dm,
                  int norb, double mu,
                  void *handle)
{
    precision_t u = fp16_fp32;
    refine_t r = no;
    finitetsp2(ham, dm, norb, mu, u, r, handle);
}
*/
void dm_goldensp2(double *ham,
                  double *dm,
                  int norb, double mu,
                  void *handle)
{
    precision_t u = fp16_fp32;
    refine_t r = no;
    goldensp2(ham, dm, norb, mu, u, r, handle);
}

void dm_movingmusp2(double *ham,
                    double *dm,
                    int norb, double mu,
                    void *handle)
{
    precision_t u = fp16_fp32;
    refine_t r = no;
    movingmusp2(ham, dm, norb, mu, u, r, handle);
}

void dm_dnnprt(double *ham, double *prt,
               double *dm, double *rsp,
               int norb, int nocc, void *handle)
{

    // precision_t u = fp16_fp32;
    // refine_t r = yes;

    dnnprt(ham, prt, dm, rsp, norb, nocc); //, handle);
}

void dm_mlsp2(double *model, double *ham, double *dm, int nlayers, int norb)
{
    precision_t u = fp16_fp32;
    refine_t r = yes;

    mlsp2(model, ham, dm, nlayers, norb, u, r);
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

     precision_t u = fp64;
     refine_t r = yes;

    // invOlapFactorize(ham, dm, kbt, bndfil, u, norb, nocc);
}

void dm_diag(double *ham, double *dm, double kbt, int norb, int nocc, double bndfil)
{

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
    CUDA_CHECK_ERR(cudaMalloc(&devptr, size));

    std::cout << devptr << std::endl;

    return (void *)devptr;
}

/*
    Wrap cudaSetDevice with python
*/
void set_device(int device)
{
    CUDA_CHECK_ERR(cudaSetDevice(device));

    std::cout << "Device set to " << device << std::endl;
}

/*
    Wrap cudaStreamCreate with python
*/
void *set_stream(void)
{
    cudaStream_t *stream = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 10);

    // create 10 cuda streams
    for (int i=0; i<10;i++){
	    CUDA_CHECK_ERR(cudaStreamCreate(&stream[i]));
    }
    
    std::cout << stream << std::endl;

    return (void*) stream;
}

/*
    Initalize cublas handle with python
*/
void *cublasInit()
{
    cublasHandle_t *handle = (cublasHandle_t*) malloc(sizeof(cublasHandle_t));
    CUBLAS_CHECK_ERR(cublasCreate(handle));
    
    // Set math mode
    CUBLAS_CHECK_ERR(cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH));
    std::cout << "cublas handle intialized" << std::endl;

    return (void *) handle;
}

/*
    Destroy cublas handle with python
*/
/*void cublasDestroy(cublasHandle_t *handle)
{
    CUBLAS_CHECK_ERR(cublasDestroy(*handle));
    
    // Set math mode
    std::cout << "cublas handle destroyed" << std::endl;
}
*/

/*
    Wrap cudaGetDevice with python
*/
int get_device()
{
    int device = 0;

    CUDA_CHECK_ERR(cudaGetDevice(&device));

    return device;
}


/*
    Wrap cudaMallocManaged with python
*/
void *dev_alloc_managed(size_t size)
{

    double *devptr;
    CUDA_CHECK_ERR(cudaMallocManaged(&devptr, size));

    std::cout << devptr << std::endl;

    return (void *)devptr;
}

/*
    Wrap cudaMallocHost with python
*/
void *host_alloc_pinned(size_t size)
{

    double *hostptr;
    CUDA_CHECK_ERR(cudaMallocHost(&hostptr, size));

    std::cout << hostptr << std::endl;

    return (void *)hostptr;
}

/*
    Wrap cudaMemcpy with python, Host to Host
*/
void memcpyHtoH(void *dest, void *source, size_t size)
{

    CUDA_CHECK_ERR(cudaMemcpy(dest, source, size, cudaMemcpyHostToHost));

    std::cout << "Memcpy H to H" << std::endl;
}

/*
    Wrap cudaMemcpy with python, Device to Host
*/
void memcpyDtoH(void *dest, void *source, size_t size)
{

    CUDA_CHECK_ERR(cudaMemcpy(dest, source, size, cudaMemcpyDeviceToHost));

    std::cout << "Memcpy D to H" << std::endl;
}

/*
    Wrap cudaMemcpy with python, Host to Device
*/
void memcpyHtoD(void *dest, void *source, size_t size)
{

    CUDA_CHECK_ERR(cudaMemcpy(dest, source, size, cudaMemcpyHostToDevice));

    std::cout << "Memcpy H to D" << std::endl;
}

/*
    Wrap cudaMemcpyAsync with python, Device to Host
*/
void memcpyasyncDtoH(void *dest, void *source, size_t size)
{

    CUDA_CHECK_ERR(cudaMemcpyAsync(dest, source, size, cudaMemcpyDeviceToHost));

    std::cout << "Memcpy Async" << std::endl;
}


/*
    Wrap cudaMemcpyAsync with python, Host to Device
*/
void memcpyasyncHtoD(void *dest, void *source, size_t size)
{

    CUDA_CHECK_ERR(cudaMemcpyAsync(dest, source, size, cudaMemcpyHostToDevice));

    std::cout << "Memcpy Async" << std::endl;
}


/*
    Wrap cudaFree with python
*/
void dev_free(void *devptr)
{

    std::cout << devptr << std::endl;
    CUDA_CHECK_ERR(cudaFree(devptr));
    std::cout << "Free" << std::endl;
}

