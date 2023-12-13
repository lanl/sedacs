#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <random>
#include <ctime>

#include "tcore_hp_emulator.cuh"

// Device function for splitting a single into two halves
__device__
void split_single(const float x, half &hi, half &lo)
{
    hi = __float2half(x);
    float y = (x - __half2float(hi));
    lo = __float2half(y * 1024.0);
}

template <typename T>
__global__
void array_split_single(const float *AF, T *AH1, T *AH2, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        half hi;
        half lo;

        split_single(AF[i], hi, lo);

        AH1[i] = hi;
        AH2[i] = lo;
    }
}
void tcoretools::tcoreSPGemmSymm (cublasHandle_t &handle
                                 ,const unsigned N
                                 ,const float* A
                                 ,half*  Ah
                                 ,half*  Al
                                 ,float* B1
                                 ,float* B2
                                 ,float* B
                                 ,cudaStream_t cuStrm) {
    // Setup kernel launch
    unsigned MAX_THREADS = 512;
    unsigned BLOCKS = int(ceil(float(N*N)/float(MAX_THREADS)));
    unsigned THREADS = MAX_THREADS;

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(A, Ah, Al, N*N);

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    //cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha = 1.0;
    float beta = 0.0;

    // Compute gemm for high
    CUBLAS_CHECK_ERR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                                  &alpha,
                                  Ah, CUDA_R_16F, N,
                                  Ah, CUDA_R_16F, N,
                                  &beta, B1, CUDA_R_32F, N, 
                                  CUBLAS_COMPUTE_32F, CUBLAS_DEFAULT_MATH));

    // Compute gemm for low
    CUBLAS_CHECK_ERR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                                  &alpha,
                                  Ah, CUDA_R_16F, N,
                                  Al, CUDA_R_16F, N,
                                  &beta, B2, CUDA_R_32F, N, 
                                  CUBLAS_COMPUTE_32F, CUBLAS_DEFAULT_MATH));

    alpha = 1.0;
    beta = 1.0;
    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 N, N,
                                 &alpha,
                                 B2, N,
                                 &beta,
                                 B2, N,
                                 B, N));

    beta = powf(2,-10);
    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &alpha,
                                 B1, N,
                                 &beta,
                                 B, N,
                                 B, N));
};

void tcoretools::tcoreSPGemmSymm1(cublasHandle_t &handle
                                 ,const unsigned N
                                 ,const float* A
                                 ,const float* B
                                 ,half*  Ah
                                 ,half*  Al
                                 ,half*  Bh
                                 ,half*  Bl
                                 ,float* C1
                                 ,float* C2
                                 ,float* C
                                 ,cudaStream_t cuStrm) {
    // Setup kernel launch
    unsigned MAX_THREADS = 1024;
    unsigned BLOCKS = ceil(N*N/float(MAX_THREADS));
    unsigned THREADS = MAX_THREADS;

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(A, Ah, Al, N*N);

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(B, Bh, Bl, N*N);
    
    // Set the math mode to allow cuBLAS to use Tensor Cores:
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha (1.0f);
    float beta  (0.0f);
    float gamma = powf(2,-10);

    // Compute gemm for high
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Ah, CUDA_R_16F, N,
                              Bh, CUDA_R_16F, N,
                              &beta, C1, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Compute gemms for low
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Ah, CUDA_R_16F, N,
                              Bl, CUDA_R_16F, N,
                              &beta, C2, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Al, CUDA_R_16F, N,
                              Bh, CUDA_R_16F, N,
                              &alpha, C2, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // add the high gemm and low gemm together
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &alpha,
                             C1, N,
                             &gamma,
                             C2, N,
                             C2, N);

    // compute C + C^T 
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             N, N,
                             &alpha,
                             C2, N,
                             &alpha,
                             C2, N,
                             C, N);
    
};

