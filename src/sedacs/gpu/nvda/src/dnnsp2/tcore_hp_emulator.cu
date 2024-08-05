#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas.h>

// device function for splitting a float into two halves
__device__
void split_single(const float x, half &hi, half &lo)
{
    hi = __float2half(x);
    float y = (x - __half2float(hi));
    lo = __float2half(y * 1024);
}


// global function for splitting a float matrix into two float halves
template <typename T>
__global__
void array_split_single(const float *AF, half *AH1, half *AH2, const unsigned N)
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


void tcoreSPGemmSymm(cublasHandle_t  handle,
                     cudaStream_t *,
                     const unsigned N,
                     const float* A,
                     half* Ah,
                     half* Al,
                     float* B1,
                     float* B2,
                     float* B)
{
    // Setup kernel launch
    unsigned num_thds = 512;
    unsigned num_blks = int(ceil(float(N*N)/float(num_thds)));

    // Split the floats into the high and low parts
    array_split_single<half><<< num_blks, num_thds >>>(A, Ah, Al, N*N);

    float alpha = 1.0;
    float beta = 0.0;

    // Compute gemmEx for high
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                                  &alpha,
                                  Ah, CUDA_R_16F, N,
                                  Ah, CUDA_R_16F, N,
                                  &beta, B1, CUDA_R_32F, N, 
                                  CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    // Compute gemmEx for low
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                                  &alpha,
                                  Ah, CUDA_R_16F, N,
                                  Al, CUDA_R_16F, N,
                                  &beta, B2, CUDA_R_32F, N, 
                                  CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    alpha = 1.0;
    beta = 1.0;
    cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 N, N,
                                 &alpha,
                                 B2, N,
                                 &beta,
                                 B2, N,
                                 B, N);

    // undo prior scaling of 2^10
    beta = powf(2,-10);
    cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &alpha,
                                 B1, N,
                                 &beta,
                                 B, N,
                                 B, N);
};

