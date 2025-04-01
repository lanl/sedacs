#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// device function for splitting a float into two halves
__device__ void split_single(const float x, half &hi, half &lo)
{
    hi = __float2half(x);
    float y = (x - __half2float(hi));
    lo = __float2half(y * 1024);
}

// global function for splitting a float matrix into two float halves
template <typename T>
__global__ void array_split_single(const float *AF, half *AH1, half *AH2, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        half hi;
        half lo;

        split_single(AF[i], hi, lo);

        AH1[i] = hi;
        AH2[i] = lo;
    }
}

void tcoreSPGemmSymm(cublasHandle_t handle,
                     const unsigned N,
                     const float *A,
                     half *Ah,
                     half *Al,
                     float *B1,
                     float *B2,
                     float *B,
		     cudaStream_t *stream)
{
    // Setup kernel launch
    unsigned num_thds = 512;
    unsigned num_blks = int(ceil(float(N * N) / float(num_thds)));

    // Split the floats into the high and low parts
    array_split_single<half><<<num_blks, num_thds>>>(A, Ah, Al, N * N);

    float alpha = 1.0;
    float beta = 0.0;

    // Compute gemmEx for high, set to stream[0]
    cublasSetStream(handle,stream[0]);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                 &alpha,
                 Ah, CUDA_R_16F, N,
                 Ah, CUDA_R_16F, N,
                 &beta, B1, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    // Compute gemmEx for low, set to stream[1]
    cublasSetStream(handle,stream[1]);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                 &alpha,
                 Ah, CUDA_R_16F, N,
                 Al, CUDA_R_16F, N,
                 &beta, B2, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    cudaStreamSynchronize(stream[0]); cudaStreamSynchronize(stream[1]);

    alpha = 1.0;
    beta = 1.0;

    // reset to stream[0]
    cublasSetStream(handle,stream[0]);
    cublasSgeam(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                N, N,
                &alpha,
                B2, N,
                &beta,
                B2, N,
                B, N);

    // undo prior scaling of 2^10
    beta = powf(2, -10);
    cublasSgeam(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N,
                &alpha,
                B1, N,
                &beta,
                B, N,
                B, N);
};

void tcoreSPGemmSymm1(cublasHandle_t handle
                     ,const unsigned N
                     ,const float* A
                     ,const float* B
                     ,half*  Ah
                     ,half*  Al
                     ,half*  Bh
                     ,half*  Bl
                     ,float* C1
                     ,float* C2
                     ,float* C)
{
    // Setup kernel launch
    unsigned MAX_THREADS = 1024;
    unsigned BLOCKS = ceil(N*N/float(MAX_THREADS));
    unsigned THREADS = MAX_THREADS;

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(A, Ah, Al, N*N);

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(B, Bh, Bl, N*N);

    float alpha (1.0f);
    float beta  (0.0f);
    float gamma = powf(2,-10);

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    

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
