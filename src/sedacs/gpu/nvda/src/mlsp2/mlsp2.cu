#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <structs.h>
#include <cusolverDn.h>
#include <tcore_hp_emulator.cuh>
#include <linalg_tools.cuh>
#include <mlsp2.cuh>
#include <error_check.cuh>
#include "nvToolsExt.h"

void mlsp2(double *model,
           double *GPU_hamiltonian,
           double *GPU_densityMatrix,
           int numLayers,
           int N,
           precision_t precision,
           refine_t refinement)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float one_f = 1.0, zero_f = 0.0, neg_one_f = -1.0;

    // Cublas Handle
    nvtxRangePushA("build cublas handle");
    cublasHandle_t handle;
    CUBLAS_CHECK_ERR(cublasCreate(&handle));
    nvtxRangePop();

    nvtxRangePushA("declare memory");
    // Declare Memory
    float *GPU_Si, *GPU_Si_squared, *GPU_identityMatrix, *sbuf1, *sbuf2, *GPU_accumulationMatrix;
    half *hbuf1, *hbuf2;

    nvtxRangePop();

    // Allocate some host memory
    nvtxRangePushA("initialize memory");

    // Allocate device memory
    CUDA_CHECK_ERR(cudaMalloc(&GPU_Si, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&GPU_Si_squared, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&GPU_identityMatrix, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&GPU_accumulationMatrix, N * N * sizeof(float)));
    // Initialize the accumulation matrix to zero
    CUDA_CHECK_ERR(cudaMemset(GPU_accumulationMatrix, 0, N * N * sizeof(float)));

    // Allocate Buffers
    CUDA_CHECK_ERR(cudaMalloc(&sbuf1, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&sbuf2, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&hbuf1, N * N * sizeof(half)));
    CUDA_CHECK_ERR(cudaMalloc(&hbuf2, N * N * sizeof(half)));
    nvtxRangePop();

    // Define blk,thd grid size
    int numthds = 512;
    int numblks = int(ceil(double(N * N) / double(numthds)));

    // Initialize Hamiltonian and identity

    // build Identity on dev
    setToIdentityMatrix<<<numblks, numthds>>>(GPU_identityMatrix, N);

    // cast GPU_hamiltonian from double to float
    doubleToFloat<<<numblks, numthds>>>(GPU_hamiltonian, GPU_Si, N);
    CUDA_CHECK_ERR(cudaMemcpy(sbuf1, GPU_Si, N * N * sizeof(float), cudaMemcpyDeviceToDevice));

    nvtxRangePushA("Affine transform");
    // Estimate sprectral bounds
    double e1, en;
    cudaStream_t streamm;
    CUDA_CHECK_ERR(cudaStreamCreate(&streamm));
    gershgorin_v2<double>(N, GPU_hamiltonian, &e1, &en, streamm);

    // zeroth-order term
    float a = float(-1 / (en - e1));
    float b = float(en / (en - e1));

    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &b,
                                 GPU_identityMatrix, N,
                                 &a,
                                 GPU_Si, N,
                                 GPU_Si, N));

    nvtxRangePop();

    nvtxRangePushA("Main loop");
    for (int iter = 0; iter < numLayers; ++iter)
    {
        nvtxRangePushA("TC matmul");
        if (precision == fp32)
        {
            CUBLAS_CHECK_ERR(cublasSgemm(handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, N, N,
                                         &one_f,
                                         GPU_Si, N,
                                         GPU_Si, N,
                                         &zero_f,
                                         GPU_Si_squared, N));
        }
        //else if (precision == fp16_fp32)
        //{
/*            tcoreSPGemmSymm(handle,
                            N,
                            GPU_Si,
                            hbuf1, hbuf2,
                            sbuf1, sbuf2,
                            GPU_Si_squared);*/
       // };
        nvtxRangePop();

        a = model[iter];
        b = model[numLayers + iter];
        float c = model[2 * numLayers + iter];
        float d = model[3 * numLayers + iter];

        CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N,
                                     &a,
                                     GPU_Si_squared, N,
                                     &b,
                                     GPU_Si, N,
                                     GPU_Si, N));
        CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N,
                                     &one_f, // Keep GPU_S₀ as-is (scaled by 1)
                                     GPU_Si, N,
                                     &c,
                                     GPU_identityMatrix, N,
                                     GPU_Si, N));

        // Accumulate d * GPU_S₀ into the accumulation matrix
        CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N,
                                     &d, // Scale factor for GPU_S₀
                                     GPU_Si, N,
                                     &one_f, // Accumulate into GPU_accumulationMatrix
                                     GPU_accumulationMatrix, N,
                                     GPU_accumulationMatrix, N));

        nvtxRangePop();
    }
    nvtxRangePop();

    // Subtract GPU_accumulationMatrix from GPU_identityMatrix and store in GPU_densityMatrix
    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &neg_one_f, // Scale factor for GPU_accumulationMatrix (-1)
                                 GPU_accumulationMatrix, N,
                                 &one_f, // Scale factor for GPU_identityMatrix (1)
                                 GPU_identityMatrix, N,
                                 GPU_accumulationMatrix, N)); // Store result in GPU_densityMatrix

    floatToDouble<<<numblks, numthds>>>(GPU_accumulationMatrix, GPU_densityMatrix, N);

    // Save GPU_densityMatrix to disk as plain text before cleaning up resources
    double *host_densityMatrix = (double *)malloc(N * N * sizeof(double));
    CUDA_CHECK_ERR(cudaMemcpy(host_densityMatrix, GPU_densityMatrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    FILE *file = fopen("density_matrix.txt", "w");
    if (file != NULL)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                fprintf(file, "%lf ", host_densityMatrix[i * N + j]);
            }
            fprintf(file, "\n"); // New line at the end of each row
        }
        fclose(file);
    }
    else
    {
        std::cerr << "Error: Could not open file for writing" << std::endl;
    }
    free(host_densityMatrix);

    nvtxRangePushA("Handle destroy");
    // Destroy CUBLAS handle
    CUBLAS_CHECK_ERR(cublasDestroy(handle));
    nvtxRangePop();

    nvtxRangePushA("cudaFree");
    // Free device memory
    CUDA_CHECK_ERR(cudaFree(GPU_Si));
    CUDA_CHECK_ERR(cudaFree(GPU_Si_squared));
    CUDA_CHECK_ERR(cudaFree(GPU_identityMatrix));
    CUDA_CHECK_ERR(cudaFree(GPU_accumulationMatrix));
    CUDA_CHECK_ERR(cudaFree(sbuf1));
    CUDA_CHECK_ERR(cudaFree(sbuf2));
    CUDA_CHECK_ERR(cudaFree(hbuf1));
    CUDA_CHECK_ERR(cudaFree(hbuf2));
    nvtxRangePop();

    // Record the stop event
    cudaEventRecord(stop);

    // Synchronize and measure elapsed time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "time = " << milliseconds / 1000.0 << " seconds" << std::endl;

    // Destroy CUDA events
    CUDA_CHECK_ERR(cudaEventDestroy(start));
    CUDA_CHECK_ERR(cudaEventDestroy(stop));
}
