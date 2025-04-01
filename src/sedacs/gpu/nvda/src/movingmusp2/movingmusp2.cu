#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <typeinfo>
#include <cmath>
#include <vector>
#include <structs.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <tcore_hp_emulator.cuh>
#include <linalg_tools.cuh>
#include <error_check.cuh>
#include "nvToolsExt.h"


/*
Inputs
------
    d_ham:       device pointer to hamiltonian
    d_dm:        device pointer to density matrix
    N:           matrix size
    Nocc:        occupation number
    precision:   which precison to use, fp16/32 or fp32
    refinement:  use refinement or not, yes or no
*/


void movingmusp2(double *d_ham,
                 double *d_dm,
                 int N,
                 double mu,
                 precision_t precision,
                 refine_t refinement,
                 void *Handle)
{


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int Stopp = 0;
    int iter = 0;
    std::vector<float> Idemp_Error;
    float one_f = 1.0, zero_f = 0.0;

    // Cublas streams
    int num_streams = 2;
    cudaStream_t stream[num_streams];

    // Cublas Handle
    nvtxRangePushA("build cublas handle");
    cublasHandle_t handle;
    CUBLAS_CHECK_ERR(cublasCreate(&handle));
    nvtxRangePop();
    
    nvtxRangePushA("declare memory");
    // Declare Memory
    double *d_TrD0, *TrD0; //*d_ham, *d_dm, *h_dm;

    float *d_S0, *d_S02, *d_TrS0, *d_TrS02, *TrS0, *TrS02, *d_S,
        *d_Sig, *d_Id, *sbuf1, *sbuf2, *Sig;

    half *hbuf1, *hbuf2;

    int *v_sgn;
    nvtxRangePop();

    // Allocate some host memory
    nvtxRangePushA("initialize memory");
    v_sgn = (int *)malloc(500 * sizeof(int));
    TrS0 = (float *)malloc(sizeof(float));
    TrS02 = (float *)malloc(sizeof(float));
    Sig = (float *)malloc(sizeof(float));
    TrD0 = (double *)malloc(sizeof(double));

    // Allocate device memory
    CUDA_CHECK_ERR(cudaMalloc(&d_S, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S0, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S02, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_Id, N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_Sig, sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrS0, sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrS02, sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrD0, sizeof(double)));

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
    setToIdentityMatrix<<<numblks, numthds>>>(d_Id, N);

    // cast d_ham from double to float
    doubleToFloat<<<numblks, numthds>>>(d_ham, d_S0, N);
    CUDA_CHECK_ERR(cudaMemcpy(sbuf1, d_S0, N * N * sizeof(float), cudaMemcpyDeviceToDevice));

    nvtxRangePushA("Affine transform");

    // Estimate sprectral bounds of H-mu*I
    float h1, hN;

    cudaStream_t streamm;
    CUDA_CHECK_ERR(cudaStreamCreate(&streamm));
    gershgorin_v2<float>(N, d_S0, &h1, &hN, streamm);

    // zeroth-order term
    float a = (hN / (hN - h1));
    float b = (-1 / (hN - h1));

    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &a,
                                 d_Id, N,
                                 &b,
                                 d_S0, N,
                                 d_S0, N));

    nvtxRangePop();
    mu = (double(hN)-mu)/double(hN-h1);
    double mu_starting = mu;
    printf("linear transformed mu = %f\n",mu);

    // compute and copy initial traces
    GPUSTrace(N, d_S0, d_TrS0);
    CUDA_CHECK_ERR(cudaMemcpy(TrS0, d_TrS0, sizeof(float), cudaMemcpyDeviceToHost));
    double mu2=0.;

    nvtxRangePushA("Main loop");
    while (Stopp == 0)
    {

        nvtxRangePushA("TC matmul");
        if (precision == fp32){
        

            CUBLAS_CHECK_ERR(cublasSgemm(handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, N, N,
                                         &one_f,
                                         d_S0, N,
                                         d_S0, N,
                                         &zero_f,
                                         d_S02, N));
        }
        else if (precision == fp16_fp32)
        {
            tcoreSPGemmSymm(handle,
                            N,
                            d_S0,
                            hbuf1, hbuf2,
                            sbuf1, sbuf2,
                            d_S02, &streamm);
        };
        mu2 = mu*mu;
	nvtxRangePop();
	
	if (abs(mu2-mu_starting) < (2*mu-mu2 - mu_starting)){
            Sig[0] =  1;
	}
	else{
            Sig[0] = -1;
        }

        nvtxRangePushA("Compute trace");

        // trace of S0^2

	float trace = 0.0;
        #pragma acc parallel loop deviceptr(d_S02) reduction(+ : trace)
        for (int i = 0; i < N; i++)
        {
            trace += d_S02[i * N + i];
        }
        TrS02[0] = double(trace);
        nvtxRangePop();

        // S0 idempotency error
        Idemp_Error.push_back(TrS0[0] - TrS02[0]);
	printf("trace = %.7f with err = %.10f at iter = %d \n", TrS0[0], Idemp_Error[iter], iter);

        // convergence control on S0
        if (TrS0[0] - TrS02[0] <= 0 and iter > 2)
        {
            //std::cout << "S0 Idempotency error = " << Idemp_Error[iter] << std::endl;
            printf("XO converged at iteration = %d \n", iter);
            break;
        }
        else if (iter > 2 && v_sgn[iter - 1] != v_sgn[iter - 2] && Idemp_Error[iter] >= 4.5 * Idemp_Error[iter - 2] * Idemp_Error[iter - 2])
        {
            //std::cout << "S0 Idempotency error = " << Idemp_Error[iter] << std::endl;
            printf("XO converged at iteration = %d \n", iter);
            break;
        };
        
        // Compute Sigma (which is determind by S0)
        nvtxRangePushA("Compute weights");

	a = Sig[0];
        b = 1.0 - Sig[0];

        // Compute S0_{n+1} = W_n*S0_n^2 + B_n = W_n*S0_n^2 + (1-W_n)S0_n
        CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N,
                                     &a,
                                     d_S02, N,
                                     &b,
                                     d_S0, N,
                                     d_S0, N));

        nvtxRangePop();
        mu = mu2*a + b*mu;
        //printf("mu = %f\n", mu);

	// Update traces
        TrS0[0] = Sig[0] * TrS02[0] + (1 - Sig[0]) * TrS0[0];
     

        // Update sign vector
        v_sgn[iter] = int(Sig[0]);

        iter += 1;
    }
    nvtxRangePop();
    // Free buffers
    CUDA_CHECK_ERR(cudaFree(sbuf1));
    CUDA_CHECK_ERR(cudaFree(sbuf2));
    CUDA_CHECK_ERR(cudaFree(hbuf1));
    CUDA_CHECK_ERR(cudaFree(hbuf2));

    nvtxRangePushA("Refinement");
    // allocate memory for density matrices
    double *d_T0;
    CUDA_CHECK_ERR(cudaMalloc(&d_T0, N * N * sizeof(double)));

    // refinement step
    if (refinement == yes)
    {
        std::cout << "Doing refinement..." << std::endl;

        floatToDouble<<<numblks, numthds>>>(d_S0, d_dm, N);

	// change dm approximation to double-prec
        //floatToDouble<<<numblks, numthds>>>(d_S0, d_T0, N);

        // do the refinement
        //doRefinement(d_T0, d_dm, N, Nocc, handle);
    }
    else
    {

        // change dm approximation to double-prec
        floatToDouble<<<numblks, numthds>>>(d_S0, d_dm, N);
    };
    nvtxRangePop();

    nvtxRangePushA("cudaFree");
    // Free device memory thats no longer needed
    CUDA_CHECK_ERR(cudaFree(d_S0));
    CUDA_CHECK_ERR(cudaFree(d_S02));
    CUDA_CHECK_ERR(cudaFree(d_Sig));
    CUDA_CHECK_ERR(cudaFree(d_TrS0));
    CUDA_CHECK_ERR(cudaFree(d_TrS02));
    CUDA_CHECK_ERR(cudaFree(d_T0));
    CUDA_CHECK_ERR(cudaFree(d_TrD0));
    CUDA_CHECK_ERR(cudaFree(d_Id));
    nvtxRangePop();

    nvtxRangePushA("free");
    // deallocate host memory
    free(v_sgn);
    free(TrD0);
    free(TrS0);
    free(TrS02);
    free(Sig);
    nvtxRangePop();

    nvtxRangePushA("Handle destroy");
    // Destroy handle
    CUBLAS_CHECK_ERR(cublasDestroy(handle));
    nvtxRangePop();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "time = " << milliseconds / 1000.0 << std::endl;
}
