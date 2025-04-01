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
#include <cuda_profiler_api.h>

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


void dnnsp2(double *d_ham,
            double *d_dm,
            double *d_T02,
            float  *d_Id,
            float  *d_S0,
            float  *d_S02,
            float  *sbuf1,
            float  *sbuf2,
            void   *vbuf1,
            void   *vbuf2,
            int N,
            int Nocc,
            precision_t precision,
            refine_t refinement,
	    void *handle_,
	    cudaStream_t *stream)
{

    int dev;
    //unsigned long long stream_id;

    cudaGetDevice(&dev);
    std::cout << "current device = "  << dev << std::endl;
    std::cout << "stream = "  << stream << std::endl;
    
    std::cout << N << ", " << Nocc << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int Stopp = 0;
    int iter = 0;
    std::vector<float> Idemp_Error;
    float one_f = 1.0, zero_f = 0.0;

    // Cast void pointer to cublasHandle_t
    nvtxRangePushA("build cublas handle");
    cublasHandle_t *temp = (cublasHandle_t*) handle_; 
    cublasHandle_t handle = *temp; 

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    nvtxRangePop();
    
    nvtxRangePushA("declare memory");
    // Declare Memory
    float *TrS0, *TrS02, *Sig;
    half *hbuf1 = (half*) vbuf1;
    half *hbuf2 = (half*) vbuf2;

    int *v_sgn;
    nvtxRangePop();

    // Allocate some host memory
    nvtxRangePushA("initialize memory");
    v_sgn = (int *)   malloc(500 * sizeof(int));
    TrS0  = (float *) malloc(sizeof(float));
    TrS02 = (float *) malloc(sizeof(float));
    Sig   = (float *) malloc(sizeof(float));

    // Define blk,thd grid size
    int numthds = 512;
    int numblks = int(ceil(double(N * N) / double(numthds)));

    // build Identity on dev (stream 0)
    setToIdentityMatrix<<<numblks, numthds, 0, stream[0]>>>(d_Id, N);

    // cast d_ham from double to float (stream 1)
    doubleToFloat<<<numblks, numthds, 0, stream[1]>>>(d_ham, d_S0, N);
    CUDA_CHECK_ERR(cudaMemcpyAsync(sbuf1, d_S0, N * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Allocate Buffers (stream 3)
    //CUDA_CHECK_ERR(cudaMalloc(&hbuf1, N * N * sizeof(half)));
    //CUDA_CHECK_ERR(cudaMalloc(&hbuf2, N * N * sizeof(half)));

    nvtxRangePop();

    // define events
    cudaEvent_t event; cudaEventCreate(&event);

    nvtxRangePushA("Affine transform");
    // Estimate sprectral bounds
    double h1, hN;
    gershgorin_v2<double>(N, d_ham, &h1, &hN, stream[0]);

    // stream 0 wait for stream 1
    cudaEventRecord(event, stream[0]); cudaStreamWaitEvent(stream[1], event, 0); 

    // input layer to DNN-SP2

    // zeroth-order term
    float a = float(-1 / (hN - h1));
    float b = float(hN / (hN - h1));
    std::cout << "a=" << a << std::endl;

    //  Set stream
    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &b,
                                 d_Id, N,
                                 &a,
                                 d_S0, N,
                                 d_S0, N));

    //cudaEventRecord(event, stream[1]); cudaStreamWaitEvent(stream[0], event, 0); 
    //cudaEventRecord(event, stream[2]); cudaStreamWaitEvent(stream[0], event, 0); 
    //cudaEventRecord(event, stream[3]); cudaStreamWaitEvent(stream[0], event, 0); 
    //cudaEventRecord(event, stream[3]); cudaStreamWaitEvent(stream[1], event, 0); 
    
    nvtxRangePop();

    // compute and copy initial traces
    float trace = 0.0;

    //int streamId = 23;
    //acc_set_cuda_stream(streamId, stream[0]); 
    
    #pragma acc parallel loop deviceptr(d_S0) reduction(+ : trace) 
    for (int i = 0; i < N; i++)
    {
        trace += d_S0[i * N + i];
    }
    TrS0[0] = trace;
    std::cout << TrS0[0] << std::endl;
    nvtxRangePop();

    //cudaEventRecord(event, stream[0]);
    //cudaEventSynchronize(event);
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
                            d_S02, stream);
        };
        nvtxRangePop();

        nvtxRangePushA("Compute trace");

        // trace of S0^2
        trace = 0.0;

        #pragma acc parallel loop deviceptr(d_S02) reduction(+ : trace)
        for (int i = 0; i < N; i++)
        {
            trace += d_S02[i * N + i];
        }
        TrS02[0] = trace;
        nvtxRangePop();

        float temp = TrS0[0] - TrS02[0];
 
	// S0 idempotency error
        Idemp_Error.push_back(temp);

        // convergence control on S0
        if (temp <= 0 and iter > 2)
        {
            printf("XO converged at iteration = %d \n", iter);
            break;
        }
        else if (iter > 2 && v_sgn[iter - 1] != v_sgn[iter - 2] && Idemp_Error[iter] >= 4.5 * Idemp_Error[iter - 2] * Idemp_Error[iter - 2])
        {
            printf("XO converged at iteration = %d \n", iter);
            break;
        };

        // Compute Sigma (which is determind by S0)
        nvtxRangePushA("Compute sigma and weights");
   
        if (fabs(TrS02[0] - float(Nocc)) < fabs(2.0 * TrS0[0] - TrS02[0] - float(Nocc)))
        {
            Sig[0] = 1;
        }
        else
        {
            Sig[0] = -1;
        }

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

        // Update traces
        TrS0[0] = Sig[0] * TrS02[0] + (1 - Sig[0]) * TrS0[0];

	if (iter%1==0) std::cout << "trace = " << TrS0[0] << " at iter = " << iter << std::endl;

        // Update sign vector
        v_sgn[iter] = int(Sig[0]);

        iter += 1;
    }
    nvtxRangePop();

    nvtxRangePushA("Refinement");
    // refinement step
    if (refinement == yes)
    {
        std::cout << "Doing refinement..." << std::endl;
   
        // shallow copy	
	double *d_T0 = d_ham;

	// change dm approximation to double-prec
        floatToDouble<<<numblks, numthds>>>(d_S0, d_T0, N);

        // do the refinement
        doRefinement(d_T0, d_T02, d_dm, N, Nocc, stream[0], handle);
    }
    else
    {
        // change dm approximation to double-prec
        floatToDouble<<<numblks, numthds>>>(d_S0, d_dm, N);
    };
    nvtxRangePop();

    nvtxRangePushA("cudaFree");
    // Free device memory thats no longer needed
    //CUDA_CHECK_ERR(cudaFree(hbuf1));
    //CUDA_CHECK_ERR(cudaFree(hbuf2));
    nvtxRangePop();

    nvtxRangePushA("free");
    // deallocate host memory
    free(v_sgn);
    free(TrS0);
    free(TrS02);
    free(Sig);
    nvtxRangePop();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "time = " << milliseconds / 1000.0 << ", N = " << N << std::endl;

    //cudaProfilerStop();
}
