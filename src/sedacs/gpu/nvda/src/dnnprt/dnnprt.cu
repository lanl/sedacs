#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <error_check.cuh>
#include <cmath>
#include <vector>
#include <linalg_tools.cuh>
#include <tcore_hp_emulator.cuh>
#include "nvToolsExt.h"


/*
Inputs
------
    d_ham:   device pointer to hamiltonian
    d_prt:   device pointer to perturbation in hamiltonian
    d_dm:    device pointer to density matrix
    d_rsp:   device pointer to response in density matrix
    N:     matrix size
    Nocc:  occupation number
*/

void dnnprt(double* d_ham, double* d_prt, 
            double* d_dm, double* d_rsp,
            int N, int Nocc)
{
    nvtxRangePushA("Declare and allocate memory");
    int Stopp = 0;
    int iter = 0;
    std::vector<float> Idemp_Error;
    std::vector<float> Idemp1_Error;
    std::vector<float> Occ1_Error;

    // Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // set math mode (deprecated)
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    double  *d_TrD0, *TrD0, *TrD1, *d_TrD1;

    float   *d_S0, *d_S02, *d_TrS0, *d_TrS02, *TrS0, *TrS02, *d_Z, *A, *A_t, *d_A,
            *d_S1, *d_S0S1, *d_TrS1, *d_TrS0S1, *TrS1, *TrS0S1,
            *d_Sig, *d_Id, *sbuf1, *sbuf2, *sbuf3,*Sig, *d_S2, *D0, *D1;

    float   a, b, c;      

    half    *hbuf1, *hbuf2, *hbuf3, *hbuf4;

    int     *v_sgn;
    
    // Allocate some host memory
    v_sgn = (int*) malloc( 500 * sizeof(int) );
    Sig   = (float*) malloc(sizeof(float));
    TrD0  = (double*) malloc(sizeof(double) );
    TrD1  = (double*) malloc(sizeof(double) );
    TrS0 = (float *)malloc(sizeof(float)); 
    TrS1 = (float *)malloc(sizeof(float));
    TrS0S1 = (float *)malloc(sizeof(float));
    TrS02 = (float *)malloc(sizeof(float));

    // Create cuda timing events
    cudaEvent_t start,stop,start_loop,stop_loop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_loop);
    cudaEventCreate(&stop_loop);
    float elapsedTime_loop;
    
    // Allocate device memory
    CUDA_CHECK_ERR(cudaMalloc(&d_S0,N*N*sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S02,N*N*sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S1,N*N*sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S0S1,N*N*sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_Id,N*N*sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_Sig,sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrS0,sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrS02,sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrD0,sizeof(double)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrD1,sizeof(double)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrS1,sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrS0S1,sizeof(float)));

    // Allocate buffers
    CUDA_CHECK_ERR(cudaMalloc(&sbuf1,  N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&sbuf2,  N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&sbuf3,  N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&hbuf1,  N * N * sizeof(half)));
    CUDA_CHECK_ERR(cudaMalloc(&hbuf2,  N * N * sizeof(half)));
    CUDA_CHECK_ERR(cudaMalloc(&hbuf3,  N * N * sizeof(half)));
    CUDA_CHECK_ERR(cudaMalloc(&hbuf4,  N * N * sizeof(half)));
        
    nvtxRangePop();

    // Define grid size
    int numthds = 512;
    int numblks = N*N/numthds+1; 

    /* Input layer to DNN-SP2 */
    
    nvtxRangePushA("Compute input layer");
    // convert double to single
    doubleToFloat<<<numblks, numthds>>>(d_ham, d_S0, N);
    doubleToFloat<<<numblks, numthds>>>(d_prt, d_S1, N);

    // Estimate spectral bounds with Gershgorin 
    double h1, hN;
    cudaStream_t stream;
    CUDA_CHECK_ERR(cudaStreamCreate(&stream));
    gershgorin_v2<double>(N, d_ham, &h1, &hN, stream);
    
    // Linear transform constants 
    a = -1/(hN-h1); 
    b = hN/(hN-h1); 
    c = 0.0;

    // build Identity on dev
    setToIdentityMatrix<<<numblks, numthds>>>(d_Id, N);

    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &b,
                                 d_Id, N,
                                 &a,
                                 d_S0, N,  
                                 d_S0, N)); 

    // Compute initial layer of first order term
    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, 
                                 &c,
                                 d_Id, N,
                                 &a,
                                 d_S1, N,  
                                 d_S1, N));

    nvtxRangePop();

    // Compute initial traces
    float trace=0.;
    /*float trace = 0.0;
    #pragma acc parallel loop deviceptr(d_S0) reduction(+:trace)
    for(int i=0; i < N; i++){
        trace += d_S0[i*N + i];
    }
    TrS0[0] = trace;

    GPUSTrace(N, d_S0, d_TrS0); 
    CUDA_CHECK_ERR(cudaMemcpy(TrS0, d_TrS0, sizeof(float), cudaMemcpyDeviceToHost)); 
    */
    openacc_trace(TrS0, d_S0, N);
  
    /*trace = 0.0;
    #pragma acc parallel loop deviceptr(d_S1) reduction(+:trace)
    for(int i=0; i < N; i++){
        trace += d_S1[i*N + i];
    }
    TrS1[0] = trace;*/
    openacc_trace(TrS1, d_S1, N);
    
    CUDA_CHECK_ERR(cudaEventRecord(start_loop, 0));

    int converged = 0;

    nvtxRangePushA("Main loop");

    while (Stopp == 0) {
        
        nvtxRangePushA("TC matmul S0^2");

        tcoreSPGemmSymm(handle,
                        N,
                        d_S0,
                        hbuf1,hbuf2,
                        sbuf1,sbuf2,
                        d_S02, &stream);
        nvtxRangePop();

        nvtxRangePushA("TC matmul S0S1");
	tcoreSPGemmSymm1(handle,
                         N,
                         d_S0,d_S1,
                         hbuf1,hbuf2,hbuf3,hbuf4,
                         sbuf1,sbuf2,
                         d_S0S1);
        nvtxRangePop();
	
        // Trace of S0^2
        nvtxRangePushA("tr[S0^2]");

        /*trace = 0.0;
        #pragma acc parallel loop deviceptr(d_S02) reduction(+:trace)
        for (int i=0;i<N;i++){
            trace += d_S02[i*N+i];
        }
        TrS02[0] = trace;*/
        openacc_trace(TrS02, d_S02, N);
   
        nvtxRangePop();
    
        // Trace of S0S1
        nvtxRangePushA("tr[S0S1]");
        
        /*trace = 0.0;
        #pragma acc parallel loop deviceptr(d_S0S1) reduction(+:trace)
        for (int i=0;i<N;i++){
            trace += d_S0S1[i*N+i];
        }
        TrS0S1[0] = trace;*/
        openacc_trace(TrS0S1, d_S0S1, N);
          
        nvtxRangePop();
    
        
	// Store S0 Idempotency error    
        Idemp_Error.push_back(TrS0[0]-TrS02[0]);
        
        // S1 Idempotency error    
        Occ1_Error.push_back(TrS1[0]);
        Idemp1_Error.push_back(TrS1[0]-TrS0S1[0]);
          
        #ifdef VERBOSE
          
            std::cout << "S0 Idempotency error = " << Idemp_Error[iter] << std::endl;	
	  
          // S1 Idempotency error    
            std::cout << "S1 idemp. err. = " <<  Idemp1_Error[iter] << std::endl;	
	  // S1 Occupation error    
            if (iter>2){
                std::cout << "S1 occ. error = " << Occ1_Error[iter] << std::endl;              
                std::cout << "Proposed error 1 = " << Occ1_Error[iter]/(Occ1_Error[iter-2]*Occ1_Error[iter-2]) << std::endl;              
                std::cout << "Proposed error 2 = " << Idemp1_Error[iter]/(Idemp1_Error[iter-2]*Idemp1_Error[iter-2]) << std::endl;              
	     }
          
        #endif

        // Convergence control 
        if (converged == 0){
	    if (TrS0[0]-TrS02[0]<=0){
                converged = 1;
            }
            else if ( iter>2 && v_sgn[iter-1]!=v_sgn[iter-2]  && Idemp_Error[iter]>= 4.5*Idemp_Error[iter-2]*Idemp_Error[iter-2] ){
                converged = 1;
            };
        };
        
        // Compute sigma 
        computeSigma(Nocc,d_TrS0,d_TrS02,d_Sig);
        cudaMemcpy(Sig, d_Sig, sizeof(float), cudaMemcpyDeviceToHost); 
            
            
        // Make sure sigma flips after convergence of S0
        if (converged != 0 ){
            if (v_sgn[iter-1] == 1){
                Sig[0] = -1.;             
            } else {
                Sig[0] = 1.;
            }
            converged++;
        }
             
        if (converged > 2){ 
            std::cout << "FINAL: iter = " << iter << std::endl ; 
            std::cout << "FINAL: S1 idemp. err  = " << Idemp1_Error[iter]  << std::endl ; 
            std::cout << "FINAL: S1 occ. err  = " << Occ1_Error[iter]  << std::endl;   
            break;
        }
 
	a = Sig[0];
	b = 1.0-Sig[0]; 
	
	    
	// Compute S0_{n+1} = W_n*S0_n^2 + B_n = W_n*S0_n^2 + (1-W_n)S0_n
        nvtxRangePushA("Compute S_{n+1}");

        CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N, 
                                     &a,
                                     d_S02, N,
                                     &b,
                                     d_S0, N,  
                                     d_S0, N));

	// Compute S1_{n+1} = W_n*(S0S1_n + S1S0_n) + (1-W_n)S1_n
        CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N, 
                                     &a,
                                     d_S0S1, N,
                                     &b,
                                     d_S1, N,  
                                     d_S1, N));
        
        nvtxRangePop();


        // Update traces
        TrS0[0] = Sig[0]*TrS02[0] + (1-Sig[0])*TrS0[0];
        TrS1[0] = Sig[0]*TrS0S1[0] + (1-Sig[0])*TrS1[0];
        
	
	// Send traces back to device
        CUDA_CHECK_ERR(cudaMemcpy(d_TrS0, TrS0, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERR(cudaMemcpy(d_TrS1, TrS1, sizeof(float), cudaMemcpyHostToDevice));

        // Update sign vector
        v_sgn[iter]=int(Sig[0]);
        
        iter += 1;
    }
    // do refinement
//    linalgtools::doSRefinement(d_S0,
//                               d_D0,
//                               N,
//                               handle);


//    linalgtools::doSRefinement_1stOrder(d_S0,
//                                        d_S1,
//                                        d_D1,
//                                        N,
//                                        handle);
    nvtxRangePop();
    
    // cast from single to double
    floatToDouble<<<numblks, numthds>>>(d_S0, d_dm, N);
    floatToDouble<<<numblks, numthds>>>(d_S1, d_rsp, N);
    
    
    CUDA_CHECK_ERR(cudaEventRecord(stop_loop, 0));
    CUDA_CHECK_ERR(cudaEventSynchronize(stop_loop));
    CUDA_CHECK_ERR(cudaEventElapsedTime(&elapsedTime_loop, start_loop, stop_loop));
    std::cout << "Time for SP2 loop = " << elapsedTime_loop << " ms " << std::endl;
    
    #ifdef SP2_SINGLE
    double TFLOPS = 2*double(N)*double(N)*double(N)*(iter+1)/(elapsedTime_loop/double(1e3))/double(1e12);
    #else
    double TFLOPS = 5*double(N)*double(N)*double(N)*(iter+1)/(elapsedTime_loop/double(1e3))/double(1e12);
    #endif
    
    std::cout << TFLOPS << " TFLOPS" <<std::endl;

    // Deallocate device memory
    // Free buffers
    CUDA_CHECK_ERR(cudaFree(sbuf1));
    CUDA_CHECK_ERR(cudaFree(sbuf2));
    CUDA_CHECK_ERR(cudaFree(hbuf1));
    CUDA_CHECK_ERR(cudaFree(hbuf2));
    CUDA_CHECK_ERR(cudaFree(hbuf3));
    CUDA_CHECK_ERR(cudaFree(hbuf4));
    CUDA_CHECK_ERR(cudaFree(d_S0));
    CUDA_CHECK_ERR(cudaFree(d_S02));
    CUDA_CHECK_ERR(cudaFree(d_S1));
    CUDA_CHECK_ERR(cudaFree(d_S0S1));
    CUDA_CHECK_ERR(cudaFree(d_Id));
    CUDA_CHECK_ERR(cudaFree(d_Sig));
    CUDA_CHECK_ERR(cudaFree(d_TrS0));
    CUDA_CHECK_ERR(cudaFree(d_TrS02));
    CUDA_CHECK_ERR(cudaFree(d_TrD0));
    CUDA_CHECK_ERR(cudaFree(d_TrD1));
    CUDA_CHECK_ERR(cudaFree(d_TrS1));
    CUDA_CHECK_ERR(cudaFree(d_TrS0S1));
    

    //Deallocate host memory
    free(v_sgn);
    free(TrD0);
    free(Sig);
    
    // Destroy handle
    CUBLAS_CHECK_ERR(cublasDestroy(handle));
}
 



