#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <typeinfo>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <vector>
#include <cusolverDn.h>
#include <tcore_hp_emulator.cuh>
#include <linalg_tools.cuh>
#include <dnnsp2.cuh>

__global__ 
void DtoF(double* X,
          float* Y,
	  int N) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
  
    if (i < N * N) {
        Y[i] = float(X[i]);
    }
}


__global__ 
void FtoD(float* X, 
          double* Y, 
	  int N) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N * N) {
        Y[i] = double(X[i]);
    }
}


__global__ 
void dev_buildIdenity(float* X, int N)
{  
    int i = threadIdx.x + blockIdx.x * blockDim.x; 
  
    if (i < N * N) {
        if ( i % (N+1) == 0) {
            X[i] = 1.0f;
        } 
        else {
            X[i] = 0.0f;
        }
    };
};

void dnnsp2(double* ham, 
            double* dm, 
            size_t N, 
            size_t Nocc,
            precision_t precision,
            refine_t refinement)
{

    int Stopp = 0;
    int iter = 0;
    int ITER =0;
    std::vector<float> Idemp_Error;
     
    // Set GPU
    //int device = 0;
    //cudaSetDevice(device);

    // Cublas Handle
    cublasHandle_t handle;
    CUBLAS_CHECK_ERR(cublasCreate(&handle));

    // Cusolver Handle
    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK_ERR(cusolverDnCreate(&cusolverH));
    
    // Set math mode
    if (precision=="fp32"){
    CUBLAS_CHECK_ERR(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    }
    else if (precision=="fp16_fp32"){
        CUBLAS_CHECK_ERR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    };

    // Declare Memory
    double *d_TrD0, *TrD0, *d_ham; 

    float  *d_S0, *d_S02, *d_TrS0, *d_TrS02, *S0, *TrS0, *TrS02, *d_S, 
           *d_Sig, *d_Id, *sbuf1, *sbuf2, *Sig, *Eig; 

    half   *hbuf1, *hbuf2;
    int    *v_sgn;
    
    // Allocate some host memory
    S0     =  (float*) malloc( N * N * sizeof(float));
    v_sgn  =    (int*) malloc( 500 * sizeof(int) );
    TrS0   =  (float*) malloc(sizeof(float));
    TrS02  =  (float*) malloc(sizeof(float));
    Sig    =  (float*) malloc(sizeof(float));
    TrD0   = (double*) malloc(sizeof(double) );
    Eig    =  (float*) malloc(N * sizeof(float));
   
    // Allocate device memory
    CUDA_CHECK_ERR(cudaMalloc(&d_ham,    N * N * sizeof(double)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S,      N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S0,     N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S02,    N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_Id,     N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_Sig,    sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrS0,   sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrS02,  sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_TrD0,   sizeof(double)));

    // Allocate Buffers
    CUDA_CHECK_ERR(cudaMalloc(&sbuf1,  N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&sbuf2,  N * N * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&hbuf1,  N * N * sizeof(half)));
    CUDA_CHECK_ERR(cudaMalloc(&hbuf2,  N * N * sizeof(half)));
    
    // Define grid size
    int num_thds = 512;
    int num_blks = int(ceil(double(N*N)/double(num_thds))); 

    // Initialize Hamiltonian and identity
    CUDA_CHECK_ERR(cudaMemcpy(d_ham, ham, N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // build Identity on dev
    dev_buildIdenity<<< num_blks, num_thds >>>(d_Id, N);

    // cast d_ham from double to float
    DtoF<<< numBlocks, numThreads >>>(d_S0, d_ham, N); 
    CUDA_CHECK_ERR(cudaMemcpy(sbuf1, d_S0, N * N * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Determine initial spectral bounds using cuSOLVER diagonalization  
    linalgtools::computeEigs(sbuf1, N, Eig);
    
    // set extremal eigenvalues, add 10% slack
    float h1, hN;
    h1 = Eig[0]; 
    hN = Eig[N-1];

    if (h1 < 0. and hN > 0){
        h1*=1.1;
        hN*=1.1;
    }
    else if (h1 > 0. and hN > 0.){
        h1*=0.9;
        hN*=1.1;
    }
    else if (h1 < 0. and hN < 0.){
        h1*=1.1;
        hN*=0.9;
    }

    // input layer to DNN-SP2
      
    // zeroth-order term
    float a = -1/(hN-h1); 
    float b = hN/(hN-h1); 
    float c = 0.;

    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &b,
                                 d_Id, N,
                                 &a,
                                 d_S0, N,  
                                 d_S0, N)); 
    

    // compute and copy initial traces
    linalgtools::GPUSTrace(N,d_S0,d_TrS0);
    CUDA_CHECK_ERR(cudaMemcpy(TrS0, d_TrS0, sizeof(float), cudaMemcpyDeviceToHost));  
    

    if (precision==fp32){
        float alphaS = 1.0, betaS = 0.0, gammaS = 1.0;
    }

    while (Stopp == 0) {
        
        if (precision==fp32){

            CUBLAS_CHECK_ERR(cublasSgemm(handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, N, N,
                                         &alphaS,
                                         d_S0, N,
                                         d_S0, N,
                                         &betaS,
                                         d_S02, N));
        
        }
        else if (precision==fp16_fp32){

            tcoretools::tcoreSPGemmSymm(handle,
                                        N,
                                        d_S0,
                                        hbuf1, hbuf2
                                        sbuf1, sbuf2
                                        d_S02);

        };
	
	// trace of S0^2
        linalgtools::GPUSTrace(N,d_S02,d_TrS02); //only works for N even
        CUDA_CHECK_ERR(cudaMemcpy(TrS02, d_TrS02, sizeof(float), cudaMemcpyDeviceToHost)); 
	
        
	// S0 idempotency error    
        Idemp_Error.push_back(TrS0[0]-TrS02[0]);
          
        #ifdef VERBOSE
          
        std::cout << "S0 Idempotency error = " << Idemp_Error[iter] << std::endl;	
	  
        #endif
	 
        // convergence control on S0
	if (TrS0[0]-TrS02[0]<=0){
            printf("XO converged at iteration = %d \n", iter);
            break;
        }
        else if ( iter>2 && v_sgn[iter-1]!=v_sgn[iter-2]  && Idemp_Error[iter]>= 4.5*Idemp_Error[iter-2]*Idemp_Error[iter-2] ){
            printf("XO converged at iteration = %d \n", iter);
            break;
        };

        // Compute Sigma (which is determind by S0)
        linalgtools::computeSigma(Nocc,d_TrS0,d_TrS02,d_Sig);
        CUDA_CHECK_ERR(cudaMemcpy(Sig, d_Sig, sizeof(float), cudaMemcpyDeviceToHost)); 
        
	a = Sig[0];
	b = 1.0-Sig[0]; 
	
	// Compute S0_{n+1} = W_n*S0_n^2 + B_n = W_n*S0_n^2 + (1-W_n)S0_n
        CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N, 
                                     &a,
                                     d_S02, N,
                                     &b,
                                     d_S0, N,  
                                     d_S0, N));

        // Update traces
        TrS0[0] = Sig[0]*TrS02[0] + (1-Sig[0])*TrS0[0];
        
	
	// Send traces back to device
	CUDA_CHECK_ERR(cudaMemcpy(d_TrS0, TrS0, sizeof(float), cudaMemcpyHostToDevice)); 

        // Update sign vector
        v_sgn[iter]=int(Sig[0]);
        
        iter += 1;


    }
    
    // Free buffers
    CUDA_CHECK_ERR(cudaFree(sbuf1));
    CUDA_CHECK_ERR(cudaFree(sbuf2));
    CUDA_CHECK_ERR(cudaFree(hbuf1));
    CUDA_CHECK_ERR(cudaFree(hbuf2));
    
    // allocate memory for density matrices 
    double *d_T0;
    CUDA_CHECK_ERR(cudaMalloc(&d_T0,N*N*sizeof(double)));
    D0 = (double*) malloc( N * N * sizeof(double));

    // refinement step
    if (refinement == yes){
    
        // change dm approximation to double-prec
        FtoD<<<numBlocks, numThreads>>>(d_S0, d_T0, N);   
    
        // do the refinement 
        linalgtools::doRefinement(d_T0,d_dm,N,Nocc,handle);
    
    }
    else {
    
        // change dm approximation to double-prec
        FtoD<<<numBlocks, numThreads>>>(d_S0, d_dm, N);
    
    };

    // copy dm back to host
    CUDA_CHECK_ERR(cudaMemcpy(dm, d_dm, N * N * sizeof(double), cudaMemcpyDeviceToHost)); 
    
    // Free device memory thats no longer needed
    CUDA_CHECK_ERR(cudaFree(d_S0));
    CUDA_CHECK_ERR(cudaFree(d_S02));
    CUDA_CHECK_ERR(cudaFree(d_Sig));
    CUDA_CHECK_ERR(cudaFree(d_TrS0));
    CUDA_CHECK_ERR(cudaFree(d_TrS02));
    CUDA_CHECK_ERR(cudaFree(d_ham));
    CUDA_CHECK_ERR(cudaFree(d_T0));
    CUDA_CHECK_ERR(cudaFree(d_D0));
    CUDA_CHECK_ERR(cudaFree(d_Id));
    CUDA_CHECK_ERR(cudaFree(d_TrD0));

    // deallocate host memory
    free(v_sgn);
    free(TrD0);
    free(TrS0);
    free(TrS02);
    free(Sig);
    
    // Destroy handle
    CUBLAS_CHECK_ERR(cublasDestroy(handle));

}



