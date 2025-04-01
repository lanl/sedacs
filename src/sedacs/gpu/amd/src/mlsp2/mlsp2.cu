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
//#include <tcore_hp_emulator.cuh>
//#include <linalg_tools.cuh>
#include <mlsp2.cuh>

__global__ 
void floatToDouble(float *X
         , double *Y
	 , int N) 
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < N * N) {
    Y[i] = double(X[i]);
    i += blockDim.x * gridDim.x; // add total number of threads to i
  }
}


__global__ 
void setToIdentityMatrix(float* X
 		     ,int N)
{  
  int i = threadIdx.x + blockIdx.x * blockDim.x; 
  
  while (i < N * N) {
    if ( i % (N+1) == 0) {
      X[i] = 1.0f;
    } 
    else {
      X[i] = 0.0f;
    }
    i += blockDim.x * gridDim.x; // add total number of threads to i
}
};

void mlsp2(double* ham, 
            double* dm, 
            int N, 
            int Nocc,
            precision_t prec,
            refine_t refi)
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
    cublasCreate(&handle);

/*    // Cusolver Handle
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    
    // Set math mode
    cublasStatus_t cublasStat \
            = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

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
    cudaMalloc(&d_ham,N*N*sizeof(double));
    cudaMalloc(&d_S,N*N*sizeof(float));
    cudaMalloc(&d_S0,N*N*sizeof(float));
    cudaMalloc(&d_S02,N*N*sizeof(float));
    cudaMalloc(&d_Id,N*N*sizeof(float));
    cudaMalloc(&d_Sig,sizeof(float));
    cudaMalloc(&d_TrS0,sizeof(float));
    cudaMalloc(&d_TrS02,sizeof(float));
    cudaMalloc(&d_TrD0,sizeof(double));

    // Allocate Buffers
    cudaMallocManaged(&sbuf1,  N * N * sizeof(float));
    cudaMallocManaged(&sbuf2,  N * N * sizeof(float));
    cudaMallocManaged(&hbuf1,  N * N * sizeof(half));
    cudaMallocManaged(&hbuf2,  N * N * sizeof(half));
    
    // Define grid size
    int numThreads = 512;
    int numBlocks = N*N/numThreads+1; 

    // Initialize Hamiltonian and identity
    std::cout << "loaded Hamiltonian" << std::endl;
    cudaMemcpy(d_S0, S0, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // build Identity on dev
    setToIdentityMatrix<<< numBlocks, numThreads >>>(d_Id, N);

    // cast ham from float to double
    floatToDouble<<<numBlocks,numThreads>>>(d_S0, d_ham, N); 
    cudaMemcpy(sbuf1, d_S0, N * N * sizeof(float), cudaMemcpyDeviceToDevice);

    //
    //
    //===================================================================
    // Determine initial spectral bounds using cuSOLVER diagonalization
    //===================================================================
    //
    //      
      
    linalgtools::computeEigs(sbuf1, N, Eig);
    
   
    // set extremal eigenvalues
    
    float h1, hN;
    
    h1 = Eig[0]*1.01; 
    hN = Eig[N-1]*1.01;
   

    //
    //
    //===================================================================
    // Input layer to DNN-SP2
    //===================================================================
    //
    //  
      
    // zeroth-order term
    float a = -1/(hN-h1); 
    float b = hN/(hN-h1); 
    float c = 0.;

    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &b,
                             d_Id, N,
                             &a,
                             d_S0, N,  
                             d_S0, N); 
    

    // Compute and copy initial traces
    linalgtools::GPUSTrace(N,d_S0,d_TrS0);
    cudaMemcpy(TrS0, d_TrS0, sizeof(float), cudaMemcpyDeviceToHost);  
    

    #ifdef SP2_SINGLE
    float alphaS = 1.0, betaS = 0.0, gammaS = 1.0;
    #endif

    int BERGA=0;
    while (Stopp == 0) {
        
        #ifdef SP2_SINGLE 
        cublasStat = cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alphaS,
                             d_S0, N,
                             d_S0, N,
                             &betaS,
                             d_S02, N);
        
        #else
        tcoretools::tcoreSPGemmSymm(handle
                                   ,N
                                   ,d_S0
                                   ,hbuf1
                                   ,hbuf2
                                   ,sbuf1
                                   ,sbuf2
                                   ,d_S02);
        #endif
	
	// Trace of S0^2
          linalgtools::GPUSTrace(N,d_S02,d_TrS02); //only works for N even
          cudaMemcpy(TrS02, d_TrS02, sizeof(float), cudaMemcpyDeviceToHost); 
	
        
	// S0 Idempotency error    
          Idemp_Error.push_back(TrS0[0]-TrS02[0]);
          
          #ifdef VERBOSE
          
          std::cout << "S0 Idempotency error = " << Idemp_Error[iter] << std::endl;	
	  
          
          #endif
	 
        // Convergence control on S0 (what about S1??)
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
          cudaMemcpy(Sig, d_Sig, sizeof(float), cudaMemcpyDeviceToHost); 
        
	  a = Sig[0];
	  b = 1.0-Sig[0]; 
	
	// Compute S0_{n+1} = W_n*S0_n^2 + B_n = W_n*S0_n^2 + (1-W_n)S0_n
        cublasStat = cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, 
                                 &a,
                                 d_S02, N,
                                 &b,
                                 d_S0, N,  
                                 d_S0, N);

        // Update traces
        TrS0[0] = Sig[0]*TrS02[0] + (1-Sig[0])*TrS0[0];
        
	
	// Send traces back to device
	cudaMemcpy(d_TrS0, TrS0, sizeof(float), cudaMemcpyHostToDevice); 

        // Update sign vector
        v_sgn[iter]=int(Sig[0]);
        
        iter += 1;


    }
    
    // Free buffers
    cudaFree(sbuf1);
    cudaFree(sbuf2);
    cudaFree(hbuf1);
    cudaFree(hbuf2);
    
    exit(0);
    // Allocate memory for density matrices 
    double *d_D0, *d_D1, *d_T0, *d_T1, *D0, *D1;
     
    
    cudaMalloc(&d_D0,N*N*sizeof(double));
    
    cudaMalloc(&d_T0,N*N*sizeof(double));
   
    D0 = (double*) malloc( N * N * sizeof(double));
    
    #ifdef REFINEMENT
    
    //
    // Change density matrix approximation to double-prec
    //

      floatToDouble<<<numBlocks, numThreads>>>(d_S0, d_T0, N);   
    
    //
    // Do the refinement 
    //

    linalgtools::doRefinement(d_T0,d_D0,N,Nocc,handle);
    
    #endif
    
    #ifdef NO_REFINEMENT
    floatToDouble<<<numBlocks, numThreads>>>(d_S0, d_D0, N);
    
    // copy density matrix back to host
    cudaMemcpy(dm, d_D0, N * N * sizeof(double), cudaMemcpyDeviceToHost); 
    
    // Free device memory thats no longer needed
    cudaFree(d_S0);
    cudaFree(d_S02);
    cudaFree(d_Sig);
    cudaFree(d_TrS0);
    cudaFree(d_TrS02);
    #endif

    
    // deallocate dev memory
    cudaFree(d_ham);
    cudaFree(d_T0);
    cudaFree(d_D0);
    cudaFree(d_Id);
    cudaFree(d_TrD0);


    // deallocate host memory
    free(v_sgn);
    free(TrD0);
    free(TrS0);
    free(TrS02);
    free(Sig);
    
    // Destroy handle
    cublasDestroy(handle);
*/
}



