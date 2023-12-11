#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <regex>
#include <typeinfo>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <vector>
#include <cusolverDn.h>

#include "tcore_hp_emulator.cuh"
#include "linalg_tools.cuh"

#define SP2_TC
#define REFINEMENT

__global__ 
void FtoD(float *X
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
void dev_buildIdenity(float* X
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

__global__ 
void 
dev_buildPerturbation(float* X
 		          ,int N)
{  
  int i = threadIdx.x + blockIdx.x * blockDim.x; 

  if (i==N+1){
	X[i] = 1.0;
  }
  else{
  	X[i]= 0.0;
  }

  if (i== 0){
	X[i] = -1.0;
  }

};

void 
build_H (const unsigned N, float *X) {
    for(int i=0; i<N; ++i) {
        for(int j=i; j<N; ++j) {
            X[i+j*N] = exp(-0.5f*abs((float)(i-j)))*sin((float)(i+1));
            X[j+i*N] = X[i+j*N];
        }
    }
};


void 
print_Dmat (const unsigned m, const unsigned N, double* x) {
     for (int i=0; i<m;i++){
          for (int j=0; j<m; j++){
              std::cout << std::setprecision(7) << x[i+j*N] << " ";
          }
       std::cout << std::endl;
   };
};

void 
print_Smat (const unsigned m, 
            const unsigned N, 
            float* x) 
{
     for (int i=0; i<m;i++){
          for (int j=0; j<m; j++){
              std::cout << std::setprecision(7) << x[i+j*N] << " ";
          }
       std::cout << std::endl;
   };
};

void 
getP(const float lambda,
     const int N,
     const int Nocc,
     double* P,
     cublasHandle_t handle)
{

    int Stopp = 0;
    int iter = 0;

    std::vector<float> Idemp_Error;
     
    // Cusolver Handle
      cusolverDnHandle_t cusolverH;
      cusolverDnCreate(&cusolverH);
    
    // Set math mode
      cublasStatus_t 
           cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Declare Memory
      double *d_TrD, *TrD, *d_H1, *d_energy_1, *d_energy_2,
             *d_H; 

      float  *d_S, *d_S2, *d_TrS, *d_TrS2, *S, *TrS, *TrS2,  
             *d_S1, *d_Sig, *d_Id, *sbuf1, *sbuf2, *Sig, *Eig;

      half   *hbuf1, *hbuf2;
      int    *v_sgn;
    
    // Allocate some host memory
      S = (float*) malloc( N * N * sizeof(float));
      v_sgn = (int*) malloc( N * sizeof(int) );
      TrS = (float*) malloc(sizeof(float));
      TrS2 = (float*) malloc(sizeof(float));
      Sig = (float*) malloc(sizeof(float));
      TrD = (double*) malloc(sizeof(double) );
      Eig = (float*) malloc(N * sizeof(float));
    
    // Allocate device memory
      cudaMalloc(&d_H,N*N*sizeof(double));
      cudaMalloc(&d_S,N*N*sizeof(float));
      cudaMalloc(&d_S2,N*N*sizeof(float));
      cudaMalloc(&d_S1,N*N*sizeof(float));
      cudaMalloc(&d_Id,N*N*sizeof(float));
      cudaMalloc(&d_Sig,sizeof(float));
      cudaMalloc(&d_TrS,sizeof(float));
      cudaMalloc(&d_TrS2,sizeof(float));
      cudaMalloc(&d_TrD,sizeof(double));

    // Allocate Buffers
      cudaMallocManaged(&sbuf1,  N * N * sizeof(float));
      cudaMallocManaged(&sbuf2,  N * N * sizeof(float));
      cudaMallocManaged(&hbuf1,  N * N * sizeof(half));
      cudaMallocManaged(&hbuf2,  N * N * sizeof(half));
   
    // get GPU specs so can adjust grid size
 
    // Define grid size
      int numThreads = 1024;
      int numBlocks = N*N/numThreads+1; 

    // Initialize Hamiltonian and identity
      build_H(N, S);
      cudaMemcpy(d_S, S, N * N * sizeof(float), cudaMemcpyHostToDevice);
      dev_buildIdenity<<< numBlocks, numThreads >>>(d_Id, N);
      dev_buildPerturbation<<< numBlocks, numThreads >>>(d_S1, N);

    // Build double-prec Hamiltonian
      FtoD<<<numBlocks,numThreads>>>(d_S, d_H, N);
    
    // introduce first-order perturbation
      float aa= 0.0, bb = 1.0;
      cublasStat = cublasSgeam(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N,
                               &aa,
                               d_Id, N,
			       &lambda,
                               d_S1, N,  
                               d_S1, N);
      
      cublasStat = cublasSgeam(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N,
                               &bb,
                               d_S, N,
			       &bb,
                               d_S1, N,  
                               sbuf1, N);
    
      cudaMemcpy(d_S, sbuf1, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    


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
        //printf("h1 = %f \n", h1);
        //printf("hN = %f \n", hN);
   
        float band_energy = 0.0;
       
        for ( int i = 0; i < Nocc; ++i)
	{ 
          band_energy += Eig[i]; 
	};
   
    //
    //
    //===================================================================
    // Input layer to DNN-SP2
    //===================================================================
    //
    //  
      
      float a = -1/(hN-h1); 
      float b = hN/(hN-h1); 

      cublasStat = cublasSgeam(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N,
                               &b,
                               d_Id, N,
                               &a,
                               d_S, N,  
                               d_S, N); 
    
    // Compute and copy initial traces
      linalgtools::GPUSTrace(N,d_S,d_TrS);

    #ifdef SP2_SINGLE
    float alphaS = 1.0, betaS = 0.0, gammaS = 1.0;
    #endif
    
    while (Stopp == 0) {
        
        #ifdef SP2_SINGLE 
        cublasStat = cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N,
                                 &alphaS,
                                 d_S, N,
                                 d_S, N,
                                 &betaS,
                                 d_S2, N);
        
         
        #else
        tcoretools::tcoreSPGemmSymm(handle
                                   ,N
                                   ,d_S
                                   ,hbuf1
                                   ,hbuf2
                                   ,sbuf1
                                   ,sbuf2
                                   ,d_S2);
        
        #endif


	// Trace of S^2
          linalgtools::GPUSTrace(N, d_S2, d_TrS2); //only works for N even
          cudaMemcpy(TrS2, d_TrS2, sizeof(float), cudaMemcpyDeviceToHost); 
	
        // S Idempotency error    
          Idemp_Error.push_back(TrS[0]-TrS2[0]);
         // std::cout << "S Idempotency error = " << Idemp_Error[iter] << std::endl;	

        // Convergence control on S
	  if (TrS[0]-TrS2[0]<=0){
          //    break;
          };
          if ( iter>2 && v_sgn[iter-1]!=v_sgn[iter-2]  && Idemp_Error[iter]>= 4.5*Idemp_Error[iter-2]*Idemp_Error[iter-2]){
              break;
          };

  
          linalgtools::computeSigma(Nocc,d_TrS,d_TrS2,d_Sig);
          cudaMemcpy(Sig, d_Sig, sizeof(float), cudaMemcpyDeviceToHost); 
	  a = Sig[0];
	  b = 1.0-Sig[0]; 
	
	// Compute S_{n+1} = W_n*S_n^2 + B_n = W_n*S_n^2 + (1-W_n)S_n
        cublasStat = cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, 
                                 &a,
                                 d_S2, N,
                                 &b,
                                 d_S, N,  
                                 d_S, N);
       
        // Update traces
        TrS[0] = Sig[0]*TrS2[0] + (1-Sig[0])*TrS[0];
	
	// Send traces back to device
	cudaMemcpy(d_TrS, TrS, sizeof(float), cudaMemcpyHostToDevice); 

        // Update sign vector
        v_sgn[iter]=int(Sig[0]);
        
        iter += 1;
    }

    // Free buffers
    cudaFree(sbuf1);
    cudaFree(sbuf2);
    cudaFree(hbuf1);
    cudaFree(hbuf2);
    
    
    // Allocate memory for density matrix 
    double *d_T;
    cudaMalloc(&d_T,N*N*sizeof(double));
    
    
    //
    // Change density matrix approximation to double-prec
    //

      FtoD<<<numBlocks, numThreads>>>(d_S, d_T, N);
      
    //
    // Do the refinement 
    //
      linalgtools::doRefinement(d_T, P, N, Nocc, handle);
   
    
    // std::cout <<"sending back P..."<< std::endl;
  
    
    //Deallocate device memory
    cudaFree(d_H);
    cudaFree(d_Id);
    
    //Deallocate host memory
    free(v_sgn);
    free(TrS);
    free(TrS2);
    free(Sig);
    
    // Destroy handle
    cublasDestroy(handle);

};



