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
#include "../include/tcore_hp_emulator.cuh"
#include "../include/linalg_tools.cuh"
#include <cusolverDn.h>

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
void dev_buildPerturbation(float* X
 		          ,int N)
{  
  int i = threadIdx.x + blockIdx.x * blockDim.x; 
  
  //while (i < N * N) {
  //   
  //    X[i] = cos( (float)i ) ;
  //    i += blockDim.x * gridDim.x; // add total number of threads to i

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

void build_H0 (const unsigned N, float *X) {
    for(int i=0; i<N; ++i) {
        for(int j=i; j<N; ++j) {
            X[i+j*N] = exp(-0.5f*abs((float)(i-j)))*sin((float)(i+1));
            X[j+i*N] = X[i+j*N];
        }
    }
};


void print_Dmat (const unsigned m, const unsigned N, double* x) {
     for (int i=0; i<m;i++){
          for (int j=0; j<m; j++){
              std::cout << std::setprecision(7) << x[i*N+j] << " ";
          }
       std::cout << std::endl;
   };
};

void print_Smat (const unsigned m, const unsigned N, float* x) {
     for (int i=0; i<m;i++){
          for (int j=0; j<m; j++){
              std::cout << std::setprecision(7) << x[i*N+j] << " ";
          }
       std::cout << std::endl;
   };
};

int main(int argc, char *argv[])
{

    // Matrix size
    size_t N = atoi(argv[1]);
    size_t Nocc = atoi(argv[2]);
    float lambda = atof(argv[3]);

    int Stopp = 0;
    int iter = 0;

    std::vector<float> Idemp_Error;
     
    // Set GPU
      int device = 0;
      cudaSetDevice(device);

    // Cublas Handle
      cublasHandle_t handle;
      cublasCreate(&handle);

    // Cusolver Handle
      cusolverDnHandle_t cusolverH;
      cusolverDnCreate(&cusolverH);
    
    // Set math mode
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Declare Memory
      double *d_TrD0, *TrD0, *TrD1, *d_TrD1, *d_H1, *d_energy_1, *d_energy_2,
             *d_H0, *d_energy_0, *energy_0, *comm_err, *idem_err, *energy_1, *energy_2, 
             *occ_err, *d_occ_err, *d_idem_err, *d_comm_err;

      float  *d_S0, *d_S02, *d_TrS0, *d_TrS02, *S0, *TrS0, *TrS02, *d_S, 
             *d_S1, *d_TrS1, *TrS1, *d_Sig, *d_Id, *sbuf1, *sbuf2, *Sig, 
	     *Eig, *d_E1, *d_S2, *d_ener;

      half   *hbuf1, *hbuf2;
      int    *v_sgn;
    
    // Allocate some host memory
      S0 = (float*) malloc( N * N * sizeof(float));
      v_sgn = (int*) malloc( N * sizeof(int) );
      TrS0 = (float*) malloc(sizeof(float));
      TrS02 = (float*) malloc(sizeof(float));
      TrS1 = (float*) malloc(sizeof(float));
      Sig = (float*) malloc(sizeof(float));
      TrD0 = (double*) malloc(sizeof(double) );
      TrD1 = (double*) malloc(sizeof(double) );
      energy_0 = (double*) malloc(sizeof(double));
      energy_1 = (double*) malloc(sizeof(double));
      energy_2 = (double*) malloc(sizeof(double));
      comm_err = (double*) malloc(sizeof(double));
      occ_err = (double*) malloc(sizeof(double));
      idem_err = (double*) malloc(sizeof(double));
      Eig = (float*) malloc(N * sizeof(float));
    
    // Create cuda timing events
      cudaEvent_t start,stop,start_loop,stop_loop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventCreate(&start_loop);
      cudaEventCreate(&stop_loop);
      float elapsedTime_loop;
    
    // Allocate device memory
      cudaMalloc(&d_H0,N*N*sizeof(double));
      cudaMalloc(&d_S,N*N*sizeof(float));
      cudaMalloc(&d_S0,N*N*sizeof(float));
      cudaMalloc(&d_S02,N*N*sizeof(float));
      cudaMalloc(&d_E1,N*N*sizeof(float));
      cudaMalloc(&d_Id,N*N*sizeof(float));
      cudaMalloc(&d_Sig,sizeof(float));
      cudaMalloc(&d_TrS0,sizeof(float));
      cudaMalloc(&d_TrS02,sizeof(float));
      cudaMalloc(&d_TrD0,sizeof(double));
      cudaMalloc(&d_TrD1,sizeof(double));
      cudaMalloc(&d_TrS1,sizeof(float));
      cudaMalloc(&d_occ_err,sizeof(double));
      cudaMalloc(&d_idem_err,sizeof(double));
      cudaMalloc(&d_ener,sizeof(float));
      cudaMalloc(&d_energy_0,sizeof(double));
      cudaMalloc(&d_energy_1,sizeof(double));
      cudaMalloc(&d_energy_2,sizeof(double));
      cudaMalloc(&d_comm_err,sizeof(double)); 

    // Allocate Buffers
      cudaMallocManaged(&sbuf1,  N * N * sizeof(float));
      cudaMallocManaged(&sbuf2,  N * N * sizeof(float));
      cudaMallocManaged(&hbuf1,  N * N * sizeof(half));
      cudaMallocManaged(&hbuf2,  N * N * sizeof(half));
    
    // Define grid size
      int numThreads = 1024;
      int numBlocks = N*N/numThreads+1; 

    // Initialize Hamiltonian and identity
      build_H0(N, S0);
      cudaMemcpy(d_S0, S0, N * N * sizeof(float), cudaMemcpyHostToDevice);
      dev_buildIdenity<<< numBlocks, numThreads >>>(d_Id, N);
      dev_buildPerturbation<<< numBlocks, numThreads >>>(d_S1, N);

    // Build double-prec Hamiltonian
      FtoD<<<numBlocks,numThreads>>>(d_S0, d_H0, N);
    
    // introduce first-order perturbation, +lambda
      float a = 1.0;

      // scale d_S1 by lambda
      cublasSscal(handle, N*N, &lambda, d_S1, 1);
    
      cublasStat = cublasSgeam(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N,
                               &a,
                               d_S0, N,
			       &a,
                               d_S1, N,  
                               sbuf1, N);

    // introduce first-order perturbation, -lambda
      float b = -1.0;
      // scale d_S1 by -lambda
      
      cublasStat = cublasSgeam(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N,
                               &a,
                               d_S0, N,
			       &b,
                               d_S1, N,  
                               sbuf2, N);
    
    for (int j = 0; j < 1; ++j)  
    
    {	    
    //
    //===================================================================
    // Determine initial spectral bounds using cuSOLVER diagonalization
    //===================================================================
    //
      
      linalgtools::computeEigs(sbuf1, N, Eig);
    
   
      // set extremal eigenvalues
    
        float h1, hN;
    
        h1 = Eig[0]*1.01; 
        hN = Eig[N-1]*1.01;
        printf("h1 = %f \n", h1);
        printf("hN = %f \n", hN);
   
        float band_energy = 0.0;
       
        for ( int i = 0; i < Nocc; ++i)
	{ 
	  //printf("e = %f \n", Eig[i]);
          band_energy += Eig[i]; 
	};
    

    //
    //===================================================================
    // Input layer to DNN-SP2
    //===================================================================
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
      cudaMemcpy(TrS, d_TrS, sizeof(float), cudaMemcpyDeviceToHost);  
    
   
    #ifdef SP2_SINGLE
    float alphaS = 1.0, betaS = 0.0, gammaS = 1.0;
    #endif

    cudaEventRecord(start_loop, 0);
 
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
          linalgtools::GPUSTrace(N,d_S2,d_TrS2); //only works for N even
          cudaMemcpy(TrS2, d_TrS2, sizeof(float), cudaMemcpyDeviceToHost); 
        
	// S Idempotency error    
          Idemp_Error.push_back(TrS[0]-TrS2[0]);
          std::cout << "S Idempotency error = " << Idemp_Error[iter] << std::endl;	
         
         
        // Convergence control on S0 (what about S1??)
	  if (TrS0[0]-TrS02[0]<=0){
              break;
          };
          if ( iter>2 && v_sgn[iter-1]!=v_sgn[iter-2]  && Idemp_Error[iter]>= 4.5*Idemp_Error[iter-2]*Idemp_Error[iter-2]){
              break;
          };
        
        // Compute Sigma (which is determind by S0)
          linalgtools::computeSigma(Nocc,d_TrS,d_TrS2,d_Sig);
          cudaMemcpy(Sig, d_Sig, sizeof(float), cudaMemcpyDeviceToHost); 
        
	  a = Sig[0];
	  b = 1.0-Sig[0]; 
	
	// Compute S0_{n+1} = W_n*S0_n^2 + B_n = W_n*S0_n^2 + (1-W_n)S0_n
        cublasStat = cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, 
                                 &a,
                                 d_S2, N,
                                 &b,
                                 d_S, N,  
                                 d_S, N);
       
      
        // Update traces
        TrS0[0] = Sig[0]*TrS02[0] + (1-Sig[0])*TrS0[0];
	
	// Send traces back to device
	cudaMemcpy(d_TrS0, TrS0, sizeof(float), cudaMemcpyHostToDevice); 

        // Update sign vector
        v_sgn[iter]=int(Sig[0]);
        
        iter += 1;
    }
    
    cudaEventRecord(stop_loop, 0);
    cudaEventSynchronize(stop_loop);
    cudaEventElapsedTime(&elapsedTime_loop, start_loop, stop_loop);
    std::cout << "Time for SP2 loop = " << elapsedTime_loop << " ms " << std::endl;
    
    #ifdef SP2_SINGLE
    double TFLOPS = double(N)*double(N)*double(N)*(iter+1)/(elapsedTime_loop/double(1e3))/double(1e12);
    #else
    double TFLOPS = 5*double(N)*double(N)*double(N)*(iter+1)/(elapsedTime_loop/double(1e3))/double(1e12);
    #endif
    
    std::cout << TFLOPS << " TFLOPS" <<std::endl;

    // Free buffers
    cudaFree(sbuf1);
    cudaFree(sbuf2);
    cudaFree(hbuf1);
    cudaFree(hbuf2);
    
    
    // Allocate memory for density matrices 
    double *d_D, *d_T, *D;
   
    cudaMalloc(&d_D,N*N*sizeof(double));
    cudaMalloc(&d_T,N*N*sizeof(double));
   
    D = (double*) malloc( N * N * sizeof(double));
    
    #ifdef REFINEMENT
    
    //
    // Change density matrix approximation to double-prec
    //

      FtoD<<<numBlocks, numThreads>>>(d_S, d_T, N);
      
    
    //
    // Do the refinement 
    //
      linalgtools::doRefinement(d_T,d_D,N,Nocc,handle);

    #endif
    

    }

/*
    #ifdef NO_REFINEMENT
    double alpha_dbl, beta_dbl, *d_TT;
    cudaMalloc(&d_TT,N*N*sizeof(double));
    
    // Typecast S0 S1 to double
    FtoD<<<numBlocks, numThreads>>>(d_S0, d_T0, N);
    cudaMemcpy(d_D0, d_T0, N * N * sizeof(double), cudaMemcpyDeviceToDevice); 
    cudaMemcpy(D0, d_D0, N * N * sizeof(double), cudaMemcpyDeviceToHost); 
    
    FtoD<<<numBlocks, numThreads>>>(d_S1, d_T1, N);
    cudaMemcpy(d_D1, d_T1, N * N * sizeof(double), cudaMemcpyDeviceToDevice); 
    cudaMemcpy(D1, d_D1, N * N * sizeof(double), cudaMemcpyDeviceToHost); 
    
    // Free device memory thats no longer needed
    cudaFree(d_S0);
    cudaFree(d_S02);
    cudaFree(d_S1);
    cudaFree(d_S0S1);
    cudaFree(d_Sig);
    cudaFree(d_TrS0);
    cudaFree(d_TrS02);

    //////////////////////////////////////////////////////
    ///////// Compute occupation error via GPU ///////////
    //////////////////////////////////////////////////////
    linalgtools::GPUDTrace(N,d_D0,d_TrD0); 
    cudaMemcpy(TrD0, d_TrD0, sizeof(double), cudaMemcpyDeviceToHost);
    occ_err[0] = abs(TrD0[0]-Nocc);
    //////////////////////////////////////////////////////'
    
    
    //
    //
    // Compute E0  
    //
    //
      alpha_dbl = 1.0; beta_dbl = 0.0;
    
      cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_D0, N,
                             d_H0, N,
                             &beta_dbl,
                             d_TT, N); 
    
      linalgtools::GPUDTrace(N,d_TT,d_energy_0);
    
      cudaMemcpy(energy_0, d_energy_0, sizeof(double), cudaMemcpyDeviceToHost);    
     
      std::cout << "0th order Energy: " << energy_0[0] << std::endl; 
    
    
    //
    //
    // Compute E1 
    //
    //
      alpha_dbl = 1.0; beta_dbl = 0.0;
      cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_D1, N,
                             d_H0, N,
                             &beta_dbl,
                             d_TT, N); 
      
      linalgtools::GPUDTrace(N,d_TT,d_energy_1);
      
      cudaMemcpy(energy_1, d_energy_1, sizeof(double), cudaMemcpyDeviceToHost);    
    
      std::cout << "1st order Energy: " << energy_1[0]/cc << std::endl; 

    //
    //
    // Compute E2
    //
    //
    alpha_dbl=0.5; beta_dbl=0.0; 
    cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_D1, N,
                             d_H1, N,
                             &beta_dbl,
                             d_TT, N);    
    linalgtools::GPUDTrace(N,d_TT,d_energy_2);
    cudaMemcpy(energy_2, d_energy_2, sizeof(double), cudaMemcpyDeviceToHost);
    //////////////////////////////////////////////////////

    std::cout << "2nd order Energy: " << energy_2[0]/cc/cc << std::endl; 
    
    
    std::cout << "Energy estimate: " << energy_0[0]+energy_1[0]+energy_2[0] << std::endl;
    std::cout << "Calculated band energy: " << band_energy << std::endl;
    std::cout << "Rel. energy error: " << (energy_0[0]+energy_1[0]+energy_2[0]-band_energy)/band_energy << std::endl;
    
    #endif
*/


    printf("\n========================================================================================================================\n\n"); 
    print_Dmat(5,N,D); 
    printf("\n========================================================================================================================\n\n"); 
   
    
    //Deallocate device memory
    cudaFree(d_H0);
    cudaFree(d_T0);
    cudaFree(d_D0);
    cudaFree(d_T1);
    cudaFree(d_D1);
    cudaFree(d_Id);
    cudaFree(d_TrD0);
    cudaFree(d_idem_err);
    cudaFree(d_energy_0);
    cudaFree(d_comm_err);



    //Deallocate host memory
    free(v_sgn);
    free(TrD0);
    free(TrS0);
    free(TrS02);
    free(Sig);
    free(energy_0);
    free(comm_err);
    free(occ_err);
    free(idem_err);
    
    // Destroy handle
    cublasDestroy(handle);

    return 0;
}



