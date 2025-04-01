#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <typeinfo>
#include <cmath>
#include <vector>
#include <structs.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <tcore_hp_emulator.h>
#include <linalg_tools.h>
#include <error_check.h>

__global__ 
void doubleToFloat(double* X,
          float* Y,
	  int N) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
  
    if (i < N * N) {
        Y[i] = float(X[i]);
    }
}


__global__ 
void floatToDouble(float* X, 
          double* Y, 
	  int N) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N * N) {
        Y[i] = double(X[i]);
    }
}


__global__ 
void setToIdentityMatrix(float* X, int N)
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




void dnnsp2(double* d_ham, 
            double* d_dm, 
            int N, 
            int Nocc,
            precision_t precision,
            refine_t refinement,
            void* Handle)
{

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    int Stopp = 0;
    int iter = 0;
    std::vector<float> Idemp_Error;


    //nvtxRangePushA("Register host memory");
    //cudaHostRegister ( ham, N * N * sizeof(double), cudaHostRegisterDefault);
    //cudaHostRegister ( dm, N * N * sizeof(double), cudaHostRegisterDefault);
    //nvtxRangePop();

    // Cublas streams
    int num_streams=2;
    hipStream_t stream[num_streams];
    //for (int i=0;i<num_streams;i++){ 
    //    cudaStreamCreate(&stream[i]);
    //}
     
    // Cublas Handle
    //nvtxRangePushA("build cublas handle");
    rocblas_handle handle;
    ROCBLAS_CHECK_ERR(rocblas_create_handle(&handle));
    //nvtxRangePop();

    //handle = (cublasHandle_t) Handle;    
    // Set stream
    //cublasSetStream(handle,stream[0]);

    nvtxRangePushA("declare memory");
    // Declare Memory
    double *d_TrD0, *TrD0; //*d_ham, *d_dm, *h_dm; 

    float  *d_S0, *d_S02, *d_TrS0, *d_TrS02, *TrS0, *TrS02, *d_S, 
           *d_Sig, *d_Id, *sbuf1, *sbuf2, *Sig; 

    half   *hbuf1, *hbuf2;

    int    *v_sgn;    
    nvtxRangePop();

    // Allocate some host memory
    nvtxRangePushA("initialize memory");
    v_sgn  =    (int*) malloc( 500 * sizeof(int) );
    TrS0   =  (float*) malloc(sizeof(float));
    TrS02  =  (float*) malloc(sizeof(float));
    Sig    =  (float*) malloc(sizeof(float));
    TrD0   = (double*) malloc(sizeof(double) );
    //CUDA_CHECK_ERR(cudaMallocHost(&h_dm,     N * N * sizeof(double)));
   
    // Allocate device memory
    //CUDA_CHECK_ERR(cudaMalloc(&d_ham,    N * N * sizeof(double)));
    //CUDA_CHECK_ERR(cudaMalloc(&d_dm,     N * N * sizeof(double)));
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
    nvtxRangePop();   
 
    // Define blk,thd grid size
    int numthds = 512;
    int numblks = int(ceil(double(N*N)/double(numthds))); 

    // Initialize Hamiltonian and identity
    //CUDA_CHECK_ERR(cudaMemcpy(d_ham, ham, N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // build Identity on dev
    setToIdentityMatrix<<< numblks, numthds >>>(d_Id, N);

    // cast d_ham from double to float
    doubleToFloat<<< numblks, numthds >>>(d_ham, d_S0, N); 
    CUDA_CHECK_ERR(cudaMemcpy(sbuf1, d_S0, N * N * sizeof(float), cudaMemcpyDeviceToDevice));
    
    nvtxRangePushA("Affine transform");
    // Estimate sprectral bounds
    double h1, hN;
    gershgorin_v2(N, d_ham, &h1, &hN);
    
    // input layer to DNN-SP2
      
    // zeroth-order term
    float a = float(-1/(hN-h1)); 
    float b = float(hN/(hN-h1)); 

    CUBLAS_CHECK_ERR(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N,
                                 &b,
                                 d_Id, N,
                                 &a,
                                 d_S0, N,  
                                 d_S0, N)); 
    
    nvtxRangePop();   

    // compute and copy initial traces
    GPUSTrace(N,d_S0,d_TrS0);
    CUDA_CHECK_ERR(cudaMemcpy(TrS0, d_TrS0, sizeof(float), cudaMemcpyDeviceToHost));  
    
    //if (precision==fp32){
    float alphaS = 1.0, betaS = 0.0;
    //}



    nvtxRangePushA("Main loop");
    while (Stopp == 0) {
        
        nvtxRangePushA("TC matmul");
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
            tcoreSPGemmSymm(handle,
                            stream,
                            N,
                            d_S0,
                            hbuf1, hbuf2,
                            sbuf1, sbuf2,
                            d_S02);
        };
        nvtxRangePop();   
	
        nvtxRangePushA("Compute trace");
	// trace of S0^2-- sucks
        //GPUSTrace(N,d_S02,d_TrS02); //only works for N even
        float trace=0.0;
        #pragma acc parallel loop deviceptr(d_S02) reduction(+:trace)
        for (int i=0;i<N;i++){
           trace += d_S02[i*N+i];
        }
        TrS02[0] = double(trace);
        nvtxRangePop();   
        
        //CUDA_CHECK_ERR(cudaMemcpy(TrS02, d_TrS02, sizeof(float), cudaMemcpyDeviceToHost)); 
        
//        nvtxRangePushA("Convergence criteria");
	// S0 idempotency error    
        Idemp_Error.push_back(TrS0[0]-TrS02[0]);
          
        //if verbose: 
        //std::cout << "S0 Idempotency error = " << Idemp_Error[iter] << std::endl;	
	 
        // convergence control on S0
	if (TrS0[0]-TrS02[0]<=0)
        {
            //printf("XO converged at iteration = %d \n", iter);
            break;
        }
        else if ( iter>2 && v_sgn[iter-1]!=v_sgn[iter-2] \
                   && Idemp_Error[iter]>= 4.5*Idemp_Error[iter-2]*Idemp_Error[iter-2] )
        {
            //printf("XO converged at iteration = %d \n", iter);
            break;
        };
        
 //       nvtxRangePop();   

        // Compute Sigma (which is determind by S0)
        nvtxRangePushA("Compute sigma and weights");
        computeSigma(Nocc,d_TrS0,d_TrS02,d_Sig);
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

        nvtxRangePop();   

        // Update traces
        TrS0[0] = Sig[0]*TrS02[0] + (1-Sig[0])*TrS0[0];
        
	
	// Send traces back to device
	CUDA_CHECK_ERR(cudaMemcpy(d_TrS0, TrS0, sizeof(float), cudaMemcpyHostToDevice)); 

        // Update sign vector
        v_sgn[iter]=int(Sig[0]);
        
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
    CUDA_CHECK_ERR(cudaMalloc(&d_T0,N*N*sizeof(double)));

    // refinement step
    if (refinement == yes){
    
        // change dm approximation to double-prec
        floatToDouble<<<numblks, numthds>>>(d_S0, d_T0, N);   
    
        // do the refinement 
        doRefinement(d_T0,d_dm,N,Nocc,handle);
    
    }
    else {
    
        // change dm approximation to double-prec
        floatToDouble<<<numblks, numthds>>>(d_S0, d_dm, N);
    
    };
    nvtxRangePop();   

    nvtxRangePushA("dm copy from DtoH");
    // copy dm back to hot buffer
    //CUDA_CHECK_ERR(cudaMemcpy(h_dm, d_dm, N * N * sizeof(double), cudaMemcpyDeviceToHost)); 
    nvtxRangePop();   
   
    nvtxRangePushA("HtoH dm copy");
    // copy cpu buffer to python allocated dm (avoids python memory issues, pagability?)
    //memcpy(dm,h_dm, N * N * sizeof(double));
    nvtxRangePop();   
     
    nvtxRangePushA("cudaFree");
    // Free device memory thats no longer needed
    CUDA_CHECK_ERR(cudaFree(d_S0));
    CUDA_CHECK_ERR(cudaFree(d_S02));
    CUDA_CHECK_ERR(cudaFree(d_Sig));
    CUDA_CHECK_ERR(cudaFree(d_TrS0));
    CUDA_CHECK_ERR(cudaFree(d_TrS02));
    //CUDA_CHECK_ERR(cudaFree(d_ham));
    CUDA_CHECK_ERR(cudaFree(d_T0));
    //CUDA_CHECK_ERR(cudaFree(d_dm));
    CUDA_CHECK_ERR(cudaFree(d_TrD0));
    CUDA_CHECK_ERR(cudaFree(d_Id));
    //CUDA_CHECK_ERR(cudaFreeHost(h_dm));
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
std::cout << "time = " << milliseconds/1000.0 << std::endl;
}



