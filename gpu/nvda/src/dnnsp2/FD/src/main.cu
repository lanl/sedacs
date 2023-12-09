#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cmath>
#include <main.cuh>

void 
printDmat (const unsigned m, 
           const unsigned N, 
           double* x) 
{
     for (int i=0; i<m;i++){
          for (int j=0; j<m; j++){
              std::cout << x[i+j*N] << " ";
          }
       std::cout << std::endl;
   };
};

void 
twonorm (const unsigned N, 
           double* x, 
           double* out) 
{

     for (int i=0; i<N;i++){
          for (int j=0; j<N; j++){
               out[0] = out[0] + x[i+j*N]*x[i+j*N];
          }
   };

   out[0]=sqrt(out[0]);
};

void
save_P1(
  const unsigned N,
  double *P1) {

  std::ofstream file_P1 ("D1.dat");

  if (file_P1.is_open())
  {
    file_P1 << "Matrix of size N = " << N << "\n" ;
    for(int i=0; i<N; ++i) {
        for(int j=i; j<N; ++j) {
            file_P1 << P1[i+j*N] << "\n" ;
        }
    }
    file_P1.close();
  }
};



int main(int argc, char *argv[])
{

    // Matrix size
    size_t N = atoi(argv[1]);
    size_t Nocc = atoi(argv[2]);
    float lambda = atof(argv[3]);

    // Set GPU
    int device = 0;
    cudaSetDevice(device);


    // Cublas Handle
      cublasHandle_t handle;
      cublasCreate(&handle);

    // Set math mode
    cublasStatus_t cublasStat 
              = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Declare Memory
    double *Pp, *Pm, *Ppp, *Pmm, *dP, *P1, *buff;

    // set device memory
    cudaMalloc( &Pp , N * N * sizeof(double) );
    cudaMalloc( &Pm , N * N * sizeof(double) );
    cudaMalloc( &Ppp , N * N * sizeof(double) );
    cudaMalloc( &Pmm , N * N * sizeof(double) );
    cudaMalloc( &dP , N * N * sizeof(double) );
    cudaMalloc( &buff , N * N * sizeof(double) );
    
    // set host memory
    P1 = (double*) malloc( N * N * sizeof(double));

    // Pp = P(lambda), Pm = P(-lambda)     
    getP(2*lambda, N, Nocc, Ppp, handle);
    getP(lambda, N, Nocc, Pp, handle);
    getP(-lambda, N, Nocc, Pm, handle);
    getP(-2*lambda, N, Nocc, Pmm, handle);

    // Calculate 1st order response approx.
    
    double a, b, c, d, e;

    // 3 point stencil
    a = 1.0 / (2.0 * lambda);
    b = -1.0/ (2.0 * lambda);
    cublasStat = cublasDgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &a,
                             Pp, N,
                             &b,
                             Pm, N,
                             dP, N); 
    // 5 point stencil
/*
    a = -1.0 / (12.0 * lambda);
    b = 8.0/ (12.0 * lambda);
    c = -8.0/ (12.0 * lambda);
    d = 1.0/ (12.0 * lambda);
    e = 1.0;
    cublasStat = cublasDgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &a,
                             Ppp, N,
                             &b,
                             Pp, N,
                             buff, N); 
    cublasStat = cublasDgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &c,
                             Pm, N,
                             &d,
                             Pmm, N,
                             dP, N); 
    cublasStat = cublasDgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &e,
                             buff, N,
                             &e,
                             dP, N,
                             dP, N);*/ 
    
    // Copy approximate first order response
    cudaMemcpy(P1, dP, N*N*sizeof(double), cudaMemcpyDeviceToHost); 

    
    // Calculate double-precision result    



    // Compare with density matrix perturbation thy

    //double *out;
    //out = (double*) malloc(sizeof(double));
    //twonorm(N, P1, out);
    //std::cout << out[0] << std::endl; 

   
    save_P1(N, P1);
    

    // Free memory
    cudaFree(Pp);
    cudaFree(Pm);
    cudaFree(dP);

    return 0;
}



