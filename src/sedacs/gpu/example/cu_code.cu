/**
  Description
  -----------
  Simple function which accepts pointer for arr_in and arr_out and integer nocc. New device variable d_ham isinitilaized 
  and data from arr_in is copied to d_ham. d_ham and d_dm are then passed into device kernel which then multiples each component of 
  d_ham by 17. This result is stored into d_dm. d_dm is then copied back to arr_out. This newly overwritten arr_out is then 
  accesible from python.


  Parameters
  ----------
  arr_in : float*
    Input array. Must be pre-allolcated in python.
  arr_out : float*
    Output array. Must be pre-allocated in python.
  nocc : int
    Integer factor meant to mimic occupation number.
**/

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <math_libs/include/cublas.h>

extern "C"{

int main (float*, float*, int );

void berga (float*, float*, int);
}

__global__
void gpu_init(float* ham, float* dm,float nocc){
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i < 16384){
    dm[i]=17*ham[i];   
    };
}

void berga(float* a,float* b,int nocc){
    //a[5]=3;
    a[37]=17;
    b[1]=2;
    //printf("%f\n",b[0]);
}

int main(float *arr_in, float *arr_out, int nocc){ //unsigned int shape) {

  unsigned int num_rows, num_cols, row, col;

  num_rows = 4;
  num_cols = 4;
  
  std::cout << arr_in[0] << std::endl;  
  float Nocc = 2;
  
  size_t N = 16384;//num_rows*num_cols;
  int blks = 16384/1024, thds = 1024;

  float *d_ham, *d_dm;
  cudaMalloc(&d_ham, N *sizeof(float));
  cudaMalloc(&d_dm, N *sizeof(float));

  cudaMemcpy(d_ham,arr_in, N * sizeof(float),cudaMemcpyHostToDevice);
  gpu_init<<<blks, thds>>>(d_ham,d_dm,Nocc);

  // Cublas Handle
  cublasHandle_t handle;
  //cublasCreate(&handle);

  // Set math mode
  cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); 
 
  cudaMemcpy(arr_out,d_dm, N * sizeof(float),cudaMemcpyDeviceToHost);

  return 0;
}
