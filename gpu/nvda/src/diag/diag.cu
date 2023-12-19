#include <iostream>
#include <string>
#include <math.h>
#include <cuda.h>
#include <error_check.cuh>
#include <cusolverDn.h>

// diagonalize a matrix
void computeEval(double *d_ham, int N, 
                 double *d_eval, 
                 double *d_evec)
{
    // cusolver handles
    cusolverDnHandle_t cusolver_H;
    CUSOLVER_CHECK_ERR(cusolverDnCreate(&cusolver_H));

    // specify cusolver diag flags 
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; //CUSOLVER_EIG_MODE_NOVECTOR if no eigvec needed
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;  //lowe triangle of A contains elementry reflectors related to diagonlize algorithm
    int lwork = 0;

    // determine amount of temp space needed
    CUSOLVER_CHECK_ERR(cusolverDnDsyevd_bufferSize(cusolver_H,
                                                   jobz, uplo, N, 
                                                   d_ham, N, 
                                                   d_eval, &lwork)); 

    // allocate temp work vars
    double *d_work = NULL;
    int *devInfo = NULL;
    CUDA_CHECK_ERR(cudaMalloc((void **) &d_work, sizeof(double)*lwork));
    CUDA_CHECK_ERR(cudaMalloc((void **) &devInfo, sizeof(int)));

    // diagonalize
    CUSOLVER_CHECK_ERR(cusolverDnDsyevd(cusolver_H, 
                                        jobz, uplo, N, 
                                        d_ham, N, 
                                        d_eval, 
                                        d_work, lwork, 
                                        devInfo));

    // copy to d_evec 
    CUDA_CHECK_ERR(cudaMemcpy(d_evec, d_ham, N*N*sizeof(double), cudaMemcpyDeviceToDevice));

    // free memory
    CUSOLVER_CHECK_ERR(cusolverDnDestroy(cusolver_H));
    CUDA_CHECK_ERR(cudaFree(d_work)); 
    CUDA_CHECK_ERR(cudaFree(devInfo));
};


// Compute the occupation factors using Fermi-Dirac
// and store in diagonal of diagonal matrix 
__global__ 
void computeOcc(double *eval,
                const unsigned int N, 
                double *beta,
                double *mu, 
                double *occ)
{
    // get thread idx
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    while (i < N*N){

        
        if (i % (N+1) == 0){

	    // calculate occupation using Fermi-Dirac, along diagonal
            occ[i] = pow(exp(beta[0]*(eval[i%N] - mu[0])) + 1, -1); 
	
        }
        else{
            // fill in zeros off-diagonal
	    occ[i] = 0.0;
	}
	
        // advance i by the grid size
        i += blockDim.x*gridDim.x;
    }

};

//Compute a density matrix from eigenvectors
void  compute_dm(double *occ, 
                 double *evec, 
                 double *dm,
                 const unsigned N)
{
    // create handles
    cublasHandle_t handle;
    CUBLAS_CHECK_ERR(cublasCreate(&handle));	
    CUBLAS_CHECK_ERR(cublasSetMathMode(handle, CUBLAS_GEMM_DEFAULT));

    // set gemm coeffs
    double a, b;
    a=1.0; b=0.0;

    // create occupation matrix
    double *occ_mat;
    CUDA_CHECK_ERR(cudaMalloc(&occ_mat, N * N * sizeof(double)));

    // evecs * occ_mat = occ_mat
    CUBLAS_CHECK_ERR(cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N,
                                 &a,
                                 evec, N,
                                 occ_mat, N,
                                 &b,
                                 occ_mat, N));

    // occ_mat * evecs^T= d_dm
    CUBLAS_CHECK_ERR(cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 N, N, N,
                                 &a,
                                 occ_mat, N,
                                 evec, N,
                                 &b,
                                 dm, N)); 

    CUBLAS_CHECK_ERR(cublasDestroy(handle)); 
    CUDA_CHECK_ERR(cudaFree(occ_mat));
}

void diagonalize(double* ham, 
                 double* dm, 
                 int N,
                 int Nocc)
{
    // kernel launch paramaters
    int nthds = 512;
    int nblks = int(ceil(float(N*N)/float(nthds))); 

    // declare vars
    double *eval, *evec, *occ, *beta, *mu;
    double *d_ham, *d_eval, *d_evec;
    double *d_dm, *d_occ, *d_beta, *d_mu;

    // allocate host memory
    eval = (double*)malloc( N * sizeof(double) );
    evec = (double*)malloc( N * N * sizeof(double) );
    occ =  (double*)malloc( N * N * sizeof(double) );
    beta = (double*)malloc( sizeof(double) );
    mu =   (double*)malloc( sizeof(double) );

    // allocate device memory
    CUDA_CHECK_ERR(cudaMalloc(&d_ham,  N * N * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_dm,   N * N * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_evec, N * N * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_occ,  N * N * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_eval,     N * sizeof(double)  ));
    //cudaMalloc(&d_mu, sizeof(double));
	
    // copy ham to device
    CUDA_CHECK_ERR(cudaMemcpy(d_ham, ham, N * N * sizeof(double), cudaMemcpyHostToDevice));	

    // call cusolver diag
    computeEval(d_ham, N, d_eval, d_evec); 

    // copy evals,evecs to host
    cudaMemcpy(eval, d_eval, N*sizeof(double), cudaMemcpyDeviceToHost); 
    // cudaMemcpy(evec, d_evec, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // compute fermi level, mu


    // compute occupations 
    computeOcc<<<nthds, nblks>>>(d_eval, N, beta, occ, mu);
    
    // build density matrix
    compute_dm(d_occ, d_evec, d_dm, N);
		
    // send dm back to host
    CUDA_CHECK_ERR(cudaMemcpy(dm, d_dm, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    // free memory
    free(eval); 
    free(evec); 
    free(occ); 
    CUDA_CHECK_ERR(cudaFree(d_ham)); 
    CUDA_CHECK_ERR(cudaFree(d_evec)); 
    CUDA_CHECK_ERR(cudaFree(d_eval)); 
    CUDA_CHECK_ERR(cudaFree(d_occ)); 
    CUDA_CHECK_ERR(cudaFree(d_dm));	
}
