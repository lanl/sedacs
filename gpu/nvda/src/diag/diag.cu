#include <iostream>
#include <string>
#include <math.h>
#include <cuda.h>
#include <error_check.cuh>
#include <cusolverDn.h>
#include <thrust/device_vector.h> 
#include <thrust/host_vector.h> 




// diagonalize a matrix
void computeEval(double *d_ham, int norb, 
                 double *d_eval, 
                 double *d_evec)
{
    // cusolver handles
    cusolverDnHandle_t cusolver_H;
    CUSOLVER_CHECK_ERR(cusolverDnCreate(&cusolver_H));

    // specify cusolver diag flags
    
    //CUSOLVER_EIG_MODE_NOVECTOR if no eigvec needed 
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
    
    //lower triangle contains elementry reflectors related to diag algorithm
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;  
    int lwork = 0;

    // determine amount of temp space needed
    CUSOLVER_CHECK_ERR(cusolverDnDsyevd_bufferSize(cusolver_H,
                                                   jobz, uplo, norb, 
                                                   d_ham, norb, 
                                                   d_eval, &lwork)); 

    // allocate temp work vars
    double *d_work = NULL;
    int *devInfo = NULL;
    CUDA_CHECK_ERR(cudaMalloc((void **) &d_work, sizeof(double)*lwork));
    CUDA_CHECK_ERR(cudaMalloc((void **) &devInfo, sizeof(int)));

    // diagonalize
    CUSOLVER_CHECK_ERR(cusolverDnDsyevd(cusolver_H, 
                                        jobz, uplo, norb, 
                                        d_ham, norb, 
                                        d_eval, 
                                        d_work, lwork, 
                                        devInfo));

    // copy to d_evec 
    CUDA_CHECK_ERR(cudaMemcpy(d_evec, d_ham, norb*norb*sizeof(double), cudaMemcpyDeviceToDevice));

    // free memory
    CUSOLVER_CHECK_ERR(cusolverDnDestroy(cusolver_H));
    CUDA_CHECK_ERR(cudaFree(d_work)); 
    CUDA_CHECK_ERR(cudaFree(devInfo));
};


// Compute the occupation factors using Fermi-Dirac
// and store along diagonal of a diagonal matrix 
__global__ 
void computeOcc(double *eval,
                double *occ,
                const unsigned int norb, 
                double kbt,
                double mu)
{
    // get thread idx
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    while (i < norb*norb){

        
        if (i % (norb+1) == 0){

	    // calculate occupation using Fermi-Dirac, along diagonal
            occ[i] = 2.0*pow(exp((eval[i%norb] - mu)/kbt) + 1, -1); 
	
        }
        else{
            // fill in zeros off-diagonal
	    occ[i] = 0.0;
	}
	
        // advance i by the grid size
        i += blockDim.x*gridDim.x;
    }

};


void get_fermilevel(double* eval,
                    double* occ,
                    int norb, 
                    double kbt,
                    double bndfil,
                    double mu,
                    int nthds, 
                    int nblks)   // may need to add error flag      
{
    double nel, mu0, f1, f2, step;
    nel= bndfil*2.0*double(norb);
    mu0 = mu;
    step = 0.1;


    // wrap occ into a thrust device vector
    thrust::device_ptr<double> thrust_occ;
    thrust_occ = thrust::device_pointer_cast(occ);

    // compute occupation with guess for mu
    computeOcc<<<nblks, nthds>>>(eval, occ, norb, kbt, mu);

    // sum the eigenvalues after applying Fermi-Dirac
    f1 = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());

    // calculate error in mu
    f1 = f1 - nel;

    //
    mu = mu0 + step;

    f2 = 0.0;

    // compute occupation with updated mu
    computeOcc<<<nthds, nblks>>>(eval, occ, norb, kbt, mu);

    // sum the eigenvalues after applying Fermi-Dirac
    f2 = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());

    // calculate error in mu
    f2 = f2 - nel;
  
    // set mu0 to previous mu
    mu0 = mu;

    //if(abs(f2 - f1) < 1e-5){
      //err = .true.
      //return;
    //}

    mu = mu0 - f2*step/(f2-f1); // newton-raphson
    f1 = f2;
    step = mu - mu0;

    for (int m = 0; m < 101; m++){
      if (m == 100){
        printf("WARNING: norbewton-raphson is not converging ...");
        //err = .true.;
        mu = mu0;
        return;
      }

      // new sum of the occupations
      f2 = 0.0;
       
      // compute occupation with updated mu and Fermi-Dirac
      computeOcc<<<nblks, nthds>>>(eval, occ, norb, kbt, mu);

      // sum the occupation factors
      f2 = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());

      /*!$omp parallel do default(none) private(i) &
      !$omp shared(eigenvalues,ef,kbt,norb) &
      !$omp reduction(+:f2)
      do i=1,norb
        f2 = f2 +  2.0_dp*fermi(eigenvalues(i),ef,kbt)
      enddo
      !$omp end parallel do*/

      // update f2
      f2 = f2-nel;
      mu0 = mu;
      mu = mu0 - f2*step/(f2-f1);
      f1 = f2;
      step = mu - mu0;
      //if (abs(f1).lt.tol)then !tolerance control
      //  return
      //endif
    }
}



//Compute a density matrix from eigenvectors
void  compute_dm(double *occ, 
                 double *evec, 
                 double *dm,
                 const unsigned norb)
{
    // create handles
    cublasHandle_t handle;
    cublasCreate(&handle);	
    //cublasSetMathMode(handle, CUBLAS_GEMM_DEFAULT);

    // set gemm coeffs
    double a, b;
    a=1.0; b=0.0;

    // create occupation matrix
    double *occ_mat;
    CUDA_CHECK_ERR(cudaMalloc(&occ_mat, norb * norb * sizeof(double)));

    // evecs * occ_mat = occ_mat
    cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 norb, norb, norb,
                                 &a,
                                 evec, norb,
                                 occ_mat, norb,
                                 &b,
                                 occ_mat, norb);

    // occ_mat * evecs^T= d_dm
    cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 norb, norb, norb,
                                 &a,
                                 occ_mat, norb,
                                 evec, norb,
                                 &b,
                                 dm, norb); 

    cublasDestroy(handle); 
    CUDA_CHECK_ERR(cudaFree(occ_mat));
}

void diagonalize(double* ham, 
                 double* dm, 
                 double kbt,
                 double bndfil,
                 int norb,
                 int nocc)
{
    // kernel launch paramaters
    int nthds = 512;
    int nblks = int(ceil(float(norb*norb)/float(nthds))); 

    // declare vars
    double *eval, *evec, *occ, mu;
    double *d_ham, *d_eval, *d_evec;
    double *d_dm, *d_occ;

    // allocate host memory
    eval = (double*)malloc( norb * sizeof(double) );
    evec = (double*)malloc( norb * norb * sizeof(double) );
    occ =  (double*)malloc( norb * norb * sizeof(double) );

    // allocate device memory
    CUDA_CHECK_ERR(cudaMalloc(&d_ham,  norb * norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_dm,   norb * norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_evec, norb * norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_occ,  norb * norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_eval,     norb * sizeof(double)  ));
	
    // copy ham to device
    CUDA_CHECK_ERR(cudaMemcpy(d_ham, ham, norb * norb * sizeof(double), cudaMemcpyHostToDevice));	

    // call cusolver diag
    computeEval(d_ham, norb, d_eval, d_evec); 

    // copy evals,evecs to host
    //cudaMemcpy(eval, d_eval, norb*sizeof(double), cudaMemcpyDeviceToHost); 
    //cudaMemcpy(evec, d_evec, norb*norb*sizeof(double), cudaMemcpyDeviceToHost);

    // guess mu
    mu = 1.0;

    // compute fermi level, mu
    get_fermilevel(d_eval, d_occ, norb, kbt, bndfil, mu, nblks, nthds);
    
    // build density matrix
    compute_dm(d_occ, d_evec, d_dm, norb);
		
    // send dm back to host
    CUDA_CHECK_ERR(cudaMemcpy(dm, d_dm, norb * norb * sizeof(double), cudaMemcpyDeviceToHost));

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
