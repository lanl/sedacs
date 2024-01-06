#include <iostream>
#include <string>
#include <math.h>
#include <cuda.h>
#include <error_check.cuh>
#include <cusolverDn.h>
#include <thrust/device_vector.h> 
#include <thrust/host_vector.h> 
#include <structs.h>

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


// Fill the diagonal of a square matrix
__global__ 
void fill_diagonal(double *mat,
                   const double *diag,
                   const int n)
{
    // get thread idx
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    //while (i < n*n){
    if (i < n*n){
        
        if (i % (n+1) == 0){

	    // along diagonal
            mat[i] = diag[i%n]; 
        
        }
        else{

            // fill in zeros off-diagonal
	    mat[i] = 0.0;

	}
	
        // advance i by the grid size
        //i += blockDim.x*gridDim.x;
    }

};

// Compute the occupation error as
// as a function of the ham eigenvalues
__global__ 
void computeOcc(double *eval,
                double *occ,
                const unsigned int norb, 
                double kbt,
                double mu)
{
    // get thread idx
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i < norb){
        // calculate occupation using Fermi-Dirac, along diagonal
        occ[i] = 2.0/(exp((eval[i] - mu)/kbt) + 1); 
    }

};

// Compute derivative wrt mu of occupation error
__global__ 
void compute_dOcc_dmu(double *eval,
                      double *docc_dmu,
                      const unsigned int norb, 
                      double kbt,
                      double mu)
{
    // get thread idx
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i < norb){
    
        // calculate occupation using Fermi-Dirac, along diagonal
        docc_dmu[i] = -2.0*exp((eval[i] - mu)/kbt)/pow((exp((eval[i] - mu)/kbt) + 1),2)/kbt; 
    }

};

void get_fermilevel_bisection(double* eval,
                              double* occ,
                              int norb, 
                              double kbt,
                              double bndfil,
                              double* mu,
                              int nthds, 
                              int nblks)   // may need to add error flag      
{
    double mu_a, mu_b, f;
    double err = 1.0;  
    double nel = bndfil*2.0*double(norb);
    mu_a = -40.;
    mu_b = 10.;

    // need to implement gershgorin circle
    // in order to get initial mu_a and mu_b

    // wrap occ into a thrust device vector
    thrust::device_ptr<double> thrust_occ;
    thrust_occ = thrust::device_pointer_cast(occ);


    while( abs(err) > 1e-6 ){
    //for (int i = 0; i<20;i++){

        // take new mu to be average of old ones
        mu[0] = (mu_b+mu_a)/2;
   
        // compute occupation with guess for mu
        computeOcc<<<nblks, nthds>>>(eval, occ, norb, kbt, mu[0]);

        // sum the occ factors
        f = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());
    
        // calculate error in sum of occupations
        err = f - nel;
         
        // halve the interval [mu_a,mu_b]
        if ( err < 0. ){

            // make mu new left endpoint
            mu_a=mu[0];

        }
        else if ( 0. < err ) {
            
            // make mu new right endpoint
            mu_b=mu[0];

        }

    }
}


void get_fermilevel_newton(double* eval,
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


    // need to implement gershgorin circle

    // wrap occ into a thrust device vector
    thrust::device_ptr<double> thrust_occ;
    thrust_occ = thrust::device_pointer_cast(occ);

    double err=0.;
    while (err<1e-5){

    // compute occupation with guess for mu
    computeOcc<<<nblks, nthds>>>(eval, occ, norb, kbt, mu);

    // sum the occ factors
    f1 = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());
    
    // calculate error in sum of occupations
    f1 = f1 - nel;
    printf("f1-nel=%.15f\n",f1);
    

    // compute occupation with guess for mu
    compute_dOcc_dmu<<<nblks, nthds>>>(eval, occ, norb, kbt, mu);

    f2 = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());


    //if(abs(f2 - f1) < 1e-5){
      //err = .true.
      //return;
    //}
    mu0=mu;
    std::cout << "mu prev = " << mu0 << std::endl;
    mu = mu0 - f1/f2; // newton-raphson, mu = mu - f(mu)/f'(mu)
    std::cout << "mu update = " << mu << std::endl;

    err = abs(mu0-mu);

    //printf("mu=%.15f = %.15f - %.15f * %.15f / (%.15f-%.15f)\n",mu,mu0,f2,step,f2,f1);

    std::cout << "mu = " <<  mu << std::endl;
    }
    for (int m = 0; m < 101; m++){
      if (m == 100){
        printf("WARNING: Newton-raphson is not converging ...");
        //err = .true.;
        mu = mu0;
        return;
      }

      // compute occupation with updated mu and Fermi-Dirac
      computeOcc<<<nblks, nthds>>>(eval, occ, norb, kbt, mu);

      // sum the occupation factors
      f1 = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());

      // update f2 sample point
      f1 = f1-nel;
      std::cout << "occ err = " << mu << std::endl;
 
      // compute occupation with guess for mu
      compute_dOcc_dmu<<<nblks, nthds>>>(eval, occ, norb, kbt, mu);

      f2 = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());

      mu0 = mu;
      // newton step
      mu = mu0 - f1/f2;
      std::cout << "mu update = " << mu << std::endl;

      //step = mu - mu0;
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

    // set gemm coeffs
    double a, b;
    a=1.0; b=0.0;

    // create occupation matrix
    double *occ_mat;
    CUDA_CHECK_ERR(cudaMalloc(&occ_mat, norb * norb * sizeof(double)));

    // fill diagonal
    fill_diagonal<<<int(ceil(float(norb*norb)/512.)),512>>>(occ_mat, occ, norb);

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
                 precision_t prec,
                 int norb,
                 int nocc)
{
    // kernel launch paramaters
    int nthds = 512;
    int nblks = int(ceil(float(norb*norb)/float(nthds))); 

    int nthds2 = 512;
    int nblks2 = int(ceil(float(norb)/float(nthds))); 

    // declare vars
    double *eval, *evec, *occ, *mu;
    double *d_ham, *d_eval, *d_evec;
    double *d_dm, *d_occ;

    // allocate host memory
    mu   = (double*)malloc( sizeof(double) );
    eval = (double*)malloc( norb * sizeof(double) );
    evec = (double*)malloc( norb * norb * sizeof(double) );
    //occ  = (double*)malloc( norb * norb * sizeof(double) );

    // allocate device memory
    CUDA_CHECK_ERR(cudaMalloc(&d_ham,  norb * norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_dm,   norb * norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_evec, norb * norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_occ,  norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_eval, norb * sizeof(double)  ));
	
    // copy ham to device
    CUDA_CHECK_ERR(cudaMemcpy(d_ham, ham, norb * norb * sizeof(double), cudaMemcpyHostToDevice));	

    // call cusolver diag
    computeEval(d_ham, norb, d_eval, d_evec); 

    // compute fermi level, mu
    get_fermilevel_bisection(d_eval, d_occ, norb, kbt, bndfil, mu, nblks2, nthds2);
    
    // build density matrix
    compute_dm(d_occ, d_evec, d_dm, norb);
		
    // send dm back to host
    CUDA_CHECK_ERR(cudaMemcpy(dm, d_dm, norb * norb * sizeof(double), cudaMemcpyDeviceToHost));

    /*for (int i=0;i<10;i++){
    for (int j=0;j<10;j++){
  
        printf("%.5f  ",dm[i*norb+j]);
    }
    printf("\n");;
    }*/
   

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
