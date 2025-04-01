#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <iostream>
//#include <boost/random/linear_congruential.hpp>
//#include <boost/random/uniform_real.hpp>
//#include <boost/random/variate_generator.hpp>
//#include <lapack.h>

#include <error_check.cuh>


void buildTest_eval(
    double* H, const int n, double* eval)
{

    for (int i = 0; i < n ; i++){
        for (int j = 0; j < n ; j++){
                if (i==j){
                        H[i + n * i] = eval[i];
                        std::cout << H[i + n * i] << std::endl;
                }else{
                        H[i + n * j] = 0;
                };

        };
    };
};

void buildTest_diag(
    double* H, const int n)
{

    for (int i = 0; i < n ; i++){
        for (int j = 0; j < n ; j++){
                if (i==j){
                        H[i + n * i] = -10.0+(double(i)/double(n))*20.0;
                        std::cout << H[i + n * i] << std::endl;
                }else{
                        H[i + n * j] = 0;
                };

        };
    };
};

void buildTest(
    double* Iden, const int n)
{

    for (int i = 0; i < n ; i++){
        for (int j = 0; j < n ; j++){
                if (i==j && i>0.5*n){
                        Iden[i + n * i] = 1;
                }else{
                        Iden[i + n * j] = 0;
                };

        };
    };
};

/*double gtod(void)
{
    struct timeval tv;
    gettimeofday(&tv, (struct timezone*)nullptr);
    return 1.e6 * tv.tv_sec + tv.tv_usec;
}
*/


/*
    Kernel to build identity matrix on device 
*/

__global__
void buildId_dev_kernel(double* Iden, int n)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (i < n*n){
        if (i%(n+i)==0){
	    Iden[i + n * i] = 1.0;
        }
        else{
	    Iden[i] = 0.0;
        };
    };
};
/*
    Kernel launcher to build identity matrix on device 
*/

void buildId_dev(double* Iden, 
                 const int n) 
{
    const int thds = 512;
    const int blks = (int) ceil(float(n) / float(thds));
     

    buildId_dev_kernel<<<blks,thds>>>(Iden, n);
    
    CUDA_CHECK_ERR(cudaDeviceSynchronize());

};



/*__global__ 
void cheby_coeffs_GPU(const double emax, const double emin, 
                      const double ef, 
                      const double kbt, 
                      const int npts, 
                      const int K, const int M, 
                      double *c_)
{

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ double berga[512];

    if (i < K * M){
        int ix = 0;
        double xj,x,Ti,fermi,Kr,mid_xj;
        double sum = 0.;
        double s;
        double dx = M_PI/(npts + 1);

        if (i == 0){
            Kr = double(npts+1) * 3.;
        }else{
            Kr = 0.5 * double(npts+1) * 3.;
        };

        while (ix < npts){

            // simpsons rule
            if (ix == 0){
                s = 1.0 ;
            }
            else if (ix%2 == 0 and ix < npts-1){
                s = 2.0;
            }
            else if (ix%2 == 1 and ix < npts-1 ){
                s = 4.0;
            }
            else if (ix == npts-1){
                s = 1.0;
            } 
            // <Ti|f> = \int_{-pi/2}^{pi/2 Ti(t)*f(t)dt
            // <Ti|f> = \int_{-1}^1 Ti(x)*f(x)/(1-x^2) dx
            xj = cos((ix+0.5)*dx);
            
            x = (emax-emin)*(xj + 1.0)/2.0 + emin;
    
            Ti = cos(i * acos(xj));
            fermi = 1.0/(1.0+exp((x-ef)/(kbt)));
   
            sum += Ti * fermi * s;
            ix += 1; 
        };


        berga[threadIdx.x] = sum / Kr;

        __syncthreads();
        c_[i] = berga[threadIdx.x];



    };

};
*/

void buildIdentity(
    double* Iden, const int n)
{

    for (int i = 0; i < n ; i++){
    	for (int j = 0; j < n ; j++){
		if (i==j){
			Iden[i + n * i] = 1.0;
		}else{
			Iden[i + n * j] = 0.0;
		};

    	};
    };
};

/*void build_SynthHam(
    double* H, const int n)
{

    typedef boost::minstd_rand rng_type;
    typedef boost::uniform_real<> distribution_type;

    int seed = 1739;
    rng_type rng(seed);
    distribution_type nd(-1., 1.);
    boost::variate_generator<rng_type, distribution_type> gen(rng, nd);

    // metal
    double rcoeff = 0.0; //gen();
    double ea = 0.0;
    double eb = 0.0;
    double decay = -0.01;
    // semiconductor
    //double rcoeff = 0.0; //gen();
    //double ea = 0.0;
    //double eb = 0.0;
    //double decay = -0.01;

    // synthetic Hamiltonian for a metal

    double dist = 0;

    // Compute the diagonal
    for (int i = 0; i < n ; i++){
        if (i%2 == 0){
            H[i+i*n] = ea + rcoeff*(2.0*gen() - 1.0);
	}else{
	    H[i+i*n] = ea + rcoeff*(2.0*gen() - 1.0);
	}
    }

    // metal
    double daiaj = -1.0;
    double dbibj = -1.0;
    double dab = 0.0;
    
    // semiconductor
    //double daiaj = -1.0;
    //double dbibj = 0.0;
    //double dab = -2.0;


    // Compute off-diagonal
    for (int i = 0; i < n; i++){
	for (int j = i+1; j < n; j++){
	    
	    if ( abs(double(i-j)) <= double(n)/2.0 ){
		dist = std::max(abs(double(i-j))-2.0, 0.0);
            }else{
		dist = std::max((-abs(double(i-j)) + double(n)) - 2.0, 0.0);
	    }

	    // A-A type
            if((i%2 != 0) and j%2 != 0){
                H[i+j*n] = (daiaj + rcoeff*(2.0*gen() - 1.0))*exp(decay*dist);
            // A-B type
            }else if( (i%2 == 0) and (j%2 == 0) ){
                H[i+j*n] = (dbibj + rcoeff*(2.0*gen() - 1.0))*exp(decay*dist);
            //B-B type
            }else{
                H[i+j*n] = (dab + rcoeff*(2.0*gen() - 1.0))*exp(decay*dist);
            }	    

            // Symmetrize
	    H[j+i*n] = H[i+j*n];
	}
    }

    
}*/
void cheby_coeffs(double emax, double emin, double ef, double kbt, int npts, int K, int M, double *c){


    // Determine cheby coefficients (code taken from progress)
    //double Int = 0.;
    //double xj,fermi,x,Kr,Ti;
    ////////////////////////
    ////////////////////////
    // CONSTRUCT VECTOR C //
    ////////////////////////
    ////////////////////////
   
    printf("Starting Cheby coeffs...\n");
    #pragma omp target map(from : c[:K * M])
    #pragma omp teams distribute

    for (int i=0; i < K * M ; i++){

        //Int = 0.0;
        double sum=0.0;    
        #pragma omp parallel for reduction(+:sum)
        for (int j=0; j < npts; j++){
    
            double xj = cos((j+0.5)*M_PI/(npts + 1));
            double x = (emax-emin)*(xj + 1.0)/2.0 + emin;
    
            // Compute integral
            double Ti = cos((double)i * acos(xj));
            double fermi = 1.0/(1.0+exp((x-ef)/(kbt)));
            //Int = Int + Ti * fermi;
            sum += Ti * fermi;
        };

	double Kr;
        if (i == 0){
                Kr = double(npts+1);
        }else{
                Kr = 0.5 * double(npts+1);
        };
        //c[i] = Int/Kr;
        c[i] = sum/Kr;
    };

    // start cheby timing
    printf("Cheby coeffs complete...\n");
};


/*
    Kernel for computing Chebyshev expansion coefficients.
*/

__global__ 
void cheby_coeffs_dev_kernel(const double emax, const double emin, 
	                     const double ef, 
	                     const double kbt, 
	                     const int npts, 
	                     const int K, const int M, 
	                     double *c_)
{

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ double berga[1024];

    if (i < K * M){
        int ix = 0;
        double xj,x,Ti,fermi,Kr,mid_xj;
        double sum = 0.;
        double s;
        double dx = M_PI/(npts + 1);

        if (i == 0){
            Kr = double(npts+1) * 3.;
        }else{
            Kr = 0.5 * double(npts+1) * 3.;
        };

        while (ix < npts){

            // simpsons rule
            if (ix == 0){
                s = 1.0 ;
            }
            else if (ix%2 == 0 and ix < npts-1){
                s = 2.0;
            }
            else if (ix%2 == 1 and ix < npts-1 ){
                s = 4.0;
            }
            else if (ix == npts-1){
                s = 1.0;
            } 
            // <Ti|f> = \int_{-pi/2}^{pi/2 Ti(t)*f(t)dt
            // <Ti|f> = \int_{-1}^1 Ti(x)*f(x)/(1-x^2) dx
            xj = cos((ix+0.5)*dx);
            
            x = (emax-emin)*(xj + 1.0)/2.0 + emin;
    
            Ti = cos(i * acos(xj));
            fermi = 1.0/(1.0+exp((x-ef)/(kbt)));
   
	    sum += Ti * fermi * s;
            ix += 1; 
        };


        berga[threadIdx.x] = sum / Kr;

        __syncthreads();
        c_[i] = berga[threadIdx.x];



    };

};

/*
    Kernel launcher for computing Chebyshev expansion coefficients.
*/
void cheby_coeffs_dev(const double emax, const double emin, 
	              const double ef, 
	              const double kbt, 
	              const int npts, 
	              const int K, const int M, const int N, 
	              double *_c)
{

    const int thds = 512;
    const int blks = (int) ceil(float(N) / float(thds));
     

    cheby_coeffs_dev_kernel<<<blks,thds>>>(emax, emin, 
	                                   ef, 
	                                   kbt, 
	                                   npts, 
	                                   K, M, 
	                                   _c);

    CUDA_CHECK_ERR(cudaDeviceSynchronize());

    printf("Cheby coeffs complete...\n");


}


void cheby_coeffs_wJackson(double emax, double emin, double ef, double kbt, int npts, int K, int M, double *c){


    // Determine cheby coefficients (code taken from progress)
    double Int = 0.;
    double xj,fermi,x,Kr,Ti;
    double jackson_coeff;
    ////////////////////////
    ////////////////////////
    // CONSTRUCT VECTOR C //
    ////////////////////////
    ////////////////////////
   
    printf("Starting Cheby coeffs...\n");

    for (int i=0; i < K * M ; i++){

        Int = 0.0;
    
        for (int j=0; j < npts; j++){
    
            xj = cos((j+0.5)*M_PI/(npts + 1));
            x = (emax-emin)*(xj + 1.0)/2.0 + emin;
    
            // Compute integral
            Ti = cos((double)i * acos(xj));
            fermi = 1.0/(1.0+exp((x-ef)/(kbt)));
            Int = Int + Ti * fermi;
        };

        if (i == 0){
                Kr = double(npts+1);
        }else{
                Kr = 0.5 * double(npts+1);
        };
        jackson_coeff =((K*M-i+1)*cos(M_PI*i/(K*M+1)) + sin(M_PI*i/(K*M+1))/tan(M_PI/(K*M+1)))/(K*M+1);  
        c[i] = Int/Kr;
        c[i] *= jackson_coeff;
    };

    // start cheby timing
    printf("Cheby coeffs complete...\n");
};



void cheby_coeffs_trap(double emax, double emin, double ef, double kbt, int npts, int K, int M, double *c){


    // Determine cheby coefficients (code taken from progress)
    double Int = 0.;
    double xj,fermi,x,Kr,Ti;

    ////////////////////////
    ////////////////////////
    // CONSTRUCT VECTOR C //
    ////////////////////////
    ////////////////////////
   
    printf("Starting Cheby coeffs...\n");

    //#pragma omp target map(from : c[:K * M])
    //#pragma omp teams distribute

    for (int i=0; i < K * M ; i++){

        Int = 0.0;
    
        //#pragma omp parallel for reduction(+:sum)
	for (int j=0; j <= npts; j++){
    
            xj = cos((j+0.5)*M_PI/(npts + 1));
            x = (emax-emin)*(xj + 1.0)/2.0 + emin;
    
            // Compute integral
            Ti = cos((double)i * acos(xj));
            fermi = 1.0/(1.0+exp((x-ef)/(kbt)));

            if(j==0 or j==npts){
                Int = Int + Ti * fermi; 
            } 
            else
            {
                Int = Int + 2 * Ti * fermi;
            };

        };

        if (i == 0){
                Kr = double(npts+1);
        }else{
                Kr = 0.5 * double(npts+1);
        };
         
        c[i] = Int/Kr/2.0;
    };

    // start cheby timing
    printf("Cheby coeffs complete...\n");


};
/*
void ps_coeffs_cheby_gpu(rocblas_handle handle, 
                         double emax, double emin,double ef,
                         double kbt,
                         const int npts, 
                         double *c, 
                         double *d_c, 
                         const int K, 
                         const int M)
{

    //input K, M: paterson-stockmeyer size parameters
    //input c: standard chebyshev expansion coeffs
    //input ps_c: paterson-stockmeyer chebyshev expansion coeffs

    // construct regular Cheby coeffs
    int thds = 1024;
    int blks = (int) ceil(float(K * M) / float(thds));
    cheby_coeffs_GPU<<<blks,thds>>>(emax, emin, ef, kbt, npts, K, M, d_c);
    HIP_API_CHECK(hipMemcpy(c,d_c, K*M*sizeof(double),hipMemcpyDeviceToHost));
    
    std::cout << c[5] << std::endl;
    // declare rocblas params
    rocblas_status  rocb_stat;
    rocblas_fill up = rocblas_fill_upper;
    rocblas_diagonal diag = rocblas_diagonal_non_unit;
    rocblas_side left = rocblas_side_left; 
    rocblas_operation no_trans = rocblas_operation_none;
    rocblas_int lda = K*M;
    rocblas_int ldb = K*M;

    // init A and X matrices
    double A[M][M];
    double X[K*M][K*M];

    // zero everything out
    for (int i=0; i < M; i++){
        for (int j=0; j < M; j++){
	    A[i][j]=0.0; 
	};
    };
    for (int i=0; i < K*M; i++){
        for (int j=0; j < K*M; j++){
	    X[i][j]=0.0; 
	};
    };
     

    // follow recursion in Eq 10 
    for (int j = 0; j < M; j++){
        for (int l = 0; l < M; l++){
            if (j==0 and l==0){
                A[j][l] = 1.;
	    }
	    else if (j==1 and l==0){
                A[j][l]=0.;
	    }
	    else if (j==1 and l==1){
                A[j][l]=1.;
	    }
	    else if (j < l){
                A[j][l]=0.;
	    }
	    else{
                if (l == 0){
                    A[j][l]=A[j-1][1]/2;
	        }
	        else if (l == 1){
                    A[j][l]=A[j-1][0]+A[j-1][2]/2;
	        }
	        else if (1 < j and l < j){
                    A[j][l]=(A[j-1][l-1]+A[j-1][l+1])/2;
	        }
	        else if (l==j){
                    A[j][l]=A[j-1][l-1]/2;
	        };
	    };
	};
    };

    
    // Build block matrix

    int idx_i, idx_j, rev_idx_i, rev_idx_j;

    // follow Eq 20
    for (int block_i = M-1; block_i > -1; block_i--){            // go backwards, start filling from lower-left corner of X 
        for (int block_j = M-1; block_j > block_i-1; block_j--){ // go backwards, enforce upper-triangularity 

	    
            for (int i = 0; i < K; i++){
                for (int j = 0; j < K; j++){

                    // get forward index for I blocks
                    idx_i=i+(block_i)*K;
                    idx_j=j+(block_j)*K;

                    // backward index for J blocks
                    rev_idx_i=(K-1-i)+(block_i)*K;
                    rev_idx_j=(K-1-j)+(block_j)*K;
                
                    // if block_i and block_j have same parity
                    if ((block_i%2) == (block_j%2)){

                        // fill the diagonal
                        if (i == j){

                            // scale block by appropriate entry of A
                            // if in first block row, do not divide I blocks by 2 
                            if (block_i == 0){
                                X[idx_i][idx_j]=A[block_j][block_i];
			    }
                            else{
                                X[idx_i][idx_j]=0.5*A[block_j][block_i]; 
			    };
            
                            // if (0,0) position, multiply by 2 (these are the scalars)
                            if (i == 0 and j == 0){
                                X[idx_i][idx_j] = A[block_j][block_i];
		            }

			};
		    };
		    //if block_i and block_j have different parity
                    if ((block_i%2) != (block_j%2)){
                            
   	                // fill the anti-diagonal
                        if ( (K-1-i) == (K-1-j-1) ){

                            // scale block by appropriate entry of A
                            X[idx_i][rev_idx_j] = 0.5*A[block_j][block_i+1];
			};

		    };

	        };
	    };
        };
    };


    // Solve upper-triangular linear problem using back-substitution
    int N = K*M;
    
    double *XX;
    XX = (double*) malloc( N * N * sizeof(double) );

    for (int i=0; i < N; i++){
        for (int j=0; j < N; j++){
    	    XX[i + N*j] = X[i][j];
	};
    };
   
    double *d_X;
    HIP_API_CHECK(hipMalloc(&d_X, N * N * sizeof(double)));
    HIP_API_CHECK(hipMemcpy(d_X, XX, N * N * sizeof(double)
                               ,hipMemcpyHostToDevice));


    // Solve c=U*ps_c
    // Call rocblas, upper-triangular solve in fp64
    // and write ps_c over c
    double alpha = 1.0;  

    rocb_stat = rocblas_dtrsm(handle, left,  up, no_trans,
                              diag,
                              N, N,
                              &alpha,
                              d_X, N,
                              d_c, N);
    
                 
    // bring back PS coeffs to host
    HIP_API_CHECK(hipMemcpy(c,d_c, K*M*sizeof(double),hipMemcpyDeviceToHost));

    std::cout << c[5] << std::endl;

    HIP_API_CHECK(hipFree(d_X));
    free(XX);
    //free(X);
    //free(A);
    
};
*/



void ps_cheby_coeffs_dev(double *ps_c, double *c, const int K, const int M, cublasHandle_t handle){

    //input K, M: paterson-stockmeyer size parameters
    //input c: standard chebyshev expansion coeffs
    //input ps_c: paterson-stockmeyer chebyshev expansion coeffs

    double A[M][M];
    double X[K*M][K*M];


    // zero everything out
    for (int i=0; i < M; i++){
        for (int j=0; j < M; j++){
	    A[i][j]=0.0; 
	};
    };
    for (int i=0; i < K*M; i++){
        for (int j=0; j < K*M; j++){
	    X[i][j]=0.0; 
	};
    };
     

    // follow recursion in Eq 10 
    for (int j = 0; j < M; j++){
        for (int l = 0; l < M; l++){
            if (j==0 and l==0){
                A[j][l] = 1.;
	    }
	    else if (j==1 and l==0){
                A[j][l]=0.;
	    }
	    else if (j==1 and l==1){
                A[j][l]=1.;
	    }
	    else if (j < l){
                A[j][l]=0.;
	    }
	    else{
                if (l == 0){
                    A[j][l]=A[j-1][1]/2;
	        }
	        else if (l == 1){
                    A[j][l]=A[j-1][0]+A[j-1][2]/2;
	        }
	        else if (1 < j and l < j){
                    A[j][l]=(A[j-1][l-1]+A[j-1][l+1])/2;
	        }
	        else if (l==j){
                    A[j][l]=A[j-1][l-1]/2;
	        };
	    };
	};
    };

    
    // Build block matrix

    int idx_i, idx_j, rev_idx_i, rev_idx_j;

    // follow Eq 20
    for (int block_i = M-1; block_i > -1; block_i--){            // go backwards, start filling from lower-left corner of X 
        for (int block_j = M-1; block_j > block_i-1; block_j--){ // go backwards, enforce upper-triangularity 

	    
            for (int i = 0; i < K; i++){
                for (int j = 0; j < K; j++){

                    // get forward index for I blocks
                    idx_i=i+(block_i)*K;
                    idx_j=j+(block_j)*K;

                    // backward index for J blocks
                    rev_idx_i=(K-1-i)+(block_i)*K;
                    rev_idx_j=(K-1-j)+(block_j)*K;
                
                    // if block_i and block_j have same parity
                    if ((block_i%2) == (block_j%2)){

                        // fill the diagonal
                        if (i == j){

                            // scale block by appropriate entry of A
                            // if in first block row, do not divide I blocks by 2 
                            if (block_i == 0){
                                X[idx_i][idx_j]=A[block_j][block_i];
			    }
                            else{
                                X[idx_i][idx_j]=0.5*A[block_j][block_i]; 
			    };
            
                            // if (0,0) position, multiply by 2 (these are the scalars)
                            if (i == 0 and j == 0){
                                X[idx_i][idx_j] = A[block_j][block_i];
		            }

			};
		    };
		    //if block_i and block_j have different parity
                    if ((block_i%2) != (block_j%2)){
                            
   	                // fill the anti-diagonal
                        if ( (K-1-i) == (K-1-j-1) ){

                            // scale block by appropriate entry of A
                            X[idx_i][rev_idx_j] = 0.5*A[block_j][block_i+1];
			};

		    };

	        };
	    };
        };
    };


    // Solve upper-triangular linear problem using back-substitution
    double *U, *d_U;
    U = (double*) malloc( K * M * K * M * sizeof(double) );

    for (int i=0; i < K*M; i++){
        for (int j=0; j < K*M; j++){
    	    U[i + K*M*j] = X[i][j];
	};
    };
   
    // Solve c=U*ps_c
    // Call lapack, upper-triangular solve in fp64
    int N = K*M, err, nrhs=1; 
    double one = 1.0;

    CUDA_CHECK_ERR(cudaMalloc(&d_U, N * N * sizeof(double)));
    CUDA_CHECK_ERR(cudaMemcpy(d_U, U, N * N * sizeof(double), cudaMemcpyHostToDevice));

    // lapack routine
    //dtrtrs_("U","N","N",&N, &nrhs, U, &N, c, &N, &err);
    CUBLAS_CHECK_ERR(cublasDtrsm(handle,
                                 CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                 N, 1,
                                 &one,
                                 d_U, N,
                                 c, N));
    
    CUDA_CHECK_ERR(cudaMemcpy(ps_c, c, N * sizeof(double), cudaMemcpyDeviceToHost));

    // copy c to ps_c
    for (int i=0; i < K*M; i++){
        std::cout << ps_c[i] << std::endl;
    };
    
    // free memory
    free(U);
};


void gershgorin_cheby(const unsigned N,
                const double *X, 
                double *h1,
                double *hN)
{
    float sum, diag_elem, minest, maxest;

    for (size_t i = 0; i < N; ++i)
    {   
        sum = 0.0; minest = 0.0; 
        diag_elem = 0.0; maxest = 0.0;
        for (size_t j = 0; j < N; ++j)
        {   
            if (i != j)
            {   
                sum += abs(X[i * N + j]);  // assuming row major, running sum
            }   
            else
            {   
                diag_elem = X[i * N + i]; 
            }   
    
        }   
    
        minest = diag_elem - sum; //sum always non-neg
        maxest = diag_elem + sum; //sum always non-neg

        if (minest < h1[0])
        {   
            h1[0] = minest;
        }   
        if (hN[0]< maxest)
        {   
            hN[0] = maxest;
        }   
    }   
};


