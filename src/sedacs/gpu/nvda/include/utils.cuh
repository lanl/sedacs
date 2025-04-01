#include <cublas_v2.h>

void gershgorin_cheby(const unsigned,
                      const double *,
                      double *,
                      double *);

void buildTest_eval(const double *,
                    const int,
                    double *);

void buildTest(double *, const int);

double gtod(void);

void buildId_dev(double *, int);

void buildIdentity(double *, const int);

void build_SynthHam(double *, const int);

void cheby_coeffs_dev(const double, const double, const double,
                      const double,
                      const int,
                      const int, const int, const int,
                      double *);

void ps_cheby_coeffs_dev(double *,
                         double *,
                         const int, const int,
                         cublasHandle_t);

/*   // Determine cheby coefficients (code taken from progress)
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
*/
/*
__global__
void cheby_coeffs_GPU(const double emax, const double emin,
                  const double ef,
                  const double kbt,
                  const int npts,
                  const int K, const int M,
                  double *c_)
{

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    //const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

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
*/

/*void ps_coeffs_cheby_gpu(rocblas_handle handle,
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

/*
void construct_ps_coeffs_new(double *ps_c, double *c, const int K, const int M){

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
    double *U;
    U = (double*) malloc( K * M * K * M * sizeof(double) );

   // double berga=1.0;

    for (int i=0; i < K*M; i++){
        for (int j=0; j < K*M; j++){
            U[i + K*M*j] = X[i][j];
    };
    };

    // Solve c=U*ps_c
    // Call lapack, upper-triangular solve in fp64
    int N = K*M, err, nrhs=1;

    //dtrtrs_("U","N","N",&N, &nrhs, U, &N, c, &N, &err);
    // copy c to ps_c
    for (int i=0; i < K*M; i++){
    ps_c[i] = c[i];
    };

    // free memory
    free(U);
};
*/
