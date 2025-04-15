#include <iostream>
#include <string>
#include <math.h>
#include <structs.h>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <error_check.cuh>
#include <cusolverDn.h>
//#include "nvToolsExt.h"

// diagonalize a matrix
void computeEval(double *d_ham, int norb,
                 double *d_eval,
                 double *d_evec)
{
    nvtxRangePushA("create handle");
    // cusolver handles
    cusolverDnHandle_t cusolver_H;
    CUSOLVER_CHECK_ERR(cusolverDnCreate(&cusolver_H));
    nvtxRangePop();

    // specify cusolver diag flags

    // CUSOLVER_EIG_MODE_NOVECTOR if no eigvec needed
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    // lower triangle contains elementry reflectors related to diag algorithm
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int lwork = 0;

    // determine amount of temp space needed
    nvtxRangePushA("buffer calc");
    CUSOLVER_CHECK_ERR(cusolverDnDsyevd_bufferSize(cusolver_H,
                                                   jobz, uplo, norb,
                                                   d_ham, norb,
                                                   d_eval, &lwork));

    nvtxRangePop();
    // allocate temp work vars
    double *d_work = NULL;
    int *devInfo = NULL;
    CUDA_CHECK_ERR(cudaMalloc((void **)&d_work, sizeof(double) * lwork));
    CUDA_CHECK_ERR(cudaMalloc((void **)&devInfo, sizeof(int)));

    // diagonalize
    nvtxRangePushA("diagonalize");
    CUSOLVER_CHECK_ERR(cusolverDnDsyevd(cusolver_H,
                                        jobz, uplo, norb,
                                        d_ham, norb,
                                        d_eval,
                                        d_work, lwork,
                                        devInfo));
    nvtxRangePop();

    // copy to d_evec
    CUDA_CHECK_ERR(cudaMemcpy(d_evec, d_ham, norb * norb * sizeof(double), cudaMemcpyDeviceToDevice));

    // free memory
    nvtxRangePushA("cusolver mem destroy");

    CUSOLVER_CHECK_ERR(cusolverDnDestroy(cusolver_H));
    CUDA_CHECK_ERR(cudaFree(d_work));
    CUDA_CHECK_ERR(cudaFree(devInfo));
    nvtxRangePop();
};

// Fill the diagonal of a square matrix
__global__ void fill_diagonal(double *mat,
                              const double *diag,
                              const int n)
{
    // get thread idx
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // while (i < n*n){
    if (i < n * n)
    {

        if (i % (n + 1) == 0)
        {

            // along diagonal
            mat[i] = diag[i % n];
        }
        else
        {

            // fill in zeros off-diagonal
            mat[i] = 0.0;
        }

        // advance i by the grid size
        // i += blockDim.x*gridDim.x;
    }
};

// Compute the occupation error as
// as a function of the ham eigenvalues
__global__ void computeOcc(double *eval,
                           double *occ,
                           const unsigned int norb,
                           double kbt,
                           double mu)
{
    // get thread idx
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < norb)
    {

        // calculate occupation, along diagonal

        if (kbt == 0.0)
        {
            // Heaviside step
            if ((eval[i] < mu) or (eval[i] == mu))
            {
                occ[i] = 2.0;
            }
            else if (eval[i] > mu)
            {
                occ[i] = 0.0;
            }
        }
        else
        {
            // Fermi-Dirac
            occ[i] = 2.0 / (exp((eval[i] - mu) / kbt) + 1);
        };
    }
};

// Compute derivative wrt mu of occupation error
__global__ void compute_dOcc_dmu(double *eval,
                                 double *docc_dmu,
                                 const unsigned int norb,
                                 double kbt,
                                 double mu)
{
    // get thread idx
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < norb)
    {

        // calculate occupation using Fermi-Dirac, along diagonal
        docc_dmu[i] = -2.0 * exp((eval[i] - mu) / kbt) / pow((exp((eval[i] - mu) / kbt) + 1), 2) / kbt;
    }
};

/*
    Determine chemical potential to use for building
    density matrix using diagonalization.
*/
void get_fermilevel_bisection(double *h_eval,
                              double *eval,
                              double *occ,
                              int norb,
                              double kbt,
                              double bndfil,
                              double *mu,
                              int nthds,
                              int nblks) // may need to add error flag
{
    double mu_a, mu_b, f;
    double err = 1.0;
    double nel = bndfil * 2.0 * double(norb);

    // copy extremal eignavlues to host
    CUDA_CHECK_ERR(cudaMemcpy(h_eval, eval, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(h_eval + (norb - 1), eval + (norb - 1), sizeof(double), cudaMemcpyDeviceToHost));

    // set these to mu_a and mu_b
    mu_a = h_eval[0];
    mu_b = h_eval[norb - 1];

    // wrap occ into a thrust device vector for reductions
    thrust::device_ptr<double> thrust_occ;
    thrust_occ = thrust::device_pointer_cast(occ);

    int iter = 0;
    int Max = 50;

    while ((abs(err) > 1e-6) and (iter < Max))
    {

        // take new mu to be average of old ones
        mu[0] = (mu_b + mu_a) / 2;

        // compute occupation with guess for mu
        nvtxRangePushA("Occuption");
        computeOcc<<<nblks, nthds>>>(eval, occ, norb, kbt, mu[0]);
        nvtxRangePop();
        // sum the occ factors
        f = thrust::reduce(thrust_occ, thrust_occ + norb, 0.0, thrust::plus<double>());

        // calculate error in sum of occupations
        err = f - nel;
        // std::cout << err<< std::endl;

        // halve the interval [mu_a,mu_b]
        if (err < 0.)
        {

            // make mu new left endpoint
            mu_a = mu[0];
        }
        else if (0. < err)
        {

            // make mu new right endpoint
            mu_b = mu[0];
        }
        // std::cout << "mu = " << mu[0] << std::endl;

        iter += 1;
    }
}

/*
    Compute a density matrix from eigenvectors
*/
void compute_dm_from_eig(double *occ,
                         double *evec,
                         double *dm,
                         const unsigned norb)
{
    // create handles
    cublasHandle_t handle;
    cublasCreate(&handle);

    // set gemm coeffs
    double two, one, zero;
    two = 2.0, one = 1.0;
    zero = 0.0;

    // create occupation matrix
    double *occ_mat;
    CUDA_CHECK_ERR(cudaMalloc(&occ_mat, norb * norb * sizeof(double)));

    // fill diagonal
    fill_diagonal<<<int(ceil(float(norb * norb) / 512.)), 512>>>(occ_mat, occ, norb);

    // evecs * occ_mat = occ_mat
    CUBLAS_CHECK_ERR(cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 norb, norb, norb,
                                 &one,
                                 evec, norb,
                                 occ_mat, norb,
                                 &zero,
                                 occ_mat, norb));

    // occ_mat * evecs^T= d_dm
    CUBLAS_CHECK_ERR(cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 norb, norb, norb,
                                 &two,
                                 occ_mat, norb,
                                 evec, norb,
                                 &zero,
                                 dm, norb));

    cublasDestroy(handle);
    CUDA_CHECK_ERR(cudaFree(occ_mat));
}

void diagonalize(double *d_ham,
                 double *d_dm,
                 double kbt,
                 double bndfil,
                 precision_t prec,
                 int norb,
                 int nocc)
{

    // Create cuda timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    cudaEventRecord(start, 0);
    // kernel launch paramaters
    int nthds = 512;
    int nblks = int(ceil(float(norb * norb) / float(nthds)));

    int nthds2 = 512;
    int nblks2 = int(ceil(float(norb) / float(nthds)));

    // declare vars
    double *eval, *evec, *occ, *mu;
    double *d_eval, *d_evec;
    double *d_occ;

    // nvtxRangePushA("Register host memory");
    // cudaHostRegister ( ham, N * N * sizeof(double), cudaHostRegisterDefault);
    // cudaHostRegister ( dm, norb * norb * sizeof(double), cudaHostRegisterDefault);
    // nvtxRangePop();

    // allocate host memory
    mu = (double *)malloc(sizeof(double));
    eval = (double *)malloc(norb * sizeof(double));
    evec = (double *)malloc(norb * norb * sizeof(double));
    // CUDA_CHECK_ERR(cudaMallocHost(&h_dm,     norb * norb * sizeof(double)));

    // allocate device memory
    // CUDA_CHECK_ERR(cudaMalloc(&d_ham,  norb * norb * sizeof(double)  ));
    // CUDA_CHECK_ERR(cudaMalloc(&d_dm,   norb * norb * sizeof(double)  ));
    CUDA_CHECK_ERR(cudaMalloc(&d_evec, norb * norb * sizeof(double)));
    CUDA_CHECK_ERR(cudaMalloc(&d_occ, norb * sizeof(double)));
    CUDA_CHECK_ERR(cudaMalloc(&d_eval, norb * sizeof(double)));

    // copy ham to device
    // CUDA_CHECK_ERR(cudaMemcpy(d_ham, ham, norb * norb * sizeof(double), cudaMemcpyHostToDevice));

    // do cusolver diag
    nvtxRangePushA("cusolver");
    computeEval(d_ham, norb, d_eval, d_evec);
    nvtxRangePop();

    // compute fermi level, mu
    nvtxRangePushA("calculate mu");
    get_fermilevel_bisection(eval, d_eval, d_occ, norb, kbt, bndfil, mu, nblks2, nthds2);
    nvtxRangePop();

    // build density matrix
    nvtxRangePushA("compute dm");
    compute_dm_from_eig(d_occ, d_evec, d_dm, norb);
    nvtxRangePop();

    // send dm back to host
    // CUDA_CHECK_ERR(cudaMemcpy(h_dm, d_dm, norb * norb * sizeof(double), cudaMemcpyDeviceToHost));

    // nvtxRangePushA("HtoH dm copy");
    //  copy cpu buffer to python allocated dm (avoids python memory issues, pagability?)
    // memcpy(dm,h_dm, norb * norb * sizeof(double));
    // nvtxRangePop();

    // free memory
    free(eval);
    free(evec);
    free(occ);
    // CUDA_CHECK_ERR(cudaFree(d_ham));
    CUDA_CHECK_ERR(cudaFree(d_evec));
    CUDA_CHECK_ERR(cudaFree(d_eval));
    CUDA_CHECK_ERR(cudaFree(d_occ));
    // CUDA_CHECK_ERR(cudaFreeHost(h_dm));
    // CUDA_CHECK_ERR(cudaFree(d_dm));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time for building DM with diag = " << elapsedTime << " ms " << std::endl;
}
