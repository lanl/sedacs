#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

//Print a matrix
void printMat(const unsigned N, double *H)
{
	using std::cout;
	using std::endl;
	for (int i = 0; i < N; i++)
        {       for (int j = 0; j < N; j++)
                {
			cout << H[i*N+j] << " ";
		}
	cout << endl;
	}	
}

//Build a Hamiltonian for test
void buildH0(const unsigned N, double *H)
{	//input dimension of the matrix in one direction
	//input pointer to the Hamiltonian array that is 1D
	//fill in the elements of this flattened matrix array
	for (int i = 0; i < N; i++)
	{	for (int j = i; j < N; j++)
		{
			H[i+j*N] = exp(-0.5f*abs(double(i-j))) * sin((double)(i+1)); // index fills columns and also symmetric elements
			H[j+i*N] = H[i+j*N];  //Hamiltonian is symmetric and Hermitian
		}
	}
};
//Read a matrix from a file
void readH0(int N, double *X) {

   std::string line;
   std::ifstream f;

   f.open ("matrix.dat");

   double val;

   int ii,jj,rows,cols,nonz;

   getline(f,line);
   f >> cols;
   f >> rows;
   f >> nonz;
   
   for (int i=0; i < nonz; i++){
	f >> ii;
        f >> jj;
        f >> val;
        X[(ii-1)+N*(jj-1)] = val;
   };
};

//Diagonalize a Hamiltonian
cudaError_t computeEigvals(double *d_H, int N, double *d_eigval, double *d_eigvec)//, cudaStream_t cuStrm)
{
	//Cusolver handles
	cusolverDnHandle_t cusolver_H;
	cusolverDnCreate(&cusolver_H);
	
	//int device = 0;
	//cudaSetDevice(device);
	//std::cout<< "eigs device = " << device << std::endl;
	
	double *temp_mat; //So that the solver doesn't overwrite the matrix with the eigenvecs
	cudaMalloc(&temp_mat, N*N*sizeof(double));
	cudaMemcpy(temp_mat, d_H, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
    	//Specify all the flags and parameters that go into the cusolver eigh function
    	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; //CUSOLVER_EIG_MODE_NOVECTOR; this if no eigvec needed
    	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;  //lowe triangle of A contains elementry reflectors related to diagonlize algorithm
    	int lwork = 0;
    	cusolverDnDsyevd_bufferSize(cusolver_H, jobz, uplo, N, temp_mat, N, d_eigval, &lwork); //specifies the amount of space needed
    	double *d_work = NULL;
    	cudaMalloc((void **) &d_work, sizeof(double)*lwork);
    	int *devInfo = NULL;
    	cudaMalloc((void **) &devInfo, sizeof(int));
	//Call the diagonalization function
    	cusolverDnDsyevd(cusolver_H, jobz, uplo, N, temp_mat, N, d_eigval, d_work, lwork, devInfo);
	cudaMemcpy(d_eigvec, temp_mat, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
    	//destroy cusolver handle
    	cusolverDnDestroy(cusolver_H);
	cudaFree(d_work); cudaFree(devInfo); cudaFree(temp_mat);
    	return cudaPeekAtLastError();
};

//Compute the occupation factors using Fermi Dirac equation and eigenvals
__global__ void computeOcc(const unsigned N, double *beta, double *mu, double *eigval, double *occ)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	while (i < N*N)
	{
		if (i % (N+1) == 0)
		{
			occ[i] = pow(exp(beta[0]*(eigval[i%N] - mu[0])) + 1, -1); //Fermi Dirac function
		}else
		{
			occ[i] = 0.0f;
		}
		i += blockDim.x*gridDim.x;
	}	
};

//Compute a density matrix from eigenvectors
cudaError_t computeDmat(const unsigned N, double *occ, double *eigvec, double *Dmat)
{
	//Create handles
	cublasHandle_t handle;
	cublasCreate(&handle);	
	cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	double alpha_dbl, beta_dbl;
	alpha_dbl=1.0; beta_dbl=0.0;
	double *occ_evec;
	cudaMalloc(&occ_evec, N*N*sizeof(occ_evec));
	cublasStat = cublasDgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,  //np.matmul(np.diag(evalR), evecR.T) reverse order to make it row major
                                N, N, N,
                                &alpha_dbl,
                                eigvec, N,
                                occ, N,
                                &beta_dbl,
                                occ_evec, N);
	cublasStat = cublasDgemm(handle,
                    		CUBLAS_OP_N, CUBLAS_OP_T, //np.matmul(evecR, np.matmul(np.diawqg(evalR), evecR.T))
                             	N, N, N,
                             	&alpha_dbl,
                             	eigvec, N,
                             	occ_evec, N,
                             	&beta_dbl,
                             	Dmat, N); 
	cublasDestroy(handle); cudaFree(occ_evec);
	return cudaPeekAtLastError();
}


//Main functions to run
int main(int argc, char *argv[])
{
	//Matrix size
	size_t N = atoi(argv[1]);
	size_t Nocc = atoi(argv[2]);
	int num_threads = atoi(argv[3]);
	std::string readTrue = argv[4];
	int num_blocks = N*N/num_threads + 1; 

	//Declare the pointers for each items needed
	double *H, *eigval, *eigvec, *Dmat, *occ, *beta, *mu;
	double *d_H, *H_buf, *d_eigval, *d_eigvec, *d_Dmat, *d_occ, *d_beta, *d_mu;

	//allocate host memory
	H = (double*)malloc(N*N*sizeof(double));
	eigval = (double*)malloc(N*sizeof(double));
	eigvec = (double*)malloc(N*N*sizeof(double));
	Dmat = (double*)malloc(N*N*sizeof(double));
	occ = (double*)malloc(N*N*sizeof(double)); //diagonal matrix
	beta = (double*)malloc(sizeof(double));
	mu = (double*)malloc(sizeof(double));
	//Setting random values for testing now
	beta[0] = 0.5f;
	mu[0] = 0.1f;

	//Check to build the hamiltonian or read it
	if (readTrue == "yes")
	{
		readH0(N, H);
	}else
	{	
		buildH0(N, H);
	}
	
	printf("Printing matrix H0 ... \n");
	printMat(N, H);
	printf("========================\n\n");
	
	//Allocate device memory
	cudaMalloc(&d_H, N*N*sizeof(double));
	cudaMalloc(&d_Dmat, N*N*sizeof(double));
	cudaMalloc(&d_eigvec, N*N*sizeof(double));
	cudaMalloc(&d_eigval, N*sizeof(double));
	cudaMalloc(&d_occ, N*N*sizeof(double));
	cudaMalloc(&d_beta, sizeof(double));
	cudaMalloc(&d_mu, sizeof(double));
	cudaMallocManaged(&H_buf, N*N*sizeof(double));
	
	//Copy host to device
	cudaMemcpy(d_H, H, N*N*sizeof(double), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_beta, beta, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mu, mu, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(H_buf, d_H, N*N*sizeof(double), cudaMemcpyDeviceToDevice);

	//Call the diagonlization func
	computeEigvals(H_buf, N, d_eigval, d_eigvec); 
        cudaMemcpy(eigval, d_eigval, N*sizeof(double), cudaMemcpyDeviceToHost); //copy eigenvals to host from device
	cudaMemcpy(eigvec, d_eigvec, N*N*sizeof(double), cudaMemcpyDeviceToHost);
		
	//---Debug------
	printf("Printing Eigenvals and Eigenvecs ...\n");
	//print eigenvals
    	for (int i = 0; i < N; i++)
    		printf("Eigenval %d is %f \n", i, eigval[i]);
	printf("\n");
	printf("Eigenvectors ...\n");
	printMat(N, eigvec);
	printf("========================\n\n");
	//---Debug------

	//Make density matrices
	printf("Making density matrix ...\n");
	computeOcc<<<num_blocks, num_threads>>>(N, d_beta, d_mu, d_eigval, d_occ);
	cudaMemcpy(occ, d_occ, N*N*sizeof(double), cudaMemcpyDeviceToHost);
 	printf("Occupation factors ...\n");
        for (int i = 0; i < N*N; i++)
	{	if (i % (N+1) == 0)
                {
			printf("Occupation at %d is %f \n", i, occ[i]);
		}
	}
	printf("\n");	
	computeDmat(N, d_occ, d_eigvec, d_Dmat);
	cudaMemcpy(Dmat, d_Dmat, N*N*sizeof(double), cudaMemcpyDeviceToHost);
	printf("Printing density matrix ...\n");
	printMat(N, Dmat);
	printf("========================\n\n");
		
	//Free memory
	free(H); free(eigval); free(eigvec); free(beta); free(occ); free(Dmat); free(mu);
       	cudaFree(d_H); cudaFree(H_buf); cudaFree(d_eigvec); cudaFree(d_eigval); 
	cudaFree(d_beta); cudaFree(d_mu); cudaFree(d_occ); cudaFree(d_Dmat);	
	return 0;
}
