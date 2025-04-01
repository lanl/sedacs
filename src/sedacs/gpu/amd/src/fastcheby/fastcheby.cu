#include <vector>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <iomanip>
#include <iostream>

#include <utils.cuh>
#include <cuda.h>
#include <cublas_v2.h>
#include <error_check.cuh>


/*
    Main Chebyshev expansion routine that computes density matrix
    using the fast cheby solver in Finkelstein et al. JCP 2023
*/

void pscheby(double *ham, double *dm, 
             int K, int M, 
             int N,
             int nocc,
             double kbt) 
{

    int ld = N;
    int thds = 512;
    int blks = (int) ceil(float(K * M) / float(thds));


    // Print info
    std::cout << "==================================" << "\n"
              << "Computing Chebychev expansion for: " << "\n"
              << "    System size = " << N << "\n"
              << "    Num expansion terms = " << K*M << "\n"
              << "==================================" << "\n "
    	      << std::endl;

    // determine device number
    int count;
    cudaGetDeviceCount(&count);
    int device=0;
    cudaSetDevice(device);

    // initialize cuda streams 
    int num_streams = std::max(K,M);
    cudaStream_t stream[num_streams];
    for (int i = 0 ; i < num_streams; i++){
        CUDA_CHECK_ERR(cudaStreamCreate(&stream[i]));
    };
    
    // define cublas handle
    cublasHandle_t handle;
    CUBLAS_CHECK_ERR(cublasCreate(&handle));

    // hip device properties
    //cudaDeviceProp_t props;
    //cudaGetDeviceProperties(&props, device);

    // Define and allocate Chebyshev polynomials
    // matrices, T_n, on device
    double *d_T[K+1];
    double *d_aux[M+1];
     
    for (int j=0; j <= K; j++){ 
        CUDA_CHECK_ERR(cudaMalloc(&d_T[j], ld * N * sizeof(double)));
    };
    
    for (int j=0; j <= M; j++){ 
        CUDA_CHECK_ERR(cudaMalloc(&d_aux[j], ld * N * sizeof(double)));
    };

    // declare device vars
    double *d_I,*d_dm,*d_temp;
    CUDA_CHECK_ERR(cudaMalloc(&d_I, N * N * sizeof(double)));
    CUDA_CHECK_ERR(cudaMalloc(&d_dm, N * N * sizeof(double)));
    CUDA_CHECK_ERR(cudaMalloc(&d_temp, N * N * sizeof(double)));

    
    // Estimate sprectral bounds
    double h1, hN;
    gershgorin_cheby(N, ham, &h1, &hN);
    std::cout << "h1 = " << h1 << ", hN = " << hN << std::endl;

    // Cheby params
    int npts = 1e4;

    // Compute Chebyshev coefficients    
    double *c, *d, *d_c;
    CUDA_CHECK_ERR(cudaMallocHost(&c, K * M * sizeof(double)));
    CUDA_CHECK_ERR(cudaMallocHost(&d, K * M * sizeof(double)));
    CUDA_CHECK_ERR(cudaMalloc(&d_c, K * M * sizeof(double)));
  
    cheby_coeffs_dev(hN, h1, 0.642348, kbt, 
                     npts,
                     K, M, N,
                     d_c);
    
    //CUDA_CHECK_ERR(cudaMemcpy(c,d_c, N * sizeof(double),cudaMemcpyDeviceToHost));

    ps_cheby_coeffs_dev(d, 
                        d_c, 
                        K, M,
                        handle);


    // initialize T0 = Id
    buildId_dev(d_T[0],N);

    // initialize T1 = H
    CUDA_CHECK_ERR(cudaMemcpy(d_T[1], ham, N * N * sizeof(double),cudaMemcpyHostToDevice));


    // Transform Hamiltonian
    // so that spectrum inside (-1,1)
    // (H - avg_eps*I)/delta_eps = H^~
    double delta_eps = hN-h1;
    double avg_eps = 0.5*(hN + h1);
    double alpha = -2*avg_eps/delta_eps;
    double beta = 2.0/delta_eps;

    // set stream before each cublas call, cannot pass
    // into call like with magma, active stream resides
    // within the handle
    CUBLAS_CHECK_ERR(cublasSetStream(handle, stream[0]));   
                                                  
    CUBLAS_CHECK_ERR(cublasDgeam(handle, 
                                 CUBLAS_OP_N, CUBLAS_OP_N, 
                                 N, N,
                                 &alpha, d_T[0], ld,
                                 &beta, d_T[1], ld,
                                 d_T[1], ld));


    //long int usec_ps_coeffs = t2 - t1;
    //std::cout << " Time for PS coeffs: time/s = " << (double)usec_ps_coeffs/1e6 << std::endl;

    
    /////////////////////////
    /////////////////////////
    /// COMPUTE THE T_N's ///
    /////////////////////////
    /////////////////////////
            
    int cnt;
    int i,j; 
    int k,kk;
    int cnt_ops = 0;
   
    long int usec_mult;
    long int usec_sum, usec_temp; 
    double stream_flops;

   
    int stages = ceil( log( (double) K ) / log(2.0) );
    std::cout << "number of stages needed = " << stages << std::endl;

    
    // time mults
    //t1 = gtod();

    // Loop over stages
    for (int i = 1; i <= stages ; i++){
	    
	//cnt = 0; 
    	if (i == 1){

 
            alpha = 1.0; beta = 0.0;
            // replace with tensor core
	    CUBLAS_CHECK_ERR(cublasDgemm(handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, N, N,
                                         &alpha,
                                         d_T[1], ld,
                                         d_T[1], ld,
                                         &beta,
                                         d_T[2], ld));

            alpha = -1.0; beta = 2.0;
	    CUBLAS_CHECK_ERR(cublasDgeam(handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, N,
	   	     	                 &alpha, 
    		    	                 d_T[0], ld, 
			                 &beta, 
			                 d_T[2], ld, 
			                 d_T[2], ld)); 


            // cnt_ops+=1;
	    //printf("Compute 2*(T_%d times T_%d)-T_%d = T_%d on queue %d \n", 1, 1, 0, 2, 0);
	    //printf("queue sync %d \n", 0);	    
	    //printf("================================== \n");

	}
        else{ 
		  
	    k  = pow(2,i-2);
	    kk = pow(2,i-1);

	    for (j = k+1; j <= pow(2,i-1); j++){
                if (j+k < K+1){

                    CUBLAS_CHECK_ERR(cublasSetStream(handle, stream[j-k-1]));   
	    
                    alpha = 1.0; beta = 0.0;
      	            CUBLAS_CHECK_ERR(cublasDgemm(handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 N, N, N,
                                                 &alpha,
                                                 d_T[j], ld,
                                                 d_T[k], ld,
                                                 &beta,
                                                 d_T[j+k], ld));

                    alpha = -1.0; beta = 2.0;
	            CUBLAS_CHECK_ERR(cublasDgeam(handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
	    	                                 N, N,
		                                 &alpha, 
		                                 d_T[abs(j-k)], ld, 
		                                 &beta, 
				                 d_T[j+k], ld,
				                 d_T[j+k], ld));
	
		    cnt_ops+=1;		

       		    //#ifdef DEBUG_ON
		    //printf("Compute 2*(T_%d times T_%d)-T_%d = T_%d on queue %d \n", j, k, abs(j-k), j+k, j-k-1);
		    //#endif
		};
	           	
	    };	   	
	
	    for (j = k+1; j <= pow(2,i-1); j++){
                if (j+kk < K+1){
		
                    CUBLAS_CHECK_ERR(cublasSetStream(handle, stream[j-1]));   
                    
 	            alpha = 1.0; beta = 0.0;
      	            CUBLAS_CHECK_ERR(cublasDgemm(handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 N, N, N,
                                                 &alpha,
                                                 d_T[j], ld,
                                                 d_T[kk], ld,
                                                 &beta,
                                                 d_T[j+kk], ld));

                    alpha = -1.0; beta = 2.0;
	            CUBLAS_CHECK_ERR(cublasDgeam(handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
	    	                                 N, N,
		                                 &alpha, 
		                                 d_T[abs(j-kk)], ld, 
		                                 &beta, 
				                 d_T[j+kk], ld,
				                 d_T[j+kk], ld));
	
		    cnt_ops+=1;

		    //printf("Compute 2*(T_%d times T_%d)-T_%d = T_%d on queue %d \n", j, kk, abs(j-kk), j+kk, j-1);
		};
	    };	
		
		
	    // sync streams for each stage
	    for (int i_s = 0; i_s < pow(2,i-1); i_s++){
            
	        CUDA_CHECK_ERR(cudaStreamSynchronize(stream[i_s]));

                //printf("queue sync %d \n", i_s);	    
	    };
       		
	    //printf("================================== \n");

        };

    }; 

    //t2 = gtod();
    //usec_mult = t2 - t1;
    //std::cout << " Time for mults: time/s = " << (double)usec_mult/1e6 << std::endl;


    ////////////////////////
    /// COMPUTE THE SUMS ///
    ////////////////////////

    // start timer
    //t1 = gtod();

    for (int k=0; k < K; k++)
    {    
        for (int j=0; j < M; j++)
        {

	    if(k==(K-1))
            {
                    
                CUBLAS_CHECK_ERR(cublasSetStream(handle, stream[j]));   
	         
                alpha = d[(K*M-1)-(j*K + k)];
                beta = 1.0;

                CUBLAS_CHECK_ERR(cublasDgeam(handle,
                                             CUBLAS_OP_N, CUBLAS_OP_N,
		                             N, N,
					     &alpha, 
					     d_T[0], ld, 
					     &beta, 
					     d_aux[j], ld, 
					     d_aux[j], ld)); 
		
            /*    std::cout << c[(K*M-1)-(j*K + k)] << " * I "  
			  << " + 1 * dev_daux[" << j << "]" 
			  << " = dev_daux[" << j << "]" 
			  << " on queue " << j << ". " 
			  << std::endl;
            */
	    }
            else
            {	
		    
                if (k==0)
                {
                    
                    CUBLAS_CHECK_ERR(cublasSetStream(handle, stream[j]));   

                    alpha = c[(K*M-1)-(j*K + k)];
                    beta = 0.0;
                    CUBLAS_CHECK_ERR(cublasDgeam(handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 N, N,
					         &alpha,
					         d_T[(K-1)-k], ld, 
					         &beta, 
                                                 d_aux[j], ld, 
                                                 d_aux[j], ld)); 
	 		
		/*    std::cout << c[(K*M-1)-(j*K + k)]  
			      << " * dev_T" << (K-1)-k
			      << " + 0 * dev_aux[" << j << "]" 
			      << " = dev_aux[" << j << "]" 
			      << " on queue " << j << ". " 
			      << std::endl;
                */
		}
                else
                {
                    CUBLAS_CHECK_ERR(cublasSetStream(handle, stream[j]));   
                        
                    alpha = d[(K*M-1)-(j*K + k)];
                    beta = 1.0;
                    CUBLAS_CHECK_ERR(cublasDgeam(handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 N, N,
					         &alpha, 
					         d_T[(K-1)-k], ld, 
					         &beta, 
                                                 d_aux[j], ld, 
                                                 d_aux[j], ld)); 

		/*    std::cout << c[(K*M-1)-(j*K + k)]  
			      << " * d_T" << (K-1)-k
			      << " + 1 * d_aux[" << j << "]" 
			      << " = d_aux[" << j << "]" 
			      << " on queue " << j << ". " 
		              << std::endl;
                */
		};

            };
		
	};

    };

    for (int i_s=0; i_s < M; i_s++){
	    
         CUDA_CHECK_ERR(cudaStreamSynchronize(stream[i_s]));

	 //std::cout << "cudaStreamSync(" << i_s << ")" << std::endl;
    };

    // final serial mults and sums on stream 0                     
    CUBLAS_CHECK_ERR(cublasSetStream(handle, stream[0]));   
 
    for (int j=0; j < M-1; j++){
    
        alpha=1.0; beta=1.0;
        CUBLAS_CHECK_ERR(cublasDgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N, N,
                                     &alpha,
                                     d_T[K], ld,
                                     d_aux[j], ld,
                                     &beta,
                                     d_aux[j+1], ld));

    		
        //#ifdef DEBUG_ON
	//    std::cout << "d_T_" << K << " * d_aux[" << j << "] + 1 * d_aux[" << j + 1 << "]" 
	//     	      << " = d_aux[" << j + 1 << "]"
	//              << " on queue " << 0 << ". " << std::endl;
	//#endif
    };
    
    // sync stream 0
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream[0]));

    //t2 = gtod();
    //usec_sum = t2 - t1;
    //std::cout << " Time for sums: time/s = " << (double)usec_sum/1e6 << std::endl;


    // copy DM back from device
    CUDA_CHECK_ERR(cudaMemcpy(dm,d_aux[M-1], N * N * sizeof(double),cudaMemcpyDeviceToHost));
};

    /*

 

    ////////////////////////
    ////////////////////////
    /// COMPUTE THE SUMS ///
    ////////////////////////
    ////////////////////////
    
    t1 = gtod();
    
  
    for (int k=0; k < K; k++){    
        for (int j=0; j < M; j++){
    
		if(k==(K-1)){
			magmablas_dgeadd2(n, n,
					  d[(K*M-1)-(j*K + k)], 
					  dev_T[0], ld, 
					  1.0, 
					  dev_aux[j], ld, 
					  queue[j]);
					  //queue[0]);
       			#ifdef DEBUG_ON
			std::cout << d[(K*M-1)-(j*K + k)] << " * I "  
				  << " + 1 * dev_daux[" << j << "]" 
				  << " = dev_daux[" << j << "]" 
				  << " on queue " << j << ". " 
				  //<< " on queue " << 0 << ". " 
				  << std::endl;
 			#endif

		}else{	
			if (k==0){
				magmablas_dgeadd2(n, n, 
					  d[(K*M-1)-(j*K + k)], 
					  dev_T[(K-1)-k], ld, 
					  0.0, dev_aux[j], 
					  ld, 
					  queue[j]);
					  //queue[0]);
                                #ifdef DEBUG_ON
	 		
				std::cout << d[(K*M-1)-(j*K + k)]  
				          << " * dev_T" << (K-1)-k
			                  << " + 0 * dev_aux[" << j << "]" 
				          << " = dev_aux[" << j << "]" 
				          << " on queue " << j << ". " 
				          //<< " on queue " << 0 << ". " 
				          << std::endl;
			        #endif
			}else{
				magmablas_dgeadd2(n, n, 
					  d[(K*M-1)-(j*K + k)], 
					  dev_T[(K-1)-k], ld, 
					  1.0, dev_aux[j], 
					  ld, 
					  queue[j]);
					  //queue[0]);
                                #ifdef DEBUG_ON
	 		
				std::cout << d[(K*M-1)-(j*K + k)]  
				          << " * dev_T" << (K-1)-k
			                  << " + 1 * dev_aux[" << j << "]" 
				          << " = dev_aux[" << j << "]" 
				          << " on queue " << j << ". " 
				          //<< " on queue " << 0 << ". " 
				          << std::endl;
			        #endif

			};


        	};
		
	};
    };
   


    for (int j=0; j < M; j++){
	    //magma_queue_sync(queue[j]);
	    magma_queue_sync(queue[0]);

	    #ifdef DEBUG_ON
	    //std::cout << "magma_queue_sync(" << j << ")" << std::endl;
	    std::cout << "magma_queue_sync(" << 0 << ")" << std::endl;
            #endif
    };


    for (int j=0; j < M-1; j++){
	    magma_dgemm(magma_trans, magma_trans, n, n, n, 1.0,
                  	    dev_T[K], ld, dev_aux[j], ld, 1.0, dev_aux[j+1], ld, queue[0]);
	    magma_queue_sync(queue[0]);
    		
	    #ifdef DEBUG_ON
	     	std::cout << "dev_T_" << K << " * dev_aux[" << j << "] + 1 * dev_aux[" << j + 1 << "]" 
		  	  << " = dev_aux[" << j + 1 << "]"
	                  << " on queue " << 0 << ". " << std::endl;
	    #endif
    };




    magma_queue_sync(queue[0]);
    
    t2 = gtod();
    usec_stream_add = t2 - t1;
    std::cout << "N = " << n
              << "M = " << M
              << "K = " << K
              << ": time/us for mults = " << (double)usec_stream_mult
              << ": time/us for additions = " << (double)usec_stream_add
              << ": time/us for all = " << (double)usec_stream_add + (double)usec_stream_mult
              << std::endl;

    magma_dgetmatrix(n, n, dev_aux[M-1], n, h_PS, ld, queue[0]);

    // Frobenius norm of difference between recursion and PS
    
    // Print cheby plot
    for (int i=0; i < n; i++){
        //printf("Ch: %.15f, %.15f, %.15f\n", eval[i], evall[i], h_PS[i+i*n]);
    }


    std::cout << "Num terms = " << K * M 
	      << ",  "   
              << "PS speed-up factor over bml diagonalization = " <<  (double)usec_stream_diag / 
   	                                                    ((double)usec_stream_add + (double)usec_stream_mult) 
              << std::endl;
   

    PS = bml_import_from_dense(dense
  		              ,matrix_precision
		  	      ,dense_column_major
		 	      ,n, n
			      ,h_PS
			      ,1.0
			      ,sequential);

    printf("============== PS ===============\n");
    bml_print_bml_matrix(PS,0,10,0,10);
    printf("================================\n");

    alpha = 1.0;
    beta = -1.0;
    bml_add(D, PS, alpha, beta,0.0);
    double fnormPSD = bml_fnorm(D);
    std::cout << K << ": Rel error = " << std::setprecision(15) << fnormPSD / fnormD << std::endl;


    ret = magma_free(dev_aux);
    ret = magma_free(dev_T);

    magma_finalize();*/



/*
int main()
{

    int n = 1000;
    int M = 15, K = 15;

    // get ham
    double *ham, *dm;
    HIP_API_CHECK(hipHostMalloc(&ham,n*n*sizeof(double)));
    HIP_API_CHECK(hipHostMalloc(&dm,n*n*sizeof(double)));

    density_mat(ham, dm, 
                n, 
                K, 
                M);

    return 0;
}

*/


