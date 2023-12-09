#ifndef __LINALG_TOOLS__
#define __LINALG_TOOLS__

namespace linalgtools {

extern float M_Trace(const unsigned N,
                     const float* A);

extern cudaError_t GPUSTrace(const unsigned N
                             ,const float* A
                             ,float* B
                             ,cudaStream_t cuStrm=0);

extern cudaError_t GPUDTrace(const unsigned N
                             ,const double* A
                             ,double* B
                             ,cudaStream_t cuStrm=0);

extern cudaError_t GPUSTrace2(const unsigned N
                             ,const float* A
                             ,float* B
                             ,cudaStream_t cuStrm=0);


extern cudaError_t computeS0np1(const unsigned N
                               ,const float* Sig
                               ,const float* A
                               ,const float* B
                               ,float* C // Assumed to be on the device
                               ,cudaStream_t cuStrm=0);

extern cudaError_t computeS1np1(const unsigned N
                               ,const float* Sig
                               ,const float* A
                               ,const float* B
                               ,float* C // Assumed to be on the device
                               ,cudaStream_t cuStrm=0);

extern cudaError_t computeSigma(unsigned Nocc
                               ,const float* TrXn
                               ,const float* TrX2n
                               ,float* Sig
                               ,cudaStream_t cuStrm=0);

extern cudaError_t computeSigma_double(unsigned Nocc
                               ,const double* TrXn
                               ,const double* TrX2n
                               ,double* Sig
                               ,cudaStream_t cuStrm=0);

extern cudaError_t computeEigs(float* d_A
             		      ,int N
			      ,float* Eigs
			      ,cudaStream_t cuStrm=0);

extern cudaError_t doRefinement(double* _dA
                               ,double* D0_
                               ,const int _N
                               ,const int _Nocc
                               ,cublasHandle_t handle
                               ,cudaStream_t cuStrm=0);

extern cudaError_t doRefinement_1stOrder(double* _dA0
                               ,double* _dA1
                               ,double* D1_
                               ,const int _N
                               ,const int _Nocc
                               ,cublasHandle_t handle
                               ,cudaStream_t cuStrm=0);
}

#endif
