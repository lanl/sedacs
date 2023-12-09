#ifndef __TCORE_TOOLS__
#define __TCORE_TOOLS__

namespace tcoretools {

extern void tcoreSPGemmSymm(cublasHandle_t &handle
                           ,const unsigned N
                           ,const float* A
                           ,half*  Ah
                           ,half*  Al
                           ,float* B1
                           ,float* B2
                           ,float* B
                           ,cudaStream_t cuStrm=0);

extern void tcoreSPGemmSymm1(cublasHandle_t &handle
                           ,const unsigned N
                           ,const float* A
                           ,const float* B
                           ,half*  Ah
                           ,half*  Al
                           ,half*  Bh
                           ,half*  Bl
                           ,float* C1
                           ,float* C2
                           ,float* C
                           ,cudaStream_t cuStrm=0);

extern void tcoreSPGemmSymmS(cublasHandle_t &handle
                           ,const unsigned N
                           ,const float* A
                           ,half*  Ah
                           ,half*  Al
                           ,float* B1
                           ,float* B2
                           ,float* B
                           ,cudaStream_t cuStrm=0);

extern void tcoreSPGemmSP2iter(cublasHandle_t &handle
                              ,const unsigned N
                              ,const float* A
                              ,half*  Ah
                              ,half*  Al
                              ,float* B1
                              ,float* B2
                              ,float* B
                              ,cudaStream_t cuStrm=0);
};

#endif
