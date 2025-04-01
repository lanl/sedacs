void tcoreSPGemmSymm(cublasHandle_t, 
                     const unsigned,
                     const float*, 
                     half*,
                     half*,
                     float*,
                     float*,
                     float*,
		     cudaStream_t*);

void tcoreSPGemmSymm1(cublasHandle_t,
                      const unsigned,
                      const float*,
                      const float*,
                      half*,
                      half*,
                      half*,
                      half*,
                      float*,
                      float*,
                      float*);

