void gershgorin(const unsigned,
                const double*, 
                double*,
                double*);

void gershgorin_v2(const unsigned,
                   const double*, 
                   double*,
                   double*);



cudaError_t GPUSTrace(const unsigned,
                      const float*,
                      float*);        // Assumed to be on the device

cudaError_t GPUDTrace(const unsigned,
                      const double*,   
                      double* );      // Assumed to be on the device

cudaError_t GPUSTrace2(const unsigned,
                       const float*,
                       float* B);     // Assumed to be on the device

cudaError_t computeSnp1(const unsigned,
                        const float*,
                        const float*,
                        const float*,
                        float*);      // Assumed to be on the device

void computeSigma(unsigned Nocc
                        ,const float* TrXn
                        ,const float* TrX2n
                        ,float* Sig);

cudaError_t doRefinement(double* 
                        ,double* 
                        ,const int                         
                        ,const int 
                        ,cublasHandle_t);


