#include <openacc.h>

__global__ void doubleToFloat(double*,
                     float*,
                     int);

__global__ void floatToDouble(float*,
                     double*,
                     int);

__global__ void setToIdentityMatrix(float*, int);


void gershgorin(const unsigned,
                const double *,
                double *,
                double *);

//template <typename T> 
//extern void gershgorin_v2(const unsigned,
//                          const T *,
//                          T *,
//                          T *);
template<typename T> 
void gershgorin_v2(const unsigned N,
                   const T *X,
                   T *h1,
                   T *hN,
		   cudaStream_t stream)
{
    //int streamId = 23;
    //acc_set_cuda_stream(streamId, stream); 

    T minest, maxest;

    minest = 0.0;
    maxest = 0.0;

    #pragma acc parallel loop reduction(min : minest) reduction(max : maxest) deviceptr(X) 
    for (size_t i = 0; i < N; ++i)
    {
        float sum = 0;
        #pragma acc loop reduction(+ : sum)
        for (size_t j = 0; j < N; ++j)
        {
            sum += abs(X[i * N + j]); // assuming row major, running sum
        }

        minest = min(2 * X[i * N + i] - sum, minest); // sum always non-neg
        maxest = sum;                                 // sum always non-neg
    }
    h1[0] = minest;
    hN[0] = maxest;
};

void openacc_trace(float*, 
		   float*, 
		   const int);

cudaError_t GPUSTrace(const unsigned,
                      const float *,
                      float *); // Assumed to be on the device

cudaError_t GPUDTrace(const unsigned,
                      const double *,
                      double *); // Assumed to be on the device

cudaError_t GPUSTrace2(const unsigned,
                       const float *,
                       float *B); // Assumed to be on the device

cudaError_t computeSnp1(const unsigned,
                        const float *,
                        const float *,
                        const float *,
                        float *); // Assumed to be on the device

void computeSigma(unsigned Nocc, const float *TrXn, const float *TrX2n, float *Sig);

cudaError_t doRefinement(double *, double *, double *, const int, const int, cudaStream_t, cublasHandle_t);
