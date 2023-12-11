#define CUDA_CHECK_ERROR(val) check((val), #val, __FILE__, __LINE__)


void remap_ghost(int *
                ,const int *
                ,const int
                ,const int);

float compute_avgCoord(const int, const int *);

int apply_PBC(float *, float *, float *
              ,const float , const float, const float 
              ,const float , const float, const int 
              ,int *, const int, int);

template <typename T>
void check(T ,const char* const , const char* const ,
           const int);

