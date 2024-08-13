

typedef enum
{
    yes = 0, // do refinement
    no = 1   // do not do refinement
} refine_t;

typedef enum
{
    fp64 = 0,     // uniform double precision
    fp32 = 1,     // uniform single precision
    fp16_fp32 = 2 // accumulate in single, compute in half
} precision_t;
