typedef enum
{
    yes,
    no
} refine_t;

typedef enum
{
    fp64,
    fp32,
    fp16_fp32
} precision_t;

void invovlp(double *,
             double *,
             int,
             precision_t,
             refine_t)
