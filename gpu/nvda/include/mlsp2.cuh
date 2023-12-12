typedef enum {
    full,
    none
} refine_t; 


typedef enum {
    fp64,
    fp32,
    fp16_fp32
} precision_t; 


void dnnsp2(double*, 
            double*, 
            size_t, 
            size_t,
            precision_t,
            refine_t);

