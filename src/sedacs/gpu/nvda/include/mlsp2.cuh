typedef enum {
    full,
    none
} refine_t; 


typedef enum {
    fp64,
    fp32,
    fp16_fp32
} precision_t; 


void mlsp2(double*, 
           double*, 
           int, 
           int,
           precision_t,
           refine_t);

