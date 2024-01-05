
/*typedef enum{
    yes       = 0,
    no        = 1   
} refine_t; 

typedef enum{
    fp64      = 0,
    fp32      = 1, 
    fp16_fp32 = 2   
} precision_t; 
*/

void dnnsp2(double*, 
            double*, 
            int, 
            int,
            precision_t,
            refine_t);

