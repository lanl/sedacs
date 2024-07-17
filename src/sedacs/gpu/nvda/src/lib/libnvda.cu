#include <stdio.h>
#include <lib.h>
#include <structs.h>
#include <diag.cuh>
#include <dnnsp2.cuh>
#include <pscheby.cuh>


void dm_dnnsp2(double* ham, double* dm, int norb, int nocc){

    printf("DNN SP2\n");
   
    precision_t u = fp16_fp32;    
 
    refine_t r = yes;


    dnnsp2(ham, dm, norb, nocc, u, r);    

}

void dm_mlsp2(double* ham, double* dm, int norb, int nocc){

    printf("ML SP2\n");
   
    precision_t u = fp16_fp32;    
 
    refine_t r = yes;


    dnnsp2(ham, dm, norb, nocc, u, r);    

}



void dm_pscheby(double* ham, double* dm, int norb, int nocc, double kbt){


    double bndfil = 0.666666;

    precision_t u = fp64;    
    
    refine_t r = yes;

    int K = 32; int M = 32;

    pscheby(ham, dm, K, M, norb, nocc, kbt);    


}


void involap(double* overlap, double* guess, double* factor, int norb){


    //precision_t u = fp64;    
 
    //refine_t r = yes;

    //invOlapFactorize(ham, dm, kbt, bndfil, u, norb, nocc);   


}



void dm_diag(double* ham, double* dm, double kbt, int norb, int nocc){


    double bndfil = 0.666666;

    precision_t u = fp64;    
 
    refine_t r = yes;

    diagonalize(ham, dm, kbt, bndfil, u, norb, nocc);   

}
