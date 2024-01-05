#include <stdio.h>
#include <lib.h>
#include <structs.h>
#include <dnnsp2.cuh>
#include <diag.cuh>

void dm_dnnsp2(double* ham, double* dm, int n, int nocc){

    printf("berga\n");
   
    precision_t u = fp32;    
 
    refine_t r = yes;


    //dnnsp2(ham, dm, n, nocc, u, r);    

}


void dm_diag(double* ham, double* dm, double kbt, int norb, int nocc){

    printf("berga\n");

    double bndfil = 0.25;

    precision_t u = fp32;    
 
    refine_t r = yes;

    diagonalize(ham, dm, kbt, bndfil, u, norb, nocc);   


}
