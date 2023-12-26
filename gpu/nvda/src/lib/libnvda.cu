#include <stdio.h>
#include <dnnsp2.cuh>
#include <diag.cuh>
#include <lib.h>

void dm_dnnsp2(double* ham, double* dm, int n, int nocc){

    printf("berga\n");


    //dnnsp2(ham, dm, n, nocc, fp32, yes);    

}


void dm_diag(double* ham, double* dm, double kbt, int norb, int nocc){

    printf("berga\n");

    double bndfil = 0.5;

    diagonalize(ham, dm, kbt, bndfil, norb, nocc);    


}
