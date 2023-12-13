//#include<cheby.cuh>
//#include<sp2.cuh>
//#include<diag.cuh>
#include <stdio.h>
#include <lib.h>
#include <dnnsp2.cuh>
#include <diag.cuh>

void dm_dnnsp2(double* ham, double* dm, int n, int nocc){

    printf("berga\n");

    dnnsp2(ham, dm, n, nocc, fp32, yes);    

}


void dm_diag(double* ham, double* dm, int n, int nocc){

    printf("berga\n");

    diagonalize(ham, dm, n, nocc);    


}
