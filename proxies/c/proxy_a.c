#include "proxy_a.h"

int main(int argc, char *argv[])
{
  double
    *coords = NULL,
    *H = NULL,
    *D = NULL;
  
  int
    *types,
    nats,
    nocc,
    i,
    j;
    
  nats = 2;

  coords = (double *)malloc(3*nats*sizeof(double));
  get_random_coordinates(nats,coords);
  
  for (i=0; i < nats; i++) {
    printf("%30.18g\n",coords[3*i]);
  }
  types = (int *)malloc(nats*sizeof(int));
  for (i = 0; i < nats; i++) types[i] = 1;
  
  H = (double *)malloc(nats*nats*sizeof(double));
  get_hamiltonian(nats,coords, types, H, true);
  printf("Hamiltonian matrix\n");
  for (i = 0; i < nats; i++) {
    size_t ofst = nats * i;
    for (j = 0; j < nats; j++) {
      printf("%g\n",H[ofst + j]);
    }
  }

  D = (double *)malloc(nats*nats*sizeof(double));
  nocc = (int)((double)nats/2.0);
  get_densityMatrix(nats,H, nocc, D, true);
  printf("Density matrix:\n");
  for (i = 0; i < nats; i++) {
    size_t ofst = nats * i;
    for (j = 0; j < nats; j++) {
      printf("%g\n",D[ofst + j]);
    }
  }
}
