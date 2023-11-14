#ifndef PROXY_A_H
#define PROXY_A_H

#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<string.h>
#include<math.h>

int get_random_coordinates(int nats, double *coords);
int get_hamiltonian(int nats, double *coords, int *atomTypes, double *H, bool verb);
int get_densityMatrix(int nats, double *H, int Nocc, double *D, bool verb);

#endif
