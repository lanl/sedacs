/*
  proxy_a_lib.c

  A prototype engine that:
     - Reads the total number of atoms 
     - Constructs a set of random coordinates 
     - Constructs a simple Hamiltonian 
     - Computes the Density matrix from the Hamiltonian

  Translated from C to Fortran by Michael E. Wall, LANL

*/

#include "proxy_a.h"
#include "lapack.h"

/*
  Simple random number generator
  This is important in order to compare across codes
  written in different languages.

  To initialize:
  \verbatim
    myRand = rand(123)
  \endverbatim
  where the argument of rand is the seed.

  To get a random number between "low" and "high":
  \verbatim
    rnd = myRand.get_rand(low,high)
  \endverbatim
*/

double proxy_rand(double low, double high, int seed,bool init)
{
  static int
    stat,
    a = 321,
    b = 231,
    c = 13;

  double
    w,
    rnd;

  int
    place;
    
  if (init) {
    stat = seed * 1000;
    rnd = 0.0;;
  }
  else {
    w = high - low;
    place = a * stat;
    place = (int)(place/b);
    rnd = ((double)(place % c))/((double)c);
    place = rnd * 1000000;
    stat = place;
    rnd = low + w*rnd;
  }
  
  return(rnd);
}

/*
  Generating random coordinates 
   @brief Creates a system of size "nats = Number of atoms" with coordindates having 
   a random (-1,1) displacement from a simple cubic lattice with parameter 2.0 Ang.
  
   @param nats The total number of atoms
   @return coordinates Position for every atom. z-coordinate of atom 1 = coords[0,2]
*/

int get_random_coordinates(int nats, double *coords)
{
  int
    *seedin,
    ssize,
    length,
    atomsCounter,
    i,
    j,
    k;

  double
    rnd,
    latticeParam;

  length = (int)(pow((double)nats,1./3.)) + 1;
  latticeParam = 2.0;
  atomsCounter = 0;
  rnd = proxy_rand(0.,0.,111,true); // set the random number seed
  for (i = 0; i < length; i++) {
    for (j = 0; j < length; j++) {
      for (k = 0; k < length; k++) {
	atomsCounter = atomsCounter + 1;
	size_t ofst = (atomsCounter - 1) * 3;
	if (atomsCounter > nats) break;
	rnd = proxy_rand(-1.,1.,0,false);
	coords[ofst + 0] = i * latticeParam + rnd;
	printf("%lg\n",coords[ofst + 0]);
	rnd = proxy_rand(-1.,1.,0,false);
	coords[ofst + 1] = j * latticeParam + rnd;
	rnd = proxy_rand(-1.,1.,0,false);
	coords[ofst + 2] = k * latticeParam + rnd;
      }
    }
  }
  return(0);
}

/*
  Computes a Hamiltonian based on a single "s-like" orbitals per atom.
  @author Anders Niklasson
  @brief Computes a hamiltonian \f$ H_{ij} = (x/m)\exp(-(y/n + decay_{min}) |R_{ij}|^2))\f$, based on distances
  \f$ R_{ij} \f$. \f$ x,m,y,n,decay_{min} \f$ are fixed parameters.
  
  @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
  @param types Index type for each atom in the system. Type for first atom = type[0] (not used yet)
  @return H 2D numpy array of Hamiltonian elements
  @param verb Verbosity. If True is passed, information is printed.
*/  

int get_hamiltonian(int nats, double *coords, int *atomTypes, double *H, bool verb)
{
  int
    *N,
    Nocc,
    m,
    n,
    hdim,
    i,
    j,
    k,
    cnt;

  double
    *xx,
    a,
    c,
    x,
    b,
    d,
    y,
    tmp,
    dvec[3],
    dist2,
    eps,
    decay_min;
  
  hdim = nats;
  Nocc = (int)((double)hdim/4.0);
  eps = 1.0e-9;
  decay_min = 0.1;
  m = 78;
  a = 3.817632; c = 0.816371; x = 1.029769; n = 13;
  b = 1.927947; d = 3.386142; y = 2.135545;
  if (H == NULL) {
    perror("get_hamiltonian requires allocated H\n");
    exit(1);
  }
  if (verb) printf("Constructing a simple Hamiltonian for the full system\n");
  cnt = 0;
  for (i = 0; i < hdim; i++) {
    size_t iofst = i * 3;
    x = fmod((a * x + c), (double)m);
    y = fmod((b * y + d), (double)n);
    for (j = i; j < hdim; j++) {
      size_t jofst = j * 3;
      dist2 = 0.0;
      for (k = 0; k < 3; k++) {
	dvec[k] = coords[iofst + k] - coords[jofst + k];
	dist2 += dvec[k]*dvec[k];
      }
      tmp = (x/(double)m) * exp(-(y/(double)n + decay_min)*dist2);
      H[i*nats + j] = tmp;
      H[j*nats + i] = tmp;
    }
  }
  return(0);
}
/*
  Computes the Density matrix from a given Hamiltonian.
  @author Anders Niklasson
  @brief This will create a "zero-temperature" Density matrix \f$ \rho \f$
  \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
  where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
  
  @param H Hamiltonian mtrix 
  @param Nocc Number of occupied orbitals
  @param verb Verbosity. If True is passed, information is printed.
  
  @return D Density matrix

*/

int get_densityMatrix(int nats, double *H, int Nocc, double *D, bool verb)
{
  double
    *Q,
    *E,
    mu,
    *work;

  int
    info,
    lwork,
    i,
    j,
    k,
    hdim,
    homoIndex,
    lumoIndex;

  char
    jobz = 'V',
    uplo = 'U';

  if (verb) printf("Computing the Density matrix");

  hdim = nats;
  lwork = 3*hdim - 1;

  Q = (double *)malloc(hdim*hdim*sizeof(double));
  work = (double *)malloc(lwork*sizeof(double));
  E = (double *)malloc(hdim*sizeof(double));
  memcpy(Q,H,hdim*hdim*sizeof(double));
  LAPACK_dsyev(&jobz,&uplo,&hdim,Q,&hdim,E,work,&lwork,&info);
  if (verb) {
    printf("Eigenvalues:\n");
    for (i = 0; i < hdim; i++) {
      printf("%g\n",E[i]);
    }
    homoIndex = Nocc;
    lumoIndex = Nocc + 1;
    mu = 0.5*(E[homoIndex] + E[lumoIndex]);
    memset(D,0,hdim*hdim*sizeof(double));
    for (i = 0; i < hdim; i++) {
      size_t iofst = i * hdim;
      for (j = 0; j < hdim; j++) {
	size_t jofst = j * hdim;
	for (k = 0; k < hdim; k++) {
	  if (E[k] < mu) {
	    D[iofst + j] = D[iofst + j] + Q[iofst + k]*Q[jofst + k];
	  }
	}
      }
    }
    if (verb) printf("Chemical potential = %g\n",mu);
    return(0);
  }
}
