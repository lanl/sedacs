"""proxy code a
A prototype engine that:
    - Reads the total number of atoms
    - Constructs a set of random coordinates
    - Constructs a simple Hamiltonian
    - Computes the Density matrix from the Hamiltonian
"""

import os
import sys

import numpy as np
import scipy.linalg as sp
import sedacs.driver
import sedacs.interface_modules
from sedacs.dev.io import src_path

try:
    import ctypes

    # import gpulibInterface as gpu

    gpuLib = True
    arch = "nvda"
    pwd = os.getcwd()

    if arch == "nvda":
        print("loading nvidia...")
        lib = ctypes.CDLL(str((src_path() / "gpu/nvda/libnvda.so").absolute()))
    if arch == "amd":
        lib = ctypes.CDLL(str((src_path() / "gpu/amd/libamd.so").absolute()))

except:
    gpuLib = False


__all__ = [
    "RandomNumberGenerator",
    "get_random_coordinates",
    "get_hamiltonian_proxy",
    "get_density_matrix_proxy",
    "get_density_matrix_gpu",
]


## Simple random number generator
# This is important in order to compare across codes
# written in different languages.
#
# To initialize:
# \verbatim
#   myRand = rand(123)
# \endverbatim
# where the argument of rand is the seed.
#
# To get a random number between "low" and "high":
# \verbatim
#   rnd = myRand.get_rand(low,high)
# \endverbatim
#
class RandomNumberGenerator:
    """To generate random numbers."""

    def __init__(self, seed):
        self.a = 321
        self.b = 231
        self.c = 13
        self.seed = seed
        self.status = seed * 1000

    def generate(self, low, high):
        """Get a random real number in between low and high."""
        w = high - low
        place = self.a * self.status
        place = int(place / self.b)
        rand = (place % self.c) / self.c
        place = int(rand * 1000000)
        self.status = place
        rand = low + w * rand

        return rand


## Generating random coordinates
# @brief Creates a system of size "nats = Number of atoms" with coordindates having
# a random (-1,1) displacement from a simple cubic lattice with parameter 2.0 Ang.
#
# @param nats The total number of atoms
# @return coordinates Position for every atom. z-coordinate of atom 1 = coords[0,2]
#
def get_random_coordinates(nats):
    """Get random coordinates"""
    length = int(nats ** (1 / 3)) + 1
    coords = np.zeros((nats, 3))
    latticeParam = 2.0
    atomsCounter = -1
    myrand = RandomNumberGenerator(111)
    for i in range(length):
        for j in range(length):
            for k in range(length):
                atomsCounter = atomsCounter + 1
                if atomsCounter >= nats:
                    break
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 0] = i * latticeParam + rnd
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 1] = j * latticeParam + rnd
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 2] = k * latticeParam + rnd
    return coords


## Computes a Hamiltonian based on a single "s-like" orbitals per atom.
# @author Anders Niklasson
# @brief Computes a hamiltonian \f$ H_{ij} = (x/m)\exp(-(y/n + decay_{min}) |R_{ij}|^2))\f$, based on distances
# \f$ R_{ij} \f$. \f$ x,m,y,n,decay_{min} \f$ are fixed parameters.
#
# @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
# @param types Index type for each atom in the system. Type for first atom = type[0] (not used yet)
# @return ham 2D numpy array of Hamiltonian elements
# @param verb Verbosity. If True is passed, information is printed
#
def get_hamiltonian_proxy(coords, atomTypes, symbols, verb=False):
    """Construct simple toy s-Hamiltonian"""


    #This is mimicking what a code will have internally
    symbols_internal = np.array([ "Bl" ,
          "H" ,          "He" ,
          "Li" ,         "Be" ,   "B" ,          "C" ,          "N" ,          "O" ,      "F" ,                  \
          ], dtype=str)
    numel_internal = np.zeros(len(symbols_internal),dtype=int)
    numel_internal[:] = 0,   \
          1 ,       2 ,  \
          1 ,       2 ,   3 ,        4 ,       5 ,    6 ,   7 ,
  
    bas_per_atom = np.zeros(len(symbols_internal),dtype=int)
    bas_per_atom[:] = 0,   \
          1 ,       1 ,\
          4 ,       4,   4,        4 ,       4,       4,    4,
  
  
    nats  = len(coords[:,0])
    numel = 0

    # Map symbols to indices in symbols_internal
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols_internal)}
  
    # Translate `symbols` to `symbols_internal` indices
    mapped_indices = np.array([symbol_to_index[symbol] for symbol in symbols])
  
    # Convert atomTypes to `symbols_internal` indices
    atom_internal_indices = mapped_indices[atomTypes]
  
    # Sum the corresponding values in bas_per_atom and numel_internal
    norbs = np.sum(bas_per_atom[atom_internal_indices])
    numel = np.sum(numel_internal[atom_internal_indices])


    nocc = int(numel/2.0)
    eps = 1e-9
    decay_min = 0.1
    m = 78
    a = 3.817632
    c = 0.816371
    x = 1.029769
    n = 13
    b = 1.927947
    d = 3.386142
    y = 2.135545
    ham = np.zeros((norbs, norbs))
    if verb:
        print("Constructing a simple Hamiltonian for the full system")
    colsh = 0
    rowsh = 0
    for i in range(0, nats):
        for ii in range(bas_per_atom[atom_internal_indices[i]]):
            x = (a * x + c) % m  # Hamiltonian parameters
            y = (b * y + d) % n
            colsh = 0
            for j in range(i, nats):
                for jj in range(bas_per_atom[atom_internal_indices[j]]):
                    dist = np.linalg.norm(coords[i, :] - coords[j, :])
                    tmp = (x / m) * np.exp(-(y / n + decay_min) * (dist**2))
                    print("cr",i,ii,j,jj,colsh,rowsh,norbs,nats)
                    ham[rowsh, colsh] = tmp
                    ham[colsh, rowsh] = tmp
                    colsh = colsh + 1
            rowsh = rowsh + 1
    return ham


sedacs.driver.get_hamiltonian = get_hamiltonian_proxy

## Computes the Density matrix from a given Hamiltonian.
# @author Anders Niklasson
# @brief This will create a "zero-temperature" Density matrix \f$ \rho \f$
# \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
# where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
#
# @param ham Hamiltonian matrix
# @param nocc Number of occupied orbitals
# @param verb Verbosity. If True is passed, information is printed.
#
# @return rho Density matrix
#
def get_density_matrix_proxy(ham, nocc, verb=False):
    """Calcualted the full density matrix from H"""
    if verb:
        print("Computing the Density matrix")
    E, Q = sp.eigh(ham)
    norbs = len(ham[:, 0])
    homoIndex = nocc - 1
    lumoIndex = nocc
    mu = 0.5 * (E[homoIndex] + E[lumoIndex])
    rho = np.zeros((norbs, norbs))
    if verb:
        print("Eigenvalues of H:", E)
    for i in range(norbs):
        if E[i] < mu:
            rho = rho + np.outer(Q[:, i], Q[:, i])
    if verb:
        print("Chemical potential = ", mu)
    return rho


sedacs.driver.get_density_matrix = get_density_matrix_proxy



def get_density_matrix_T(H, Nocc, Tel, mu0, coreSize, core_ham_dim, S=None, verb=False):
  
  kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
  if(verb): print("Computing the renormalized Density matrix")

  if S is not None:
    E_val,Q = scipy.linalg.eigh(H) ### need S? not ones with S $$$
  else:
    E_val,Q = np.linalg.eigh(H)
  N = len(H[:,0])

  #print('Q\n', Q[:,0])

  homoIndex = Nocc - 1
  lumoIndex = Nocc
  print('HOMO, LUMO:', E_val[homoIndex], E_val[lumoIndex])
  mu_test = 0.5*(E_val[homoIndex] + E_val[lumoIndex]) #don't need it 
  print(N, Nocc,)
  print('!!!! mu test:\n', mu_test)

  # use mu0 as a guess

  OccErr = 1.0
  beta = 1./(kB*Tel)
  f = np.array([])
  for i in range(N):
     f_i = 1/(np.exp(beta*(E_val[i] - mu0)) + 1) # save fi to f
     f = np.append(f,f_i)
     #Occ = Occ + f_i*E_occ[i,k]


  D = sum(np.outer(Q[:, i],Q[:, i]*f[i]) for i in range(Nocc))*2
  #np.savetxt('co2_32_dm.txt',D)


  # rho = Q@f_vector@Q.T
  # or
  # rho_ij = SUM_k Q_ik * f_kk * Q_jk


  print('core_ham_dim', core_ham_dim)
  dVals = np.array([])
  for i in range(N):
    dVals = np.append(dVals, np.inner(Q[:core_ham_dim,i],Q[:core_ham_dim, i]))

  return D, E_val, dVals



## Computes the Density matrix from a given Hamiltonian.
# @author Josh Finkelstein
# @brief This will create a "zero-temperature" Density matrix \f$ \rho \f$
# \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
# where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
# using GPU/AI accelerator library
#
# @param H Hamiltonian matrix
# @param Nocc Number of occupied orbitals
# @param verb Verbosity. If True is passed, information is printed.
#
# @return D Density matrix
#
def get_density_matrix_gpu(H, N, Nocc, lib, verb=False):
    """Calcualted the full density matrix from H"""
    if verb:
        print("Computing the Density matrix using GPU/AI accel library")

    # init DM
    D = np.zeros((N, N))
    kbt = 0.1

    # get DM from cusolver diag
    # dm = gpu.dmDNNSP2(H,D,N,Nocc,lib)
    # dm = gpu.dmCheby(H,D,N,Nocc,kbt,lib)
    print("Density matrix=", D)
    # dm = gpu.dmDiag(H,D,N,Nocc,kbt,lib)
    # print("Density matrix=",dm)
    dm = gpu.dmMLSP2(H, D, N, Nocc, lib)
    return D


## Get TB forces from H and D
# \brief Get TB forces from H and D

## Main program for proxy a
# \brief It will read the number of atoms, contruct
# a set of random coordinates and give back a Density matrix.
#
if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print("Give the total number of atoms. Example:\n")
        print("proxy_a 100\n")
        sys.exit(0)
    else:
        nats = int(sys.argv[1])

    verb = True
    coords = get_random_coordinates(nats)

    H = get_hamiltonian_proxy(coords)

    gpuLibIn = True  ## need to pass from input file or command line
    occ = int(float(nats) / 2.0)  # Get the total occupied orbitals

    if gpuLibIn == False:
        print("Using CPU for DM construction. Consider installing accelerator library...")
        D = get_density_matrix(H, occ)

    # print("Hamiltonian matrix=",H)
    # print("Density matrix=",D)
