"""proxy code a
A prototype engine that:
    - Reads the total number of atoms
    - Constructs a set of random coordinates
    - Constructs a simple Hamiltonian
    - Computes the Density matrix from the Hamiltonian
"""

import os
import sys
import time

import numpy as np
import scipy.linalg as sp

global gpuLib


try:
    import ctypes

    import gpulibInterface as gpu

    gpuLib = True
    arch = "nvda"
    pwd = os.getcwd()

    if arch == "nvda":
        print("loading nvidia...")
        lib = ctypes.CDLL(str(pwd) + "../gpu/nvda/libnvda.so")
    if arch == "amd":
        lib = ctypes.CDLL(str(pwd) + "../gpu/amd/libamd.so")

except:
    gpuLib = False

__all__ = [
    "RandomNumberGenerator",
    "get_random_coordinates",
    "get_hamiltonian",
    "get_density_matrix",
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
# @return H 2D numpy array of Hamiltonian elements
# @param verb Verbosity. If True is passed, information is printed
#
def get_hamiltonian(coords, atomTypes=np.zeros((1), dtype=int), verb=False):
    """Construct simple toy s-Hamiltonian"""
    N = len(coords[:, 1])
    Nocc = int(N / 4)
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
    H = np.zeros((N, N))
    if verb:
        print("Constructing a simple Hamiltonian for the full system")
    cnt = 0
    for i in range(0, N):
        x = (a * x + c) % m  # Hamiltonian parameters
        y = (b * y + d) % n
        for j in range(i, N):
            dist = np.linalg.norm(coords[i, :] - coords[j, :])
            tmp = (x / m) * np.exp(-(y / n + decay_min) * (dist**2))
            H[i, j] = tmp
            H[j, i] = tmp
    return H


## Computes the Density matrix from a given Hamiltonian.
# @author Anders Niklasson
# @brief This will create a "zero-temperature" Density matrix \f$ \rho \f$
# \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
# where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
#
# @param H Hamiltonian matrix
# @param Nocc Number of occupied orbitals
# @param verb Verbosity. If True is passed, information is printed.
#
# @return D Density matrix
#
def get_density_matrix(H, Nocc, verb=False):
    """Calcualted the full density matrix from H"""
    if verb:
        print("Computing the Density matrix")
    E, Q = sp.eigh(H)
    N = len(H[:, 0])
    homoIndex = Nocc - 1
    lumoIndex = Nocc
    mu = 0.5 * (E[homoIndex] + E[lumoIndex])
    D = np.zeros((N, N))
    if verb:
        print("Eigenvalues of H:", E)
    for i in range(N):
        if E[i] < mu:
            D = D + np.outer(Q[:, i], Q[:, i])
    if verb:
        print("Chemical potential = ", mu)
    return D


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

    H = get_hamiltonian(coords)

    gpuLibIn = True  ## need to pass from input file or command line
    occ = int(float(nats) / 2.0)  # Get the total occupied orbitals

    if gpuLibIn == False:
        print("Using CPU for DM construction. Consider installing accelerator library...")
        D = get_density_matrix(H, occ)

    # print("Hamiltonian matrix=",H)
    # print("Density matrix=",D)
