"""hamiltonian 
A prototype engine code that:
    - Constructs different Hamiltonians
This code is only used to guide implemetations and understand which are the 
basic elements needed to interface with the sedacs driver.
"""

import os
import sys

import numpy as np
import sedacs.driver
from sedacs.dev.io import src_path
import scipy.linalg as sp
from hamiltonian_elements import *  
from dnnprt import *
from proxies.python.proxy_global import *
#from proxy_global import *

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
    "get_hamiltonian_proxy",
    "get_random_hamiltonian",
    "build_coul_ham_proxy"
]


## Computes a Hamiltonian based on exponential decay of orbital couplings.
# @author Anders Niklasson
# @brief Computes a hamiltonian based on exponential decays.
# @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
# @param atomTypes Index type for each atom in the system. Type for first atom = type[0] (not used yet)
# @param symbols Symbols for every atom type
# @param verb Verbosity. If True is passed, information is printed
# @para get_overlap If overlap needs to be returned
# @return ham 2D array of Hamiltonian elements
#
def get_hamiltonian_proxy(coords, atomTypes, symbols, verb=False, get_overlap=False):
    """Constructs a simple tight-binding Hamiltonian"""

    # Internal periodic table for the code
    symbols_internal = np.array(["Bl", "H", "C", "N", "O", "P"], dtype=str)
    numel_internal = np.zeros(len(symbols_internal), dtype=int)
    numel_internal[:] = 0, 1, 4, 5, 6, 5
    bas_per_atom = np.zeros(len(symbols_internal), dtype=int)
    bas_per_atom[:] = 0, 1, 4, 4, 4, 4
    spOrbTypes = ["s", "px", "py", "pz"]
    sOrbTypes = ["s"]

    nats = len(coords[:, 0])

    # Map symbols to indices in symbols_internal
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols_internal)}

    # Translate `symbols` to `symbols_internal` indices
    mapped_indices = np.array([symbol_to_index[symbol] for symbol in symbols])

    # Convert atomTypes to `symbols_internal` indices
    atom_internal_indices = mapped_indices[atomTypes]

  
    # Sum the corresponding values in bas_per_atom and numel_internal
    norbs = np.sum(bas_per_atom[atom_internal_indices])
    numel = np.sum(numel_internal[atom_internal_indices])

    ham = np.zeros((norbs, norbs))
    if get_overlap:
        over = np.zeros((norbs, norbs))
    if verb:
        print("Constructing a simple Hamiltonian for the full system")

    colsh = 0
    rowsh = 0
    tbparams = bring_tbparams()
    for i in range(0, nats):
        for ii in range(bas_per_atom[atom_internal_indices[i]]):
            colsh = rowsh
            parI = tbparams[atomTypes[i]][ii]
            for j in range(i, nats):
                if i == j:
                    llimit = ii
                else:
                    llimit = 0
                for jj in range(llimit,bas_per_atom[atom_internal_indices[j]]):
                    parJ = tbparams[atomTypes[j]][jj]
                    hval, sval = get_integral_v1(coords[i],coords[j],parI,parJ)
                    if(get_overlap):
                        over[rowsh,colsh] = sval
                        over[colsh,rowsh] = sval
                    
                    ham[rowsh, colsh] = hval
                    ham[colsh, rowsh] = hval
                    colsh = colsh + 1
            rowsh = rowsh + 1

    if get_overlap:
        return ham, over
    else:
        return ham


sedacs.driver.get_hamiltonian = get_hamiltonian_proxy

## Add coulombic potentials to the Hamiltonian
# @param ham0 No-SCF Hamiltonian
# @param vcouls Coulombic potentials for every atomic site
# @pparam orbital_based If set to True, coulombic potentials for every orbitals will be
# expected.
# @param hindex will give the orbital index for each atom
# The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`
# @param overlap Overlap matrix for nonorthogonal formulations.
# @param verb Verbosity switch.
#
def build_coul_ham_proxy(ham0, vcouls, types, charges, orbital_based, hindex, overlap=None, verb=False):
    norbs = len(ham0[:, 0])
    vcouls_orbs = np.zeros((norbs), dtype=float)  # Expanded coulombic potentials
    nats = len(hindex[:]) - 1

    tbparams = bring_tbparams()

    if orbital_based:
        error_at("build_coul_ham", "Orbital-based coulombic potential not implemented")
    else:
        for i in range(nats):
            for ii in range(hindex[i], hindex[i + 1]):
                k = ii - hindex[i]
                vcouls_orbs[ii] = vcouls[i] + tbparams[types[i]][k].u * charges[i]
        if overlap is None:
            ham = ham0 + np.diag(vcouls_orbs)
        else:
            vmat = np.diag(vcouls_orbs)
            ham = ham0 + 0.5 * (np.dot(overlap, vmat) + np.dot(vmat, overlap))
    return ham


## Computes a Hamiltonian based on a single "s-like" orbitals per atom.
# @author Anders Niklasson 
# @brief Computes a hamiltonian \f$ H_{ij} = (x/m)\exp(-(y/n + decay_{min}) |R_{ij}|^2))\f$, based on distances
# \f$ R_{ij} \f$. \f$ x,m,y,n,decay_{min} \f$ are fixed parameters.
#
# @param ndim Size of the Hamiltonian matrix
# @param verb Verbosity. If True is passed, information is printed
# @return H 2D numpy array of Hamiltonian elements
#
def get_random_hamiltonian(coords,verb=False):
  """Construct simple toy s-Hamiltonian """
  norbs = len(coords[:,0])
  eps = 1e-9; decay_min = 0.1; m = 78;
  a = 3.817632; c = 0.816371; x = 1.029769; n = 13;
  b = 1.927947; d = 3.386142; y = 2.135545;
  ham = np.zeros((norbs,norbs))
  if(verb): print("Constructing a simple Hamiltonian for the full system")
  cnt = 0
  for i in range(0,norbs):
    x = (a*x+c)%m       #Hamiltonian parameters
    y = (b*y+d)%n
    for j in range(i,norbs):
      dist = np.linalg.norm(coords[i,:]-coords[j,:])
      tmp = (x/m)*np.exp(-(y/n + decay_min)*(dist**2))
      ham[i,j] = tmp
      ham[j,i] = tmp
    ham[i,i] = x*y
  return ham

