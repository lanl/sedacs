"""random number generator
This code is only used to guide implemetations and understand which are the 
basic elements needed to interface with the sedacs driver.
"""

import numpy as np
import sys
from random_numbers import RandomNumberGenerator
from coordinates import get_random_coordinates


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


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print("Give the total number of atoms/orbitals. Example:\n")
        print("python hamiltonian_random.py 100\n")
        sys.exit(0)
    else:
        nats = int(sys.argv[1])

    verb = True

    coords = get_random_coordinates(nats)
    ham = get_random_hamiltonian(coords,verb=verb)

    print("Hamiltonian:",ham)

