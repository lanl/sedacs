"""coulombic 
Some functions to copute coulombic interactions

So far: get_coulvs 
"""

from sedacs.message import *
from sedacs.periodic_table import PeriodicTable
import numpy as np

# from sdc_out import *
try:
    from mpi4py import MPI

    mpiLib = True
except ImportError as e:
    mpiLib = False
from multiprocessing import Pool

if mpiLib:
    from sedacs.mpi import *
import time

__all__ = [
    "get_coulvs",
]

## Get short-range (non periodic) Coulombic potentials 
# @param charges Excess electronic ocupation (this is the negative of the charge vector)
# @param coords Atomic positions
# @pram Unit factor
# @param verb Verbosity level.
#
def get_coulvs(charges,coords,unit_factor=14.3996437701414,verb=False):

    nats = len(charges)

    coulvs = np.zeros(nats)
    for i in range(nats):
        #Coulombic Forces
        for j in range(nats):
            if(i != j):
                distance =  np.linalg.norm(coords[j,:] - coords[i,:])
                coulvs[i] = coulvs[i] - (unit_factor*charges[j])/(distance)

    return coulvs


## Add coulombic potentials to the Hamiltonian
# @param ham0 No-SCF Hamiltonian
# @param vcouls Coulombic potentials for every atomic site 
# @pparam orbital_based If set to True, coulombic potentials for every orbitals will be 
# expected.
# @param hindex will give the orbital index for each atom
# The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`
# 
def build_coul_ham(ham0,vcouls,orbital_based,hindex,overlap=None,verb=False):

    norbs = len(ham0[:,0])

    vcouls_orbs = np.zeros((norbs),dtype=float) #Expanded coulombic potentials

    nats = len(hindex[:]) - 1
 
    if(orbital_based):
        pass 
    else:
        if(overlap is None):
            for i in range(nats):
                for ii in range(hindex[i],hindex[i+1]):
                    vcouls_orbs[ii] = vcouls[i]
        else: 
            pass
            

    ham = ham0 + np.diag(vcouls_orbs)

    return ham 

