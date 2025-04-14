"""coulombic 
Some functions to copute Coulombic interactions

So far: get_coulvs and build_coul_ham
"""

from sedacs.message import *
from sedacs.periodic_table import PeriodicTable
from sedacs.interface_modules import build_coul_ham_module
import numpy as np

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
    "get_coulvs","build_coul_ham","get_coulombic_forces"
]

## Get short-range (non periodic) Coulombic potentials 
# @param charges Excess electronic ocupation (this is the negative of the charge vector)
# @param coords Atomic positions
# @param unit_facto Unit factor to account for proper units.
# @param verb Verbosity level.
#
def get_coulvs(charges,coords,unit_factor=14.3996437701414,verb=False):

    nats = len(charges)
    coulvs = np.zeros(nats)
    for i in range(nats):
        for j in range(nats):
            if(i != j):
                distance =  np.linalg.norm(coords[j,:] - coords[i,:])
                coulvs[i] = coulvs[i] - (unit_factor*charges[j])/(distance)
            #else: #We do not know how Hubbard Us are treated bu guest codes
                #coulvs[i] = coulvs[i] - hubbard[types[i]]*q[i]


    return coulvs


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
def build_coul_ham(engine,ham0,vcouls,types,charges,orbital_based,hindex,overlap=None,verb=False):
    # Call the proper interface
    # If there is no interface, one should write its own Hamiltonian
    if engine.interface == "None":
        raise ValueError("ERROR!!! - Write your own coulombic Hamiltonian.")
    # Tight interface using modules or an external code compiled as a library
    elif engine.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        ham = build_coul_ham_module(engine,ham0,vcouls,types,charges,orbital_based,hindex,overlap=overlap,verb=False)
    # Using any available library. 
    elif engine.interface == "MDI":
        raise NotImplemented("MDI interface not implemented yet")
    # Using unix sockets to interface the codes
    elif engine.interface == "Socket":
        raise NotImplemented("Sockets not implemented yet")
    # Using files as a form of communication and transfering data.
    elif engine.interface == "File":
        raise NotImplemented("File interface not implemented yet")
    else:
        raise ValueError(f"ERROR!!!: Interface type not recognized: '{engine.interface}'. " +
                     f"Use any of the following: Module,File,Socket,MDI")


    return ham 


def get_coulombic_forces(charges,coords,atomTypes,symbols,factor=14.39964377014,field=None):
    nats = len(charges)
    forces_coul = np.zeros((nats,3))
    forces_field = np.zeros((nats,3))
    forces = np.zeros((nats,3))
    for i in range(nats):
        #Coulombic Forces
        for j in range(nats):
            if(i != j):
                distance =  np.linalg.norm(coords[i,:] - coords[j,:])
                direction = (coords[i,:] - coords[j,:])/distance
                forces_coul[i,:] = forces_coul[i,:] - (factor*direction*charges[i]*charges[j])/(distance**2)

        #Field forces
        if(field is not None):
            forces_field[i,:] = forces_field[i,:] + field*charges[i]
            forces = forces_field + forces_coul
        else:
            forces = forces_coul

    return forces
