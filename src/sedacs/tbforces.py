#TB forces 
"""TB forces 
Some functions to get the TB forces

So far: get_tb_forces
"""

from sedacs.message import *
from sedacs.periodic_table import PeriodicTable
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
    "get_tb_forces"
]


## Get TB forces
# \brief Get TB and Coulombic forces from the Hamiltonian, Density matrix
# and charges.
# \param ham Hamiltonian Matrix 
# \param rho Density Matrix 
# \param field External field 
# \param coords Coordinates 
# \param atomTypes Atomic types 
# \param Symbols Atomic symbols for every type.
##
def get_tb_forces(engine,ham,rho,charges,field,coords,atomTypes,symbols,overlap=None,verb=False):

    # Call the proper interface
    # If there is no interface, one could write its own tb forces
    if engine.interface == "None":
        raise ValueError("ERROR!!! - Write your own TB forces.")
    # Tight interface using modules or an external code compiled as a library
    elif engine.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        if(overlap is not None):
            raise NotImplemented("TB forces with overlap not implemented in proxy code")
        else:
            return get_tb_forces_module(ham,rho,charges,field,coords,atomTypes,symbols,overlap=overlap,verb=verb)
    # Using any available library. We will use MDI here.
    elif engine.interface == "MDI":
        raise NotImplemented("MDI interface not implemented yet")
    # Using unix sockets to interface the codes
    elif engine.interface == "Socket":
        raise NotImplemented("Sockets not implemented yet")
    # Using files as a form of communication and transfering data.
    elif engine.interface == "File":
        raise NotImplemented("Sockets not implemented yet")
    else:
        raise ValueError(f"ERROR!!!: Interface type not recognized: '{engine.interface}'. " +
                        f"Use any of the following: Module,File,Socket,MDI")



    return forces



