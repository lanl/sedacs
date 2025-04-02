"""Density matrix
Routines to build a density matrix. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_modules import get_density_matrix_modules

__all__ = ["get_density_matrix"]


## Build the density matrix.
# @brief This will build a density matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verb Verbosity
#
def get_density_matrix(eng,partIndex,nparts,norbs,latticeVectors, coords, types, symbols,ham,vcouls,nocc,norbsInCore=None,mu=None,etemp=0.0,overlap=None,full_data=False,verb=False,newsystem=True,keepmem=False):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        if eng.name == "LATTE":
            rho, charges = get_density_matrix_modules(eng,partIndex,nparts,norbs,latticeVectors, coords, types, symbols,ham,vcouls,nocc,norbsInCore=norbsInCore,mu=mu,overlap=overlap,full_data=full_data,verb=verb,newsystem=newsystem,keepmem=keepmem)
        elif full_data:
            rho,evals,dvals = get_density_matrix_modules(eng,partIndex,nparts,norbs,latticeVectors, coords, types, symbols,ham,vcouls,nocc,norbsInCore=norbsInCore,mu=mu,overlap=overlap,full_data=full_data,verb=verb,newsystem=newsystem,keepmem=keepmem)
        else:
            rho = get_density_matrix_modules(eng,partIndex,nparts,norbs,latticeVectors, coords, types, symbols,ham,vcouls,nocc,norbsInCore=norbsInCore,mu=mu,overlap=overlap,full_data=False,verb=verb,newsystem=newsystem,keepmem=keepmem)
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
    if eng.name == "LATTE":
        return rho, charges
    elif full_data :
        return rho,evals,dvals
    else:
        return rho
