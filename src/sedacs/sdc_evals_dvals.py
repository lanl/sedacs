"""Evals and dvals
Routines to compute evals and dvals. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_modules import get_evals_dvals_modules

__all__ = ["get_evals_dvals"]


## Compute evals and dvals 
# @brief This will compute evals and dvals. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verb Verbosity
#
def get_evals_dvals(eng,partIndex,nparts,latticeVectors, coords, types, symbols,ham,vcouls,nocc,norbsInCore=None,mu=None,etemp=0.0,overlap=None,full_data=False,verb=False, newsystem=True):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        evals,dvals = get_evals_dvals_modules(eng,partIndex,nparts,latticeVectors, coords, types, symbols,ham,vcouls,nocc,norbsInCore=norbsInCore,mu=mu,overlap=overlap,full_data=full_data,verb=verb, newsystem=newsystem)
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")

    return evals,dvals
