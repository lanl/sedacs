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
# @verbose Verbosity
#
def get_density_matrix(eng, nocc, ham, verbose):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        rho = get_density_matrix_modules(eng, nocc, ham, verb=False)
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
    return rho
