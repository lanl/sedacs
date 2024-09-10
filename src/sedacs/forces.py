"""eVals dVals
Routines to build eVals and dVals. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_modules import get_density_matrix_modules
from sedacs.interface_pyseqm import get_molecule_pyseqm, get_coreHalo_ham_inds, get_eVals_pyseqm
import numpy as np
import time


__all__ = ["get_eVals"]


## Build the density matrix.
# @brief This will build a density matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verbose Verbosity
#
def get_forces(eng, obj, eTot):
    if eng.interface == "None":
        print("ERROR!!! - Write your own forces")
        exit()
    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print('TBD')
        exit()
    elif eng.interface == "PySEQM":
        tic = time.perf_counter()
        L = eTot.sum()
        eTot.backward()
        force = -obj.molecule_whole.coordinates.grad.detach()
        obj.molecule_whole.coordinates.grad.zero_()
        print("Time to compute forces", time.perf_counter() - tic,"(s)")
        return force



    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
