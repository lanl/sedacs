"""eVals dVals
Routines to build eVals and dVals. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface.pyseqm import pyseqmObjects
import numpy as np

__all__ = ["get_eVals"]


## Build the density matrix.
# @brief This will build a density matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verbose Verbosity
#
def get_molSysData(eng, sdc, coords,symbols,atomTypes, do_large_tensors=True, device='cpu'):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Data")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print('TBD')
        exit()
    elif eng.interface == "PySEQM":
        return pyseqmObjects(sdc, coords,symbols,atomTypes, do_large_tensors=do_large_tensors, device=device)

    return
