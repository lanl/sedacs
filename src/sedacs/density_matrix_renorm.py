"""Density matrix
Routines to build a density matrix. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_modules import get_density_matrix_modules
from sedacs.interface.pyseqm import get_densityMatrix_renormalized_pyseqm
import torch
try:
    import seqm; PYSEQM = True
    from seqm.seqm_functions.pack import pack
except: PYSEQM = False


__all__ = ["get_density_matrix"]


## Build the density matrix.
# @brief This will build a density matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verbose Verbosity
#
def get_density_matrix_renorm(sdc, eng, Tel, mu0, P_contr, graph_for_pairs,
                              eVals, Q, NH_Nh_Hs, core_indices_in_sub_expandedm, verbose=False):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        rho = get_density_matrix_modules(eng, ham, verb=False)
    elif eng.interface == "PySEQM":
        if(PYSEQM == False):
            print("ERROR: No PySEQM installed")
            exit()
        rho = get_densityMatrix_renormalized_pyseqm(sdc, eVals, Q, Tel, mu0, NH_Nh_Hs)
        maxDif = None
        sumDif = None

    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return rho, maxDif, sumDif
