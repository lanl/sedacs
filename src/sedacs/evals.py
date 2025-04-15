"""eVals dVals
Routines to build eVals and dVals. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_modules import get_density_matrix_modules
from sedacs.interface.pyseqm import get_molecule_pyseqm, get_coreHalo_ham_inds, get_eVals_pyseqm
import numpy as np
import torch

__all__ = ["get_eVals"]


## Build the density matrix.
# @brief This will build a density matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verbose Verbosity
#
def get_eVals(eng, sdc, sy, ham, coords, symbols, types, Tel, mu0,
              core_indices_in_sub, core_indices_in_sub_expanded, hindex_sub,
              coreSize, subSy, subSyCore,
              partIndex, partCoreHaloIndex, verbose):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print('TBD')
        exit()
    elif eng.interface == "PySEQM":
        symbols_internal = np.array([ "Bl" ,                               
            "H" ,                                     "He",        
            "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne",          \
            "Na", "Mg", "Al", "Si", "P" , "S" , "Cl", "Ar",
            ], dtype=str)
        numel_internal = np.zeros(len(symbols_internal),dtype=int)
        numel_internal[:] = 0,   \
            1 ,                  2,   \
            1 ,2 ,3 ,4 ,5 ,6 ,7, 8,   \
            1 ,2 ,3 ,4 ,5 ,6 ,7, 8,

        bas_per_atom = np.zeros(len(symbols_internal),dtype=int)
        bas_per_atom[:] =   0,   \
            1 ,                   1 ,\
            4 ,4 ,4 ,4 ,4 ,4 ,4 , 4,  \
            4 ,4 ,4 ,4 ,4 ,4 ,4 , 4,  \

        molecule_sub, occ = get_molecule_pyseqm(sdc, coords, symbols, types, do_large_tensors=False)
        
        core_ham_dim_list = [torch.arange(s, e) for s, e in zip(hindex_sub[core_indices_in_sub], hindex_sub[core_indices_in_sub + 1])]
        core_indices_in_sub_expanded_packed = torch.cat(core_ham_dim_list).to(ham.device)

        # the difference between core_indices_in_sub_expanded and core_indices_in_sub_expanded_packed is that
        # core_indices_in_sub_expanded is core indices of core+halo hamiltonian in 4x4 blocks form (pyseqm format)
        # core_indices_in_sub_expanded_packed is core indices of core+halo hamiltonian in normal form corresponding to the number of AOs per atom. We need this one for parsing eigenvectors.

        eVals, dVals, Q = get_eVals_pyseqm(sdc, ham, occ, core_indices_in_sub_expanded_packed, molecule=molecule_sub, verb=False)
        # We will call proxyA directly as it will be loaded as a module.
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return eVals, dVals, Q, [molecule_sub.nHeavy, molecule_sub.nHydro, ham.shape[-1], molecule_sub.nocc]
