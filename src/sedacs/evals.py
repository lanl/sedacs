"""eVals dVals
Routines to build eVals and dVals. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_modules import get_density_matrix_modules
from sedacs.interface_pyseqm import get_molecule_pyseqm, get_coreHalo_ham_inds, get_eVals_pyseqm
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
def get_eVals(eng, sdc, sy, nocc, ham, coords, symbols, types, Tel, mu0,
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


        molecule_sub, occ = get_molecule_pyseqm(coords, symbols, types)

        symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols_internal)}
        # Translate `symbols` to `symbols_internal` indices
        mapped_indices = np.array([symbol_to_index[symbol] for symbol in subSyCore.symbols])
        # Convert atomTypes to `symbols_internal` indices
        atom_internal_indices = mapped_indices[subSyCore.types]
        # Sum the corresponding values in bas_per_atom and numel_internal
        core_ham_dim = np.sum(bas_per_atom[atom_internal_indices])
        
        core_indices_in_sub, core_indices_in_sub_expanded, hindex_sub, I, I_halo = \
            get_coreHalo_ham_inds(partIndex, partCoreHaloIndex, sdc, sy, subSy, device=ham.device)
        #core_ham_dim = np.hstack([np.arange(s, e) for s, e in zip(hindex_sub[core_indices_in_sub], hindex_sub[core_indices_in_sub+1])])
        core_ham_dim_list = [torch.arange(s, e) for s, e in zip(hindex_sub[core_indices_in_sub], hindex_sub[core_indices_in_sub + 1])]
        core_ham_dim = torch.cat(core_ham_dim_list).to(ham.device)

        eVals, dVals, Q, NH_Nh_Hs = get_eVals_pyseqm(ham, occ, Tel, mu0, coreSize, core_ham_dim, molecule=molecule_sub, verb=False)
        # We will call proxyA directly as it will be loaded as a module.
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return eVals, dVals, Q, NH_Nh_Hs, I, I_halo, core_indices_in_sub_expanded
