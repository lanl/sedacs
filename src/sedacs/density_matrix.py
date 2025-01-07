"""Density matrix
Routines to build a density matrix. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_modules import get_density_matrix_modules
try:
    import seqm; PYSEQM = True
    from seqm.seqm_functions.make_dm_guess import make_dm_guess
    from seqm.seqm_functions.pack import pack, unpack

except: PYSEQM = False
import torch


__all__ = ["get_density_matrix"]


## Build the density matrix.
# @brief This will build a density matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verbose Verbosity
#
def get_density_matrix(eng, nocc, ham, coords, symbols, types, Tel, verbose):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        rho = get_density_matrix_modules(eng, nocc, ham, verb=False)
    elif eng.interface == "PySEQM":
        if(PYSEQM == False):
            print("ERROR: No PySEQM installed")
            exit()
        
        # We will call proxyA directly as it will be loaded as a module.
        rho = get_density_matrix_modules(eng, nocc, ham, verb=False)
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return rho

def get_initDM(eng, sdc, coords, symbols, types, molecule_whole):
    if eng.interface == "None":
        print("ERROR!!! - Write your own dmInit")
        exit()
    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print("ERROR!!! - Write your own dmInit")
        exit()
    elif eng.interface == "PySEQM":
        if(PYSEQM == False):
            print("ERROR: No PySEQM installed")
            exit()
        dm = make_dm_guess(molecule_whole, molecule_whole.seqm_parameters, mix_homo_lumo=False, mix_coeff=0.3, overwrite_existing_dm=True, assignDM = False)[0];
        #dm = torch.load("/home/maxim/Projects/SEDACS_1/sedacs/examples/pyseqm/w_4_dm.pt", weights_only=True)
        #dm = torch.load("/home/maxim/Projects/SEDACS_1/sedacs/examples/pyseqm/nanostar_dm.pt", weights_only=True)
        #molSysData.molecule_whole.dm = torch.load("/home/maxim/Projects/SEDACS_1/sedacs/examples/pyseqm/gs_solvated_cell_dm.pt", weights_only=True)
        #dm = torch.load("/home/maxim/Projects/SEDACS_1/sedacs/examples/pyseqm/gs_10k_dm_128.pt", weights_only=True)
        #molSysData.molecule_whole.dm = torch.load("/home/maxim/Projects/SEDACS_1/sedacs/examples/pyseqm/overlap_whole.pt")
        #molSysData.molecule_whole.dm = unpack(torch.tensor(torch.load("/home/maxim/Projects/SEDACS_1/sedacs/examples/pyseqm/overlap_whole.pt")).unsqueeze(0),
        #                           molSysData.molecule_whole.nHeavy, molSysData.molecule_whole.nHydro, molSysData.molecule_whole.species.shape[-1]*4)
        return dm

    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return rho

def get_dmErrs(eng, dm1, dm2):
    if eng.interface == "None":
        print("ERROR!!! - Write your own dmDif")
        exit()
    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print("ERROR!!! - Write your own dmDif")
        exit()
    elif eng.interface == "PySEQM":
        if(PYSEQM == False):
            print("ERROR: No PySEQM installed")
            exit()
        #dif = torch.abs(dm1 - dm2)            
        maxDif = torch.max(torch.abs(dm1[0,:32000,:32000] - dm2[0,:32000,:32000])).numpy()
        sumDif = torch.sum(torch.abs(dm1[0,:32000,:32000] - dm2[0,:32000,:32000])).numpy()
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return maxDif, sumDif

def get_dmTrace(eng, dm):
    if eng.interface == "None":
        print("ERROR!!! - Write your own trace")
        exit()
    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print("ERROR!!! - Write your own trace")
        exit()
    elif eng.interface == "PySEQM":
        if(PYSEQM == False):
            print("ERROR: No PySEQM installed")
            exit()
        trace = torch.trace(dm[0])
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return trace

