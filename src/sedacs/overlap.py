import torch

try:
    import seqm; PYSEQM = True
    from sedacs.interface.pyseqm import get_overlap_pyseqm

except: PYSEQM = False

__all__ = ["get_overlap"]


## Build the overlap matrix.
# @brief This will build a density matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verbose Verbosity
#
def get_overlap(eng, coords, symbols, types, hindex):
    if eng.interface == "None":
        print("ERROR!!! - Write your own overlap")
        exit()
    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print("ERROR!!! - Write your own overlap")
        exit()
    elif eng.interface == "PySEQM":

        with torch.no_grad():
            return get_overlap_pyseqm(coords, symbols, types, hindex).detach().numpy()
        
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return rho
