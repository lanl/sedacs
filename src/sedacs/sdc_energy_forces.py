"""Energy and force
Routines to calculate energy and force. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_modules import get_energy_forces_modules

__all__ = ["get_energy_forces"]


## Calculate the energy and force. 
# @brief This will calculate energy and force. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verb Verbosity
#
def get_energy_forces(eng,partIndex,nparts,norbs,hamiltonian,latticeVectors,coords,atomTypes,symbols,vcouls,nocc,norbsInCore=None,numberOfCoreAtoms=None,mu=None,etemp=0.0,verb=False,newsystem=True,keepmem=False):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        energy, forces = get_energy_forces_modules(eng,partIndex,nparts,norbs,hamiltonian,latticeVectors,coords,atomTypes,symbols,vcouls,nocc,norbsInCore=norbsInCore,numberOfCoreAtoms=numberOfCoreAtoms,mu=mu,etemp=etemp,verb=verb,newsystem=newsystem,keepmem=keepmem)
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
    
    return energy, forces 
