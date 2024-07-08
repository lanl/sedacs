"""Hamiltonian
Routines to build a Hamiltonian matrix. Typically
this will be done interfacing with an engine.

"""

import sys

import numpy as np
from sedacs.interface_files import *
from sedacs.interface_modules import *


## Build the non-scf Hamiltonian matrix.
# @brief This will build a Hamiltonian matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param coords Positions for every atom. z-coordinate of atom 1 = coords[0,2]
# @param types Index type for each atom in the system. Type for first atom = type[0]
# @param symbols Symbols for every atom type
# @verb Verbosity
#
def sdc_get_hamiltonian(eng, coords, types, symbols, verb):
    # Call the proper interface
    # If there is no interface, one should write its own Hamiltonian
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        ham = sdc_get_hamiltonian_module(eng, coords, types, symbols, verb=False)

    # Using any available library. We will use MDI here.
    elif eng.interface == "MDI":
        print("MDI interface not implemented yet")
        sys.exit(0)

    # Using unix sockets to interface the codes
    elif eng.interface == "Socket":
        print("Sockets not implemented yet")
        sys.exit(0)

    # Using files as a form of communication and transfering data.
    elif eng.interface == "File":
        ham = get_hamiltonian_files(eng, coords, types, symbols, verb=False)

    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        sys.exit(0)

    return ham
