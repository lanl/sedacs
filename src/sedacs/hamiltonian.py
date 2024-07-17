"""Hamiltonian
Routines to build a Hamiltonian matrix. Typically
this will be done interfacing with an engine.

"""

import sys

from sedacs.interface_files import get_hamiltonian_files
from sedacs.interface_modules import get_hamiltonian_module

__all__ = ["get_hamiltonian"]


## Build the non-scf Hamiltonian matrix.
# @brief This will build a Hamiltonian matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param coords Positions for every atom. z-coordinate of atom 1 = coords[0,2]
# @param types Index type for each atom in the system. Type for first atom = type[0]
# @param symbols Symbols for every atom type
# @verb Verbosity
#
def get_hamiltonian(engine, coords, types, symbols, verbose=False):
    # Call the proper interface
    # If there is no interface, one should write its own Hamiltonian
    if engine.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")
        return None

    # Tight interface using modules or an external code compiled as a library
    if engine.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        return get_hamiltonian_module(engine, coords, types, symbols, verb=verbose)

    # Using any available library. We will use MDI here.
    if engine.interface == "MDI":
        print("MDI interface not implemented yet")
        return None

    # Using unix sockets to interface the codes
    if engine.interface == "Socket":
        print("Sockets not implemented yet")
        return None

    # Using files as a form of communication and transfering data.
    if engine.interface == "File":
        return get_hamiltonian_files(engine, coords, types, symbols, verb=verbose)

    print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
    return None
