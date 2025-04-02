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
# @verbose Verbosity
#
def get_hamiltonian(
    engine,
    partIndex,
    nparts,
    norbs,
    latticeVectors,
    coords,
    types,
    symbols,
    get_overlap=True,
    verbose=False,
    newsystem=True,
    keepmem=False
):
    # Call the proper interface
    # If there is no interface, one should write its own Hamiltonian
    if engine.interface == "None":
        raise ValueError("ERROR!!! - Write your own Hamiltonian.")
    # Tight interface using modules or an external code compiled as a library
    elif engine.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        if get_overlap:
            ham, overlap = get_hamiltonian_module(
                engine,
                partIndex,
                nparts,
                norbs,
                latticeVectors,
                coords,
                types,
                symbols,
                get_overlap=get_overlap,
                verb=verbose,
                newsystem=newsystem,
                keepmem=keepmem
            )
            return ham, overlap
        else:
            return get_hamiltonian_module(
                engine,
                partIndex,
                nparts,
                norbs,
                latticeVectors,
                coords,
                types,
                symbols,
                get_overlap=get_overlap,
                verb=verbose,
                newsystem=newsystem,
                keepmem=keepmem
            )
    # Using any available library. We will use MDI here.
    elif engine.interface == "MDI":
        raise NotImplemented("MDI interface not implemented yet")
    # Using unix sockets to interface the codes
    elif engine.interface == "Socket":
        raise NotImplemented("Sockets not implemented yet")
    # Using files as a form of communication and transfering data.
    elif engine.interface == "File":
        return get_hamiltonian_files(engine, coords, types, symbols, verb=verbose)

    raise ValueError(
        f"ERROR!!!: Interface type not recognized: '{engine.interface}'. "
        + f"Use any of the following: Module,File,Socket,MDI"
    )
