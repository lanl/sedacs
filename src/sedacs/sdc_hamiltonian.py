"""
sdc_hamiltonian.py
====================================
Routines to build a Hamiltonian matrix.
Typically this will be done interfacing with an engine.
"""

from sedacs.interface_files import get_hamiltonian_files
from sedacs.interface_modules import get_hamiltonian_module

__all__ = ["get_hamiltonian"]


def get_hamiltonian(
    eng,
    partIndex,
    nparts,
    norbs,
    latticeVectors,
    coords,
    types,
    symbols,
    get_overlap=True,
    verb=False,
    newsystem=True,
    keepmem=False,
):
    """
    Constructs a non-SCF Hamiltonian matrix, typically by interfacing with an external engine.

    Parameters
    ----------
    eng : Engine object
        Refer to engine.py for detailed information.
    partIndex : int
        Index of the current partition in the graph-partitioned system.
    nparts : int
        Total number of partitions in the graph-partitioned system.
    norbs : int
        Total number of orbitals in the current partition.
    latticeVectors : 2D numpy array, dtype: float
        A 3x3 matrix representing lattice vectors for periodic boundary conditions.
    coords : 2D numpy array, dtype: float
        Atomic coordinates. For example, the z-coordinate of the first atom is coords[0,2].
    types : 1D numpy array, dtype: int
        Type indices for all atoms in the system. For example, the type of the first atom is types[0].
    symbols : list of str
        Chemical symbols corresponding to each type index.
    get_overlap : bool, optional
        If True, computes the overlap matrix in addition to the Hamiltonian matrix.
    verb : bool, optional
        If True, enables verbose output.
    newsystem : bool, optional
        If True, notifies the engine that the provided system is new.
    keepmem : bool, optional
        If True, notifies the engine to retrieve stored electronic structure data from memory.

    Returns
    -------
    ham: 2D numpy array
        The non-SCF Hamiltonian matrix.
    overlap: 2D numpy array, optional
        The overlap matrix, if requested.
    """
    # Call the proper interface
    # If there is no interface, one should write its own Hamiltonian
    if eng.interface == "None":
        raise ValueError("ERROR!!! - Write your own Hamiltonian.")
    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        if get_overlap:
            ham, overlap, zmat = get_hamiltonian_module(
                eng,
                partIndex,
                nparts,
                norbs,
                latticeVectors,
                coords,
                types,
                symbols,
                get_overlap=get_overlap,
                verb=verb,
                newsystem=newsystem,
                keepmem=keepmem,
            )
            return ham, overlap, zmat
        else:
            return get_hamiltonian_module(
                eng,
                partIndex,
                nparts,
                norbs,
                latticeVectors,
                coords,
                types,
                symbols,
                get_overlap=get_overlap,
                verb=verb,
                newsystem=newsystem,
                keepmem=keepmem,
            )
    # Using any available library. We will use MDI here.
    elif eng.interface == "MDI":
        raise NotImplemented("MDI interface not implemented yet")
    # Using unix sockets to interface the codes
    elif eng.interface == "Socket":
        raise NotImplemented("Sockets not implemented yet")
    # Using files as a form of communication and transfering data.
    elif eng.interface == "File":
        return get_hamiltonian_files(eng, coords, types, symbols, verb=verb)

    raise ValueError(
        f"ERROR!!!: Interface type not recognized: '{eng.interface}'. "
        + f"Use any of the following: Module,File,Socket,MDI"
    )
