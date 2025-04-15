"""
sdc_evals_dvals.py
======================================
Routines to compute evals and dvals.
Typically this will be done interfacing with an engine.
"""

from sedacs.interface_modules import get_evals_dvals_modules

__all__ = ["get_evals_dvals"]


def get_evals_dvals(
    eng,
    partIndex,
    nparts,
    latticeVectors,
    coords,
    types,
    symbols,
    ham,
    vcouls,
    nocc,
    norbsInCore=None,
    mu=None,
    etemp=0.0,
    verb=False,
    newsystem=True,
):
    """
    Computes eigenvalues and contributions (dvals) from each core+halo part to the eigenvectors of the full system, typically by interfacing with an external engine.

    Parameters
    ----------
    eng : Engine object
        Refer to engine.py for detailed information.
    partIndex : int
        Index of the current partition in the graph-partitioned system.
    nparts : int
        Total number of partitions in the graph-partitioned system.
    latticeVectors : 2D numpy array, dtype: float
        A 3x3 matrix representing lattice vectors for periodic boundary conditions.
    coords : 2D numpy array, dtype: float
        Atomic coordinates. For example, the z-coordinate of the first atom is coords[0,2].
    types : 1D numpy array, dtype: int
        Type indices for all atoms in the system. For example, the type of the first atom is types[0].
    symbols : list of str
        Chemical symbols corresponding to each type index.
    ham : 2D numpy array, dtype: float
        The Hamiltonian matrix.
    vcouls : numpy 1D array, dtype: float
        The Coulomb potential for each atom.
    nocc : int
        Number of occupied orbitals.
    norbsInCore : int, optional
        Number of orbitals in the core region.
    mu : float, optional
        Chemical potential for the system.
    etemp : float, optional
        Electronic temperature for the system.
    verb : bool, optional
        If True, enables verbose output.
    newsystem : bool, optional
        If True, notifies the engine that the provided system is new.
    keepmem : bool, optional
        If True, notifies the engine to retrieve stored electronic structure data from memory.

    Returns
    -------
    evals: 1D numpy array, dtype: float
        The eigenvalues of the input hamiltonian matrix, if requested.
    dvals: 1D numpy array, dtype: float
        The dvals of the input hamiltonian matrix, if requested.
    """
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        evals, dvals = get_evals_dvals_modules(
            eng,
            partIndex,
            nparts,
            latticeVectors,
            coords,
            types,
            symbols,
            ham,
            vcouls,
            nocc,
            norbsInCore=norbsInCore,
            mu=mu,
            verb=verb,
            newsystem=newsystem,
        )
    else:
        print(
            "ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI"
        )

    return evals, dvals
