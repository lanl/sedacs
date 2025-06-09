"""
sdc_density_matrix.py
====================================
Routines to build a density matrix.
Typically this will be done interfacing with an engine.

"""

from sedacs.interface_modules import get_density_matrix_modules

__all__ = ["get_density_matrix"]


def get_density_matrix(
    eng,
    partIndex,
    nparts,
    norbs,
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
    overlap=None,
    full_data=False,
    verb=False,
    newsystem=True,
    keepmem=False,
):
    """
    Constructs a density matrix, typically by interfacing with an external engine.

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
    ham : 2D numpy array, dtype: float
        The Hamiltonian matrix.
    vcouls : 1D numpy array, dtype: float
        The Coulomb potential for each atom.
    nocc : int
        Number of occupied orbitals.
    norbsInCore : int, optional
        Number of orbitals in the core region.
    method : str, optional
        Type of algorithm used to compute density matrix.
    accel : str, optional
        Type of accelerator/special device used to compute DM. Default is "No".
    mu : float, optional
        Chemical potential for the system.
    etemp : float, optional
        Electronic temperature for the system.
    overlap : 2D numpy array, dtype: float, optional
        The overlap matrix.
    full_data : bool, optional
        If True, retrieves additional data such as eigenvalues and contributions (dvals) from each core+halo part to the eigenvectors of the full system.
    verb : bool, optional
        If True, enables verbose output.
    newsystem : bool, optional
        If True, notifies the engine that the provided system is new.
    keepmem : bool, optional
        If True, notifies the engine to retrieve stored electronic structure data from memory.

    Returns
    -------
    rho: 2D numpy array, dtype: float
        The density matrix.
    charges: 1D numpy array, dtype: float, optional
        The Mulliken charges for each atom, if requested.
    evals: 1D numpy array, dtype: float, optional
        The eigenvalues of the input hamiltonian matrix, if requested.
    dvals: 1D numpy array, dtype: float, optional
        The dvals of the input hamiltonian matrix, if requested.
    """
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        if eng.name == "LATTE":
            rho, charges = get_density_matrix_modules(
                eng,
                partIndex,
                nparts,
                norbs,
                latticeVectors,
                coords,
                types,
                symbols,
                ham,
                vcouls,
                nocc,
                norbsInCore=norbsInCore,
                mu=mu,
                etemp=etemp,
                overlap=overlap,
                full_data=full_data,
                verb=verb,
                newsystem=newsystem,
                keepmem=keepmem,
            )
        elif full_data:
            rho, evals, dvals = get_density_matrix_modules(
                eng,
                partIndex,
                nparts,
                norbs,
                latticeVectors,
                coords,
                types,
                symbols,
                ham,
                vcouls,
                nocc,
                norbsInCore=norbsInCore,
                mu=mu,
                etemp=etemp,
                overlap=overlap,
                full_data=full_data,
                verb=verb,
                newsystem=newsystem,
                keepmem=keepmem,
            )
        else:
            rho = get_density_matrix_modules(
                eng,
                partIndex,
                nparts,
                norbs,
                latticeVectors,
                coords,
                types,
                symbols,
                ham,
                vcouls,
                nocc,
                norbsInCore=norbsInCore,
                mu=mu,
                etemp=etemp,
                overlap=overlap,
                full_data=False,
                verb=verb,
                newsystem=newsystem,
                keepmem=keepmem,
            )
    else:
        print(
            "ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI"
        )
    if eng.name == "LATTE":
        return rho, charges
    elif full_data:
        return rho, evals, dvals
    else:
        return rho
