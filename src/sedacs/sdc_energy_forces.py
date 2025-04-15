"""
sdc_energy_forces.py
====================================
Routines to calculate energy and force.
Typically this will be done interfacing with an engine.

"""

from sedacs.interface_modules import get_energy_forces_modules

__all__ = ["get_energy_forces"]


def get_energy_forces(
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
    numberOfCoreAtoms=None,
    mu=None,
    etemp=0.0,
    verb=False,
    newsystem=True,
    keepmem=False,
):
    """
    Computes the total energy and forces acting on each atom, typically by interfacing with an external engine.

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
    numberOfCoreAtoms : int, optional
        Number of atoms in the core region.
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
    energy : float
        The total energy of the system.
    forces : 2D numpy array, dtype: float
        Forces acting on each atom in the system. The forces are given in the same order as the coordinates.
    """
    if eng.interface == "None":
        print("ERROR!!! - Write your own Hamiltonian")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        energy, forces = get_energy_forces_modules(
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
            numberOfCoreAtoms=numberOfCoreAtoms,
            mu=mu,
            etemp=etemp,
            verb=verb,
            newsystem=newsystem,
            keepmem=keepmem,
        )
    else:
        print(
            "ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI"
        )

    return energy, forces
