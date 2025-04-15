"""
interface_modules.py
====================================
Routines to handle the interface with the engines.
"""

import ctypes
import os

import numpy as np
from sedacs.message import *
from sedacs.globals import *
from sedacs.periodic_table import PeriodicTable
import sys 
from sedacs.engine import Engine
from sedacs.globals import *
from sedacs.types import ArrayLike

# import the shared library
try:
    fortlibFileName = os.environ["PROXYA_FORTRAN_PATH"] + "/proxya_fortran.so"
    fortlib = True
except Exception as e:
    fortlib = False

try:
    pylibFileName = os.environ["PROXYA_PYTHON_PATH"]
    pylib = True
except Exception as e:
    pylib = False

if (not fortlib) and (not pylib):
    print(fortlib, pylib)
    error_at("interface_modules", "No specific fortran or python library exists")
    raise e

try:
    from proxies.python.hamiltonian import get_hamiltonian_proxy
    from proxies.python.density_matrix import get_density_matrix_proxy
    from proxies.python.evals_dvals import get_evals_dvals_proxy

    from proxies.python.init_proxy import init_proxy_proxy
    from proxies.python.hamiltonian import build_coul_ham_proxy
except Exception as e:
    pythlib = None
    raise e


__all__ = [
    "get_hamiltonian_module",
    "get_density_matrix_module",
    "get_evals_dvals_module",
    "get_energy_forces_module",
    "get_ppot_energy_expo",
    "get_ppot_forces_expo",
    "init_proxy",
    "get_tb_forces_module",
    "build_coul_ham_module",
]


#Initialize the proxy code
def init_proxy(symbols,orbs):
    """
    Initialize the proxy code.
    """

    init_proxy_proxy(symbols,orbs)


def build_coul_ham_module(
    eng, ham0, vcouls, types, charges, orbital_based, hindex, overlap=None, verb=False
):
    """
    Interface to call external engine for adding Coulomb potential to the Hamiltonian matrix.

    Parameters
    ----------
    eng : Engine object
        Refer to engine.py for detailed information.
    ham0 : 2D numpy array, dtype: float
        The non-SCF Hamiltonian matrix.
    vcouls : 1D numpy array, dtype: float
        The Coulomb potential for each atom.
    types : 1D numpy array, dtype: int
        Type indices for all atoms in the system. For example, the type of the first atom is types[0].
    charges : 1D numpy array, dtype: float
        The Mulliken charges for each atom.
    orbital_based : bool
        If True, Coulomb potential for every orbitals will be expected.
    hindex : list of int
        The orbital index for each atom. The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`.
    overlap : 2D numpy array, dtype: float, optional
        Overlap matrix for nonorthogonal formulations.
    verb : bool, optional
        If True, enables verbose output.

    Returns
    -------
    ham : 2D numpy array, dtype: float
        The Hamiltonian matrix with the Coulomb potential added.
    """
    if eng.name == "ProxyAPython":
        ham = build_coul_ham_proxy(
            ham0,
            vcouls,
            types,
            charges,
            orbital_based,
            hindex,
            overlap=overlap,
            verb=False,
        )
    elif eng.name == "ProxyAFortran":
        error_at("build_coul_ham_module", "ProxyAFortran version not implemented yet")
    elif eng.name == "ProxyAC":
        error_at("build_coul_ham_module", "ProxyAC version not implemented yet")
    elif eng.name == "LATTE":
        # If using LATTE as engine, the coulombic potential would be added directly in the LATTE code.
        ham = ham0
    else:
        error_at("build_coul_ham_module", "No specific engine type defined")

    return ham


def get_hamiltonian_module(
    eng: Engine,
    partIndex: int,
    nparts: int,
    norbs: int,
    latticeVectors: ArrayLike,
    coords: ArrayLike,
    types: ArrayLike,
    symbols: list[str],
    get_overlap: bool = True,
    verb: bool = False,
    newsystem: bool = True,
    keepmem: bool = False,
):
    """
    Interface to call external engine for constructing a non-SCF Hamiltonian matrix.

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
    ham: 2D numpy array, dtype: float
        The non-SCF Hamiltonian matrix.
    overlap: 2D numpy array, dtype: float, optional
        The overlap matrix, if requested.
    """
    if eng.name == "ProxyAPython":
        if get_overlap:
            hamiltonian, overlap = get_hamiltonian_proxy(
                coords, types, symbols, get_overlap=get_overlap, verb=verb
            )
        else:
            hamiltonian = get_hamiltonian_proxy(
                coords, types, symbols, get_overlap=get_overlap, verb=verb
            )

    elif eng.name == "ProxyAFortran":
        nats = len(coords[:, 0])
        norbs = nats

        coords_in = np.zeros(3 * nats)  # Vectorized coordinates
        for i in range(nats):
            coords_in[3 * i] = coords[i, 0]
            coords_in[3 * i + 1] = coords[i, 1]
            coords_in[3 * i + 2] = coords[i, 2]

        # Specify arguments type as a pointers
        get_hamiltonian_fortran.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_bool,
        ]
        # Passing a pointer to Fotran
        coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = types.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        hamiltonian = np.zeros((norbs, norbs))
        overlap = np.zeros((norbs, norbs))
        ham_ptr = hamiltonian.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overlap.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        err = get_hamiltonian_fortran(
            ctypes.c_int(nats),
            ctypes.c_int(norbs),
            coords_ptr,
            atomTypes_ptr,
            ham_ptr,
            over_ptr,
            ctypes.c_bool(verb),
        )
    elif eng.name == "LATTE":

        # Import the shared library
        latteLibFileName = os.environ["LATTE_PATH"] + "/liblatte.so"

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_compute

        # Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        # In LATTE, compflag = 1 means we only want compute the non-SCF hamiltonian
        compflag = 1
        # Getting the number of atoms through the coordinates array
        nats = len(coords[:, 0])
        # Here we don't care about ncores, so we set it to norbs
        ncores = norbs

        # Getting atomic numbers
        # Get the number of distinct atom types through counting the elements in the symbols list
        nTypes = len(symbols)
        # Initializing the atomic numbers array
        atomicNumbers = np.zeros((nTypes), dtype=np.int32)
        # Initializing the atomTypes array
        atomTypes32 = np.zeros((nats), dtype=np.int32)
        # Filling the atomTypes array with the types array
        atomTypes32[:] = types
        # Filling the atomic numbers array with the atomic numbers corresponding to the symbols
        for i in range(len(symbols)):
            atomicNumbers[i] = pt.get_atomic_number(symbols[i])

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros(3 * nats)  # Vectorized forces
        hamFlat_out = np.zeros(norbs * norbs)  # Vectorized hamiltonian
        overFlat_out = np.zeros(norbs * norbs)  # Vectorized overlap
        dmFlat_out = np.zeros(norbs * norbs)  # Vectorized density matrix
        evalsFlat_out = np.zeros(norbs)  # We call this one Flat just for consistency
        dvalsFlat_out = np.zeros(norbs)  # Same here
        chargesFlat_out = np.zeros(nats)  # Same here
        energyFlat_out = np.zeros(1)  # Same here

        # Converting the coordinates array to a flat array for C-Fortran interoperability
        for i in range(nats):
            coordsFlat_in[3 * i] = coords[i, 0]
            coordsFlat_in[3 * i + 1] = coords[i, 1]
            coordsFlat_in[3 * i + 2] = coords[i, 2]
        # Converting the lattice vectors array to a flat array for C-Fortran interoperability
        latticeVectorsFlat = np.zeros((9))
        latticeVectorsFlat[0] = latticeVectors[0, 0]
        latticeVectorsFlat[1] = latticeVectors[0, 1]
        latticeVectorsFlat[2] = latticeVectors[0, 2]

        latticeVectorsFlat[3] = latticeVectors[1, 0]
        latticeVectorsFlat[4] = latticeVectors[1, 1]
        latticeVectorsFlat[5] = latticeVectors[1, 2]

        latticeVectorsFlat[6] = latticeVectors[2, 0]
        latticeVectorsFlat[7] = latticeVectors[2, 1]
        latticeVectorsFlat[8] = latticeVectors[2, 2]
        # Initializing the Coulomb potential array with zeros
        vcoulsFlat = np.zeros(nats)

        # Getting pointers to the input arrays
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        atomicNumbers_ptr = atomicNumbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        vcouls_ptr = vcoulsFlat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Getting pointers to the output arrays
        ham_ptr = hamFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dm_ptr = dmFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evals_ptr = evalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dvals_ptr = dvalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # Call to the fortran funtion
        err = latte_compute_f(
            ctypes.c_int(compflag),
            ctypes.c_int(partIndex + 1),
            ctypes.c_int(nparts),
            ctypes.c_int(norbs),
            ctypes.c_int(ncores),
            ctypes.c_int(nats),
            ctypes.c_int(nats),
            ctypes.c_int(nTypes),
            ctypes.c_double(0.0),
            ctypes.c_double(0.0),
            vcouls_ptr,
            coords_ptr,
            latticeVectors_ptr,
            atomTypes_ptr,
            atomicNumbers_ptr,
            ham_ptr,
            over_ptr,
            dm_ptr,
            charges_ptr,
            evals_ptr,
            dvals_ptr,
            energy_ptr,
            forces_ptr,
            ctypes.c_int(verb),
            ctypes.c_int(newsystem),
            ctypes.c_int(keepmem),
        )

        # Initializing 2D numpy arrays for the hamiltonian and overlap matrices
        hamiltonian = np.zeros((norbs, norbs))
        overlap = np.zeros((norbs, norbs))
        # Filling the hamiltonian and overlap matrices with the flattened output arrays from the Fortran function
        for i in range(norbs):
            hamiltonian[:, i] = hamFlat_out[norbs * i : norbs + norbs * i]
            overlap[:, i] = overFlat_out[norbs * i : norbs + norbs * i]

    else:
        error_at("get_hamiltonian_module", "No specific engine type defined")

    if get_overlap:
        return hamiltonian, overlap
    else:
        return hamiltonian


def get_evals_dvals_modules(
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
    keepmem=False,
):
    """
    Interface to call external engine for computing eigenvalues and contributions (dvals) from each core+halo part to the eigenvectors of the full system.

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
    if eng.name == "ProxyAPython":
        evals, dvals = get_evals_dvals_proxy(
            ham,
            nocc,
            norbsInCore=norbsInCore,
            mu=mu,
            etemp=etemp,
            verb=verb,
        )

    elif eng.name == "ProxyAFortran":
        error_at("get_evals_dvals_modules", "Not implemented yet.")

    elif eng.name == "LATTE":

        # Import the shared library
        latteLibFileName = os.environ["LATTE_PATH"] + "/liblatte.so"

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_compute

        # Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        # In LATTE, compflag = 2 means we want compute evals and dvals in addition to the hamiltonian (compflag = 1)
        compflag = 2
        # Getting the number of atoms through the coordinates array
        nats = len(coords[:, 0])
        # Getting the number of orbitals through the hamiltonian array
        norbs = len(ham[:, 0])

        # Getting atomic numbers
        nTypes = len(symbols)
        # Initializing the atomic numbers array
        atomicNumbers = np.zeros((nTypes), dtype=np.int32)
        # Initializing the atomTypes array
        atomTypes32 = np.zeros((nats), dtype=np.int32)
        # Filling the atomTypes array with the types array
        atomTypes32[:] = types
        # Filling the atomic numbers array with the atomic numbers corresponding to the symbols
        for i in range(len(symbols)):
            atomicNumbers[i] = pt.get_atomic_number(symbols[i])

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros(3 * nats)  # Vectorized forces
        hamFlat_out = np.zeros(norbs * norbs)  # Vectorized hamiltonian
        overFlat_out = np.zeros(norbs * norbs)  # Vectorized overlap
        dmFlat_out = np.zeros(norbs * norbs)  # Vectorized density matrix
        evalsFlat_out = np.zeros(norbs)  # We call this one Flat just for consistency
        dvalsFlat_out = np.zeros(norbs)  # Same here
        chargesFlat_out = np.zeros(nats)  # Same here
        energyFlat_out = np.zeros(1)  # Same here

        # Converting the coordinates array to a flat array for C-Fortran interoperability
        for i in range(nats):
            coordsFlat_in[3 * i] = coords[i, 0]
            coordsFlat_in[3 * i + 1] = coords[i, 1]
            coordsFlat_in[3 * i + 2] = coords[i, 2]
        # Converting the lattice vectors array to a flat array for C-Fortran interoperability
        latticeVectorsFlat = np.zeros((9))
        latticeVectorsFlat[0] = latticeVectors[0, 0]
        latticeVectorsFlat[1] = latticeVectors[0, 1]
        latticeVectorsFlat[2] = latticeVectors[0, 2]

        latticeVectorsFlat[3] = latticeVectors[1, 0]
        latticeVectorsFlat[4] = latticeVectors[1, 1]
        latticeVectorsFlat[5] = latticeVectors[1, 2]

        latticeVectorsFlat[6] = latticeVectors[2, 0]
        latticeVectorsFlat[7] = latticeVectors[2, 1]
        latticeVectorsFlat[8] = latticeVectors[2, 2]

        # Getting pointers to the input arrays
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        atomicNumbers_ptr = atomicNumbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        vcouls_ptr = vcouls.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Getting pointers to the output arrays
        ham_ptr = hamFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dm_ptr = dmFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evals_ptr = evalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dvals_ptr = dvalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call to the fortran funtion
        err = latte_compute_f(
            ctypes.c_int(compflag),
            ctypes.c_int(partIndex + 1),
            ctypes.c_int(nparts),
            ctypes.c_int(norbs),
            ctypes.c_int(int(norbsInCore)),
            ctypes.c_int(nats),
            ctypes.c_int(nats),
            ctypes.c_int(nTypes),
            ctypes.c_double(etemp),
            ctypes.c_double(mu),
            vcouls_ptr,
            coords_ptr,
            latticeVectors_ptr,
            atomTypes_ptr,
            atomicNumbers_ptr,
            ham_ptr,
            over_ptr,
            dm_ptr,
            charges_ptr,
            evals_ptr,
            dvals_ptr,
            energy_ptr,
            forces_ptr,
            ctypes.c_int(verb),
            ctypes.c_int(newsystem),
            ctypes.c_int(keepmem),
        )

        # Initializing 1D numpy arrays for the eigenvalues and dvals
        evals = np.zeros((norbs))
        dvals = np.zeros((norbs))
        # Filling the eigenvalues and dvals arrays with the output arrays from the Fortran function
        evals[:] = evalsFlat_out[:]
        dvals[:] = dvalsFlat_out[:]

    return evals, dvals


def get_density_matrix_modules(
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
    Interface to call external engine for constructing a density matrix.

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
    if eng.name == "ProxyAPython":
        method = eng.method
        accel = eng.accel
        if full_data:
            density_matrix, evals, dvals = get_density_matrix_proxy(
                ham,
                nocc,
                norbsInCore=None,
                method=method,
                accel=accel,
                mu=mu,
                overlap=overlap,
                full_data=full_data,
                verb=False,
            )
        else:
            density_matrix = get_density_matrix_proxy(
                ham,
                nocc,
                norbsInCore=None,
                method=method,
                accel=accel,
                mu=mu,
                overlap=overlap,
                full_data=full_data,
                verb=False,
            )
    elif eng.name == "ProxyAFortran":
        # H needs to be flattened
        norbs = len(ham[:, 0])
        ht = ham.T
        # Specify arguments type as a pointers
        get_density_matrix_fortran.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_bool,
        ]
        # Passing a pointer to Fortran
        hamiltonian_ptr = ham.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        density_matrix = np.zeros((norbs, norbs))
        density_matrix_ptr = density_matrix.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )

        err = get_density_matrix_fortran(
            ctypes.c_int(norbs),
            ctypes.c_int(nocc),
            hamiltonian_ptr,
            density_matrix_ptr,
            ctypes.c_bool(verb),
        )

    elif eng.name == "LATTE":

        # Import the shared library
        latteLibFileName = os.environ["LATTE_PATH"] + "/liblatte.so"

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_compute

        # Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        # In LATTE, compflag = 3 means we want compute the density matrix in addition to compflag = 1 and 2
        compflag = 3
        # Getting the number of atoms through the coordinates array
        nats = len(coords[:, 0])

        # Getting atomic numbers
        nTypes = len(symbols)
        # Initializing the atomic numbers array
        atomicNumbers = np.zeros((nTypes), dtype=np.int32)
        # Initializing the atomTypes array
        atomTypes32 = np.zeros((nats), dtype=np.int32)
        # Filling the atomTypes array with the types array
        atomTypes32[:] = types
        # Filling the atomic numbers array with the atomic numbers corresponding to the symbols
        for i in range(len(symbols)):
            atomicNumbers[i] = pt.get_atomic_number(symbols[i])

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros(3 * nats)  # Vectorized forces
        hamFlat_out = np.zeros(norbs * norbs)  # Vectorized hamiltonian
        overFlat_out = np.zeros(norbs * norbs)  # Vectorized overlap
        dmFlat_out = np.zeros(norbs * norbs)  # Vectorized density matrix
        evalsFlat_out = np.zeros(norbs)  # We call this one Flat just for consistency
        dvalsFlat_out = np.zeros(norbs)  # Same here
        chargesFlat_out = np.zeros(nats)  # Same here
        energyFlat_out = np.zeros(1)  # Same here

        # Converting the coordinates array to a flat array for C-Fortran interoperability
        for i in range(nats):
            coordsFlat_in[3 * i] = coords[i, 0]
            coordsFlat_in[3 * i + 1] = coords[i, 1]
            coordsFlat_in[3 * i + 2] = coords[i, 2]
        # Converting the lattice vectors array to a flat array for C-Fortran interoperability
        latticeVectorsFlat = np.zeros((9))
        latticeVectorsFlat[0] = latticeVectors[0, 0]
        latticeVectorsFlat[1] = latticeVectors[0, 1]
        latticeVectorsFlat[2] = latticeVectors[0, 2]

        latticeVectorsFlat[3] = latticeVectors[1, 0]
        latticeVectorsFlat[4] = latticeVectors[1, 1]
        latticeVectorsFlat[5] = latticeVectors[1, 2]

        latticeVectorsFlat[6] = latticeVectors[2, 0]
        latticeVectorsFlat[7] = latticeVectors[2, 1]
        latticeVectorsFlat[8] = latticeVectors[2, 2]

        # Getting pointers to the input arrays
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        atomicNumbers_ptr = atomicNumbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        vcouls_ptr = vcouls.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Getting pointers to the output arrays
        ham_ptr = hamFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dm_ptr = dmFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evals_ptr = evalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dvals_ptr = dvalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call to the fortran funtion
        err = latte_compute_f(
            ctypes.c_int(compflag),
            ctypes.c_int(partIndex + 1),
            ctypes.c_int(nparts),
            ctypes.c_int(norbs),
            ctypes.c_int(int(norbsInCore)),
            ctypes.c_int(nats),
            ctypes.c_int(nats),
            ctypes.c_int(nTypes),
            ctypes.c_double(etemp),
            ctypes.c_double(mu),
            vcouls_ptr,
            coords_ptr,
            latticeVectors_ptr,
            atomTypes_ptr,
            atomicNumbers_ptr,
            ham_ptr,
            over_ptr,
            dm_ptr,
            charges_ptr,
            evals_ptr,
            dvals_ptr,
            energy_ptr,
            forces_ptr,
            ctypes.c_int(verb),
            ctypes.c_int(newsystem),
            ctypes.c_int(keepmem),
        )

        # Initializing 2D numpy arrays for the density matrix
        density_matrix = np.zeros((norbs, norbs))
        # Initializing 1D numpy arrays for charges
        charges = np.zeros((nats))
        # Filling the density matrix and charges arrays with the output arrays from the Fortran function
        for i in range(norbs):
            density_matrix[:, i] = dmFlat_out[norbs * i : norbs + norbs * i]
        charges[:] = chargesFlat_out[:]

        return density_matrix, charges

    else:
        method = eng.method
        accel = eng.accel
        if full_data:
            density_matrix, evals, dvals = get_density_matrix_proxy(
                ham,
                nocc,
                norbsInCore=None,
                method=method,
                accel=accel,
                mu=mu,
                overlap=overlap,
                full_data=full_data,
                verb=False,
            )
        else:
            density_matrix = get_density_matrix_proxy(
                ham,
                nocc,
                norbsInCore=None,
                method=method,
                accel=accel,
                mu=mu,
                overlap=overlap,
                full_data=full_data,
                verb=False,
            )

    if full_data:
        return density_matrix, evals, dvals
    else:
        return density_matrix


def get_energy_forces_modules(
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
    Interface to call external engine for computing the total energy and forces acting on each atom.

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
    if eng.name == "ProxyAPython":
        error_at("get_energy_force_modules", "Not implemented yet.")

    elif eng.name == "ProxyAFortran":
        error_at("get_energy_force_modules", "Not implemented yet.")

    elif eng.name == "LATTE":

        # Import the shared library
        latteLibFileName = os.environ["LATTE_PATH"] + "/liblatte.so"

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_compute

        # Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        # In LATTE, compflag = 4 means we want compute the energy and forces in addition to compflag = 1, 2 and 3
        compflag = 4
        # Getting the number of atoms through the coordinates array
        nats = len(coords[:, 0])

        # Getting atomic numbers
        nTypes = len(symbols)
        # Initializing the atomic numbers array
        atomicNumbers = np.zeros((nTypes), dtype=np.int32)
        # Initializing the atomTypes array
        atomTypes32 = np.zeros((nats), dtype=np.int32)
        # Filling the atomTypes array with the types array
        atomTypes32[:] = types
        # Filling the atomic numbers array with the atomic numbers corresponding to the symbols
        for i in range(len(symbols)):
            atomicNumbers[i] = pt.get_atomic_number(symbols[i])

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros(3 * nats)  # Vectorized forces
        hamFlat_out = np.zeros(norbs * norbs)  # Vectorized hamiltonian
        overFlat_out = np.zeros(norbs * norbs)  # Vectorized overlap
        dmFlat_out = np.zeros(norbs * norbs)  # Vectorized density matrix
        evalsFlat_out = np.zeros(norbs)  # We call this one Flat just for consistency
        dvalsFlat_out = np.zeros(norbs)  # Same here
        chargesFlat_out = np.zeros(nats)  # Same here
        energyFlat_out = np.zeros(1)  # Same here

        # Converting the coordinates array to a flat array for C-Fortran interoperability
        for i in range(nats):
            coordsFlat_in[3 * i] = coords[i, 0]
            coordsFlat_in[3 * i + 1] = coords[i, 1]
            coordsFlat_in[3 * i + 2] = coords[i, 2]
        # Converting the lattice vectors array to a flat array for C-Fortran interoperability
        latticeVectorsFlat = np.zeros((9))
        latticeVectorsFlat[0] = latticeVectors[0, 0]
        latticeVectorsFlat[1] = latticeVectors[0, 1]
        latticeVectorsFlat[2] = latticeVectors[0, 2]

        latticeVectorsFlat[3] = latticeVectors[1, 0]
        latticeVectorsFlat[4] = latticeVectors[1, 1]
        latticeVectorsFlat[5] = latticeVectors[1, 2]

        latticeVectorsFlat[6] = latticeVectors[2, 0]
        latticeVectorsFlat[7] = latticeVectors[2, 1]
        latticeVectorsFlat[8] = latticeVectors[2, 2]

        # Getting pointers to the input arrays
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        atomicNumbers_ptr = atomicNumbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        latticeVectors_ptr = latticeVectorsFlat.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        vcouls_ptr = vcouls.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Getting pointers to the output arrays
        ham_ptr = hamFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        over_ptr = overFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dm_ptr = dmFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evals_ptr = evalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dvals_ptr = dvalsFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call to the fortran funtion
        err = latte_compute_f(
            ctypes.c_int(compflag),
            ctypes.c_int(partIndex + 1),
            ctypes.c_int(nparts),
            ctypes.c_int(norbs),
            ctypes.c_int(norbsInCore),
            ctypes.c_int(numberOfCoreAtoms),
            ctypes.c_int(nats),
            ctypes.c_int(nTypes),
            ctypes.c_double(etemp),
            ctypes.c_double(mu),
            vcouls_ptr,
            coords_ptr,
            latticeVectors_ptr,
            atomTypes_ptr,
            atomicNumbers_ptr,
            ham_ptr,
            over_ptr,
            dm_ptr,
            charges_ptr,
            evals_ptr,
            dvals_ptr,
            energy_ptr,
            forces_ptr,
            ctypes.c_int(verb),
            ctypes.c_int(newsystem),
            ctypes.c_int(keepmem),
        )

        # Initializing 2D numpy arrays for the forces
        forces = np.zeros((nats, 3))
        # Filling the forces array with the output arrays from the Fortran function
        for i in range(nats):
            forces[i, 0] = forcesFlat_out[i * 3 + 0]
            forces[i, 1] = forcesFlat_out[i * 3 + 1]
            forces[i, 2] = forcesFlat_out[i * 3 + 2]

    else:
        error_at("get_energy_force_modules", "No specific engine type defined")

    return energyFlat_out[0], forces

def get_ppot_energy_expo(coords: ArrayLike,
                         types: ArrayLike) -> float:
    """
    Get the potential energy from a potential.

    Parameters
    ----------
    coords : ArrayLike (Natoms, 3)
        The coordinates of the atoms.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.

    Returns
    -------
    energy : float
        The potential energy.
    """

    energy = get_ppot_energy_expo_proxy(coords,types)

    return energy

def get_ppot_forces_expo(coords: ArrayLike,
                         types: ArrayLike) -> ArrayLike:
    """
    Get the forces from a potential.

    Parameters
    ----------
    coords : ArrayLike (Natoms, 3)
        The coordinates of the atoms.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.

    Returns
    -------
    forces : ArrayLike (Natoms, 3)
        The forces on each atom.
    """

    forces = get_ppot_forces_expo_proxy(coords,types) 

    return forces


def call_latte_modules(eng, Sy, verb=False, newsystem=True):

    if eng.name == "LATTE":

        coords = Sy.coords
        latticeVectors = Sy.latticeVectors
        symbols = np.array(Sy.symbols)[Sy.types]
        types = Sy.types

        # Import the shared library
        latteLibFileName = os.environ["LATTE_PATH"] + "/liblatte.so"

        latteLib = ctypes.CDLL(latteLibFileName)
        latte_compute_f = latteLib.latte_c_bind

        # Periodic table: We use this to pass the chemical atom types as integer instead of characters.
        pt = PeriodicTable()
        compflag = np.zeros(5)
        nats = len(coords[:, 0])
        norbs = Sy.norbs
        err = True

        #    Getting atomic numbers
        # nTypes = len(symbols)
        nTypes = len(Sy.symbols)
        atomTypes32 = np.zeros((nats), dtype=np.int32)
        atomTypes32[:] = Sy.types + 1
        masses = np.zeros(len(Sy.symbols), dtype=np.float64)
        for i in range(len(Sy.symbols)):
            masses[i] = pt.mass[pt.get_atomic_number(Sy.symbols[i])]

        # Vectorizing 2D arrays for C-Fortran interoperability
        coordsFlat_in = np.zeros(3 * nats)  # Vectorized coordinates
        forcesFlat_out = np.zeros((3, nats), order="F")  # Vectorized forces
        chargesFlat_out = np.zeros(nats)  # Same here
        velFlat_out = np.zeros((3, nats), order="F")
        energyFlat_out = np.zeros(1)
        virialFlat_out = np.zeros((6,), order="F")
        for i in range(nats):
            coordsFlat_in[3 * i] = coords[i, 0]
            coordsFlat_in[3 * i + 1] = coords[i, 1]
            coordsFlat_in[3 * i + 2] = coords[i, 2]

        #        latticeVectorsFlat = np.zeros((9))
        xlo = np.zeros(3)
        xhi = np.zeros(3)
        xhi[0] = latticeVectors[0, 0]
        xhi[1] = latticeVectors[1, 1]
        xhi[2] = latticeVectors[2, 2]
        # xlo[:] = -100
        # xhi[:] = 100
        xy, xz, yz = 0.0, 0.0, 0.0
        maxiter = -1

        # Inputs
        coords_ptr = coordsFlat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        atomTypes_ptr = atomTypes32.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        xlo_ptr = xlo.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        xhi_ptr = xhi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        masses_ptr = masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        compflag_ptr = compflag.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        vel_ptr = velFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Outputs
        charges_ptr = chargesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        forces_ptr = forcesFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        energy_ptr = energyFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        virial_ptr = virialFlat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # Call to the fortran funtion
        latte_compute_f(
            compflag_ptr,
            ctypes.c_int(nats),
            coords_ptr,
            atomTypes_ptr,
            ctypes.c_int(nTypes),
            masses_ptr,
            xlo_ptr,
            xhi_ptr,
            ctypes.c_double(xy),
            ctypes.c_double(xz),
            ctypes.c_double(yz),
            forces_ptr,
            ctypes.c_int(maxiter),
            energy_ptr,
            vel_ptr,
            ctypes.c_double(0.5),
            virial_ptr,
            charges_ptr,
            ctypes.c_int(1),
            ctypes.c_bool(err),
        )

        # Back to a 2D array for the forces
        charges = np.zeros((nats))
        charges[:] = chargesFlat_out[:]

        return charges

    else:
        error_at("call_latte_module", "Wrong engine assigned, must be LATTE")

def get_tb_forces_module(ham: ArrayLike,
                         rho: ArrayLike,
                         charges: ArrayLike,
                         field: ArrayLike, # ?
                         coords: ArrayLike,
                         atomTypes: ArrayLike,
                         symbols: ArrayLike,
                         overlap: ArrayLike = None,
                         verb: bool = False):

    """

    Obtain forces from a tight binding model.

    Parameters
    ----------
    ham : ArrayLike (Norb, Norb)
        The Hamiltonian matrix.
    rho : ArrayLike (Norb, Norb)
        The density matrix.
    charges : ArrayLike (Natoms)
        The charges.
    field : ArrayLike
        The applied field.
    coords : ArrayLike (Natoms, 3)
        The coordinates of the atoms.
    symbols: ArrayLike
        The unique chemical elements in the structure.
    atomTypes: ArrayLike (Natoms, )
        The element type of each atom in the system.
    overlap : ArrayLike (Norb, Norb)
        The overlap matrix.
    verb : bool
        Whether to print verbose output.

    Returns
    -------
    forces : ArrayLike (Natoms, 3)
        The forces on each atom.
    """

    forces = get_tb_forces_proxy(ham,rho,charges,field,coords,atomTypes,symbols,overlap=None,verb=False)

    return forces

