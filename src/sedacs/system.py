"""system
Some functions to create, read, and manupulate coordinates of a chemical system

So far: Creates random coordinates; reads and writes xyz and pdb files;
creates a neighbor list.
"""

import sys

import numpy as np

global aseLib
try:
    import ase1.io

    aseLib = True
except:
    aseLib = False

global mdtrajLib
try:
    import mdtraj as md

    mdtrajLib = True
except:
    mdtrajLib = False

from sedacs.message import *
from sedacs.periodic_table import PeriodicTable

# from sdc_out import *
try:
    from mpi4py import MPI

    mpiLib = True
except ImportError as e:
    mpiLib = False
from multiprocessing import Pool

if mpiLib:
    from sedacs.mpi import *
import time

from sedacs.types import ArrayLike


__all__ = [
    "System",
    "Trajectory",
    "RandomNumberGenerator",
    "get_random_coordinates",
    "parameters_to_vectors",
    "vectors_to_parameters",
    "coords_cart_to_frac",
    "coords_frac_to_cart",
    "get_volBox",
    "coords_dvec_nlist",
    "build_nlist",
    "build_nlist_small",
]


## Chemical system type
# @brief To be used only when really needed!
#
class System:
    """The sedacs System class.

    Attributes
    ----------
    nats : int
        Number of atoms
    ncores : int
        Number of core atoms
    ntypes : int
        Number of atom types
    types : ArrayLike (Natoms, )
        Type for each atom, e.g., the first atom is of type "types[0]"
    coords : ArrayLike (Natoms, 3)
        Coordinates for each atom, e.g., z-coordinate of the first atom is coords[0,2]
    charges : ndarray (Natoms, )
        Charged (orbital base population) for each atom
    vels : ArrayLike (Natoms, 3)
        Velocities for each atom
    latticeVectors : ndarray
        3x3 matrix containing the lattice vectors for the simulation box.
        latticeVectors[1,:] = first lattice vector.
    symbols : list
        Symbols for each atom type, e.g, the element symbol of the first atom is symbols[types[0]]
    coulvs : ArrayLike (Natoms, )
        Coulombic potentials
    norbs : int
        Number of orbitals
    orbs : ArrayLike (Ntypes, )
        Number of atomic orbital for each type
    znuc : (Ntypes, )
        Number of total valence electrons for each type
    hindex : ndarray or None
        Orbital indices. The orbital indices for atom i goes from `hindex[i]` to `hindex[i+1]-1`
    resNames : list or None
        Residue names/Molecule name
    resIds : list or None
        Residue/molecule id


    
    """

    def __init__(self, nats=1):
        """
        Constructor for the sedacs System object. This constructor simply initializes all 
        
        """
        ## Number of atoms
        self.nats = nats
        ## Number of core atoms
        self.ncores = self.nats
        ## Number of atom types
        self.ntypes = 0
        ## Type for each atom, e.g., the first atom is of type "types[0]"
        self.types = np.zeros(self.nats, dtype=int)
        ## Coordinates for each atom, e.g., z-coordinate of the frist atom is coords[0,2]
        self.coords = np.zeros((self.nats, 3), dtype=float)
        ## Hubbard U for each atom
        self.hubbard_u = np.zeros((self.nats), dtype=float)
        ## Charged (orbital base population) for each atom
        self.charges = np.zeros((self.nats), dtype=float)
        ## Coordinates for each atom, e.g., z-coordinate of the frist atom is coords[0,2]
        self.vels = np.zeros((self.nats, 3), dtype=float)
        ## LatticeVectors. 3x3 matrix containing the lattice vectors for the simulation box.
        # latticeVectors[1,:] = first lattice vector.
        self.latticeVectors = np.zeros((3, 3), dtype=float)
        ## Symbols for each atom type, e.g, the element symbol of the first atom is symbols[types[0]]
        self.symbols = PeriodicTable().symbols[self.types]
        ## Coulombic potentials
        self.coulvs = np.zeros(self.nats, dtype=int)
        ## Number of orbitals
        self.norbs = 0
        ## Number of atomic orbital for each type
        self.orbs = np.zeros(self.ntypes)
        ## Number of totatal valence electrons for each type
        self.znuc = np.zeros(self.ntypes)
        ## Orbital indices. The orbital indices for atom i goes from `hindex[i]` to `hindex[i+1]-1`
        self.hindex = None
        ## Residue names/Molecule name 
        self.resNames = None
        ## Residue/molecule id 
        self.resIds = None 
        ## List of subsystems
        self.subSy_list = None
        ## Kernel preconditioner
        self.ker = None
        ## Hamiltonian matrix
        self.ham = None
        ## Overlap matrix
        self.over = None
        ## Congruence transformation
        self.zmat = None
        ## Eigenvectors matrix
        self.evects = None
        ## Eigenvalues
        self.evals = None

    def extract_types_and_symbols(self,symbols_list_for_all_atoms):
        """
            Returns a new list containing only the unique symbols.
        """
        self.symbols = [None]*self.nats
        self.types = [None]*self.nats
        tmp = set()
        for item in symbols_list_for_all_atoms:
            if item not in tmp:
                self.symbols.append(item)
                tmp.add(item)
        for i in range(self.nats):
            for j in range(len(self.symbols)):
                if(symbols_list_for_all_atoms[i] == self.symbols[j]):
                    self.types[i] = j



    def print_summary(self) -> None:
        """
        Prints a summary of the sedacs System object.
        """
        s = """nats = {nats}
ncores = {ncores}
ntypes = {ntypes}
coords[0] = {coords}
latticeVectors = {latticeVectors}
symbols = {symbols}
orbs = {orbs}"""
        print(
            s.format(
                nats=self.nats,
                ncores=self.ncores,
                ntypes=self.ntypes,
                coords=self.coords[0],
                latticeVectors=self.latticeVectors,
                symbols=self.symbols,
                orbs=self.orbs,
            )
        )

    if mdtrajLib:

        def from_mdtraj(self, traj, frame_idx=0):
            table, bonds = traj.topology.to_dataframe()
            self.symbols = table["element"].to_numpy().tolist()
            self.nats = len(self.symbols)
            self.ncores = self.nats
            self.ntypes = self.nats
            # multiply the following two by 10. to convert to Angstroms
            self.coords = 10.0 * traj.xyz[frame_idx].astype(float)
            if traj.unitcell_vectors is not None:
                self.latticeVectors = 10.0 * traj.unitcell_vectors[frame_idx].astype(float)
            else:
                import warnings

                warnings.warn(
                    "No unit cell information in this mdtraj trajectory. If unit cell information is desired, it can be obtained by loading a .pdb file as a trajectory."
                )
            self.orbs = np.ones(self.nats, dtype=int)


## Trajectory type
# @brief To handle simulation results
# @param system The system description with topology info
# @param coords Coordinates at each snapshot (Angstrom)
# @param latticeVectors Simulation box vectors (Angstrom)
# @param value Generic atom value (e.g. electron population) at each time point
# @param timestep Time difference between frames, for uniform sampling (ps)
# @param time Time at each frame (ps)


class Trajectory:
    """A prototype for the trajectory type."""

    def __init__(self,
                 sys: System = None,
                 nats: int = 1,
                 nframes: int = 1,
                 timestep: float = 0.00025):
        """
        Constructor for the sedacs Trajectory class.

        Parameters
        ----------
        sys : System
            The system description with topology info
        nats : int
            Number of atoms.
        nframes : int
            Number of frames.
        timestep : float
            Time step in picoseconds.
        """
        if sys is None:
            self.system = System(nats)
        else:
            self.system = sys
            nats = sys.nats
        self.coords = np.zeros((nframes, nats, 3), dtype=float)
        self.latticeVectors = None
        self.values = None
        self.timestep = timestep
        self.time = np.ones(nframes, dtype=float) * self.timestep

    if mdtrajLib:

        def from_mdtraj(self, traj):
            self.system = System()
            self.system.from_mdtraj(traj)
            self.coords = 10.0 * traj.xyz.astype(float)
            if traj.unitcell_vectors is not None:
                self.latticeVectors = 10.0 * traj.unitcell_vectors.astype(float)
                self.time = traj.time.astype(float)
            if traj.n_frames >= 2:
                self.timestep = traj.timestep

    def slice(self,
              first: int = 0,
              last: int = None,
              skip: int = 1) -> None:
        """
        Slices the trajectory.

        Parameters
        ----------
        first : int
            The first frame to slice.
        last : int
            The last frame to slice.
        skip : int
            The number of frames to skip between images in the slice.

        Returns
        -------
        None
        """

        if last is None:
            last = len(self.coords)
        self.coords = self.coords[first : last + 1 : skip]
        self.time = self.time[first : last + 1 : skip]
        if self.values is not None:
            self.values = self.values[first : last + 1 : skip]
        if self.latticeVectors is not None:
            self.latticeVectors = self.latticeVectors[first : last + 1 : skip]
        self.timestep = self.timestep * skip

    def load_prg_xyz(self,
                     fname: str) -> None:
        """
        Loads a PRG XYZ file containing an MD trajectory.

        Parameters
        ----------
        fname : str
            The name of the PRG XYZ file.

        Returns
        -------
        None
        """
        with open(fname) as f:
            lines = np.array(f.readlines())
            nats = int(lines[0])
            if nats != self.system.nats:
                raise Exception("Number of atoms must be same as that in system")
            mask = np.ones(len(lines), dtype=bool)
            mask[np.arange(0, len(lines), nats + 2)] = False
            mask[np.arange(1, len(lines), nats + 2)] = False
            lines = lines[mask]
            xyzc = np.loadtxt(lines.tolist(), usecols=range(1, 5)).astype(float)
            nframes = int(len(xyzc) / nats)
            xyzc = np.reshape(xyzc, (nframes, nats, 4))
            self.coords = xyzc[:, :, 0:3]
            self.values = xyzc[:, :, 3]

    if mdtrajLib:

        def save_xtc(self, fname):
            from mdtraj.formats import XTCTrajectoryFile

            with XTCTrajectoryFile(fname, "w") as f:
                if self.latticeVectors is not None:
                    f.write(self.coords / 10.0, box=self.latticeVectors / 10.0)
                else:
                    f.write(
                        self.coords / 10.0,
                        box=np.repeat(self.system.latticeVectors[np.newaxis, :, :] / 10.0, len(self.coords), axis=0),
                    )

        def save_dcd(self, fname):
            from mdtraj.formats import DCDTrajectoryFile

            with DCDTrajectoryFile(fname, "w") as f:
                if self.latticeVectors is not None:
                    latticeVectors = self.latticeVectors
                else:
                    latticeVectors = np.repeat(
                        self.system.latticeVectors[np.newaxis, :, :] / 10.0, len(self.coords), axis=0
                    )
                latticeParams = vectors_to_parameters(latticeVectors)
                f.write(self.coords, cell_lengths=latticeParams[:, 0:3], cell_angles=latticeParams[:, 3:6])

        def save_netcdf(self, fname):
            from mdtraj.formats import NetCDFTrajectoryFile

            with NetCDFTrajectoryFile(fname, "w") as f:
                if self.latticeVectors is not None:
                    latticeVectors = self.latticeVectors
                else:
                    latticeVectors = np.repeat(
                        self.system.latticeVectors[np.newaxis, :, :] / 10.0, len(self.coords), axis=0
                    )
                latticeParams = vectors_to_parameters(latticeVectors)
                f.write(self.coords, cell_lengths=latticeParams[:, 0:3], cell_angles=latticeParams[:, 3:6])


def parameters_to_vectors(paramA: float,
                          paramB: float,
                          paramC: float,
                          angleAlpha: float,
                          angleBeta: float,
                          angleGamma: float,
                          latticeVectors: ArrayLike,
                          verb: bool = False) -> ArrayLike:

    """
    Transforms lattice parameters to vectors. The inverse function of vectors_to_parameters.

    Parameters
    ----------
    paramA : float
        The first parameter.
    paramB : float
        The second parameter.
    paramC : float
        The third parameter.
    angleAlpha : float
        The angle between the second and third lattice vectors.
    angleBeta : float
        The angle between the first and third lattice vectors.
    angleGamma : float
        The angle between the first and second lattice vectors.
    latticeVectors : np.ndarray
        The lattice vectors.
    verb : bool
        Whether to print verbose output.

    Returns
    -------
    latticeVectors : np.ndarray (3, 3)
        The lattice vectors.

    """

    # pi = 3.1415926535897932384626433832795
    pi = np.pi

    angleAlpha = 2.0 * pi * angleAlpha / 360.0
    angleBeta = 2.0 * pi * angleBeta / 360.0
    angleGamma = 2.0 * pi * angleGamma / 360.0

    latticeVectors[0, 0] = paramA
    latticeVectors[0, 1] = 0
    latticeVectors[0, 2] = 0

    latticeVectors[1, 0] = paramB * np.cos(angleGamma)
    latticeVectors[1, 1] = paramB * np.sin(angleGamma)
    latticeVectors[1, 2] = 0

    latticeVectors[2, 0] = paramC * np.cos(angleBeta)
    latticeVectors[2, 1] = paramC * (np.cos(angleAlpha) - np.cos(angleGamma) * np.cos(angleBeta)) / np.sin(angleGamma)
    latticeVectors[2, 2] = np.sqrt(paramC**2 - latticeVectors[2, 0] ** 2 - latticeVectors[2, 1] ** 2)

    return latticeVectors


## Transforms the lattice vectors to lattice parameers
# @param latticeVectors 3x3 array containing the lattice vectors
# @param verb Verbosity level.
#
def vectors_to_parameters(latticeVectors, verb=False):
    """
    Transforms latticeVectors to parameters. The inverse function of parameters_to_vectors.

    Parameters
    ----------
    latticeVectors : np.ndarray (3, 3)
        The lattice vectors.
    verb : bool
        Whether to print verbose output.

    Returns
    -------
    parameters : np.ndarray (6,)
        The lattice parameters.
    """
    if latticeVectors.ndim == 3:
        a = np.sqrt(np.einsum("ij,ij->i", latticeVectors[:, 0], latticeVectors[:, 0]))
        b = np.sqrt(np.einsum("ij,ij->i", latticeVectors[:, 1], latticeVectors[:, 1]))
        c = np.sqrt(np.einsum("ij,ij->i", latticeVectors[:, 2], latticeVectors[:, 2]))
        adotb = np.einsum("ij,ij->i", latticeVectors[:, 0], latticeVectors[:, 1])
        adotc = np.einsum("ij,ij->i", latticeVectors[:, 0], latticeVectors[:, 2])
        bdotc = np.einsum("ij,ij->i", latticeVectors[:, 1], latticeVectors[:, 2])
        alpha = np.arccos(bdotc / b / c) * 180.0 / np.pi
        beta = np.arccos(adotc / a / c) * 180.0 / np.pi
        gamma = np.arccos(adotb / a / b) * 180.0 / np.pi
        alpha[np.abs(alpha - 90.0) <= 1.0e-5] = 90.0
        beta[np.abs(alpha - 90.0) <= 1.0e-5] = 90.0
        gamma[np.abs(alpha - 90.0) <= 1.0e-5] = 90.0
    else:
        a = np.sqrt(np.inner(latticeVectors[0], latticeVectors[0]))
        b = np.sqrt(np.inner(latticeVectors[1], latticeVectors[1]))
        c = np.sqrt(np.inner(latticeVectors[2], latticeVectors[2]))
        adotb = np.inner(latticeVectors[0], latticeVectors[1])
        adotc = np.inner(latticeVectors[0], latticeVectors[2])
        bdotc = np.inner(latticeVectors[1], latticeVectors[2])
        alpha = np.arccos(bdotc / b / c) * 180.0 / np.pi
        beta = np.arccos(adotc / a / c) * 180.0 / np.pi
        gamma = np.arccos(adotb / a / b) * 180.0 / np.pi
        if abs(alpha - 90.0) <= 1.0e-5:
            alpha = 90.0
        if abs(beta - 90.0) <= 1.0e-5:
            beta = 90.0
        if abs(gamma - 90.0) <= 1.0e-5:
            gamma = 90.0
    return np.transpose(np.array((a, b, c, alpha, beta, gamma)))


## Simple random number generator
# This is important in order to compare across codes
# written in different languages.
#
# To initialize:
# \verbatim
#   myRand = rand(123)
# \endverbatim
# where the argument of rand is the seed.
#
# To get a random number between "low" and "high":
# \verbatim
#   rnd = myRand.get_rand(low,high)
# \endverbatim
#
class RandomNumberGenerator:
    """To generate random numbers."""

    def __init__(self, seed: int):

        """
        Constructor for the RandomNumberGenerator class.

        Parameters
        ----------
        seed : int
            The seed for the random number generator.
        """

        self.a = 321
        self.b = 231
        self.c = 13
        self.seed = seed
        self.status = seed * 1000

    def generate(self, low: float, high: float) -> float:

        """
        Get a random real number in between low and high.

        Parameters
        ----------
        low : float
            The lower bound of the random number.
        high : float
            The upper bound of the random number.

        Returns
        -------
        rand : float
            The random number.
        """

        w = high - low
        place = self.a * self.status
        place = int(place / self.b)
        rand = (place % self.c) / self.c
        place = int(rand * 1000000)
        self.status = place
        rand = low + w * rand

        return rand


## Generating random coordinates
# Creates a system of size length^3 with coorindates having
# a random (-1,1) displacement from a simple cubic lattice
# with parameter 2.0 Ang.
#
# @param lenght The total number of point in x, y, and z directions.
# @return coordinates Position for every atoms. z-coordinate of atom 1 = coords[0,2]
#
# \verbatim
# NumberOfAtoms = len(coordinates[:,0])
# \endverbatim
#
def get_random_coordinates(length):
    """Get random coordinates real number in betwee low and high."""
    nats = length**3
    coords = np.zeros((nats, 3))
    latticeParam = 2.0
    atomsCounter = -1
    myrand = RandomNumberGenerator(123)
    for i in range(length):
        for j in range(length):
            for k in range(length):
                atomsCounter = atomsCounter + 1
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 0] = i * latticeParam + rnd
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 1] = j * latticeParam + rnd
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 2] = k * latticeParam + rnd
    return coords


## Extract subsystem
# @brief Extracs a chemical subsystem (coordinates and atomic types)
# from a larger system using a set of indices.
# @param coords Position for every atom. z-coordinate of atom 1 = coords[0,2]
# @param types Index type for each atom in the system. Type for first atom = type[0]
# @param symbols Symbols for every atom type
# @param part list of index for the part to be extracted
# @return subSyCoords Subsystem atomic coordinates
# @return subSyTypes Subsystem atomic types
#
# @todo Add test!
def extract_subsystem(coords, types, symbols, part):
    subSyNats = len(part)
    subSyCoords = np.zeros((subSyNats, 3))
    subSyTypes = np.zeros((subSyNats), dtype=int)
    for k in range(subSyNats):
        i = part[k]
        subSyCoords[k, :] = coords[i, :]
        subSyTypes[k] = types[i]
    return subSyCoords, subSyTypes


## Gets the volume of the simulation box
# @brief Given an array of lattice vectors, it return the box volume
# @param latticeVector Lattice vectors in an array. latice_vectors[0,2] means the z-coordinate
# of the first lattice vector.
# @return volBox Volume of the cell.
#
def get_volBox(latticeVectors, verb=False):
    volBox = 0.0

    pi = 3.14159265358979323846264338327950
    a1xa2 = np.zeros((3))
    a2xa3 = np.zeros((3))
    a3xa1 = np.zeros((3))

    a1xa2[0] = latticeVectors[0, 1] * latticeVectors[1, 2] - latticeVectors[0, 2] * latticeVectors[1, 1]
    a1xa2[1] = -latticeVectors[0, 0] * latticeVectors[1, 2] + latticeVectors[0, 2] * latticeVectors[1, 0]
    a1xa2[2] = latticeVectors[0, 0] * latticeVectors[1, 1] - latticeVectors[0, 1] * latticeVectors[1, 0]

    a2xa3[0] = latticeVectors[1, 1] * latticeVectors[2, 2] - latticeVectors[1, 2] * latticeVectors[2, 1]
    a2xa3[1] = -latticeVectors[1, 0] * latticeVectors[2, 2] + latticeVectors[1, 2] * latticeVectors[2, 0]
    a2xa3[2] = latticeVectors[1, 0] * latticeVectors[2, 1] - latticeVectors[1, 1] * latticeVectors[2, 0]

    a3xa1[0] = latticeVectors[2, 1] * latticeVectors[0, 2] - latticeVectors[2, 2] * latticeVectors[0, 1]
    a3xa1[1] = -latticeVectors[2, 0] * latticeVectors[0, 2] + latticeVectors[2, 2] * latticeVectors[0, 0]
    a3xa1[2] = latticeVectors[2, 0] * latticeVectors[0, 1] - latticeVectors[2, 1] * latticeVectors[0, 0]

    # Get the volume of the cell
    volBox = latticeVectors[0, 0] * a2xa3[0] + latticeVectors[0, 1] * a2xa3[1] + latticeVectors[0, 2] * a2xa3[2]

    return volBox


def coords_cart_to_frac(cart_coords, latticeVectors):
    A_transpose = latticeVectors
    A_transpose_inv = np.linalg.inv(A_transpose)
    frac_coords = np.matmul(cart_coords, A_transpose_inv)
    return frac_coords


def coords_frac_to_cart(frac_coords, latticeVectors):
    A_transpose = latticeVectors
    cart_coords = np.matmul(frac_coords, A_transpose)
    return cart_coords


def coords_dvec_nlist(coords_in, nn, nl, nlTr, latticeVectors, rank=0, numranks=1, api="include_dr"):
    mpiON = False
    if mpiLib and (numranks > 1):
        mpiON = True

    nats = len(coords_in[:, 0])
    if mpiON:
        comm = MPI.COMM_WORLD
    natsPerRank = int(nats / numranks)
    if rank == numranks - 1:
        natsInRank = nats - natsPerRank * (numranks - 1)
    else:
        natsInRank = natsPerRank

    if np.allclose(np.diag(np.diagonal(latticeVectors)), latticeVectors) == False:
        coords = coords_cart_to_frac(coords_in, latticeVectors)
        method = "nonortho"
    else:
        coords = coords_in
        latticeLengths = np.diagonal(latticeVectors)
        method = "ortho"

    dvecChunk = np.zeros([natsInRank, nl.shape[1], 3], dtype=coords.dtype)
    drChunk = np.zeros([natsInRank, nl.shape[1]], dtype=coords.dtype)
    dvec = np.zeros((nl.shape[0], nl.shape[1], 3), dtype=coords_in.dtype)
    dr = np.zeros(nl.shape, dtype=coords_in.dtype)
    for j in range(natsInRank):
        i = natsPerRank * rank + j
        for k in range(3):
            if method == "ortho":
                dvecChunk[i, 0 : nn[i], k] = (coords[i, k] - coords[nl[i, 0 : nn[i]], k]) - nlTr[
                    i, 0 : nn[i], k
                ] * latticeLengths[k]
            else:
                dvecChunk[i, 0 : nn[i], k] = (coords[i, k] - coords[nl[i, 0 : nn[i]], k]) - nlTr[i, 0 : nn[i], k]
        if method == "nonortho":
            dvecChunk[i, 0 : nn[i]] = coords_frac_to_cart(dvecChunk[i, 0 : nn[i]], latticeVectors)
        drChunk[i, 0 : nn[i]] = np.linalg.norm(dvecChunk[i, 0 : nn[i]], axis=1)
    if mpiON:
        dr = collect_matrix_from_chunks(drChunk, nats, natsPerRank, rank, numranks, comm)
        dvec[:, :, 0] = collect_matrix_from_chunks(dvecChunk[:, :, 0], nats, natsPerRank, rank, numranks, comm)
        dvec[:, :, 1] = collect_matrix_from_chunks(dvecChunk[:, :, 1], nats, natsPerRank, rank, numranks, comm)
        dvec[:, :, 2] = collect_matrix_from_chunks(dvecChunk[:, :, 2], nats, natsPerRank, rank, numranks, comm)

    else:
        dr = drChunk
        dvec = dvecChunk
    if api == "include_dr":
        return dvec, dr
    else:
        return dvec


## Neighbor list for small systems
# @brief It will bild a neighbor list using an "all to all" approach
# @param coords System coordinates. coords[7,1]: y-coordinate of atom 7.
# @param latticeVectors. Lattice vectors of the system box. latticeVectors[1,2]: z-coordinate of vector 1.
# @param nl neighbor list type: a simple 2D array indicating the neighbors of each atom.
# @param rank MPI rank
#
# @todo Add test!
def build_nlist_small(coords, latticeVectors, rcut, rank=0, numranks=1, verb=False):
    if verb:
        print("Building neighbor list for small systems ...")
    
    nats = len(coords[:,0])
    nl = np.zeros((nats,nats+1),dtype=int)
    nlTrX = np.zeros((nats),dtype=int)
    nlTrY = np.zeros((nats),dtype=int)
    nlTrZ = np.zeros((nats),dtype=int)

    for i in range(nats):
        #print(np.arange(0,i,1),np.arange(i+1,nats,1))

        nl[i,1:i+1] = np.arange(0,i,1,dtype = int)
        nl[i,i+1:nats] = np.arange(i+1,nats,1,dtype = int)
        nl[i,0] = nats - 1

        #print("nl",i,nl[i,1:nats+1])    

    return nl, nlTrX, nlTrY, nlTrZ


## Neighbor list
# @brief It will bild a neighbor list using an "all to all" approach
# @param coords System coordinates. coords[7,1]: y-coordinate of atom 7.
# @param latticeVectors. Lattice vectors of the system box. latticeVectors[1,2]: z-coordinate of vector 1.
# @param nl neighbor list type: a simple 2D array indicating the neighbors of each atom.
# @param rank MPI rank
#
# @todo Add test!
def build_nlist_integer(coords, latticeVectors, rcut, rank=0, numranks=1, verb=False):
    if verb:
        print("Building neighbor list ...")

    mpiON = False
    if mpiLib and (numranks > 1):
        mpiON = True

    nats = len(coords[:, 0])
    if mpiON:
        comm = MPI.COMM_WORLD
    natsPerRank = int(nats / numranks)
    if rank == numranks - 1:
        natsInRank = nats - natsPerRank * (numranks - 1)
    else:
        natsInRank = natsPerRank
    natsInBuff = max(natsInRank, nats - natsPerRank * (numranks - 1))

    # We will have approximatly [(4/3)*pi * rcut^3 * atomic density] number of neighbors.
    # A very large atomic density could be 1 atom per (1.0 Ang)^3 = 1 atoms per Ang^3
    volBox = get_volBox(latticeVectors, verb=False)
    density = 1.0
    maxneigh = np.min([int(3.14592 * (4.0 / 3.0) * density * rcut**3), nats])

    # We assume the box is orthogonal
    maxx = np.max(coords[:, 0])
    maxy = np.max(coords[:, 1])
    maxz = np.max(coords[:, 2])
    minx = np.min(coords[:, 0])
    miny = np.min(coords[:, 1])
    minz = np.min(coords[:, 2])

    smallReal = 0.1  # To ensure the borders are contained in the limiting boxes

    # This part is for trying integer discretization of the coordinates
    dr = 0.1  # Discretization param
    cx = np.zeros((nats), dtype=int)
    cy = np.zeros((nats), dtype=int)
    cz = np.zeros((nats), dtype=int)
    lx = latticeVectors[0, 0] / dr
    ly = latticeVectors[1, 1] / dr
    lz = latticeVectors[2, 2] / dr
    for i in range(nats):
        cx[i] = int(coords[i, 0] / dr)
        cy[i] = int(coords[i, 1] / dr)
        cz[i] = int(coords[i, 2] / dr)

    nx = int((maxx - minx) / (rcut))
    ny = int((maxy - miny) / (rcut))
    nz = int((maxz - minz) / (rcut))
    dx = (maxx - minx + smallReal) / float(nx)
    dy = (maxy - miny + smallReal) / float(ny)
    dz = (maxz - minz + smallReal) / float(nz)

    ix = int((maxx - minx + smallReal) / (dx))  # small box x-index of atom i
    iy = int((maxy - miny + smallReal) / (dy))  # small box y-index
    iz = int((maxz - minz + smallReal) / (dz))  # small box z-index

    nBox = nx * ny * nz
    maxInBox = int(density * (rcut) ** 3)  # Upper bound for the max number of atoms per box
    inbox = np.zeros((nBox, maxInBox), dtype=int)
    inbox[:, :] = -1
    totPerBox = np.zeros((nBox), dtype=int)
    totPerBox[:] = -1
    boxOfI = np.zeros((nats), dtype=int)
    xBox = np.zeros((nBox), dtype=int)
    yBox = np.zeros((nBox), dtype=int)
    zBox = np.zeros((nBox), dtype=int)
    ithFromXYZ = np.zeros((nx, ny, nz), dtype=int)

    # Search for the box coordinate and index of every atom
    for i in range(nats):
        # Index every atom respect to the discretized position on the simulation box.
        # tranlation = coords[i,:] - origin !For the general case we need to make sure coords are > 0
        ix = int((coords[i, 0] - minx) / (dx))  # small box x-index of atom i
        iy = int((coords[i, 1] - miny) / (dy))  # small box y-index
        iz = int((coords[i, 2] - minz) / (dz))  # small box z-index

        if ix > nx or ix < 0:
            print("Error in box index")
            sys.exit(0)
        if iy > ny or iy < 0:
            print("Error in box index")
            sys.exit(0)
        if iz > nz or iz < 0:
            print("Error in box index")
            sys.exit(0)

        ith = ix + iy * nx + iz * nx * ny  # Get small box index
        boxOfI[i] = ith

        # From index to box coordinates
        # print("ith",ith,nBox,ix,iy,iz,nx,ny)
        xBox[ith] = ix
        yBox[ith] = iy
        zBox[ith] = iz

        # From box coordinates to index
        ithFromXYZ[ix, iy, iz] = ith

        totPerBox[ith] = totPerBox[ith] + 1  # How many per box
        if totPerBox[ith] >= maxInBox:
            print("Exceeding the max in box allowed")
            sys.exit(0)
        inbox[ith, totPerBox[ith]] = i  # Who is in box ith

    for i in range(nBox):  # Correcting - from indexing to
        totPerBox[i] = totPerBox[i] + 1

    rcut2 = rcut * rcut

    # For each atom we will look around to see who are its neighbors
    def get_neighs_of(i, boxOfI, ithFromXYZ, inbox, latticeVectors):
        nlVect = np.zeros((maxneigh), dtype=int)
        nlTrVectX = np.zeros((maxneigh), dtype=int)
        nlTrVectY = np.zeros((maxneigh), dtype=int)
        nlTrVectZ = np.zeros((maxneigh), dtype=int)
        translation = np.zeros((3))
        cnt = 0
        # Which box it beongs to
        ibox = boxOfI[i]
        # Look inside the box and the neighboring boxes
        xBoxIbox = xBox[ibox]
        yBoxIbox = yBox[ibox]
        zBoxIbox = zBox[ibox]
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    # Get neigh box coordinate
                    jxBox = xBoxIbox + ix
                    jyBox = yBoxIbox + iy
                    jzBox = zBoxIbox + iz
                    tx = 0.0
                    ty = 0.0
                    tz = 0.0
                    tr = False
                    if jxBox < 0:
                        jxBox = nx - 1
                        tx = -1
                        tr = True
                    elif jxBox == nx:
                        jxBox = 0
                        tx = 1
                        tr = True
                    if jyBox < 0:
                        jyBox = ny - 1
                        ty = -1
                        tr = True
                    elif jyBox == ny:
                        jyBox = 0
                        ty = 1
                        tr = True
                    if jzBox < 0:
                        jzBox = nz - 1
                        tz = -1
                        tr = True
                    elif jzBox == nz:
                        jzBox = 0
                        tz = 1
                        tr = True

                    # Get the neigh box index
                    jbox = ithFromXYZ[jxBox, jyBox, jzBox]
                    # if (tr):
                    #    translation = tx*latticeVectors[0,:] + ty*latticeVectors[1,:] + tz*latticeVectors[2,:]
                    # else:
                    #    translation[:] = 0.0

                    trlx = tx * lx
                    trly = ty * ly
                    trlz = tz * lz
                    # Now loop over the atoms in the jbox
                    for j in range(totPerBox[jbox]):
                        jj = inbox[jbox, j]  # Get atoms in box j
                        if tr:
                            #   coordsNeigh = coords[jj,:] + translation
                            cnx = cx[jj] + trlx
                            cny = cy[jj] + trly
                            cnz = cz[jj] + trlz
                        else:
                            #    coordsNeigh = coords[jj,:]
                            cnx = cx[jj]
                            cny = cy[jj]
                            cnz = cz[jj]

                        # distance = (coords[i,0] - coordsNeigh[0])**2 + \
                        #        (coords[i,1] - coordsNeigh[1])**2 + \
                        #        (coords[i,2] - coordsNeigh[2])**2

                        distance = float((cx[i] - cnx) ** 2 + (cy[i] - cny) ** 2 + (cz[i] - cnz) ** 2) * dr**2

                        if (distance < rcut2) and (distance > 1.0e-12):
                            cnt = cnt + 1
                            nlVect[cnt] = jj  # jj is a neighbor of i by some translation
                            nlTrVectX[cnt] = tx
                            nlTrVectY[cnt] = ty
                            nlTrVectZ[cnt] = tz
        nlVect[0] = cnt
        return (nlVect, nlTrVectX, nlTrVectY, nlTrVectZ)

    nlChunk = np.empty([natsInRank, maxneigh], dtype=int)
    nlTrChunkX = np.empty([natsInRank, maxneigh], dtype=int)
    nlTrChunkY = np.empty([natsInRank, maxneigh], dtype=int)
    nlTrChunkZ = np.empty([natsInRank, maxneigh], dtype=int)

    for k in range(natsInRank):
        i = natsPerRank * (rank) + k
        nlVect, nlTrVectX, nlTrVectY, nlTrVectZ = get_neighs_of(i, boxOfI, ithFromXYZ, inbox, latticeVectors)
        nlChunk[k, :] = nlVect[:]
        nlTrChunkX[k, :] = nlTrVectX[:]
        nlTrChunkY[k, :] = nlTrVectY[:]
        nlTrChunkZ[k, :] = nlTrVectZ[:]

    nl = np.empty([nats, maxneigh], dtype=int)
    nlTrX = np.empty([nats, maxneigh], dtype=int)
    nlTrY = np.empty([nats, maxneigh], dtype=int)
    nlTrZ = np.empty([nats, maxneigh], dtype=int)

    if mpiON:
        nl = collect_matrix_from_chunks(nlChunk, nats, natsPerRank, rank, numranks, comm)
        nlTrX = collect_matrix_from_chunks(nlTrChunkX, nats, natsPerRank, rank, numranks, comm)
        nlTrY = collect_matrix_from_chunks(nlTrChunkY, nats, natsPerRank, rank, numranks, comm)
        nlTrZ = collect_matrix_from_chunks(nlTrChunkZ, nats, natsPerRank, rank, numranks, comm)
    else:
        nl = nlChunk
        nlTrX = nlTrChunkX
        nlTrY = nlTrChunkY
        nlTrZ = nlTrChunkZ

    return (nl, nlTrX, nlTrY, nlTrZ)


## Vectorized neighbor list
# @brief It will bild a neighbor list using an "all to all" approach
# @param coords System coordinates. coords[7,1]: y-coordinate of atom 7.
# @param latticeVectors. Lattice vectors of the system box. latticeVectors[1,2]: z-coordinate of vector 1.
# @param rcut Distance cutoff
# @param nl neighbor list type: a simple 2D array indicating the neighbors of each atom.
# @param rank MPI rank
#
# @todo Add test!
def build_nlist(coords_cart, latticeVectors, rcut, rank=0, numranks=1, verb=False, api="old"):
    # Use fractional coords for everything before the distance cutoff

    coords = coords_cart_to_frac(coords_cart, latticeVectors)

    if verb:
        print("Building neighbor list ...")

    mpiON = False
    if mpiLib and (numranks > 1):
        mpiON = True

    nats = len(coords[:, 0])
    if mpiON:
        comm = MPI.COMM_WORLD
    natsPerRank = int(nats / numranks)
    if rank == numranks - 1:
        natsInRank = nats - natsPerRank * (numranks - 1)
    else:
        natsInRank = natsPerRank

    # We will have approximatly [(4/3)*pi * rcut^3 * atomic density] number of neighbors.
    # A very large atomic density could be 1 atom per (1.0 Ang)^3 = 1 atoms per Ang^3
    volBox = get_volBox(latticeVectors, verb=False)
    density = 1.0
    maxneigh = np.min([int(3.14592 * (4.0 / 3.0) * density * rcut**3), nats])
    boxSize = rcut

    latticeLength = np.linalg.norm(latticeVectors, axis=1)
    boxVectors = rcut * np.array([
        latticeVectors[0] / latticeLength[0],
        latticeVectors[1] / latticeLength[1],
        latticeVectors[2] / latticeLength[2],
    ])

    nx = int(latticeLength[0] / boxSize)
    ny = int(latticeLength[1] / boxSize)
    nz = int(latticeLength[2] / boxSize)
    nBox = nx * ny * nz + 1  # Box Zero will be null box with no neighbors
    maxInBox = int(density * get_volBox(boxVectors, verb=False))  # Upper bound for the max number of atoms per box
    inbox = -np.ones((nBox, maxInBox), dtype=int)
    totPerBox = -np.ones((nBox), dtype=int)
    boxOfI = np.zeros((nats), dtype=int)
    xBox = np.zeros((nBox), dtype=int)
    yBox = np.zeros((nBox), dtype=int)
    zBox = np.zeros((nBox), dtype=int)
    ithFromXYZ = np.zeros((nx, ny, nz), dtype=int)  # Null box (= 0) unless there are atoms
    neighbox = np.zeros((nBox, 27), dtype=int)

    # Search for the box coordinate and index of every atom

    for i in range(nats):
        # Index every atom respect to the discretized position on the simulation box.
        ix = int(coords[i, 0] * nx) % nx  # small box x-index of atom i
        iy = int(coords[i, 1] * ny) % ny  # small box x-index of atom i
        iz = int(coords[i, 2] * nz) % nz  # small box x-index of atom i

        ith = ix + iy * nx + iz * nx * ny + 1  # Get small box index, leave zero for null box
        boxOfI[i] = ith

        # From index to box coordinates
        xBox[ith] = ix
        yBox[ith] = iy
        zBox[ith] = iz

        # From box coordinates to index
        ithFromXYZ[ix, iy, iz] = ith

        totPerBox[ith] = totPerBox[ith] + 1  # For now this is the atom index in the box
        if totPerBox[ith] >= maxInBox:
            print("Exceeding the max in box allowed")
            sys.exit(0)
        inbox[ith, totPerBox[ith]] = i  # Who is in box ith

    for i in range(nBox):  # Now this array will hold the total number of atoms in each box
        totPerBox[i] = totPerBox[i] + 1

    # For each box get a flat list of neighboring boxes (including self)
    for i in range(nBox):
        neighbox[i, 0] = i
        j = 1
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    if not (ix == 0 and iy == 0 and iz == 0):
                        # Get neigh box coordinate
                        neighx = xBox[i] + ix
                        neighy = yBox[i] + iy
                        neighz = zBox[i] + iz
                        jxBox = neighx % nx
                        jyBox = neighy % ny
                        jzBox = neighz % nz

                        # Get the neigh box index
                        neighbox[i, j] = ithFromXYZ[jxBox, jyBox, jzBox]
                        j += 1

    # Vectorized neighbor list calc for atom i
    def get_neighs_of(i, coords, neighbox, boxOfI, inbox, latticeVectors):
        cnt = -1
        # Get the list of all atoms in neighboring boxes
        boxneighs = inbox[neighbox[boxOfI[i]]]
        # Shorten the long dimension for speedup on CPU
        # Eliminate the atom itself from the neighbor list
        boxneighs = boxneighs[np.logical_and(boxneighs != -1, boxneighs != i)]
        # Calculate the distances to all atoms in neighboring boxes
        dvec = np.zeros((len(boxneighs), 3), dtype=coords.dtype)
        nlTrBoxneigh = np.zeros(dvec.shape, dtype=int)
        nlVect = np.zeros(maxneigh, dtype=int)
        nlTrVect = np.zeros((maxneigh, 3), dtype=int)
        dvecVect = np.zeros((maxneigh, 3), dtype=coords.dtype)
        drVect = np.zeros((maxneigh), dtype=coords.dtype)
        for k in range(3):
            # Compute the integer lattice vector translation first
            dvec[:, k] = coords[i, k] - coords[boxneighs, k] + 0.5
            nlTrBoxneigh[:, k] = np.floor(dvec[:, k])
            # Now use the translation to compute the periodic displacement
            dvec[:, k] = dvec[:, k] - nlTrBoxneigh[:, k] - 0.5
        dvec = coords_frac_to_cart(dvec, latticeVectors)
        distance = np.linalg.norm(dvec, axis=1)
        # Filter the list according to the threshold
        nlSel = np.where(distance < rcut)[0]
        cnt = len(nlSel)
        nlVect[1 : cnt + 1] = boxneighs[nlSel]
        nlTrVect[1 : cnt + 1] = nlTrBoxneigh[nlSel]
        dvecVect[1 : cnt + 1] = dvec[nlSel]
        drVect[1 : cnt + 1] = distance[nlSel]
        nlVect[0] = cnt
        nlTrVect[0] = cnt
        return (nlVect, nlTrVect[:, 0], nlTrVect[:, 1], nlTrVect[:, 2], dvecVect, drVect)

    nlChunk = np.empty([natsInRank, maxneigh], dtype=int)
    nlTrChunkX = np.empty([natsInRank, maxneigh], dtype=int)
    nlTrChunkY = np.empty([natsInRank, maxneigh], dtype=int)
    nlTrChunkZ = np.empty([natsInRank, maxneigh], dtype=int)
    if api == "include_dvec":
        dvecChunkX = np.empty([natsInRank, maxneigh], dtype=coords.dtype)
        dvecChunkY = np.empty([natsInRank, maxneigh], dtype=coords.dtype)
        dvecChunkZ = np.empty([natsInRank, maxneigh], dtype=coords.dtype)
        drChunk = np.empty([natsInRank, maxneigh], dtype=coords.dtype)

    for k in range(natsInRank):
        i = natsPerRank * (rank) + k
        nlVect, nlTrVectX, nlTrVectY, nlTrVectZ, dvecVect, drVect = get_neighs_of(
            i, coords, neighbox, boxOfI, inbox, latticeVectors
        )
        nlChunk[k, :] = nlVect[:]
        nlTrChunkX[k, :] = nlTrVectX[:]
        nlTrChunkY[k, :] = nlTrVectY[:]
        nlTrChunkZ[k, :] = nlTrVectZ[:]
        if api == "include_dvec_dr":
            drChunk[k, :] = drVect[:]
            dvecChunkX[k, :] = dvecVect[:, 0]
            dvecChunkY[k, :] = dvecVect[:, 1]
            dvecChunkZ[k, :] = dvecVect[:, 2]

    # Gather the neighbor list across MPI ranks
    nl = np.empty([nats, maxneigh], dtype=int)
    nlTrX = np.empty([nats, maxneigh], dtype=int)
    nlTrY = np.empty([nats, maxneigh], dtype=int)
    nlTrZ = np.empty([nats, maxneigh], dtype=int)
    if api == "include_dvec_dr":
        dvecX = np.empty([nats, maxneigh], dtype=coords.dtype)
        dvecY = np.empty([nats, maxneigh], dtype=coords.dtype)
        dvecZ = np.empty([nats, maxneigh], dtype=coords.dtype)
        dr = np.empty([nats, maxneigh], dtype=coords.dtype)

    if mpiON:
        tic = time.perf_counter()
        nl = collect_matrix_from_chunks(nlChunk, nats, natsPerRank, rank, numranks, comm)
        nlTrX = collect_matrix_from_chunks(nlTrChunkX, nats, natsPerRank, rank, numranks, comm)
        nlTrY = collect_matrix_from_chunks(nlTrChunkY, nats, natsPerRank, rank, numranks, comm)
        nlTrZ = collect_matrix_from_chunks(nlTrChunkZ, nats, natsPerRank, rank, numranks, comm)
        if api == "include_dvec_dr":
            dr = collect_matrix_from_chunks(drChunk, nats, natsPerRank, rank, numranks, comm)
            dvecX = collect_matrix_from_chunks(dvecChunkX, nats, natsPerRank, rank, numranks, comm)
            dvecY = collect_matrix_from_chunks(dvecChunkY, nats, natsPerRank, rank, numranks, comm)
            dvecZ = collect_matrix_from_chunks(dvecChunkZ, nats, natsPerRank, rank, numranks, comm)

        t_gather_nl = time.perf_counter() - tic
        if rank == 0 and verb:
            print("Time for gathering nl arrays= ", t_gather_nl, " sec")
    else:
        nl = nlChunk
        nlTrX = nlTrChunkX
        nlTrY = nlTrChunkY
        nlTrZ = nlTrChunkZ
        if api == "include_dvec_dr":
            dr = drChunk
            dvecX = dvecChunkX
            dvecY = dvecChunkY
            dvecZ = dvecChunkZ

    if api == "new":
        return (nl[:, 0], nl[:, 1:], np.moveaxis(np.array([nlTrX[:, 1:], nlTrY[:, 1:], nlTrZ[:, 1:]]), 0, -1))
    elif api == "include_dvec_dr":
        return (
            nl[:, 0],
            nl[:, 1:],
            np.moveaxis(np.array([nlTrX[:, 1:], nlTrY[:, 1:], nlTrZ[:, 1:]]), 0, -1),
            np.moveaxis(np.array([dvecX[:, 1:], dvecY[:, 1:], dvecZ[:, 1:]]), 0, -1),
            dr[:, 1:],
        )
    elif api == "old":
        return (nl, nlTrX, nlTrY, nlTrZ)
    else:
        raise_error("build_nlist", "api must be new, old, or dvec")


## Get hindex
# @brief hindex will give the orbital index for each atom
# in the system.
# The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`
# @param orbs A dictionary that give the total orbitals (basis set size)
# for each atomic type.
# @param symbols Symbol for each atom type. Symbol for first atom type = symbols[0]
# @param types Index type for each atom in the system. Type for first atom = type[0]
# @return norbs Total number of orbitals
# @return hindex Orbital index for each atom in the system
# @retunn numel Total number of electrons
#
# def get_hindex(orbs_for_every_symbol, valency, symbols, types, verb=False):
def get_hindex(orbs_for_every_symbol, symbols, types, valency=None, verb=False):
    nats = len(types[:])
    ntypes = len(symbols)
    hindex = np.zeros((nats + 1), dtype=int)
    norbs = 0
    ptable = PeriodicTable()
    numel = 0
    #verb = True

    norbs_for_every_type = np.zeros((ntypes),dtype=int)
    numel_for_every_type = np.zeros((ntypes),dtype=int)
    cnt = 0
    for symbol in symbols:
        atomic_number = ptable.get_atomic_number(symbol)
        try:
            norbs_for_atom = orbs_for_every_symbol[symbol]
        except:
            #If there is no specified basis set we default econftb in ptable
            econftb = ptable.econftb[atomic_number]
            if(econftb == "s"):
                norbs_for_atom = 1 #1(s)
            elif(econftb == "sp"):
                norbs_for_atom = 4 #1(s) + 3(p)
            elif(econftb == "spd"):
                norbs_for_atom = 10 #2(s) + 3(p) + 5(d)
            elif(econftb == "spdf"):
                norbs_for_atom = 17 #2(s) + 3(p) + 5(d) + 7(f)

            msg = "No number of orbitals provided for species " + symbol \
                    +", Using maxbonds in periodic table instead"
            warning_at("get_hindex",msg)
        numel_for_atom = ptable.numel[atomic_number]
        norbs_for_every_type[cnt] = norbs_for_atom
        numel_for_every_type[cnt] = numel_for_atom
        print(verb)
        if verb:
            print("type,symb,orb,valence",cnt, symbol,norbs_for_atom,numel_for_atom)
        cnt += 1

    norbs = 0
    for i in range(nats):
        hindex[i] = norbs
        norbs_for_atom = norbs_for_every_type[types[i]]        
        numel = numel + numel_for_every_type[types[i]]
        norbs = norbs + norbs_for_atom
    hindex[nats] = norbs
   
    return norbs,norbs_for_every_type,hindex,numel,numel_for_every_type
