"""proxy code a
A prototype engine code that:
    - Computes TB + coulombic forces
    - Coputes band and coulombic energies
This code is only used to guide implemetations and understand which are the 
basic elements needed to interface with the sedacs driver.
"""

import os
import sys

import numpy as np
import scipy.linalg as sp
import sedacs.driver
from sedacs.dev.io import src_path
from proxies.python.hamiltonian_elements import *  
from sedacs.file_io import read_coords_file, write_xyz_coordinates
from proxies.python.dnnprt import *
from proxies.python.proxy_global import *

__all__ = [
    "get_random_coordinates",
    "get_hamiltonian_proxy",
    "get_density_matrix_proxy",
    "get_density_matrix_gpu",
    "get_charges_proxy",
    "get_tb_forces_proxy",
    "get_ppot_energy_expo_proxy",
    "init_proxy_proxy",
    "get_ppot_forces_expo_proxy",
    "build_coul_ham_proxy",
]


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

    def __init__(self, seed):
        self.a = 321
        self.b = 231
        self.c = 13
        self.seed = seed
        self.status = seed * 1000

    def generate(self, low, high):
        """Get a random real number in between low and high."""
        w = high - low
        place = self.a * self.status
        place = int(place / self.b)
        rand = (place % self.c) / self.c
        place = int(rand * 1000000)
        self.status = place
        rand = low + w * rand

        return rand


## Generating random coordinates
# @brief Creates a system of size "nats = Number of atoms" with coordindates having
# a random (-1,1) displacement from a simple cubic lattice with parameter 2.0 Ang.
# This funtion is only used for testing purposes.
# @param nats The total number of atoms
# @return coordinates Position for every atom. z-coordinate of atom 1 = coords[0,2]
#
def get_random_coordinates(nats):
    """Get random coordinates"""
    length = int(nats ** (1 / 3)) + 1
    coords = np.zeros((nats, 3))
    latticeParam = 2.0
    atomsCounter = -1
    myrand = RandomNumberGenerator(111)
    for i in range(length):
        for j in range(length):
            for k in range(length):
                atomsCounter = atomsCounter + 1
                if atomsCounter >= nats:
                    break
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 0] = i * latticeParam + rnd
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 1] = j * latticeParam + rnd
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 2] = k * latticeParam + rnd
    return coords

## Initialize proxy code 
# @brief We will read all the parameters needed for the 
# guest or proxy code to run. Every guest code will need to 
# set up an initialization function and save parameters that 
# need to be read from file only once. Bond integral parameter, 
# pair potential, etc. will be stored in memory by the guest code.
#
def init_proxy_proxy(symbols,bas_per_atom):
    #Some codes will have their own input file
    #read_proxy_input_file()
    #Read pair potentials
    read_ppots("ppots.dat",symbols)
    print_ppots()
    #Read tb parameters
    filename = "aosa_parameters.dat"
    read_tbparams(filename, symbols, bas_per_atom)


## Computes a Hamiltonian based on exponential decay of orbital couplings.
# @author Anders Niklasson
# @brief Computes a hamiltonian based on exponential decays.
# @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
# @param atomTypes Index type for each atom in the system. Type for first atom = type[0] (not used yet)
# @param symbols Symbols for every atom type
# @param verb Verbosity. If True is passed, information is printed
# @para get_overlap If overlap needs to be returned
# @return ham 2D array of Hamiltonian elements
#
def get_hamiltonian_proxy(coords, atomTypes, symbols, verb=False, get_overlap=False):
    """Constructs a simple tight-binding Hamiltonian"""

    # Internal periodic table for the code
    symbols_internal = np.array(["Bl", "H", "C", "N", "O", "P"], dtype=str)
    numel_internal = np.zeros(len(symbols_internal), dtype=int)
    numel_internal[:] = 0, 1, 4, 5, 6, 5
    bas_per_atom = np.zeros(len(symbols_internal), dtype=int)
    bas_per_atom[:] = 0, 1, 4, 4, 4, 4
    spOrbTypes = ["s", "px", "py", "pz"]
    sOrbTypes = ["s"]

    nats = len(coords[:, 0])

    # Map symbols to indices in symbols_internal
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols_internal)}

    # Translate `symbols` to `symbols_internal` indices
    mapped_indices = np.array([symbol_to_index[symbol] for symbol in symbols])

    # Convert atomTypes to `symbols_internal` indices
    atom_internal_indices = mapped_indices[atomTypes]

  
    # Sum the corresponding values in bas_per_atom and numel_internal
    norbs = np.sum(bas_per_atom[atom_internal_indices])
    numel = np.sum(numel_internal[atom_internal_indices])

    ham = np.zeros((norbs, norbs))
    if get_overlap:
        over = np.zeros((norbs, norbs))
    if verb:
        print("Constructing a simple Hamiltonian for the full system")

    colsh = 0
    rowsh = 0
    tbparams = bring_tbparams()
    for i in range(0, nats):
        for ii in range(bas_per_atom[atom_internal_indices[i]]):
            colsh = rowsh
            parI = tbparams[atomTypes[i]][ii]
            for j in range(i, nats):
                if i == j:
                    llimit = ii
                else:
                    llimit = 0
                for jj in range(llimit,bas_per_atom[atom_internal_indices[j]]):
                    parJ = tbparams[atomTypes[j]][jj]
                    hval, sval = get_integral_v1(coords[i],coords[j],parI,parJ)
                    if(get_overlap):
                        over[rowsh,colsh] = sval
                        over[colsh,rowsh] = sval
                    
                    ham[rowsh, colsh] = hval
                    ham[colsh, rowsh] = hval
                    colsh = colsh + 1
            rowsh = rowsh + 1

    if get_overlap:
        return ham, over
    else:
        return ham


sedacs.driver.get_hamiltonian = get_hamiltonian_proxy


## Estimates mu from a matrix using the Girshgorin centers
# @brief It will use the diegonal elements as an approximation 
# for eigenvalues. 
def estimate_mu(ham,elec_temp,kb):

    diag = ham.np.diagonal(ham)


## Add coulombic potentials to the Hamiltonian
# @param ham0 No-SCF Hamiltonian
# @param vcouls Coulombic potentials for every atomic site
# @pparam orbital_based If set to True, coulombic potentials for every orbitals will be
# expected.
# @param hindex will give the orbital index for each atom
# The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`
# @param overlap Overlap matrix for nonorthogonal formulations.
# @param verb Verbosity switch.
#
def build_coul_ham_proxy(ham0, vcouls, types, charges, orbital_based, hindex, overlap=None, verb=False):
    norbs = len(ham0[:, 0])
    vcouls_orbs = np.zeros((norbs), dtype=float)  # Expanded coulombic potentials
    nats = len(hindex[:]) - 1

    tbparams = bring_tbparams()

    if orbital_based:
        error_at("build_coul_ham", "Orbital-based coulombic potential not implemented")
    else:
        for i in range(nats):
            for ii in range(hindex[i], hindex[i + 1]):
                k = ii - hindex[i]
                vcouls_orbs[ii] = vcouls[i] + tbparams[types[i]][k].u * charges[i]
        if overlap is None:
            ham = ham0 + np.diag(vcouls_orbs)
        else:
            vmat = np.diag(vcouls_orbs)
            ham = ham0 + 0.5 * (np.dot(overlap, vmat) + np.dot(vmat, overlap))
    return ham


sedacs.driver.build_coul_ham = build_coul_ham_proxy


## Computes the Density matrix from a given Hamiltonian.
# @author Anders Niklasson
# @brief This will create a Density matrix \f$ \rho \f$
# \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
# where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
#
# @param ham Hamiltonian matrix
# @param nocc Number of occupied orbitals
# @param method Type of algorithm used to compute DM
# @param accel Type of accelerator/special device used to compute DM. Default is No and
# will only use numpy.
# @param mu Chemical potential. If set to none, the calculation will use nocc
# @param elect_temp Electronic temperature
# @param overlap Overlap matrix
# @param verb Verbosity. If True is passed, information is printed.
#
# @return rho Density matrix
#
def get_density_matrix_proxy(ham, nocc, method="Diag", accel="No", mu=None, elect_temp=0.0, overlap=None, verb=False):
    """Calcualtion of the full density matrix from H"""
    if verb:
        print("Computing the Density matrix")

    norbs = len(ham[:, 0])
    ham_orth = np.zeros((norbs, norbs))
    if overlap is not None:
        # Get the inverse overlap factor
        zmat = get_xmat(overlap, method="Diag", accel="No", verbose=False)

        # Orthogonalize Hamiltonian
        ham_orth = np.matmul(np.matmul(np.transpose(zmat), ham), zmat)
    else:
        ham_orth[:, :] = ham[:, :]

    if method == "Diag" and accel == "No":
        evals, evects = sp.eigh(ham_orth)
        homoIndex = nocc - 1
        lumoIndex = nocc
        mu = 0.5 * (evals[homoIndex] + evals[lumoIndex])
        rho = np.zeros((norbs, norbs))
        if verb:
            print("Eigenvalues of H:", evals)
        for i in range(norbs):
            if evals[i] < mu:
                rho = rho + np.outer(evects[:, i], evects[:, i])
        if verb:
            print("Chemical potential = ", mu)
    elif method == "SP2" and accel == "No":
        rho = dnnprt(ham_orth, norbs, nocc, H1=None, refi=False)
        print("No method yet")
    elif method == "SP2" and accel == "PBML":
        print("No method yet")
    else:
        print("The combination of method and accelerator is unknown")
        exit(0)

    if(overlap is not None):
        rho = np.matmul(np.matmul(zmat,rho),np.transpose(zmat)) 
   
    return rho


sedacs.driver.get_density_matrix = get_density_matrix_proxy


## Computes the finite temperature density matrix (DRAFT)
# @todo Write this routine.
def get_density_matrix_T(H, Nocc, Tel, mu0, coreSize, core_ham_dim, S=None, verb=False):
    kB = 8.61739e-5  # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
    if verb:
        print("Computing the renormalized Density matrix")

    if S is not None:
        E_val, Q = scipy.linalg.eigh(H)  ### need S? not ones with S $$$
    else:
        E_val, Q = np.linalg.eigh(H)
    N = len(H[:, 0])

    # print('Q\n', Q[:,0])

    homoIndex = Nocc - 1
    lumoIndex = Nocc
    print("HOMO, LUMO:", E_val[homoIndex], E_val[lumoIndex])
    mu_test = 0.5 * (E_val[homoIndex] + E_val[lumoIndex])  # don't need it
    print(
        N,
        Nocc,
    )
    print("!!!! mu test:\n", mu_test)

    # use mu0 as a guess

    OccErr = 1.0
    beta = 1.0 / (kB * Tel)
    f = np.array([])
    for i in range(N):
        f_i = 1 / (np.exp(beta * (E_val[i] - mu0)) + 1)  # save fi to f
        f = np.append(f, f_i)
        # Occ = Occ + f_i*E_occ[i,k]

    D = sum(np.outer(Q[:, i], Q[:, i] * f[i]) for i in range(Nocc)) * 2
    # np.savetxt('co2_32_dm.txt',D)

    # rho = Q@f_vector@Q.T
    # or
    # rho_ij = SUM_k Q_ik * f_kk * Q_jk

    print("core_ham_dim", core_ham_dim)
    dVals = np.array([])
    for i in range(N):
        dVals = np.append(dVals, np.inner(Q[:core_ham_dim, i], Q[:core_ham_dim, i]))

    return D, E_val, dVals


## Computes the Density matrix from a given Hamiltonian.
# @author Josh Finkelstein
# @brief This will create a "zero-temperature" Density matrix \f$ \rho \f$
# \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
# where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
# using GPU/AI accelerator library
#
# @param H Hamiltonian matrix
# @param Nocc Number of occupied orbitals
# @param verb Verbosity. If True is passed, information is printed.
#
# @return D Density matrix
#
def get_density_matrix_gpu(H, N, Nocc, lib, verb=False):
    """Calcualted the full density matrix from H"""
    if verb:
        print("Computing the Density matrix using GPU/AI accel library")

    # init DM
    D = np.zeros((N, N))
    kbt = 0.1

    # get DM from cusolver diag
    # dm = gpu.dmDNNSP2(H,D,N,Nocc,lib)
    # dm = gpu.dmCheby(H,D,N,Nocc,kbt,lib)
    print("Density matrix=", D)
    # dm = gpu.dmDiag(H,D,N,Nocc,kbt,lib)
    # print("Density matrix=",dm)
    dm = gpu.dmMLSP2(H, D, N, Nocc, lib)
    return D

## Inverse overlap factorization
# @brief Constructs inverse overlap factors given the overlap matrix
# @param over Overlap matrix  
# @param method If a particular needs to be used
# @param accel If an accelerater (hardwar/library/programing model) is used.
# @verb Verbosity switch
##
def get_xmat(over,method="Diag",accel="No",verb=False):

    if(verbose):
        print("In get_xmat ...")

    hdim = len(over[0,:])
    if(method == "Diag" and accel == "No"):
        e,v = sp.eigh(over)
        s = 1./np.sqrt(e)
        zmat = np.zeros((hdim,hdim))
        for i in range(hdim):
            zmat[i, :] = s[i] * v[:, i]
        zmat = np.matmul(v, zmat)
    elif method == "Cholesky":
        pass
    else:
        print("ERROR: Method not implemented")
        exit(0)

    if verbose:
        print("\nZmat Matrix")
        print(zmat)

    return zmat

## Get charges (Mulliken)
# @brief gets the Mulliken charges based on rho
# @param density_matrix Density matrix
def get_charges_proxy(density_matrix,ncores,hindex,overlap=None,verb=False):
    if(verb):
        status_at("get_charges","Getting charges from density matrix")

    charges = np.zeros((ncores))

    if overlap is None:
        fullDiag = np.diag(density_matrix)
        for i in range(ncores):
            for ii in range(hindex[i], hindex[i + 1]):
                charges[i] = charges[i] + (1.0 - 2.0 * fullDiag[ii])
    else:  # S x D
        aux = np.dot(overlap, density_matrix)
        fullDiag = np.diag(aux)
        for i in range(ncores):
            for ii in range(hindex[i], hindex[i + 1]):
                charges[i] = charges[i] + (1.0 - 2.0 * fullDiag[ii])

    if verb:
        msg = "Total Charge for part= " + str(sum(charges))
        status_at("get_charges", msg)

    return charges


## Get TB forces
# \brief Get TB and Coulombic forces from the Hamiltonian, Density matrix
# and charges.
# \param ham Hamiltonian Matrix
# \param rho Density Matrix
# \param field External field
# \param coords Coordinates
# \param atomTypes Atomic types
# \param Symbols Atomic symbols for every type.
##
def get_tb_forces_proxy(ham, rho, charges, field, coords, atomTypes, symbols):
    nats = len(coords[:, 0])
    forces = np.zeros((nats, 3))
    forces_coul = np.zeros((nats, 3))
    forces_field = np.zeros((nats, 3))
    forces_band = np.zeros((nats, 3))
    dl = 0.0001
    coordsp = np.zeros((nats, 3))
    coordsm = np.zeros((nats, 3))
    ham = get_hamiltonian_proxy(coords, atomTypes, symbols, verb=False)
    vaux = np.ones((nats))
    vaux[:] = 0.5
    rho0 = np.diag(vaux)

    for i in range(len(ham[:, 0])):
        # Band Forces from tr(rho dH/dr)
        for k in range(3):
            coordsp[:, :] = coords[:, :]
            coordsp[i, k] = coords[i, k] + dl
            hamp = get_hamiltonian_proxy(coordsp, atomTypes, symbols, verb=False)
            # Hmu = get_pert(field,coordsp,nats)
            # Hp[:,:] = Hp[:,:] + Hmu[:,:]

            coordsm[:, :] = coords[:, :]
            coordsm[i, k] = coords[i, k] - dl
            hamm = get_hamiltonian_proxy(coordsm, atomTypes, symbols, verb=False)
            # Hmu = get_pert(field,coordsm,nats)
            # Hm[:,:] = Hm[:,:] + Hmu[:,:]

            dHdx = (hamp - hamm) / (2 * dl)
            aux = 2 * np.matmul(rho - rho0, dHdx)
            forces_band[i, k] = np.trace(aux)
            print("dHdx", dHdx)

    return forces_band


def get_ppot_forces_expo_proxy(coords, types):
    nats = len(coords[:, 0])
    forces = np.zeros((nats, 3))
    direction = np.zeros(3)
    ppots_in = bring_ppots()
    ppots = np.zeros((4))
    for i in range(nats):
        for j in range(nats):
            if i != j:
                ii = types[i]
                jj = types[j]
                ppots[:] = ppots_in[ii, jj, :]
                direction = coords[i, :] - coords[j, :]
                d = np.linalg.norm(direction)
                direction = direction / d
                arg = ppots[0] + ppots[1] * d + ppots[2] * d**2 + ppots[3] * d**3
                argPrime = ppots[1] + 2 * ppots[2] * d + 3 * ppots[3] * d**2
                forces[i, :] = -direction * argPrime * (np.exp(arg))
                print(forces[i, :])

    return forces


def get_ppot_energy_expo_proxy(coords, types):
    nats = len(coords[:, 0])
    forces = np.zeros((nats, 3))
    direction = np.zeros(3)
    energy = 0.0
    ppots_in = bring_ppots()
    ppot = np.zeros((4))
    for i in range(nats):
        for j in range(nats):
            if i != j:
                ii = types[i]
                jj = types[j]
                ppot[:] = ppots_in[ii, jj, :]
                direction = coords[i, :] - coords[j, :]
                d = np.linalg.norm(direction)
                arg = ppot[0] + ppot[1] * d + ppot[2] * d**2 + ppot[3] * d**3
                energy = energy + np.exp(arg)

    return energy


## Get band energy
# @brief Get the band energy from the density matrix and the non-SCF Hamiltonian
# @param ham0 Non-scf Hamiltonian matrix 
# @param rho0 Atomized density matrix 
# @param rho Density matrix
# @return energy Band energy 
def get_band_energy(ham0,rho0,rho):
    
    energy = 2*np.trace(np.dot(ham0,(rho-rho0)))  # Single-particle/band energy
    return energy

## Get the electronic free energy
# @brief This computed from the entropy of the Fermi distribution
# @param fvals A vector containing the Fermi function for every eigenenergy
# @param kB Boltzan constant (default is in eV/K)
# @param elect_temp Electronic temperature
#
def get_electron_entropy_energy(fvals,kB=8.61739e-5,elec_temp=0):
    if(elec_temp > 1.0E-10):
        entropy = 0.0
        tol = 1.0E-9
        entropy = -kB*np.sum( fvals[:]*np.log10(fvals[:]) + (1.0 - fvals[:])*np.log10(1.0 - fvals[:]) )
    else:
        entropy = 0.0
    energy = -2.0*elec_temp*entropy
    return energy 


## Get pair potential forces
# \brief We will use a simple LJ
# \param coords
#
def get_ppot_forces_LJ(coords):
    # Following Levine
    # VLJ = epsilon*( (a/d)^12 - 2*(b/d)^6 )
    # FLJ = espilon*( -12*( (a**12)/(d**13) ) - 12*( (b**6)/(d**7) ) )
    epsilon = 0.1
    a = 1.0
    b = 1.0
    nats = len(coords[:, 0])
    forces = np.zeros((nats, 3))
    direction = np.zeros(3)
    for i in range(nats):
        for j in range(nats):
            if i != j:
                direction = coords[i, :] - coords[j, :]
                d = np.linalg.norm(direction)
                direction = direction / d
                forces[i, :] = -direction * epsilon * (-12 * ((a**12) / (d**13)) + 12 * ((b**6) / (d**7)))

    return forces


def get_ppot_energy(coords):
    epsilon = 0.1
    a = 1.0
    b = 1.0
    nats = len(coords[:, 0])
    direction = np.zeros(3)
    energy = 0
    for i in range(nats):
        for j in range(nats):
            if i != j:
                direction = coords[i, :] - coords[j, :]
                d = np.linalg.norm(direction)
                energy = energy + epsilon * ((a / d) ** 12 - 2 * (b / d) ** 6)

    energy = energy / 2.0

    return energy


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print("Give the total number of atoms. Example:\n")
        print("proxy_a 100\n")
        sys.exit(0)
    else:
        nats = int(sys.argv[1])

    verb = True
    # coords = get_random_coordinates(nats)
    # atomTypes = np.zeros((nats),dtype=int)
    # symbols = []*nats
    # symbols[:] = "H"
    latticeVectors, symbols, atomTypes, coords = read_coords_file("methane.xyz", lib="None", verb=True)
    read_ppots("ppots.dat", symbols)
    print_ppots()

    bas_per_atom = [4, 1]
    filename = "aosa_parameters.dat"
    tbparams = read_tbparams(filename, symbols, bas_per_atom)
    print("tbparam", tbparams[0].symbol)
    exit(0)

    ham, over = get_hamiltonian_proxy(coords, atomTypes, symbols, get_overlap=True)

    with np.printoptions(precision=3, suppress=True):
        print(ham)
        print(over)
    exit(0)
    gpuLibIn = False  ## need to pass from input file or command line
    occ = int(float(nats) / 2.0)  # Get the total occupied orbitals

    if gpuLibIn == False:
        print("Using CPU for DM construction. Consider installing accelerator library...")
        rho = get_density_matrix_proxy(ham, occ)
    npart = len(coords[:, 0])
    hindex = np.arange(npart + 1, dtype=int)
    field = np.zeros(3)
    charges = get_charges_proxy(rho, npart, hindex, overlap=None, verb=False)
    eforces = get_tb_forces(ham, rho, charges, field, coords, atomTypes, symbols)
    nforces = get_ppot_forces(coords)

    # eenergy = get_tb_energy()
    nenergy = get_ppot_energy(coords)

    coords[0, 0] = coords[0, 0] + 0.001
    nenergyp = get_ppot_energy(coords)

    print((nenergyp - nenergy) / 0.001, nforces[0, 0])
    exit(0)

    print(nforces)
    exit(0)
    print("Hamiltonian matrix=", ham)
    print("Density matrix=", rho)
    print("Forces=", forces)
