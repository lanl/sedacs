"""coulombic 
Some functions to copute Coulombic interactions

So far: get_coulvs and build_coul_ham
"""

from sedacs.message import *
from sedacs.periodic_table import PeriodicTable
from sedacs.interface_modules import build_coul_ham_module
from sedacs.ewald import ewald_self_energy, calculate_PME_ewald
from sedacs.ewald import init_PME_data, calculate_alpha_and_num_grids, CONV_FACTOR
from sedacs.neighbor_list import NeighborState
import numpy as np
import torch

try:
    from mpi4py import MPI
    mpiLib = True
except ImportError as e:
    mpiLib = False
from multiprocessing import Pool

if mpiLib:
    from sedacs.mpi import *
import time

__all__ = [
    "get_coulvs","get_PME_coulvs","build_coul_ham","get_coulombic_forces"
]

## Get short-range (non periodic) Coulombic potentials 
# @param charges Excess electronic ocupation (this is the negative of the charge vector)
# @param coords Atomic positions
# @param unit_facto Unit factor to account for proper units.
# @param verb Verbosity level.
#
def get_coulvs(charges,coords,atomtypes,latticeVectors,unit_factor=14.3996437701414,verb=False):

    nats = len(charges)
    coulvs = np.zeros(nats)
    alpha = 0.4
    for i in range(nats):
        for j in range(nats):
            if(i != j):
                for nx in range(-1,2):
                    for ny in range(-1,2):
                        for nz in range(-1,2):
                            translation = nx * latticeVectors[0, :] + ny * latticeVectors[1, :] + nz * latticeVectors[2, :]
                            distance =  np.linalg.norm(coords[j,:] - coords[i,:] + translation)
                            erf = torch.erf(alpha * torch.Tensor([distance]))
                            coulvs[i] = coulvs[i] + erf * (unit_factor*charges[j])/(distance)
            else: #We do not know how Hubbard Us are treated bu guest codes
                #coulvs[i] = coulvs[i] + hubbard[types[i]]*q[i]
                if atomtypes[i] == 1:
#                    coulvs[i] = coulvs[i] #+ 5.876141*charges[i]
                    coulvs[i] = coulvs[i] + 11.876141*charges[i]
                elif atomtypes[i] == 0:
                    #coulvs[i] = coulvs[i] #+ 6.054683*charges[i]
                    coulvs[i] = coulvs[i] + 12.054683*charges[i]


    return coulvs

@torch.compile(dynamic=True)
def calculate_dist_dips(pos_T, long_nbr_state):
    disps = long_nbr_state.calculate_displacement(pos_T)
    dists = torch.norm(disps, dim=0)
    nbr_inds = torch.where((dists > long_nbr_state.cutoff) | (dists == 0.0), -1, long_nbr_state.nbr_inds)
    dists = torch.where(dists == 0, 1, dists)

    return disps, dists, nbr_inds

## Get Periodic Coulombic potentials 
# @param charges Excess electronic occupation (this is the negative of the charge vector)
# @param coords Atomic positions
# @param unit_facto Unit factor to account for proper units.
# @param verb Verbosity level.
#
def get_PME_coulvs(charges_np,hubbard_u_np,coords_np,atomtypes_np,lattice_vecs_np,calculate_forces=0):
    np_dtype = np.float64
    dtype = torch.float64
    device = "cuda"
    # NOTE: cutoff <= 0.5 * min(box lengths)
    # so if box lengths are [10.0, 10.0, 10.0], cutoff shuold be at most 5.0.
    # because of the minumum image convention.

    cutoff = 5.0 # real space cutoff
    if cutoff > lattice_vecs_np[0][0]:
        cutoff = float(lattice_vecs_np[0][0]) / 2
    buffer = 1.0 # buffer room
    t_err = 5e-4 # force error
    PME_order = 6

    lattice_vecs = torch.from_numpy(lattice_vecs_np).to(device).to(dtype)
    lattice_lengths = torch.norm(lattice_vecs, dim=1)
    coords = torch.from_numpy(coords_np).to(device).to(dtype)
#    coords = coords - lattice_lengths * torch.floor(coords / lattice_lengths)
    charges = torch.from_numpy(charges_np).to(device).to(dtype)
    hubbard_u = torch.from_numpy(hubbard_u_np).to(device).to(dtype)
    atomtypes = torch.from_numpy(atomtypes_np).to(device).to(torch.int)

    # init PME grid size and related data
    alpha, grid_dimensions = calculate_alpha_and_num_grids(lattice_vecs_np, cutoff, t_err)
    PME_data = init_PME_data(grid_dimensions, lattice_vecs, alpha, PME_order)

    # coords_T: [3, N], coords: [N, 3]
    coords_T = coords.T.contiguous()

    nbr_state = NeighborState(coords_T, lattice_vecs, None, cutoff, is_dense=True, buffer=buffer)

    disps, dists, nbr_inds = calculate_dist_dips(coords_T, nbr_state)
#    breakpoint()
    # When this is first run, torch.compile might give bunch of warnings about complex numbers
    # and overall tuning process, they are safe to ignore
    ewald_e, forces, coulvs  =  calculate_PME_ewald(coords, charges, lattice_vecs,
                            nbr_inds, disps, dists,
                            alpha,
                            cutoff,
                            PME_data, hubbard_u, atomtypes, calculate_forces = calculate_forces, calculate_dq = 1, screening = 1)

    # unit conversio and adding self energy (needed for energy conservation)
    # self_energy, self_energy_dq = ewald_self_energy(charges, alpha, calculate_dq = 1)
    # ewald_e = (ewald_e + self_energy) * CONV_FACTOR
    ewald_e = ewald_e + 0.5 * torch.sum( hubbard_u * charges**2)
    # coulvs = (coulvs + self_energy_dq) * CONV_FACTOR
    coulvs = coulvs + hubbard_u * charges
    ewald_e = ewald_e.double().cpu().detach().numpy()
    coulvs = coulvs.double().cpu().numpy()
    if calculate_forces:
        return coulvs, ewald_e, forces.double().cpu().numpy() #* CONV_FACTOR 
    else:
        return coulvs, ewald_e 

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
def build_coul_ham(engine,ham0,vcouls,types,charges,orbital_based,hindex,overlap=None,verb=False):
    # Call the proper interface
    # If there is no interface, one should write its own Hamiltonian
    if engine.interface == "None":
        raise ValueError("ERROR!!! - Write your own coulombic Hamiltonian.")
    # Tight interface using modules or an external code compiled as a library
    elif engine.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        ham = build_coul_ham_module(engine,ham0,vcouls,types,charges,orbital_based,hindex,overlap=overlap,verb=False)
    # Using any available library. 
    elif engine.interface == "MDI":
        raise NotImplemented("MDI interface not implemented yet")
    # Using unix sockets to interface the codes
    elif engine.interface == "Socket":
        raise NotImplemented("Sockets not implemented yet")
    # Using files as a form of communication and transfering data.
    elif engine.interface == "File":
        raise NotImplemented("File interface not implemented yet")
    else:
        raise ValueError(f"ERROR!!!: Interface type not recognized: '{engine.interface}'. " +
                     f"Use any of the following: Module,File,Socket,MDI")


    return ham 


def get_coulombic_forces(charges,coords,atomTypes,symbols,factor=14.39964377014,field=None):
    nats = len(charges)
    forces_coul = np.zeros((nats,3))
    forces_field = np.zeros((nats,3))
    forces = np.zeros((nats,3))
    for i in range(nats):
        #Coulombic Forces
        for j in range(nats):
            if(i != j):
                distance =  np.linalg.norm(coords[i,:] - coords[j,:])
                direction = (coords[i,:] - coords[j,:])/distance
                forces_coul[i,:] = forces_coul[i,:] - (factor*direction*charges[i]*charges[j])/(distance**2)

        #Field forces
        if(field is not None):
            forces_field[i,:] = forces_field[i,:] + field*charges[i]
            forces = forces_field + forces_coul
        else:
            forces = forces_coul

    return forces
