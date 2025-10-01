"""
graph_adaptive_sp_energy_forces.py
====================================
Graph adaptive single-point charge, energy and force solver

"""

import time
from pathlib import Path
import nvtx 
import pickle

from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph, multiply_graphs, adaptive_halo_expansion, symmetrize_graph
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.sdc_hamiltonian import get_hamiltonian
from sedacs.sdc_density_matrix import get_density_matrix
from sedacs.sdc_energy_forces import get_energy_forces
from sedacs.sdc_evals_dvals import get_evals_dvals
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.mpi import (
    collect_and_sum_matrices,
    collect_and_sum_matrices_float,
    collect_and_sum_vectors_float,
    collect_and_concatenate_vectors,
)
from sedacs.system import System, extract_subsystem, get_hindex
from sedacs.coulombic import get_coulvs, get_PME_coulvs, build_coul_ham
from sedacs.charges import get_charges, collect_charges
from sedacs.evals_dvals import collect_evals, collect_dvals
from sedacs.energy_forces import collect_energy, collect_forces
from sedacs.message import status_at, error_at, warning_at
from sedacs.mixer import diis_mix, linear_mix
from sedacs.chemical_potential import get_mu
from sedacs.entropy import get_entropy
from sedacs.file_io import read_latte_tbparams
import numpy as np

try:
    from mpi4py import MPI

    is_mpi_available = True
except ModuleNotFoundError as e:
    is_mpi_available = False
    error_at(
        "get_adaptiveSCFDM_scf",
        "mpi4py not found, parallelization will not be available",
    )
    raise e

__all__ = ["get_singlePoint_energy_forces", "get_adaptive_sp_energy_forces"]


def get_singlePoint_energy_forces(
    sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu=0.0, alpha=0.7,
):
    """
    Get the single point charges, energy, and forces for the full system from graph-partitioned subsystems.
    This function is called from the graph adaptive SCF loop.
    For each SCF iteration, it follows the following steps:
    1. For each subsystem:
        a. Extract the subsystem from the full system.
        b. Compute the Hamiltonian matrix for the subsystem.
        c. Compute the evals and dvals from the Hamiltonian matrix for the subsystem.
    2. Collect the full evals and dvals across all subsystems.
    3. Compute the global chemical potential for the full system using the collected full evals and dvals.
    4. For each subsystem:
        a. Compute the density matrix for the subsystem.
        b. Compute the charges for the subsystem.
        c. Compute the energy and forces for the subsystem.
    5. Collect the full graph, charges, energy, and forces across all subsystems.

    Parameters
    ----------
    sdc : sedacs driver object
        Refer to driver/init.py for detailed information.
    eng : engine object
        Refer to engine.py for detailed information.
    rank: int
        Rank of the current process in the MPI communicator.
    numranks: int
        Total number of processes in the MPI communicator.
    comm: MPI communicator
        MPI communicator for parallelization.
    parts: list of lists of int
        List of partitions of the full system.
    partsCoreHalo: list of lists of int
        List of core and halo indices for each partition.
    sy: System object
        Refer to system.py for detailed information.
    hindex: list of int
        Orbital index for each atom in the system. The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`.
    mu: float
        Chemical potential for the full system. Default is 0.0.

    Returns
    -------
    fullGraphRho: 2D numpy array, dtype: float
        The full graph collected from all subsystems.
    fullCharges: 1D numpy array, dtype: float
        The mulliken charges for the full system.
    subSysOnRank: list of System objects
        List of subsystem objects for each partition on the current rank.
    mu: float
        The chemical potential for the full system.
    """
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None
    graphRhoOnRank = None
    chargesOnRank = None
    evalsOnRank = None
    dvalsOnRank = None
    energyOnRank = None
    forcesOnRank = None
    subSysOnRank = []
    sy.subSy_list = [None] * partsPerRank

    for partIndex in range(partIndex1, partIndex2):
        numberOfCoreAtoms = len(parts[partIndex])
        subSy = System(len(partsCoreHalo[partIndex]))
        sy.subSy_list[partIndex - partIndex1] = subSy
        subSy.symbols = sy.symbols
        tic = time.perf_counter()
        subSy.coords, subSy.types = extract_subsystem(
            sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex]
        )
        subSy.ncores = len(parts[partIndex])
        toc = time.perf_counter()
        print("Time for extract_subsystem", toc - tic, "(s)")
        partFileName = "subSy" + str(rank) + "_" + str(partIndex) + ".pdb"
        write_pdb_coordinates(partFileName, subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates(
            "subSy" + str(rank) + "_" + str(partIndex) + ".xyz",
            subSy.coords,
            subSy.types,
            subSy.symbols,
        )
        write_pdb_coordinates(
            "subSy_core" + str(rank) + "_" + str(partIndex) + ".pdb",
            subSy.coords[: len(parts[partIndex]), :],
            subSy.types[: len(parts[partIndex])],
            subSy.symbols,
        )
        tic = time.perf_counter()

        # Get some electronic structure elements for the sybsystem
        # This could eventually be computed in the engine if no basis set is
        # provided in the SEDACS input file.
        subSy.norbs, subSy.orbs, subSy.hindex, subSy.numel, subSy.znuc = get_hindex(
            sdc.orbs, subSy.symbols, subSy.types, verb=True
        )
        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        tmpArray = np.zeros(numberOfCoreAtoms)
        tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]
        norbsInCore = np.sum(tmpArray)
        print("Number of orbitals in the core =", norbsInCore)
        nocc = int(float(subSy.numel) / 2.0)  # Get the total occupied orbitals
        nvtx.push_range("get_hamiltonian", color="blue", domain="get_singlePoint_charges")
        subSy.ham, subSy.over, subSy.zmat = get_hamiltonian(
            eng,
            partIndex,
            sdc.nparts,
            norbs,
            sy.latticeVectors,
            subSy.coords,
            subSy.types,
            subSy.symbols,
            verb=False,
            get_overlap=True,
            newsystem=True,
        )
        nvtx.pop_range("get_singlePoint_charges")
        toc = time.perf_counter()
        print("Time for get_hamiltonian", toc - tic, "(s)")

        tic = time.perf_counter()
        nvtx.push_range("get_evals_dvals", color="orange", domain="get_singlePoint_charges")
        subSy.evects, evalsInPart, dvalsInPart = get_evals_dvals(
            eng,
            partIndex,
            sdc.nparts,
            sy.latticeVectors,
            subSy.coords,
            subSy.types,
            subSy.symbols,
            subSy.ham,
            sy.coulvs[partsCoreHalo[partIndex]],
            nocc=nocc,
            norbsInCore=norbsInCore,
            mu=mu,
            etemp=sdc.etemp,
            verb=False,
            newsystem=False,
        )
        nvtx.pop_range("get_singlePoint_charges")
        subSy.evals = evalsInPart
        toc = time.perf_counter()
        print("Time for get_evals_dvals", toc - tic, "(s)")

        evalsOnRank = collect_evals(evalsOnRank, evalsInPart, verb=True)
        dvalsOnRank = collect_dvals(dvalsOnRank, dvalsInPart, verb=True)
    nvtx.push_range("collect_evals_dvals", color="green", domain="get_singlePoint_charges")
    if is_mpi_available and numranks > 1:
        fullEvals = collect_and_concatenate_vectors(evalsOnRank, comm)
        fullDvals = collect_and_concatenate_vectors(dvalsOnRank, comm)
        comm.Barrier()
    else:
        fullEvals = evalsOnRank
        fullDvals = dvalsOnRank
    nvtx.pop_range("get_singlePoint_charges")
    # Calculate the global chemical potential from the evals and dvals collected from all subsystems
    nvtx.push_range("get_mu", color="purple", domain="get_singlePoint_charges")
    mu = get_mu(
        mu,
        fullEvals,
        sdc.etemp,
        int(sy.numel / 2),
        dvals=fullDvals,
        kB=8.61739e-5,
        verb=True,
    )
    nvtx.pop_range("get_singlePoint_charges")
    # Calculate the electronic entropy from the evals and dvals collected from all subsystems
    nvtx.push_range("get_entropy", color="red", domain="get_singlePoint_charges")
    fullEntropy = get_entropy(
        mu,
        fullEvals,
        sdc.etemp,
        fullDvals,
        kB=8.61739e-5,
        verb=True,
    )
    nvtx.pop_range("get_singlePoint_charges")
    for partIndex in range(partIndex1, partIndex2):
        numberOfCoreAtoms = len(parts[partIndex])
        subSy = sy.subSy_list[partIndex - partIndex1]

        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        tmpArray = np.zeros(numberOfCoreAtoms)
        tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]

        norbsInCore = int(np.sum(tmpArray))
        nocc = int(float(subSy.numel) / 2.0)  # Get the total occupied orbitals
        subSy.latticeVectors = sy.latticeVectors

        tic = time.perf_counter()
        nvtx.push_range("get_density_matrix", color="red", domain="get_singlePoint_charges")
        rho, chargesInPart = get_density_matrix(
            eng,
            partIndex,
            sdc.nparts,
            norbs,
            sy.latticeVectors,
            subSy.coords,
            subSy.types,
            subSy.symbols,
            subSy.ham,
            sy.coulvs[partsCoreHalo[partIndex]],
            nocc=nocc,
            norbsInCore=norbsInCore,
            mu=mu,
            etemp=sdc.etemp,
            overlap=subSy.over,
            full_data=False,
            verb=False,
            newsystem=True,
            keepmem=True,
        )
        nvtx.pop_range("get_singlePoint_charges")
        nvtx.push_range("get_energy_forces", color="brown", domain="get_singlePoint_charges")
        energyInPart, forcesInPart = get_energy_forces(
            eng,
            partIndex,
            sdc.nparts,
            norbs,
            sy.latticeVectors,
            subSy.coords,
            subSy.types,
            subSy.symbols,
            subSy.ham,
            sy.coulvs[partsCoreHalo[partIndex]],
            nocc=nocc,
            norbsInCore=norbsInCore,
            numberOfCoreAtoms=numberOfCoreAtoms,
            mu=mu,
            etemp=sdc.etemp,
            verb=False,
            newsystem=False,
            keepmem=False,
        )
        nvtx.pop_range("get_singlePoint_charges")
        chargesInPart = chargesInPart[: len(parts[partIndex])]
        subSy.charges = chargesInPart

        forcesInPart = forcesInPart[: len(parts[partIndex])]

        # Save the subsystems list for returning them
        subSysOnRank.append(subSy)

        print("TotalCharge in part", partIndex, sum(chargesInPart))
        # print("Charges in part", chargesInPart)

        toc = time.perf_counter()
        print("Time to get_densityMatrix", toc - tic, "(s)")
        # Adaptively expand the halo for the subsystem
        nvtx.push_range("adaptive_halo_expansion", color="pink", domain="get_singlePoint_charges")
        graphOnRank = adaptive_halo_expansion(
            graphOnRank,
            rho,
            sdc.gthresh,
            sy.nats,
            sdc.maxDeg,
            partsCoreHalo[partIndex],
            parts[partIndex],
            subSy.hindex,
            sy.coords,
            alpha=alpha,
        )
        graphRhoOnRank = collect_graph_from_rho(
            graphRhoOnRank,
            rho,
            sdc.gthresh,
            sy.nats,
            sdc.maxDeg,
            partsCoreHalo[partIndex],
            len(parts[partIndex]),
            hindex,
        )
        nvtx.pop_range("get_singlePoint_charges")
        chargesOnRank = collect_charges(
            chargesOnRank, chargesInPart, parts[partIndex], sy.nats, verb=True
        )
        energyOnRank = collect_energy(energyOnRank, energyInPart, verb=True)
        forcesOnRank = collect_forces(
            forcesOnRank, forcesInPart, parts[partIndex], sy.nats, verb=True
        )
    nvtx.push_range("collect_graph_and_charges", color="green", domain="get_singlePoint_charges")
    if is_mpi_available and numranks > 1:
        fullGraphHalo = collect_and_sum_matrices_float(graphOnRank, comm)
        fullGraphRho = collect_and_sum_matrices_float(graphRhoOnRank, comm)
        fullCharges = collect_and_sum_vectors_float(chargesOnRank, rank, numranks, comm)
        fullEnergy = collect_and_sum_vectors_float(energyOnRank, rank, numranks, comm)
        fullForces = collect_and_sum_matrices_float(forcesOnRank, comm)
        comm.Barrier()
    else:
        fullGraphHalo = graphOnRank
        fullGraphRho = graphRhoOnRank
        fullCharges = chargesOnRank
        fullEnergy = energyOnRank
        fullForces = forcesOnRank
    nvtx.pop_range("get_singlePoint_charges")
    # print_graph(fullGraphRho)
    return (
        fullGraphHalo,
        fullGraphRho,
        fullCharges,
        fullEnergy[0],
        fullEntropy,
        fullForces,
        subSysOnRank,
        mu,
    )


def get_adaptive_sp_energy_forces(
    sdc, eng, comm, rank, numranks, sy, parts, partsCoreHalo, hindex, graph, mu, alpha=0.7, shadow_md=True, device="cuda",
):
    nvtx.push_range("SP energy forces", color="blue", domain="get_adaptiveSCFDM")
    charges = sy.charges
    nvtx.push_range("PME solver", color="green", domain="get_adaptiveSCFDM")
    if rank == 0:
        sy.coulvs, ecoul, fcoul, nbr_inds, disps, dists, PME_alpha, PME_data = get_PME_coulvs(
            charges,
            sy.hubbard_u,
            sy.coords,
            sy.types,
            sy.latticeVectors,
            calculate_forces=1,
            device=device,
        )
    else:
        ecoul = None
        fcoul = None

    if is_mpi_available and numranks > 1:
        sy.coulvs = comm.bcast(sy.coulvs, root=0)
        ecoul = comm.bcast(ecoul, root=0)
        fcoul = comm.bcast(fcoul, root=0)
    nvtx.pop_range("get_adaptiveSCFDM")
    
    nvtx.push_range("get_singlePoint_charges", color="orange", domain="get_adaptiveSCFDM")
    fullGraphHalo, fullGraphRho, charges, energy, entropy, forces, subSysOnRank, mu = (
        get_singlePoint_energy_forces(
            sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu, alpha=alpha,
        )
    )
    nvtx.pop_range("get_adaptiveSCFDM")
    if shadow_md:
        fcoul = ((2 * charges - sy.charges) / sy.charges)[:, None] * fcoul
        
    energy = energy - ecoul
    forces = forces + fcoul

    fullGraphHalo = symmetrize_graph(fullGraphHalo)
    fullGraphRho = symmetrize_graph(fullGraphRho)
    fullGraph = add_graphs(fullGraphRho, fullGraphHalo)
    AtToPrint = 0

    subSy = System(fullGraph[AtToPrint, 0])
    subSy.symbols = sy.symbols
    subSy.coords, subSy.types = extract_subsystem(
        sy.coords,
        sy.types,
        sy.symbols,
        fullGraph[AtToPrint, 1 : fullGraph[AtToPrint, 0] + 1],
    )
    if rank == 0:
        write_pdb_coordinates(
            "subSyG_fin.pdb", subSy.coords, subSy.types, subSy.symbols
        )
        write_xyz_coordinates(
            "subSyG_fin.xyz", subSy.coords, subSy.types, subSy.symbols
        )

    # if rank == 0:
    #    with open('energy_forces.log', 'w') as f:
    #        f.write(str(energy) + '\n')
    #        f.write(np.array2string(forces[0], separator=', ') + '\n')
    #    with open('forces.pkl', 'wb') as f:
    #        pickle.dump(forces, f)
        # with open('energy.log', 'a') as f:
        #     f.write(str(band_energy) + ',' + str(ecoul) + '\n')
    nvtx.pop_range("get_adaptiveSCFDM")

    return fullGraph, charges, energy, entropy, forces, mu, parts, partsCoreHalo, subSysOnRank
