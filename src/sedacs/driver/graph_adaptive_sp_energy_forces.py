"""
graph_adaptive_sp_energy_forces.py
====================================
Graph adaptive single-point charge, energy and force solver

"""

import time
from pathlib import Path

from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph, multiply_graphs
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
    sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu=0.0
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

        toc = time.perf_counter()
        print("Time for get_hamiltonian", toc - tic, "(s)")

        tic = time.perf_counter()
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
        subSy.evals = evalsInPart
        toc = time.perf_counter()
        print("Time for get_evals_dvals", toc - tic, "(s)")

        evalsOnRank = collect_evals(evalsOnRank, evalsInPart, verb=True)
        dvalsOnRank = collect_dvals(dvalsOnRank, dvalsInPart, verb=True)

    if is_mpi_available and numranks > 1:
        fullEvals = collect_and_concatenate_vectors(evalsOnRank, comm)
        fullDvals = collect_and_concatenate_vectors(dvalsOnRank, comm)
        comm.Barrier()
    else:
        fullEvals = evalsOnRank
        fullDvals = dvalsOnRank
    # Calculate the global chemical potential from the evals and dvals collected from all subsystems
    mu = get_mu(
        mu,
        fullEvals,
        sdc.etemp,
        int(sy.numel / 2),
        dvals=fullDvals,
        kB=8.61739e-5,
        verb=True,
    )

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

        chargesInPart = chargesInPart[: len(parts[partIndex])]
        subSy.charges = chargesInPart

        forcesInPart = forcesInPart[: len(parts[partIndex])]

        # Save the subsystems list for returning them
        subSysOnRank.append(subSy)

        print("TotalCharge in part", partIndex, sum(chargesInPart))
        # print("Charges in part", chargesInPart)

        toc = time.perf_counter()
        print("Time to get_densityMatrix", toc - tic, "(s)")
        # Building a graph from DMs
        graphOnRank = collect_graph_from_rho(
            graphOnRank,
            rho,
            sdc.gthresh,
            sy.nats,
            sdc.maxDeg,
            partsCoreHalo[partIndex],
            len(parts[partIndex]),
            hindex,
        )

        chargesOnRank = collect_charges(
            chargesOnRank, chargesInPart, parts[partIndex], sy.nats, verb=True
        )
        energyOnRank = collect_energy(energyOnRank, energyInPart, verb=True)
        forcesOnRank = collect_forces(
            forcesOnRank, forcesInPart, parts[partIndex], sy.nats, verb=True
        )

    if is_mpi_available and numranks > 1:
        fullGraphRho = collect_and_sum_matrices(graphOnRank, rank, numranks, comm)
        fullCharges = collect_and_sum_vectors_float(chargesOnRank, rank, numranks, comm)
        fullEnergy = collect_and_sum_vectors_float(energyOnRank, rank, numranks, comm)
        fullForces = collect_and_sum_matrices_float(forcesOnRank, comm)
        # fullForces = collect_and_sum_matrices(forcesOnRank, rank, numranks, comm, dtype=float)
        comm.Barrier()
    else:
        fullGraphRho = graphOnRank
        fullCharges = chargesOnRank
        fullEnergy = energyOnRank
        fullForces = forcesOnRank

    # print_graph(fullGraphRho)
    return (
        fullGraphRho,
        fullCharges,
        fullEnergy[0],
        fullForces,
        subSysOnRank,
        mu,
    )


def get_adaptive_sp_energy_forces(
    sdc, eng, comm, rank, numranks, sy, parts, partsCoreHalo, hindex, graphNL, mu, shadow_md=True
):
    fullGraph = graphNL

    charges = sy.charges
    # Get the path of the current file
    curr_file_path = Path(__file__)
    # Get the directory of the current file
    curr_dir = curr_file_path.parent
    # Get the path of latte parameters
    latte_param_path = curr_dir / Path(
        "../../../parameters/latte/TBparam/electrons.dat"
    )
    latte_tbparams = read_latte_tbparams(latte_param_path)
    hubbard_u = [latte_tbparams[symbol]["HubbardU"] for symbol in sy.symbols]
    hubbard_u = np.array(hubbard_u)[sy.types]
    # Partition the graph
#    parts = graph_partition(
#        sdc, eng, fullGraph, sdc.partitionType, sdc.nparts, sy.coords, True
#    )
#    njumps = 2
#    partsCoreHalo = []
#    numCores = []

    # print("\nCore and halos indices for every part:")
#    for i in range(sdc.nparts):
#        coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps)
#        partsCoreHalo.append(coreHalo)
#        numCores.append(nc)
        #print("core,halo size:", i, "=", nc, nh)
    #    print("coreHalo for part", i, "=", coreHalo)

    sy.coulvs, ecoul, fcoul = get_PME_coulvs(
        charges,
        hubbard_u,
        sy.coords,
        sy.types,
        sy.latticeVectors,
        calculate_forces=1,
    )

    fullGraphRho, charges, energy, forces, subSysOnRank, mu = (
        get_singlePoint_energy_forces(
            sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu
        )
    )
    if shadow_md:
        fcoul = ((2 * charges - sy.charges) / sy.charges)[:, None] * fcoul
        
    energy = energy - ecoul
    forces = forces + fcoul

    # fullGraph = add_graphs(fullGraphRho, graphNL)
    fullGraph = multiply_graphs(fullGraphRho, graphNL)

    AtToPrint = 0

    subSy = System(fullGraphRho[AtToPrint, 0])
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

    return fullGraph, charges, energy, forces, mu, parts, partsCoreHalo, subSysOnRank
