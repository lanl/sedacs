"""
graph_adaptive_kernel_scf.py
====================================
Graph adaptive self-consistent charge solver with kernel

"""

import time
import torch
import numpy as np
from copy import deepcopy

from sedacs.graph import (
    add_graphs,
    collect_graph_from_rho,
    print_graph,
    multiply_graphs,
    adaptive_halo_expansion,
)
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.sdc_hamiltonian import get_hamiltonian
from sedacs.sdc_density_matrix import get_density_matrix
from sedacs.sdc_evals_dvals import get_evals_dvals
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.mpi import (
    collect_and_sum_matrices,
    collect_and_sum_vectors_float,
    collect_and_concatenate_vectors,
    collect_and_sum_matrices_float,
)
from sedacs.system import System, extract_subsystem, get_hindex
from sedacs.coulombic import get_PME_coulvs, build_coul_ham
from sedacs.charges import get_charges, collect_charges
from sedacs.evals_dvals import collect_evals, collect_dvals
from sedacs.message import status_at, error_at, warning_at
from sedacs.mixer import diis_mix, linear_mix
from sedacs.chemical_potential import get_mu
from sedacs.file_io import read_latte_tbparams

# from sedacs.driver.graph_kernel_byparts import get_kernel_byParts, apply_kernel_byParts, rankN_update_byParts
from sedacs.driver.graph_kernel_byparts import (
    get_kernel_byParts,
    apply_kernel_byParts,
    rankN_update_byParts,
)

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

__all__ = ["get_singlePoint_charges", "get_adaptiveSCFDM"]


def get_singlePoint_charges(
    sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, gscf, mu=0.0, alpha=0.7,
):
    """
    Get the single point charges for the full system from graph-partitioned subsystems.
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
    5. Collect the full graph and charges across all subsystems.

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
    gscf: int
        Graph adaptive SCF iteration number.
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
        norbsInCore = int(np.sum(tmpArray))
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
            etemp=sdc.etemp,
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

        norbsInCore = np.sum(tmpArray)
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

        chargesInPart = chargesInPart[: len(parts[partIndex])]
        #        subSy.charges = chargesInPart

        # Save the subsystems list for returning them
        subSysOnRank.append(subSy)

        print("TotalCharge in part", partIndex, sum(chargesInPart))
        # print("Charges in part", chargesInPart)

        toc = time.perf_counter()
        print("Time to get_densityMatrix and get_charges", toc - tic, "(s)")
        # Adaptively expand the halo for the subsystem
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
        chargesOnRank = collect_charges(
            chargesOnRank, chargesInPart, parts[partIndex], sy.nats, verb=True
        )

    if is_mpi_available and numranks > 1:
        fullGraphHalo = collect_and_sum_matrices_float(graphOnRank, comm)
        fullCharges = collect_and_sum_vectors_float(chargesOnRank, rank, numranks, comm)
        comm.Barrier()
    else:
        fullGraphHalo = graphOnRank
        fullCharges = chargesOnRank

    # print_graph(fullGraphRho)
    return fullGraphHalo, fullCharges, subSysOnRank, mu


def get_adaptive_KernelSCFDM(
    sdc,
    eng,
    comm,
    rank,
    numranks,
    sy,
    hindex,
    graphNL,
    mu,
    alpha=0.7,
    graphweights=None,
    device="cuda",
    dtype=torch.float64,
):
    fullGraph = graphNL
    # Initial guess for the excess ocupation vector. This is the negative of
    # the charge!
    charges = sy.charges
    chargesOld = None
    chargesIn = None
    chargesOld = None
    chargesOut = None
    scfError = 1.0
    kernel = 0
    # Partition the graph
    parts = graph_partition(
        sdc, eng, fullGraph, sdc.partitionType, sdc.nparts, sy.coords, latticeVectors=sy.latticeVectors, graphweights=graphweights, verb=True
    )
    for gscf in range(sdc.numAdaptIter):
        msg = "Graph-adaptive iteration" + str(gscf)
        status_at("get_adaptiveSCFDM", msg)
        njumps = 1
        partsCoreHalo = []
        numCores = []
        # print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)
            print("core,halo size:", i, "=", nc, nh)
        #    print("coreHalo for part", i, "=", coreHalo)
        if rank == 0 and gscf == 0 and sum(charges == 0) != 0:
            sy.coulvs = np.zeros(len(charges))
        elif rank == 0:
            sy.coulvs, ewald_e = get_PME_coulvs(
                charges,
                sy.hubbard_u,
                sy.coords,
                sy.types,
                sy.latticeVectors,
                device=device,
            )
        if is_mpi_available and numranks > 1:
            sy.coulvs = comm.bcast(sy.coulvs, root=0)
        chargesOld = charges
        fullGraphHalo, charges, subSysOnRank, mu = get_singlePoint_charges(
            sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, gscf, mu, alpha=alpha,
        )
        # print("Collected charges", charges)
        if scfError > 0.0001:
            scfError, charges, chargesOld, chargesIn, chargesOut = diis_mix(
                charges, chargesOld, chargesIn, chargesOut, gscf, verb=False
            )
        #             scfError,charges,chargesOld = linear_mix(0.1,charges,chargesOld,gscf)
        # if gscf == 0:
        #    scfError = sy.numel
        else:
            scfError = np.linalg.norm(charges - chargesOld) / np.sqrt(sy.nats)
            if kernel == 0:
                print("start building the full kernel...")
                get_kernel_byParts(
                    sdc, rank, numranks, parts, partsCoreHalo, sy, mu, device=device
                )
                syk = deepcopy(sy)
                # KRes = apply_kernel_byParts(
                #     charges, chargesOld, sdc, rank, numranks, comm, parts, sy
                # )
                kernel = 1
            else:
                for i, subSy in enumerate(sy.subSy_list):
                    subSy.ker = syk.subSy_list[i].ker.clone()
            KRes = rankN_update_byParts(
                torch.from_numpy(charges).to(device).to(dtype),
                torch.from_numpy(chargesOld).to(device).to(dtype),
                6,
                sdc,
                rank,
                numranks,
                comm,
                parts,
                partsCoreHalo,
                sy,
                mu=mu,
                device=device,
            )
            KRes = KRes.cpu().numpy()
            
            # charges = charges - KRes
            charges = chargesOld - KRes
            # charges = charges - np.dot(sy.subSy_list[0].ker, g_charges - charges)
        # print_graph(fullGraphRho)

        fullGraph = add_graphs(fullGraphHalo, fullGraph)
        # fullGraph = multiply_graphs(fullGraphRho, graphNL)
        # fullGraph = fullGraphRho
        for i in range(sy.nats):
            print("Charges:", i, sy.symbols[sy.types[i]], charges[i])
        print("SCF ERR =", scfError)
        print("TotalCharge", sum(charges))

        if scfError < sdc.scfTol:
            status_at(
                "get_adaptiveSCFDM", "SCF converged with SCF error = " + str(scfError)
            )
            break

        if gscf == sdc.numAdaptIter - 1:
            warning_at("get_adaptiveSCFDM", "SCF did not converged ... ")

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

    return fullGraph, charges, mu, parts, partsCoreHalo, subSysOnRank
