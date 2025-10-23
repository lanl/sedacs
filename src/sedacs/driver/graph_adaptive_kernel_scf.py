"""
graph_adaptive_kernel_scf.py
====================================
Graph adaptive self-consistent charge solver with kernel

"""

import time
import torch
import numpy as np
from copy import deepcopy
import nvtx

from sedacs.graph import (
    add_graphs,
    collect_graph_from_rho,
    print_graph,
    multiply_graphs,
    adaptive_halo_expansion,
    symmetrize_graph,
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
    sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, mu=0.0, alpha=0.7, write_parts=False
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
        if write_parts:
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
            etemp=sdc.etemp,
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
    comm.Barrier()
    if is_mpi_available and numranks > 1:
        nvtx.push_range("collect_evals_dvals", color="green", domain="get_singlePoint_charges")
        fullEvals = collect_and_concatenate_vectors(evalsOnRank, comm)
        fullDvals = collect_and_concatenate_vectors(dvalsOnRank, comm)
        nvtx.pop_range("get_singlePoint_charges")
        comm.Barrier()
    else:
        fullEvals = evalsOnRank
        fullDvals = dvalsOnRank
    
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
    # mu = -6
    nvtx.pop_range("get_singlePoint_charges")
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
        chargesInPart = chargesInPart[: len(parts[partIndex])]
        #        subSy.charges = chargesInPart

        # Save the subsystems list for returning them
        subSysOnRank.append(subSy)

        print("TotalCharge in part", partIndex, sum(chargesInPart))
        # print("Charges in part", chargesInPart)

        toc = time.perf_counter()
        print("Time to get_densityMatrix and get_charges", toc - tic, "(s)")
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
            sy.latticeVectors,
            sy.nl.cpu().numpy(),
            alpha=alpha,
        )
        nvtx.pop_range("get_singlePoint_charges")
        nvtx.push_range("collect_graph_from_rho", color="brown", domain="get_singlePoint_charges")
        # if set(parts[partIndex]) != set(partsCoreHalo[partIndex][:len(parts[partIndex])]):
        #     raise ValueError("Wrong Core Halo List!!!")
        graphOnRank = collect_graph_from_rho(
            graphOnRank,
            rho,
            sdc.gthresh,
            sy.nats,
            sdc.maxDeg,
            partsCoreHalo[partIndex],
            len(parts[partIndex]),
            subSy.hindex,
        )
        
        nvtx.pop_range("get_singlePoint_charges")
        chargesOnRank = collect_charges(
            chargesOnRank, chargesInPart, parts[partIndex], sy.nats, verb=True
        )
    comm.Barrier()
    nvtx.push_range("collect_graph_and_charges", color="green", domain="get_singlePoint_charges")
    if is_mpi_available and numranks > 1:
        fullGraph = collect_and_sum_matrices_float(graphOnRank, comm)
        fullCharges = collect_and_sum_vectors_float(chargesOnRank, rank, numranks, comm)
        comm.Barrier()
    else:
        fullGraph = graphOnRank
        fullCharges = chargesOnRank
    nvtx.pop_range("get_singlePoint_charges")
    # print_graph(fullGraphRho)

    return fullGraph, fullCharges, subSysOnRank, mu


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
    write_parts=False
):
    fullGraph = graphNL
    if rank == 0:
        with open('scf.log', 'w') as f:
            f.write('scf starts...\n')
        with open("edge.log", "w") as f:
            f.write('scf starts...\n')
            f.write(f"graphNL edge count: {np.sum(graphNL[:, 0])}\n")
        with open('corehalo.log', 'w') as f:
            f.write('scf starts...\n')
    # Initial guess for the excess ocupation vector. This is the negative of
    # the charge!
    charges = sy.charges
    chargesOld = None
    chargesIn = None
    chargesOld = None
    chargesOut = None
    parts = None
    scfError = 10.0
    kernel = 0
    # Partition the graph
    nvtx.push_range("graph_partition", color="purple", domain="get_adaptiveSCFDM")
    if rank == 0:
        parts = graph_partition(
            sdc, eng, fullGraph, sdc.partitionType, sdc.nparts, sy.coords, latticeVectors=sy.latticeVectors, graphweights=graphweights, verb=True
        )
    parts = comm.bcast(parts, root=0)
    nvtx.pop_range("get_adaptiveSCFDM")
    njumps = 1
    partsCoreHalo = [[] for _ in range(sdc.nparts)]
    for gscf in range(sdc.numAdaptIter):
        nvtx.push_range("SCF Iter", color="blue", domain="get_adaptiveSCFDM")
        msg = "Graph-adaptive iteration" + str(gscf)
        status_at("get_adaptiveSCFDM", msg)
        nvtx.push_range("get_coreHaloIndices", color="blue", domain="get_adaptiveSCFDM")
        
        total = 0
        # print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps, partsCoreHalo[i])
            total += nh
            partsCoreHalo[i] = coreHalo
            print("core,halo size:", i, "=", nc, nh)
            if rank == 0:
                with open('corehalo.log', 'a') as f:
                    f.write(f"gscf: {gscf}\n")
                    f.write(f"core,halo size: {i} = {nc} {nh}\n")
        if rank == 0:
            with open('corehalo.log', 'a') as f:
                f.write(f"avg. halo size: {total / sdc.nparts}\n")
        nvtx.pop_range("get_adaptiveSCFDM")
        nvtx.push_range("PME solver", color="green", domain="get_adaptiveSCFDM")
        if gscf == 0 and sum(charges == 0) != 0:
            sy.coulvs = np.zeros(len(charges))
        else:
            if (rank == 0) or (not is_mpi_available):
                sy.coulvs, ewald_e = get_PME_coulvs(
                    charges,
                    sy.hubbard_u,
                    sy.coords,
                    sy.types,
                    sy.latticeVectors,
                    sy.nl,
                    sy.nl_disps,
                    sy.nl_dists,
                    sy.ewald_alpha,
                    sdc.coulcut,
                    sy.PME_data,
                    device=device,
                )

            if is_mpi_available and numranks > 1:
                # Broadcast the result to all ranks on the same node
                sy.coulvs = comm.bcast(sy.coulvs, root=0)

        nvtx.pop_range("get_adaptiveSCFDM")
        chargesOld = charges
        nvtx.push_range("get_singlePoint_charges", color="orange", domain="get_adaptiveSCFDM")
        fullGraph, charges, subSysOnRank, mu = get_singlePoint_charges(
            sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, mu=mu, alpha=alpha,
        )
        nvtx.pop_range("get_adaptiveSCFDM")
        nvtx.push_range("symmetrize_graphs", color="pink", domain="get_adaptiveSCFDM")
        # fullGraph = torch.from_numpy(fullGraph).to(device)
        fullGraph = symmetrize_graph(fullGraph)
        nvtx.pop_range("get_adaptiveSCFDM")
        # print("Collected charges", charges)
        scfError = np.linalg.norm(charges - chargesOld) / np.sqrt(sy.nats)
        if scfError > 0.1:
            scfError, charges, chargesOld, chargesIn, chargesOut = diis_mix(
                charges, chargesOld, chargesIn, chargesOut, gscf, verb=False
            )
        #             scfError,charges,chargesOld = linear_mix(0.1,charges,chargesOld,gscf)
        # if gscf == 0:
        #    scfError = sy.numel
        else:
            if kernel == 0:
                print("start building the full kernel...")
                nvtx.push_range("get_kernel_byParts", color="red", domain="get_adaptiveSCFDM")
                get_kernel_byParts(
                    sdc, rank, numranks, parts, partsCoreHalo, sy, mu=mu, identity=True, nbr_inds=sy.nl, disps=sy.nl_disps, dists=sy.nl_dists, alpha=sy.ewald_alpha, PME_data=sy.PME_data, device=device
                )
                nvtx.pop_range("get_adaptiveSCFDM")
                syk = deepcopy(sy)
                # KRes = apply_kernel_byParts(
                #     charges, chargesOld, sdc, rank, numranks, comm, parts, sy
                # )
                kernel = 1
            else:
                for i, subSy in enumerate(sy.subSy_list):
                    subSy.ker = syk.subSy_list[i].ker.clone()
            nvtx.push_range("rankN_update_byParts", color="purple", domain="get_adaptiveSCFDM")
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
                nbr_inds=sy.nl,
                disps=sy.nl_disps,
                dists=sy.nl_dists,
                alpha=sy.ewald_alpha,
                PME_data=sy.PME_data,
                device=device,
            )
            nvtx.pop_range("get_adaptiveSCFDM")
            KRes = KRes.cpu().numpy()
            
            # charges = charges - KRes
            charges = chargesOld - KRes
            # charges = charges - np.dot(sy.subSy_list[0].ker, g_charges - charges)
        #for i in range(sy.nats):
        #    print("Charges:", i, sy.symbols[sy.types[i]], charges[i])
        print("SCF ERR =", scfError)
        print("TotalCharge", sum(charges))
        nvtx.pop_range("get_adaptiveSCFDM")
        if rank == 0:
            with open('scf.log', 'a') as f:
                f.write(f'{scfError}\n')
            with open("edge.log", "a") as f:
                f.write(f"fullGraph edge count: {np.sum(fullGraph[:, 0])}\n")

        if scfError < sdc.scfTol:
            status_at(
                "get_adaptiveSCFDM", "SCF converged with SCF error = " + str(scfError)
            )
            break

        if gscf == sdc.numAdaptIter - 1:
            warning_at("get_adaptiveSCFDM", "SCF did not converged ... ")
            break

    if write_parts and rank == 0:
        AtToPrint = 0

        subSy = System(fullGraph[AtToPrint, 0])
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(
            sy.coords,
            sy.types,
            sy.symbols,
            fullGraph[AtToPrint, 1 : fullGraph[AtToPrint, 0] + 1],
        )

        write_pdb_coordinates(
            "subSyG_fin.pdb", subSy.coords, subSy.types, subSy.symbols
        )
        write_xyz_coordinates(
            "subSyG_fin.xyz", subSy.coords, subSy.types, subSy.symbols
        )

    return fullGraph, charges, mu, parts, partsCoreHalo, subSysOnRank
