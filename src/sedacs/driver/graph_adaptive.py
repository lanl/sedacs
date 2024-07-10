"""Graph adaptive solver"""

import sys
import time

from sedacs.driver import *
from sedacs.graph_partition import *
from sedacs.hamiltonian import *
from sedacs.system import System


def get_density_matrix(*args, **kwargs):
    pass


## Single point calculation
# @brief Construct a connectivity graph based on constructing density matrices
# of parts of the system.
#
def get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex):
    # computing DM for core+halo part
    #
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None
    for partIndex in range(partIndex1, partIndex2):
        print("Rank, part", rank, partIndex)
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        tic = time.perf_counter()
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])
        toc = time.perf_counter()
        print("Time for extract_subsystem", toc - tic, "(s)")
        partFileName = "subSy" + str(rank) + "_" + str(partIndex) + ".pdb"
        write_pdb_coordinates(partFileName, subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates(
            "subSy" + str(rank) + "_" + str(partIndex) + ".xyz", subSy.coords, subSy.types, subSy.symbols
        )
        tic = time.perf_counter()

        # for kk in range(subSy.nats):
        #    subSy.coords[0,kk] = subSy.coords[0,kk] + sy.latticeVectors[0,:] * nlTrX[partsCoreHalo[partIndex][kk]]

        ham = sdc_get_hamiltonian(eng, subSy.coords, subSy.types, subSy.symbols, verb=False)
        toc = time.perf_counter()
        print("Time for get_hamiltonian", toc - tic, "(s)")
        norbs = subSy.nats
        occ = int(float(norbs) / 2.0)  # Get the total occupied orbitals
        tic = time.perf_counter()
        rho = get_density_matrix(ham, occ)
        # print(rho)

        ## MAKSIM
        # Function needed inside the proxy A code
        # coreSize = len(parts[partIndex]) # previous m
        # rho,eVals,dVals = get_densityMatrix_renormalized_???(H, Nocc,mu0, coreSize, kbT,verb)
        # dVals -

        # rho - DM
        #

        toc = time.perf_counter()
        print("Time for get_densityMatrix", toc - tic, "(s)")
        # Building a graph from DMs
        graphOnRank = collect_graph_from_rho(
            graphOnRank, rho, sdc.gthresh, sy.nats, sdc.maxDeg, parts[partIndex], hindex
        )
        # Gather all the eval and dvals from all the coreHalos within this execution rank
        # dvalsOnRank = collect_dValsOnRank(dVals)
        # evalsOnRank = collect_eValsOnRank(eVals)

    if mpiON:
        fullGraphRho = collect_and_sum_matrices(graphOnRank, rank, numranks, comm)
        # dValsFull = collect_dValsFull(dValsOnRank) #MPI functions # Newton-Raphosn from graph paper???
        # eValsFull = collect_eValsFull(dValsOnRank) #MPI functions # Newton-Raphosn from graph paper???

        # Compute the new mu given dvalsFull and evalsFull!
        # With a NR scheme.  The function to be minimized will be
        # muFull = get_muFromParts(dValsFull,eValsFll,T,mu)
        # nocc - Sum_i Fermi(evalsFull_i,T,mu)*dvalsFull_i = 0

        comm.Barrier()
    else:
        fullGraphRho = graphOnRank

    return fullGraphRho


def get_adaptiveDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL):
    fullGraph = graphNL
    for gsc in range(sdc.numAdaptIter):
        # Partition the graph
        parts = graph_partition(fullGraph, sdc.partitionType, sdc.nparts, sdc.verb)
        njumps = 1
        partsCoreHalo = []
        numCores = []
        print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)
            print("coreHalo for part", i, "=", coreHalo)

        fullGraphRho = get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex)
        fullGraph = add_graphs(fullGraphRho, graphNL)

    AtToPrint = 0
    print("graphNL", graphNL[AtToPrint])
    print("fullGraphRho:", fullGraphRho[AtToPrint])
    print("fullGraph", fullGraph[AtToPrint])

    # print(graphNL)
    # print(fullGraph)
    # Get the neighbors of atom 1234 (by the graph)
    subSy = System(fullGraphRho[AtToPrint, 0])
    subSy.symbols = sy.symbols
    subSy.coords, subSy.types = extract_subsystem(
        sy.coords, sy.types, sy.symbols, fullGraph[AtToPrint, 1 : fullGraph[AtToPrint, 0] + 1]
    )

    if rank == 0:
        write_pdb_coordinates("subSyG_fin.pdb", subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates("subSyG_fin.xyz", subSy.coords, subSy.types, subSy.symbols)
    sys.exit(0)
    if rank == 0:
        print_graph(graphOnRank)
