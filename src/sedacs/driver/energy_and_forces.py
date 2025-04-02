"""Graph adaptive self-consistenf charge solver"""

import time

from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.hamiltonian import get_hamiltonian
from sedacs.density_matrix import get_density_matrix
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.mpi import collect_and_sum_matrices
from sedacs.system import System, extract_subsystem, get_hindex
from sedacs.coulombic import get_coulvs, build_coul_ham, get_coulombic_forces
from sedacs.charges import get_charges, collect_charges
from sedacs.tbforces import get_tb_forces
from sedacs.message import status_at, error_at, warning_at
from sedacs.mixer import diis_mix
import numpy as np
try:
    from mpi4py import MPI
    is_mpi_available = True
except ModuleNotFoundError:
    is_mpi_available = False

__all__ = ["get_occ_singlePoint", "get_adaptiveSCFDM_force"]


## Single point calculation
# @brief Construct a connectivity graph based on constructing density matrices
# of parts of the system.
#
def get_singlePoint_charges(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex):
    # computing DM for core+halo part
    #
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None
    chargesOnRank = None
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

        #ham0, over = get_hamiltonian(eng, subSy.coords, subSy.types, subSy.symbols, verbose=False, get_overlap=True)
        ham0 = get_hamiltonian(eng, subSy.coords, subSy.types, subSy.symbols, verbose=False, get_overlap=False)
        
        #Get some electronic structure elements for the sybsystem 
        #This could eventually be computed in the engine if no basis set is 
        #provided in the SEDACS input file.
        subSy.norbs, subSy.orbs, subSy.hindex, subSy.numel, subSy.znuc = get_hindex(sdc.orbs, subSy.symbols, subSy.types,verb=True)

        #ham = build_coul_ham(ham0,sy.coulvs[partsCoreHalo[partIndex]],False,subSy.hindex,overlap=over,verb=True)
        ham = build_coul_ham(ham0,sy.coulvs[partsCoreHalo[partIndex]],False,subSy.hindex,overlap=None,verb=True)

        toc = time.perf_counter()
        print("Time for get_hamiltonian", toc - tic, "(s)")
        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        nocc = int(float(subSy.numel) / 2.0)  # Get the total occupied orbitals
        tic = time.perf_counter()
        #rho = get_density_matrix(eng,ham,nocc=nocc,mu=None,elect_temp=0.0,overlap=over,verb=False)
        rho = get_density_matrix(eng,ham,nocc=nocc,mu=None,elect_temp=0.0,overlap=None,verb=False)
        #chargesInPart = get_charges(rho,subSy.znuc,subSy.types,parts[partIndex],subSy.hindex,over=over,verb=True)
        chargesInPart = get_charges(rho,subSy.znuc,subSy.types,parts[partIndex],subSy.hindex,over=None,verb=True)
        print("TotalCharge in part",partIndex,sum(chargesInPart))
        print("Charges in part",chargesInPart)
        
        toc = time.perf_counter()
        print("Time to get_densityMatrix", toc - tic, "(s)")
        # Building a graph from DMs
        graphOnRank = collect_graph_from_rho(
            graphOnRank, rho, sdc.gthresh, sy.nats, sdc.maxDeg, parts[partIndex], hindex
        )

        chargesOnRank = collect_charges(chargesOnRank,chargesInPart,parts[partIndex],sy.nats,verb=True)


    if (is_mpi_available and numranks > 1):
        fullGraphRho = collect_and_sum_matrices(graphOnRank, rank, numranks, comm)
        fullCharges = collect_and_sum_vector()
        comm.Barrier()
    else:
        fullGraphRho = graphOnRank
        fullCharges = chargesOnRank

    return fullGraphRho, fullCharges


def get_adaptiveSCFDM_force(sdc, eng, comm, rank, numranks, sy, hindex, graphNL):
    fullGraph = graphNL

    #Iitial guess for the excess ocupation vector. This is the negative of 
    #the charge! 
    charges = np.zeros(sy.nats) 
    chargesOld = np.zeros(sy.nats) 
    chargesIn = None
    chargesOld = None
    chargesOut = None
    for gscf in range(sdc.numAdaptIter):
        msg = "Graph-adaptive iteration" + str(gscf)
        status_at("get_adaptiveSCFDM",msg)
        # Partition the graph
        parts = graph_partition(fullGraph, sdc.partitionType, sdc.nparts, False)
        njumps = 1
        partsCoreHalo = []
        numCores = []
        print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)
            print("coreHalo for part", i, "=", coreHalo)

        sy.coulvs = get_coulvs(charges,sy.coords)
        #print(sy.coulvs)

        fullGraphRho,charges = get_singlePoint_charges(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex)

        scfError, charges, chargesOld, chargesIn, chargesOut = diis_mix(charges,chargesOld,chargesIn,chargesOut,gscf,False)

        fullGraph = add_graphs(fullGraphRho, graphNL)
        print("Charges",charges)
        print("ChargesOld",chargesOld)
        print("SCF ERR =",scfError)
        print("TotalCharge",sum(charges))
        print("All variables",dir())
        
        if(scfError < sdc.scfTol):
            status_at("get_adaptiveSCFDM","SCF converged with SCF error = "+str(scfError))
            break

        if(gscf == sdc.numAdaptIter - 1):
            warning_at("get_adaptiveSCFDM","SCF did not converged ... ")

        #Get forces after SCF convergence. 
        
        #Coulombic forces will be computed for the whole system 
        forces = get_coulombic_forces(charges,sy.coords,sy.types,sy.symbols,factor=14.39964377014,field=None)
        forces = forces[:,:] + forces_tb[:,:]
        print(forces)
        exit(0)

        
        
    AtToPrint = 0

    subSy = System(fullGraphRho[AtToPrint, 0])
    subSy.symbols = sy.symbols
    subSy.coords, subSy.types = extract_subsystem(
        sy.coords, sy.types, sy.symbols, fullGraph[AtToPrint, 1 : fullGraph[AtToPrint, 0] + 1]
    )

    if rank == 0:
        write_pdb_coordinates("subSyG_fin.pdb", subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates("subSyG_fin.xyz", subSy.coords, subSy.types, subSy.symbols)
