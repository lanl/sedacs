"""Graph adaptive self-consistenf charge solver"""

import time

from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.sdc_hamiltonian import get_hamiltonian
from sedacs.sdc_density_matrix import get_density_matrix
from sedacs.sdc_evals_dvals import get_evals_dvals
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.mpi import collect_and_sum_matrices, collect_and_sum_vectors_float, collect_and_concatenate_vectors
from sedacs.system import System, extract_subsystem, get_hindex
from sedacs.coulombic import get_coulvs, get_PME_coulvs, build_coul_ham
from sedacs.charges import get_charges, collect_charges
from sedacs.evals_dvals import collect_evals, collect_dvals
from sedacs.message import status_at, error_at, warning_at
from sedacs.mixer import diis_mix, linear_mix
from sedacs.chemical_potential import get_mu
import numpy as np

try:
    from mpi4py import MPI
    is_mpi_available = True
except ModuleNotFoundError:
    is_mpi_available = False

__all__ = ["get_singlePoint_charges", "get_adaptiveSCFDM"]


def get_singlePoint_charges(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, gscf, mu=0.0):
    # computing DM for core+halo part
    #
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None
    chargesOnRank = None
    evalsOnRank = None
    dvalsOnRank = None
    subSysOnRank = []
    ham_list = [None] * partsPerRank
    over_list = [None] * partsPerRank

    for partIndex in range(partIndex1, partIndex2):
        idx = partIndex - partIndex1
        numberOfCoreAtoms = len(parts[partIndex])
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        tic = time.perf_counter()
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])
        subSy.ncores = len(parts[partIndex])
        if(gscf == 0): 
            subSy.charges = np.zeros(len(subSy.types))
        toc = time.perf_counter()
        print("Time for extract_subsystem", toc - tic, "(s)")
        partFileName = "subSy" + str(rank) + "_" + str(partIndex) + ".pdb"
        write_pdb_coordinates(partFileName, subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates(
            "subSy" + str(rank) + "_" + str(partIndex) + ".xyz", subSy.coords, subSy.types, subSy.symbols
        )
        tic = time.perf_counter()

        #Get some electronic structure elements for the sybsystem 
        #This could eventually be computed in the engine if no basis set is 
        #provided in the SEDACS input file.
        subSy.norbs, subSy.orbs, subSy.hindex, subSy.numel, subSy.znuc = get_hindex(sdc.orbs, subSy.symbols, subSy.types,verb=True)

        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        if(sdc.over):
            # eng.name = "LATTE" 
            ham0, over = get_hamiltonian(eng, partIndex, sdc.nparts, norbs, sy.latticeVectors, subSy.coords, subSy.types, subSy.symbols, verbose=sdc.verb, get_overlap=True, newsystem=True)
            #eng.name = "ProxyAPython" 
        else:
            ham0 = get_hamiltonian(eng, partIndex, sdc.nparts, norbs, sy.latticeVectors, subSy.coords, subSy.types, subSy.symbols, verbose=sdc.verb, get_overlap=False, newsystem=True)
        over_list[idx] = over
        # ham = ham0
        tmpArray = np.zeros(numberOfCoreAtoms)
        tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]

        norbsInCore = int(np.sum(tmpArray))

        print("Number of orbitals in the core =",norbsInCore)
#        print("ham0:", ham0)
        # if gscf == 0:
        if(sdc.over):
           ham = build_coul_ham(eng,ham0,sy.coulvs[partsCoreHalo[partIndex]],subSy.types,subSy.charges,False,subSy.hindex,overlap=over,verb=True)
        else:
           ham = build_coul_ham(eng,ham0,sy.coulvs[partsCoreHalo[partIndex]],subSy.types,subSy.charges,False,subSy.hindex,overlap=None,verb=True)
        ham_list[idx] = ham
        toc = time.perf_counter()
        print("Time for get_hamiltonian", toc - tic, "(s)")
       #norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        nocc = int(float(subSy.numel) / 2.0)  # Get the total occupied orbitals

        tic = time.perf_counter()
        evalsInPart, dvalsInPart = get_evals_dvals(eng,partIndex,sdc.nparts,sy.latticeVectors, subSy.coords, subSy.types, subSy.symbols,ham,sy.coulvs[partsCoreHalo[partIndex]],nocc=nocc,norbsInCore=norbsInCore,mu=mu,etemp=sdc.etemp,overlap=over,full_data=False,verb=True,newsystem=False)
        #breakpoint()
        toc = time.perf_counter()
        print("Time for get_evals_dvals", toc - tic, "(s)")

        # subSy.latticeVectors = sy.latticeVectors
        # if(sdc.over):
        #     rho, evalsInPart, dvalsInPart = get_density_matrix(eng,subSy,ham,sy.coulvs[partsCoreHalo[partIndex]],nocc=nocc,mu=mu,etemp=sdc.etemp,overlap=over,full_data=True,verb=False)
        # else:
        #     rho, evals, dvals = get_density_matrix(eng,subSy,ham,nocc=nocc,norbsInCore=norbsInCore,mu=None,etemp=sdc.etemp,overlap=None,full_data=True,verb=False, newsystem=False)

        evalsOnRank = collect_evals(evalsOnRank,evalsInPart,verb=True)
        dvalsOnRank = collect_dvals(dvalsOnRank,dvalsInPart,verb=True)

    if (is_mpi_available and numranks > 1):
        fullEvals = collect_and_concatenate_vectors(evalsOnRank, comm)
        fullDvals = collect_and_concatenate_vectors(dvalsOnRank, comm)
        comm.Barrier()
    else:
        fullEvals = evalsOnRank
        fullDvals = dvalsOnRank

    mu = get_mu(mu, fullEvals, sdc.etemp, int(sy.numel/2), dvals=fullDvals, kB=8.61739e-5, verb=True) 

    for partIndex in range(partIndex1, partIndex2):
        idx = partIndex - partIndex1
        numberOfCoreAtoms = len(parts[partIndex])
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        tic = time.perf_counter()
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])
        subSy.ncores = len(parts[partIndex])
        ham = ham_list[idx]
        over = over_list[idx]
        if(gscf == 0): 
            subSy.charges = np.zeros(len(subSy.types))
        toc = time.perf_counter()
        print("Time for extract_subsystem", toc - tic, "(s)")
        partFileName = "subSy" + str(rank) + "_" + str(partIndex) + ".pdb"
        write_pdb_coordinates(partFileName, subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates(
            "subSy" + str(rank) + "_" + str(partIndex) + ".xyz", subSy.coords, subSy.types, subSy.symbols
        )
        tic = time.perf_counter()

        #Get some electronic structure elements for the sybsystem 
        #This could eventually be computed in the engine if no basis set is 
        #provided in the SEDACS input file.
        subSy.norbs, subSy.orbs, subSy.hindex, subSy.numel, subSy.znuc = get_hindex(sdc.orbs, subSy.symbols, subSy.types,verb=True)

        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        tmpArray = np.zeros(numberOfCoreAtoms)
        tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]

        norbsInCore = np.sum(tmpArray)
        nocc = int(float(subSy.numel) / 2.0)  # Get the total occupied orbitals
        subSy.latticeVectors = sy.latticeVectors

        tic = time.perf_counter()
        
        if eng.name == "LATTE":
            rho, chargesInPart = get_density_matrix(eng,partIndex,sdc.nparts,norbs,sy.latticeVectors, subSy.coords, subSy.types, subSy.symbols,ham,sy.coulvs[partsCoreHalo[partIndex]],nocc=nocc,norbsInCore=norbsInCore,mu=mu,etemp=sdc.etemp,overlap=over,full_data=False,verb=True,newsystem=True,keepmem=True)
        else:
            rho = get_density_matrix(eng,partIndex,sdc.nparts,norbs,sy.latticeVectors, subSy.coords, subSy.types, subSy.symbols,ham,sy.coulvs[partsCoreHalo[partIndex]],nocc=nocc,norbsInCore=norbsInCore,mu=mu,etemp=sdc.etemp,overlap=over,full_data=False,verb=True)
        
            if(sdc.over):
                chargesInPart = get_charges(rho,subSy.znuc,subSy.types,parts[partIndex],subSy.hindex,over=over_list[partIndex],verb=True)
            else:
                chargesInPart = get_charges(rho,subSy.znuc,subSy.types,parts[partIndex],subSy.hindex,over=None,verb=True)

        chargesInPart = chargesInPart[:len(parts[partIndex])]
        subSy.charges = chargesInPart

        #Save the subsystems list for returning them
        subSysOnRank.append(subSy)

        print("TotalCharge in part",partIndex,sum(chargesInPart))
        print("Charges in part",chargesInPart)
        
        toc = time.perf_counter()
        print("Time to get_densityMatrix and get_charges", toc - tic, "(s)")
        # Building a graph from DMs
        graphOnRank = collect_graph_from_rho(
            graphOnRank, rho, sdc.gthresh, sy.nats, sdc.maxDeg, partsCoreHalo[partIndex],len(parts[partIndex]), hindex
        )

        chargesOnRank = collect_charges(chargesOnRank,chargesInPart,parts[partIndex],sy.nats,verb=True)

    if (is_mpi_available and numranks > 1):
        fullGraphRho = collect_and_sum_matrices(graphOnRank, rank, numranks, comm)
        fullCharges = collect_and_sum_vectors_float(chargesOnRank, rank, numranks, comm)
        comm.Barrier()
    else:
        fullGraphRho = graphOnRank
        fullCharges = chargesOnRank

    # print_graph(fullGraphRho)
    return fullGraphRho, fullCharges, fullEvals, fullDvals, subSysOnRank, mu


def get_adaptiveSCFDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu):
    fullGraph = graphNL

    #Iitial guess for the excess ocupation vector. This is the negative of 
    #the charge! 
    charges = sy.charges
#    charges = np.zeros(sy.nats) 
    chargesOld = None 
    chargesIn = None
    chargesOld = None
    chargesOut = None
#    mu = -0.0
    sdc.etemp = 1000
    # Partition the graph
    parts = graph_partition(sdc, eng, fullGraph, sdc.partitionType, sdc.nparts, sy.coords, True)
    njumps = 2
    partsCoreHalo = []
    numCores = []
    for gscf in range(sdc.numAdaptIter):
        msg = "Graph-adaptive iteration" + str(gscf)
        status_at("get_adaptiveSCFDM",msg)
        #print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)
            print("core,halo size:", i, "=", nc, nh)
        #    print("coreHalo for part", i, "=", coreHalo)
        symbols = np.array(sy.symbols)[sy.types]
        hubbard_u = np.where(symbols == 'H', 12.054683, 0.0) + np.where(symbols == 'O', 11.876141, 0.0)
        if gscf == 0 and sum(charges == 0) != 0:
            sy.coulvs = np.zeros(len(charges))
        else:
            sy.coulvs, ewald_e = get_PME_coulvs(charges, hubbard_u, sy.coords, sy.types, sy.latticeVectors)
        
        fullGraphRho,charges,evals,dvals,subSysOnRank,mu = get_singlePoint_charges(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex,gscf,mu)
        print("Collected charges",charges)

        scfError, charges, chargesOld, chargesIn, chargesOut = diis_mix(charges,chargesOld,chargesIn,chargesOut,gscf,verb=True)
        #scfError,charges,chargesOld = linear_mix(0.25,charges,chargesOld,gscf)
        #if gscf == 0:
        #    scfError = sy.numel

        #print_graph(fullGraphRho)

        fullGraph = add_graphs(fullGraphRho, graphNL)
        for i in range(sy.nats):
            print("Charges:",i,sy.symbols[sy.types[i]],charges[i])
        print("SCF ERR =",scfError)
        print("TotalCharge",sum(charges))
        
        if(scfError < sdc.scfTol):
            status_at("get_adaptiveSCFDM","SCF converged with SCF error = "+str(scfError))
            break

        if(gscf == sdc.numAdaptIter - 1):
            warning_at("get_adaptiveSCFDM","SCF did not converged ... ")

    AtToPrint = 0

    subSy = System(fullGraphRho[AtToPrint, 0])
    subSy.symbols = sy.symbols
    subSy.coords, subSy.types = extract_subsystem(
        sy.coords, sy.types, sy.symbols, fullGraph[AtToPrint, 1 : fullGraph[AtToPrint, 0] + 1]
    )

    if rank == 0:
        write_pdb_coordinates("subSyG_fin.pdb", subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates("subSyG_fin.xyz", subSy.coords, subSy.types, subSy.symbols)

    return fullGraph,charges,mu,parts,subSysOnRank
