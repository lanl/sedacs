"""Graph adaptive self-consistenf charge solver"""

import time

from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.sdc_hamiltonian import get_hamiltonian
from sedacs.sdc_density_matrix import get_density_matrix
from sedacs.sdc_energy_forces import get_energy_forces
from sedacs.sdc_evals_dvals import get_evals_dvals
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.mpi import collect_and_sum_matrices, collect_and_sum_vectors_float, collect_and_concatenate_vectors
from sedacs.system import System, extract_subsystem, get_hindex
from sedacs.coulombic import get_coulvs, get_PME_coulvs, build_coul_ham
from sedacs.charges import get_charges, collect_charges
from sedacs.evals_dvals import collect_evals, collect_dvals
from sedacs.energy_forces import collect_energy, collect_forces
from sedacs.message import status_at, error_at, warning_at
from sedacs.mixer import diis_mix, linear_mix
from sedacs.chemical_potential import get_mu
import numpy as np

try:
    from mpi4py import MPI
    is_mpi_available = True
except ModuleNotFoundError:
    is_mpi_available = False

__all__ = ["get_occ_singlePoint", "get_adaptiveSCFDM"]


## Single point calculation
# @brief Construct a connectivity graph based on constructing density matrices
# of parts of the system.
#
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
    energyOnRank = None
    forcesOnRank = None
    subSysOnRank = []

    for partIndex in range(partIndex1, partIndex2):
#         print("Rank, part", rank, partIndex)
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
        ham, over = get_hamiltonian(eng, partIndex, sdc.nparts, norbs, sy.latticeVectors, subSy.coords, subSy.types, subSy.symbols, verbose=sdc.verb, get_overlap=True, newsystem=True)

        tmpArray = np.zeros(numberOfCoreAtoms)
        tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]

        norbsInCore = np.sum(tmpArray)
        nocc = int(float(subSy.numel) / 2.0)  # Get the total occupied orbitals
        print("Number of orbitals in the core =",norbsInCore)

        toc = time.perf_counter()
        print("Time for get_hamiltonian", toc - tic, "(s)")

        tic = time.perf_counter()
        evalsInPart, dvalsInPart = get_evals_dvals(eng,partIndex,sdc.nparts,sy.latticeVectors, subSy.coords, subSy.types, subSy.symbols,ham,sy.coulvs[partsCoreHalo[partIndex]],nocc=nocc,norbsInCore=norbsInCore,mu=mu,etemp=sdc.etemp,overlap=over,full_data=False,verb=sdc.verb,newsystem=False)
        toc = time.perf_counter()
        print("Time for get_evals_dvals", toc - tic, "(s)")

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
#         print("Rank, part", rank, partIndex)
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
        tmpArray = np.zeros(numberOfCoreAtoms)
        tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]

        norbsInCore = int(np.sum(tmpArray))
        nocc = int(float(subSy.numel) / 2.0)  # Get the total occupied orbitals
        subSy.latticeVectors = sy.latticeVectors
        tic = time.perf_counter()

        rho, chargesInPart = get_density_matrix(eng,partIndex,sdc.nparts,norbs,sy.latticeVectors, subSy.coords, subSy.types, subSy.symbols,ham,sy.coulvs[partsCoreHalo[partIndex]],nocc=nocc,norbsInCore=norbsInCore,mu=mu,etemp=sdc.etemp,overlap=over,full_data=False,verb=sdc.verb,newsystem=True,keepmem=True)

        energyInPart, forcesInPart = get_energy_forces(eng,partIndex,sdc.nparts,norbs,ham,sy.latticeVectors,subSy.coords,subSy.types,subSy.symbols,sy.coulvs[partsCoreHalo[partIndex]],nocc=nocc,norbsInCore=norbsInCore,numberOfCoreAtoms=numberOfCoreAtoms,mu=mu,etemp=sdc.etemp,verb=sdc.verb,newsystem=False,keepmem=False)

        chargesInPart = chargesInPart[:len(parts[partIndex])]
        subSy.charges = chargesInPart

        forcesInPart = forcesInPart[:len(parts[partIndex])]

        #Save the subsystems list for returning them
        subSysOnRank.append(subSy)

        print("TotalCharge in part",partIndex,sum(chargesInPart))
        print("Charges in part",chargesInPart)
        
        toc = time.perf_counter()
        print("Time to get_densityMatrix", toc - tic, "(s)")
        # Building a graph from DMs
        graphOnRank = collect_graph_from_rho(
            graphOnRank, rho, sdc.gthresh, sy.nats, sdc.maxDeg, partsCoreHalo[partIndex],len(parts[partIndex]), hindex
        )

        chargesOnRank = collect_charges(chargesOnRank,chargesInPart,parts[partIndex],sy.nats,verb=True)
        energyOnRank = collect_energy(energyOnRank,energyInPart,verb=True)
        forcesOnRank = collect_forces(forcesOnRank,forcesInPart,parts[partIndex],sy.nats,verb=True)

    if (is_mpi_available and numranks > 1):
        fullGraphRho = collect_and_sum_matrices(graphOnRank, rank, numranks, comm)
        fullCharges = collect_and_sum_vectors_float(chargesOnRank, rank, numranks, comm)
        fullEnergy = collect_and_sum_vectors_float(energyOnRank, rank, numranks, comm)
        fullForces = collect_and_sum_matrices(forcesOnRank, rank, numranks, comm)
        comm.Barrier()
    else:
        fullGraphRho = graphOnRank
        fullCharges = chargesOnRank
        fullEnergy = energyOnRank
        fullForces = forcesOnRank

    print_graph(fullGraphRho)
    return fullGraphRho, fullCharges, fullEvals, fullDvals, fullEnergy[0], fullForces, subSysOnRank, mu


def get_adaptiveSCFDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu):
    fullGraph = graphNL

    #Initial guess for the excess ocupation vector. This is the negative of 
    #the charge! 
    #charges = np.zeros(sy.nats) 
    #charges = sy.charges
    chargesOld = np.zeros(sy.nats) 
    chargesIn = None
    chargesOld = None
    chargesOut = None
    sdc.etemp = 1000
    for gscf in range(sdc.numAdaptIter):
        msg = "Graph-adaptive iteration" + str(gscf)
        status_at("get_adaptiveSCFDM",msg)
        # Partition the graph
        parts = graph_partition(fullGraph, sdc.partitionType, sdc.nparts, True)
        njumps = 2
        partsCoreHalo = []
        numCores = []

        #print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)
            print("core,halo size:", i, "=", nc, nh)
        #    print("coreHalo for part", i, "=", coreHalo)

        symbols = np.array(sy.symbols)[sy.types]
        hubbard_u = np.where(symbols == 'H', 12.054683, 0.0) + np.where(symbols == 'O', 11.876141, 0.0)
        #if gscf == 0:
        if np.sum(sy.charges == 0) == sy.nats:
            sy.coulvs = np.zeros(len(sy.charges))
            ecoul = 0.0
            fcoul = np.zeros([len(sy.charges), 3])
        else:
            sy.coulvs, ecoul, fcoul = get_PME_coulvs(sy.charges, hubbard_u, sy.coords, sy.types, sy.latticeVectors, calculate_forces=1)
        fullGraphRho,sy.charges,evals,dvals,energy,forces,subSysOnRank,mu = get_singlePoint_charges(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, gscf, mu)
        energy = energy - ecoul
#        fcoul = fcoul * ((2*sy.charges - olddeltaq) / olddeltaq)[:,None] 
        forces = forces + fcoul
#        with open('ecoul.log', 'a') as f:
#            f.write(f'energy: {energy}, ecoul: {ecoul}\n')
#            f.write(f'{ecoul}\n')
#            f.write(f'forces: {forces}\n')
#            f.write(f'fcoul:  {fcoul}\n')
#            f.write(f'sy.coulvs: {sy.coulvs}\n')
        print("Collected charges",sy.charges)

        scfError, sy.charges, chargesOld, chargesIn, chargesOut = diis_mix(sy.charges,chargesOld,chargesIn,chargesOut,gscf,verb=True)
        #scfError,charges,chargesOld = linear_mix(0.25,charges,chargesOld,gscf)

        #print_graph(fullGraphRho)


        fullGraph = add_graphs(fullGraphRho, graphNL)
        for i in range(sy.nats):
            print("Charges:",i,sy.symbols[sy.types[i]],sy.charges[i])
        print("SCF ERR =",scfError)
        print("TotalCharge",sum(sy.charges))
        
        if(scfError < sdc.scfTol):
            status_at("get_adaptiveSCFDM","SCF converged with SCF error = "+str(scfError))
            with open('ecoul.log', 'a') as f:
                f.write(f'{ecoul}\n')
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

    return fullGraph,sy.charges,energy,forces,mu,parts,subSysOnRank


