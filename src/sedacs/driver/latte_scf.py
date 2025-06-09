"""Graph adaptive self-consistenf charge solver"""

import time

from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.interface_modules import call_latte_modules
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.mpi import collect_and_sum_matrices, collect_and_sum_vectors_float, collect_and_concatenate_vectors
from sedacs.system import System, extract_subsystem, get_hindex
from sedacs.coulombic import get_coulvs, get_PME_coulvs, build_coul_ham
from sedacs.charges import get_charges, collect_charges
from sedacs.evals_dvals import collect_evals, collect_dvals
from sedacs.message import status_at, error_at, warning_at
from sedacs.mixer import diis_mix, linear_mix
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
def get_singlePoint_charges(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex):
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

    for partIndex in range(partIndex1, partIndex2):
#         print("Rank, part", rank, partIndex)
        numberOfCoreAtoms = len(parts[partIndex])
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        tic = time.perf_counter()
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])
        subSy.ncores = len(parts[partIndex])
        subSy.charges = np.zeros(len(subSy.types))
        toc = time.perf_counter()
        #print("Time for extract_subsystem", toc - tic, "(s)")
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
        #print("Number of orbitals in the core =",norbsInCore)

        subSy.latticeVectors = sy.latticeVectors
        chargesInPart = call_latte_modules(eng,subSy,verb=True,newsystem=True)
        chargesInPart = chargesInPart[:len(parts[partIndex])]
        subSy.charges = chargesInPart

        #Save the subsystems list for returning them
        subSysOnRank.append(subSy)

        print("TotalCharge in part",partIndex,sum(chargesInPart))
        print("Charges in part",chargesInPart)
        

        chargesOnRank = collect_charges(chargesOnRank,chargesInPart,parts[partIndex],sy.nats,verb=True)

    if (is_mpi_available and numranks > 1):
        fullCharges = collect_and_sum_vectors_float(chargesOnRank, rank, numranks, comm)
        comm.Barrier()
    else:
        fullCharges = chargesOnRank

    return fullCharges, subSysOnRank


def get_adaptiveSCFDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL):
    fullGraph = graphNL

    #Iitial guess for the excess ocupation vector. This is the negative of 
    #the charge! 
    charges = np.zeros(sy.nats) 
    chargesOld = np.zeros(sy.nats) 
    chargesIn = None
    chargesOld = None
    chargesOut = None
    sdc.etemp = 1000

    parts = graph_partition(fullGraph, sdc.partitionType, sdc.nparts, True)
    njumps = 2
    partsCoreHalo = []
    numCores = []

    #print("\nCore and halos indices for every part:")
    for i in range(sdc.nparts):
        coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps)
        partsCoreHalo.append(coreHalo)
        numCores.append(nc)
        #print("core,halo size:", i, "=", nc, nh)

    symbols = np.array(sy.symbols)[sy.types]

    charges,subSysOnRank = get_singlePoint_charges(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex)

    print("Collected charges",charges)

    #print_graph(fullGraphRho)

    for i in range(sy.nats):
        print("Charges:",i,sy.symbols[sy.types[i]],charges[i])
    print("TotalCharge",sum(charges))
    
    return charges,parts,subSysOnRank
