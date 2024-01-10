#!/usr/bin/env python3
""" Graph self-consistent solver

"""
from sdc_loadMods import *
from sdc_partition import *
import time

def get_singlePoit(sdc,rank,numranks,comm,parts,partsCoreHalo,sy,hindex): 
    partsPerRank = int(sdc.nparts/numranks)
    partIndex1 = rank*partsPerRank 
    partIndex2 = (rank+1)*partsPerRank 
    graphOnRank = None
    for partIndex in range(partIndex1,partIndex2):
        print("Rank, part",rank,partIndex)
        subSy = system(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        tic = time.perf_counter()
        subSy.coords,subSy.types = extract_subsystem(sy.coords,sy.types,sy.symbols,partsCoreHalo[partIndex])
        toc = time.perf_counter()
        print("Time for extract_subsystem", toc - tic,"(s)") 
        partFileName = "subSy"+str(rank)+"_"+str(partIndex)+".pdb"
        write_pdb_coordinates(partFileName,subSy.coords,subSy.types,subSy.symbols)
        tic = time.perf_counter()
        ham = get_hamiltonian(subSy.coords,atomTypes=np.zeros((1),dtype=int),verb=False)
        toc = time.perf_counter()
        print("Time for get_hamiltonian", toc - tic,"(s)") 
        norbs = subSy.nats
 #       ham = get_hamiltonian(subSy.coords)
        occ = int(float(norbs)/2.0) #Get the total occupied orbitals
        tic = time.perf_counter()
        rho = get_densityMatrix(ham,occ)
        toc = time.perf_counter()
        print("Time for get_densityMatrix", toc - tic,"(s)") 
        #Building a graph from DMs
        graphOnRank = collect_graph_from_rho(graphOnRank,rho,sdc.gthresh,sy.nats,sdc.maxDeg,parts[partIndex],hindex)
    fullGraphRho = collect_and_sum_matrices(graphOnRank,rank,numranks,comm)
    comm.Barrier()
    return fullGraphRho


def get_adaptiveDM(sdc,comm,rank,numranks,sy,hindex,graphNL):
    fullGraph = graphNL
    for gsc in range(sdc.numAdaptIter):
        #Partition the graph 
        parts = partition(fullGraph,sdc.partitionType,sdc.nparts,True)
        njumps = 1; partsCoreHalo = []; numCores = []
        print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo,nc,nh = get_coreHaloIndices(parts[i],fullGraph,njumps)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)
            print("coreHalo for part",i,"=",coreHalo)

        fullGraphRho = get_singlePoit(sdc,rank,numranks,comm,parts,partsCoreHalo,sy,hindex)

        fullGraph = add_graphs(fullGraphRho,graphNL) 

    #Get the neighbors of atom 1234 (by the graph) 
    subSy = system(fullGraphRho[1234,0])
    subSy.symbols = sy.symbols
    subSy.coords,subSy.types = extract_subsystem(sy.coords,sy.types,sy.symbols,fullGraph[1234,1:fullGraph[1234,0]])

    if rank == 0:
        write_pdb_coordinates("subSyG.pdb",subSy.coords,subSy.types,subSy.symbols)
    exit(0)
    if(rank == 0):
        print_graph(graphOnRank)







