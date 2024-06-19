"""partition
Some functions for partition a graph
 
"""
import os, sys
try: import networkx as nx; nxLib = True
except: nxLib = False
import numpy as np
global metisLib
try: import metis; metisLib = True 
except: metisLib = False
from sdc_graph import *

## Partition
# @brief This will partition a graph based on a defined method
# @param graph Graph to be partition
# @param partitionType Method or type of partition to be used
# @param nparts Number of total parts
# @param verb Verbosity level
# @return parts Partition containing a "list of parts" where every 
# part is a list of nodes
#
def graph_partition(graph,partitionType,nparts,verb=False):
    if(partitionType == "Regular"):
        parts = regular_partition(graph,nparts,verb)
    elif(partitionType == "Metis"):
        parts = metis_partition(graph,nparts,verb)
    elif(partitionType == "MinCut"):
        parts = mincut_partition(graph,nparts,verb)
    return parts

## Partitioning by using atomic positions.
# @brief This will use the atomic positions/coordinates in order to 
# generate fragments of the system returned as a list of list of indices.
# @param coords Atomic postitions.
# @param partitionType Method or type of partition to be uses
# @param nx Number of points in the x direction 
# @param ny Number of points in the y direction
# @param nz Number of points in the z direction
# @param verb Verbosity option
# @return parts Partition containing a "list of parts" where every 
# part is a list of nodes
#
def coords_partition(coords,partitionType,nx,ny,nz,verb=False):
    if(partitionType == "Space"):
        parts = space_partition(coords,partitionType,nx,ny,nz,verb)

    return parts

## Get total cuts
# @brief Get the total edge cuts from a given partition. 
# @param whichPart A vector where whichPart[i] indicates the partition
# that i belongs to.
# @param graph Graph to be partition. graph[i,0] = degree of node i. 
# graph[i,j>0] = the node conected to node i.
# @return cut The total cut.
#
def get_cut(whichPart,graph):
    cut = 0
    for i in range(len(whichPart)):
        partIndexI = whichPart[i]
        #Look at the neighbors to see if they are in different part
        for j in range(1,graph[i,0]+1):
            index = graph[i,j]
            partIndexJ = whichPart[index]
            if((partIndexI - partIndexJ) != 0):
                cut = cut + 1
    return cut

def test_get_cut(exit1):
    passed = True
    nnodes = 9
    whichPart = np.zeros((nnodes),dtype=int)
    whichPart[0:4] = 0 ; whichPart[4:7] = 1 ; whichPart[7:9] = 2 
    graph = np.zeros((nnodes,nnodes),dtype=int)
    graph[:,0] = 1
    # A cyclic graph
    for i in range(nnodes-1):
        graph[i,1] = i + 1 
    graph[8,1] = 0
    #3 segments will cut the graph in 3 points
    result = 3 
    try:
        cut = get_cut(whichPart,graph)
        if((result - cut) == 0):
            passed = True
        else:
            passed = False
    except:
        passed = False
    return passed

## Get partition indices
# @brief Get a vector indicating which is the part index of a particular 
# node. 
# @param parts Partition containing a "list of parts" where every 
# part is a list of nodes
# nnodes Number of nodes in the graph.
# 
def get_parts_indices(parts,nnodes):
    whichPart = np.zeros((nnodes),dtype=int)
    partIndex = -1
    for part in parts:
        partIndex = partIndex + 1
        for node in part:
            whichPart[node] = partIndex
    return whichPart

def test_get_parts_indices(exit1):
    passed = True
    parts = [[0,1,2,3],[4,5,6],[7,8]]
    nnodes = 9
    result = np.zeros((nnodes),dtype=int)
    result[0:4] = 0 ; result[4:7] = 1 ; result[7:9] = 2 
    try:
        whichPart = get_parts_indices(parts,nnodes)
        if(np.linalg.norm(result - whichPart) == 0.0): 
            passed = True
        else:
            passed = False
    except:
        passed = False
    return passed

## Get the partition list from the index vector
# @param whichPart part index vector for every node
# @param nparts Number of parts
# @return part Partition list. Every element of the list 
# is a list of node on every part
#
def get_parts_from_indices(whichPart,nparts):
    parts = [[] for i in range(nparts)]
    for i in range(len(whichPart)):
        partIndex = whichPart[i]
        parts[partIndex].append(i)

    return(parts)

def test_get_parts_from_indices(exit1):
    nnodes = 9
    whichPart = np.zeros((nnodes),dtype=int)
    whichPart[0:4] = 0; whichPart[4:7] = 1 ; whichPart[7:9] = 2
    partsRef = [[0,1,2,3],[4,5,6],[7,8]]
    parts = get_parts_from_indices(whichPart,3)
    passed = True
    for element in parts:
        if element in partsRef:
            pass
        else:
            passed = False
    return(passed)

## Get graph partition balance.
# @brief This will return the partitioning balance defined as the quotient
# between the max and min partition cardinals.
# If partition is \f$ \Pi \f$, then:
# \f[
#    \mathrm{bal} = \frac{max_i|\pi_i|}{min_i|\pi_i|}
# \f]
# where \f$ \pi_i \f$ is a part of the graph (a set of node indices)
# @return bal Balance of the partition.
#
def get_balancing(parts):
    bal = 0
    largest = 1
    smallest = 10**9
    for part in parts:
        largest = max(largest,len(part))
        smallest = min(smallest,len(part))
    bal = largest/smallest
    return bal 

def test_get_balancing(exit1):
    parts = [[0,1,2,3],[0,1]]
    result = 2 
    try:
        bal = get_balancing(parts)
        if((bal - result) == 0):
            passed = True
        else:
            passed = False
    except:
        passed = False
    return passed 

## Get partition balanging.
# @brief Same as get_balancing except this uses the partitioning 
# vector. 
# @param whichPart partition indexing vector. 
# @param nparts Number of total parts.
#
def get_balance_from_indices(whichPart,nparts):
    partsSizes = np.zeros((nparts),dtype=int)
    for i in range(len(whichPart)):
        partsSizes[whichPart[i]] = partsSizes[whichPart[i]] + 1
    bal = np.max(partsSizes)/np.min(partsSizes)
    return bal

def test_get_balance_from_indices(exit1):
    nnodes = 9
    whichPart = np.zeros((nnodes),dtype=int)
    whichPart[0:4] = 0 ; whichPart[4:7] = 1 ; whichPart[7:9] = 2
    nparts = 3 
    try:
        bal = get_balance_from_indices(whichPart,nparts)
        if(bal - 2.0 == 0.0):
            passed = True
        else:
            passed = False
    except:
        passed  = False
    return passed 

## Do node partition flips with precomputed cuts.
# @brief This function is a special case of do_flips where
# the cuts around a node are precomputed for all possible 
# part index that same node could have. This will differ from the do_flips
# since everytime there is a flip, there is no actualization of the cuts. The
# price to pay is the need of more iterations until convergence. 
# @param whichPart partition indexing vector.
# @param graph Graph to be partition. graph[i,0] = degree of node i. 
# graph[i,j>0] = the node conected to node i.
# @param nnodes Number of nodes.
# @param nparts Number of parts.
# @return whichPartNew New partition indexing verctor.
#
def do_flips_precomp(whichPart,graph,nnodes,nparts,bal=None):
    #Precompute all the possible cut vals O(nnodes*deg) 
    cutsI = np.zeros((nnodes,nparts),dtype=int)
    for i in range(nnodes):
        deg = graph[i,0] 
        #Get the max cut a node could have
        cutsI[i,:] = deg 
        #Lets look at every neighbor
        for ii in range(1,deg+1):
            index = graph[i,ii]
            partIndexII = whichPart[index]
            #Everytime there is a neighbor in a certain part
            #it will decrese the cut of I if I would be on that 
            #same part.
            cutsI[i,partIndexII] = cutsI[i,partIndexII] - 1

    
    #whichPartNew = whichPart 
    #if(bal != None):
    #    if(bal < 1.1):
    #        for i in range(nnodes):
    #            whichPartNew[i] = np.argmax(cutsI[i,:])
    

    #Now do the flips O(nnodes*nnodes/2)
    whichPartNew = whichPart 
    for i in range(nnodes):
        partIndexI = whichPart[i]
        for j in range(i+1,nnodes):
            partIndexJ = whichPart[j]
            if(partIndexI != partIndexJ):
                #Look at their neighbors and count the cuts
                origCut = 0
                newCut = 0
                #Now we know the cut when I is in partIndexI and J
                origCut = cutsI[i,partIndexI]
                newCut = cutsI[i,partIndexJ]
                #Same for J
                origCut = origCut + cutsI[j,partIndexJ]
                newCut = newCut + cutsI[j,partIndexI]
                if(newCut < origCut):
                    whichPartNew[i] = partIndexJ
                    whichPartNew[j] = partIndexI
                    partIndexI = partIndexJ
                    cutsI[i,partIndexI]=0
                    for ii in range(1,graph[i,0]+1):
                        index = graph[i,ii]
                        partIndexII = whichPart[index]
                        if((partIndexI - partIndexII) != 0):
                            cutsI[i,partIndexI] = cutsI[i,partIndexI] + 1
    
    return whichPartNew

def test_do_flips_precomp(exit1):
    nnodes = 6 
    graph = get_a_small_graph()
    whichPart = np.zeros((nnodes),dtype=int)
    result = np.zeros((nnodes),dtype=int)
    result[0:3] = 1
    whichPart[0] = 1 ; whichPart[3] = 1 ; whichPart[2] = 1
    nparts = 2 
    for i in range(10): 
        whichPartNew = do_flips_precomp(whichPart,graph,nnodes,nparts)
        whichPart = whichPartNew
        cut = get_cut(whichPart,graph)
    if(np.linalg.norm(whichPartNew - result) == 0):
        passed = True
    else:
        passed = False
    return passed

## Do node partition flips.
# @brief This function does the same as the do_flips_precomp. It will converge
# in less iterations but with a lower scaling.
# @param whichPart partition indexing vector.
# @param graph Graph to be partitioned. graph[i,0] = degree of node i. 
# graph[i,j>0] = the node conected to node i.
# @return whichPartNew New partition indexing verctor.
#
def do_flips(whichPart,graph):
    
    whichPartNew = whichPart
    totNewCut = 0
    #Now flip the pairs
    for i in range(len(graph)):
        partIndexI = whichPart[i]
        
        for j in range(i+1,len(graph)):
            partIndexJ = whichPart[j]
            if(partIndexI != partIndexJ):
                #Look at their neighbors and count the cuts
                origCut = 0
                newCut = 0
                for ii in range(1,graph[i,0]+1):
                    index = graph[i,ii]
                    partIndexII = whichPart[index]
                    if((partIndexI - partIndexII) != 0):
                        origCut = origCut + 1
                    #Alternative cut when fliped partIndexI and partIndexJ 
                    if((partIndexJ - partIndexII) != 0):
                        newCut = newCut + 1

                #Look at their neighbors and count the cuts
                for jj in range(1,graph[j,0]+1):
                    index = graph[j,jj]
                    #Original cut for J
                    partIndexJJ = whichPart[index]
                    if((partIndexJ - partIndexJJ) != 0):
                        origCut = origCut + 1
                    #Alternative cut if J would be I
                    if((partIndexI - partIndexJJ) != 0):
                        newCut = newCut + 1

                if(newCut < origCut):
                    whichPartNew[i] = partIndexJ
                    whichPartNew[j] = partIndexI
                    partIndexI = partIndexJ

                totNewCut = totNewCut + newCut 

    return whichPartNew 

def test_do_flips(exit1):
    nnodes = 6 
    graph = get_a_small_graph()
    whichPart = np.zeros((nnodes),dtype=int)
    result = np.zeros((nnodes),dtype=int)
    result[0:3] = 1
    whichPart[0] = 1 ; whichPart[3] = 1 ; whichPart[2] = 1
    nparts = 2 
    for i in range(10): 
        whichPartNew = do_flips(whichPart,graph)
        whichPart = whichPartNew
        cut = get_cut(whichPart,graph)
    if(np.linalg.norm(whichPartNew - result) == 0):
        passed = True
    else:
        passed = False
    return passed

## MinCut local partition optimization.
# @brief This will optimize a given partition based on a mincut algorithm.
# @param graph Graph to be partition
# @param nparts Number of total parts
# @param verb Verbosity level
# @return parts Partition containing a "list of parts" where every 
# part is a list of nodes
#
def mincut_partition(graph,nparts,verb):

    #Do a first partition
    nnodes = len(graph[:,0])
    parts = regular_partition(graph,nparts,verb)

    #Get part indices
    whichPart = get_parts_indices(parts,nnodes)
    print(whichPart)
    
    #Evaluate the cut
    cut = get_cut(whichPart,graph) 
    print("First cut",cut)
    
    #Evaluate the balancing
    bal = get_balancing(parts)
    print("First balance",bal)

    cutOld = 10**10
    for i in range(20):
        #whichPartNew     = do_flips(whichPart,graph)
        whichPartNew  = do_flips_precomp(whichPart,graph,nnodes,nparts,bal=bal)
        whichPart = whichPartNew
        cut = get_cut(whichPartNew,graph)
        bal = get_balance_from_indices(whichPartNew,nparts)
        print(cut,bal)
        if(cut == cutOld):
            break
        else:
            cutOld = cut
        
    parts = get_parts_from_indices(whichPart,nparts)
    return(parts)                    

## Regular partition
# @brief This will partition a graph in the most
# trivial way. Partition \f$ \Pi \f$ being:
# \f[
#    \Pi = \{\{1,...k\},\{k+1,...,2k\},...,\{(n-2)(k+1),...,(n-1)k\},\{(n-1)(k+1),...,N\}\}
# \f]
# where \f$ N = \f$ total nodes, and \f$ k = E(N/n) \f$.
# @param graph Graph to be partition
# @param nparts Number of total parts
# @param verb Verbosity level
# @return parts Partition containing a "list of parts" where every 
# part is a list of nodes
#
def regular_partition(graph,nparts,verb=False):
    if(verb):print("\nRegular partition:")
    nnodes = len(graph[:,0])
    nnodesInPart = int(nnodes/nparts)
    parts = []
    for i in range(nparts):
        parti = []
        for k in range(i*nnodesInPart,(i+1)*nnodesInPart):
            parti.append(k)
        parts.append(parti)
        if(verb):print("part",i,"=",parti)
    if(nnodesInPart*nparts < nnodes):
        for k in range(nnodesInPart*nparts,nnodes):
            parti.append(k)
        parts[nparts-1] = parti

    return parts


## Metis partition 
# @brief This will partition the graph according to the Metis method.
# Details about the metis method can be find in
# <a href="http://glaros.dtc.umn.edu/gkhome/views/metis">Metis site</a> 
# @param graph Graph to be partition
# @nparts Number of total parts
# @param verb Verbosity level
# @return parts Partition containing a "list of parts" where every 
# part is a list of nodes
#
def metis_partition(graph,nparts,verb=False):
    """ Partitions using metis """
    if(metisLib == False):
         print("\n ERROR: Consider installing Metis library \n")
         exit(0)
    if(verb):print("\nMetis partition:")
    nxGraph = get_nx_graph(graph,1.0) 
    #Metis partition metis call
    #Metis returns nxParts which is a list of every's part (or "color") 
    #to where they belong. Node "i" belongs to "metisParts[i]" part.
    edgecuts, metisParts = metis.part_graph(nxGraph, nparts)

    #The next lines will transform from metis to our partition format
    parts = []
    for k in range(nparts):
        parts.append([])
    nnodes = len(graph[:,0])
    for k in range(nnodes):
        parts[metisParts[k]].append(k)
    if(verb):
        for i in range(nparts):
            print("part",i,"=",parts[i])
    
    #plot_graph(nxGraph)
    return parts

#This test is disabled for now
def testNo_metis_partition(exit1):
    nnodes = 6 
    graph = get_a_small_graph()
    whichPart = np.zeros((nnodes),dtype=int)
    result = np.zeros((nnodes),dtype=int)
    nparts = 2 
    try:
        parts = metis_partition(graph,nparts)
        whichPart = get_parts_indices(parts,nnodes)
        cut = get_cut(whichPart,graph)
        if(cut == 1):
            passed = True
        else:
            passed = False
    except:
        passed = False
    return passed

## Get the core and halo indices
# @brief Gets the halos given a list of cores and a graph
# @param core list of cores 
# @param graph Graph to extract the halos from
# @param njumps It will search the halos among the "njumps" nearest neighbors
#
def get_coreHaloIndices(core,graph,njumps):
    coreHalo = core
    nc = len(coreHalo)
    nch = nc
    nnodes = len(graph[:,0])
    nx = np.zeros((nnodes),dtype=bool)
    nx[:] = False # Logical mask 

    for k in range(nc):
        i = coreHalo[k]
        if(i != -1): nx[i] = True
    #Add halos from graph
    for jump in range(njumps):
        nc1 = nch 
        for k in range(nc1):
            i = coreHalo[k]
            degI = len(graph[i,:])
            for kk in range(1, degI):
                                      # $$$ also this cycles needs to be interrupted when reaching -1 ???
                j = graph[i,kk]
                if((j != -1) & (nx[j] == False)):
                    #print(i,j)
                    nch = nch + 1
                    coreHalo.append(j)
                    nx[j] = True
    return coreHalo, nc, nch


