"""partition
Some functions for partition a graph
 
So far: Regular and metis partition
"""
import os, sys
import networkx as nx
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
def partition(graph,partitionType,nparts,verb=False):
    if(partitionType == "Regular"):
        parts = regular_partition(graph,nparts,verb)
    elif(partitionType == "Metis"):
        parts = metis_partition(graph,nparts,verb)
    return parts

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
    return parts

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
    nx[:] = False # $$$ ??? what is nx ???

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


class Part:
    def __init__(self):
        self.name = "myPart"
        self.adj = np.zeros((1,1))
        self.nparts = 0
        self.submats = []
        self.verbose = True
        self.parts = []
        self.edgecuts = 0
        self.sizes = 0
        self.nnodes = 0
        self.nc = []
        self.nch = []
        self.coreHalo = []



    def metis(self,nparts=1,verbose=0):
        """ Partitions using metis """
        self.nparts = nparts
        edgecuts, parts = metis.part_graph(self.G, self.nparts)
        self.edgecuts = edgecuts 
        self.parts = parts
        self.sizes = []
        for i in range(nparts): 
            self.sizes.append(0)
            for j in range(len(self.parts)):
                if(self.parts[j] == i):
                    self.sizes[i] = self.sizes[i] + 1


    def help(self):
        print("\nGraph partition class:")
        print("To instanciate: gp = Part()")

    def print(self):
        print("\nGraph partition data:")
        print("Number of parts =",self.nparts)


    def getGraph(self,A,thr):
        """Builds the adjacency matrix from any given
        square matrix.
        """
        n = len(A)
        nnz = 0
        adj = np.zeros((n,n))
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(A[i,j])  > thr :
                    nnz = nnz + 1
                    adj[i,j] = 1
                    adj[j,i] = 1

        self.adj = adj            
        self.nnz = nnz 

        G = nx.Graph()
        for i in range(n-1):
            for j in range(i+1,n):
                if (adj[i,j] == 1):
                    G.add_nodes_from([i,j])
                    G.add_edge(i,j,weight=1.0)
        self.G = G
        self.nnodes = n

        return G

    def getSubmats(self,A):

        parts = self.parts
        nparts = self.nparts
        nnodes = self.nnodes

        for i in range(nparts):

            n = self.sizes[i]
            submat = np.zeros((n,n))

            jj = -1 ; kk = -1 
            for j in range(nnodes):
                if( parts[j] == i):
                    jj = jj + 1
                    kk = -1
                    for k in range(nnodes):
                        if(parts[k] == i):
                            kk = kk + 1
                            submat[jj,kk] = A[j,k]

            self.submats.append(submat)

    def getCoreHaloIndices(self,ipart,A,doubleJump):

        nparts = self.nparts
        nnodes = self.nnodes
        parts = self.parts
        adj = self.adj
        nx = np.zeros((nnodes))

        #Get the core indices from graph
        coreHalo = []
        nch = 0
        for i in range(nnodes):
            if(parts[i] == ipart):
                coreHalo.append(i)
                nch = nch + 1 
                nx[i] = 1

        nc = nch 

        #Add halos from graph
        for ii in range(nc):
            i = coreHalo[ii]
            for j in range(nnodes):
                if((adj[i,j] > 0.1) & (nx[j] == 0)): 
                    nch = nch + 1
                    coreHalo.append(j)
                    nx[j] = 1

        #Add halos again (if double jump) 
        if(doubleJump):
            nch1 = nch 
            for ii in range(nch1):
                i = coreHalo[ii]
                for j in range(nnodes):
                    if((adj[i,j] > 0.1) & (nx[j] == 0)):
                        nch = nch + 1
                        coreHalo.append(j)
                        nx[j] = 1

        return coreHalo, nc, nch         

    def getCoreHalos(self,ham,doubleJump):
        
        for i in range(self.nparts):
            coreHalo, nc, nch = self.getCoreHaloIndices(i,ham,doubleJump)
            self.nc.append(nc)
            self.nch.append(nch)
            self.coreHalo.append(coreHalo)

    def printCoreHalos(self,i):
       print("Core and halos list for part ",i)
       print("Nodes in the core ",self.nc[i])
       print("Nodes in the core+halos ",self.nch[i])
       print("Cores+halo list ",self.coreHalo[i])


    def getSubmatrix(self,nodeList,A):
        
        N = len(A) 
        n = len(nodeList)
        sMat = np.zeros((n,n))
        if(n > N):
            print("ERROR: Node list lenght larges than matrix size ...")
            exit(0)
        for ii in range(n):
            i = nodeList[ii]
            for jj in range(n):
                j = nodeList[jj]
                sMat[ii,jj] = A[i,j]

        return sMat
        
    def getSubmats(self,A):

        for i in range(self.nparts):
            nodesList = self.coreHalo[i]
            print(nodesList)

            sMat = self.getSubmatrix(nodesList,A)
            self.submats.append(sMat)
        












    

