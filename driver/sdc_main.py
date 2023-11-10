#!/usr/bin/env python3
""" Main sedacs prototype driver

"""

from sdc_parser import *
from sdc_system import *
from proxy_a import *
try:
    from mpi4py import MPI
    mpi = True
except ImportError as e:
    mpi = False
from sdc_graph import *
from sdc_partition import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#Initialize the code by reading the input file
sdc = sdc_input("input.in",True)

#Read the coordinates
latticeVectors,symbols,types,coords = \
        read_coords_file(sdc.coordsFileName,lib="None",verb=False)

sy = system(); sy.coords = coords; sy.latticeVectors = latticeVectors
sy.symbols = symbols; sy.types = types

#Get initial graph (from a neighbor list)
graph = get_initial_graph(coords,sdc.rcut,sdc.maxDeg,True)
print_graph(graph)

#Partition the graph 
parts = partition(graph,sdc.partitionType,sdc.nparts,True)

njumps = 1
coreHalos,nc,nh = get_coreHaloIndices(parts[0],graph,njumps)

print("coreHalos",coreHalos)


#Get adjacency matrix
#Initiate partition 
#gp = Part()
#gp.getGraph(ham,0.01)
#gp.metis(nparts=2)
#print(gp.parts)
#print(gp.sizes)

#gp.getCoreHalos(ham,True)

#gp.printCoreHalos(1)

#gp.getSubmats(ham)

sy = system() 
#nxGraph = get_nx_graph(graph,1.0)
#print_nx_graph(nxGraph)
subSy = []
subSy.append(sy)
subSy.append(sy)

print("At rank",rank)
ham = get_hamiltonian(subSy[rank].coords,atomTypes=np.zeros((1),dtype=int),verb=False)








