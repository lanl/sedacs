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

#Get initial graph (neighbor list)
graph = get_initial_graph(coords,sdc.rcut,sdc.maxDeg,True)

sy = system() 
nxGraph = get_nx_graph(graph,1.0)
#print_nx_graph(nxGraph)
subSy = []
subSy.append(sy)
subSy.append(sy)

print("At rank",rank)
ham = get_hamiltonian(subSy[rank].coords,atomTypes=np.zeros((1),dtype=int),verb=False)








