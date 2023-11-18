#!/usr/bin/env python3
""" Main sedacs prototype driver

"""

import argparse
from sdc_parser import *
from sdc_system import *
from proxy_a import *
import time
try:
    from mpi4py import MPI
    mpi = True
except ImportError as e:
    mpi = False
from sdc_graph import *
from sdc_partition import *


parser = argparse.ArgumentParser(description='Test driver for sedacs')

parser.add_argument("--use-torch",help="Use pytorch",required=False,action="store_true")
    
args=parser.parse_args()
if args.use_torch:
    try:
        import torch as tc
        if tc.cuda.is_available():
            print("Using CUDA")
            args.device = tc.device('cuda')
        elif tc.backends.mps.is_available():
            print("Using MPS")
            args.device = tc.device('mps')
        else:
            args.device = tc.device('cpu')
        from sdc_torch import *
    except ImportError as e:
        raise ImportError("Unable to import pytorch")
            
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numranks = comm.Get_size()

#Initialize the code by reading the input file
sdc = sdc_input("input.in",True)

#Read the coordinates
sy = system(1)
sy.latticeVectors,sy.symbols,sy.types,sy.coords = \
    read_coords_file(sdc.coordsFileName,lib="None",verb=True)
sy.nats = len(sy.coords[:,0])

tic = time.perf_counter()
nl,nlTrx,nlTry,nlTrz = build_nlist(sy.coords,sy.latticeVectors,sdc.rcut,rank=rank,numranks=numranks,verb=False)
comm.Barrier()
toc = time.perf_counter()
print("Time for build_nlist", toc - tic,"(s)")
exit(0)
if args.use_torch:
    nl = build_nlist_torch(sy.coords,sy.latticeVectors,5.0,rank=rank,numranks=numranks,verb=False)
else:    
    nl,nlTrX,nlTrY,nlTrZ = build_nlist(sy.coords,sy.latticeVectors,5.0,rank=rank,numranks=numranks,verb=False)

#Get the neighbors of atom 1234 
subSy = system(nl[1234,0])
subSy.symbols = sy.symbols
subSy.coords,subSy.types = extract_subsystem(sy.coords,sy.types,sy.symbols,nl[1234,1:nl[1234,0]])
write_pdb_coordinates("subSy.pdb",subSy.coords,subSy.types,subSy.symbols)
exit(0)

#Get initial graph (from a neighbor list)
graph = get_initial_graph(sy.coords,sdc.rcut,sdc.maxDeg,True)
print_graph(graph)

print("3")
#Partition the graph 
parts = partition(graph,sdc.partitionType,sdc.nparts,True)

njumps = 1
partsCoreHalo = []
numCores = []

print("\nCore and halos indices for every part:")
for i in range(sdc.nparts):
    coreHalo,nc,nh = get_coreHaloIndices(parts[i],graph,njumps)
    partsCoreHalo.append(coreHalo)
    numCores.append(nc)
    print("coreHalo for part",i,"=",coreHalo)

## Every rank will do a subset of the list of coreHalos
# @todo We will need to "reshuffle" the list so that the work-load 
# gets distributed. 

subSy = system(len(partsCoreHalo[1]))
subSy.symbols = sy.symbols
subSy.coords,subSy.types = extract_subsystem(sy.coords,sy.types,sy.symbols,partsCoreHalo[1])

write_pdb_coordinates("subSy.pdb",subSy.coords,subSy.types,subSy.symbols)


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

sy = system(3) 
#nxGraph = get_nx_graph(graph,1.0)
#print_nx_graph(nxGraph)
subSy = []
subSy.append(sy)
subSy.append(sy)

print("At rank",rank)
ham = get_hamiltonian(subSy[rank].coords,atomTypes=np.zeros((1),dtype=int),verb=False)








