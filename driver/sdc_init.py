#!/usr/bin/env python3
""" Initialize sedac driver

"""
import argparse
from sdc_parser import *
from sdc_system import *
from proxy_a import *
import time
try:
    from mpi4py import MPI
    mpiON = True
except ImportError as e:
    mpiON = False
from sdc_graph import *


def init(args):

    if(mpiON):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        numranks = comm.Get_size()
    else:
        comm = None
        rank = 0
        numraks = 1

    #Initialize the code by reading the input file
    sdc = sdc_input("input.in",True)

    #Read the coordinates
    sy = system(1)
    sy.latticeVectors,sy.symbols,sy.types,sy.coords = \
        read_coords_file(sdc.coordsFileName,lib="None",verb=True)
    sy.nats = len(sy.coords[:,0])

    #Get hindex, the orbital index for each atom in the system
    sy.norbs, hindex = get_hindex(sdc.orbs,sy.symbols,sy.types)

    tic = time.perf_counter()
    if args.use_torch:
        nl = build_nlist_torch(sy.coords,sy.latticeVectors,5.0,rank=rank,numranks=numranks,verb=False)
    else:    
        nl,nlTrX,nlTrY,nlTrZ = build_nlist(sy.coords,sy.latticeVectors,5.0,rank=rank,numranks=numranks,verb=False)
    comm.Barrier()
    toc = time.perf_counter()
    print("Time for build_nlist", toc - tic,"(s)")
    if rank == 0:
        with open('neighborinfo.txt','w') as of:
            for kk in range(sy.nats):
                print("Neighs (x-coords) of {} = ".format(kk),nl[kk,1:nl[kk,0]],"(",sy.coords[nl[kk,1:nl[kk,0]],0],")",file=of)

    #Get the neighbors of atom 1234 
    subSy = system(nl[1234,0])
    subSy.symbols = sy.symbols
    subSy.coords,subSy.types = extract_subsystem(sy.coords,sy.types,sy.symbols,nl[1234,1:nl[1234,0]])
    if rank == 0:
        write_pdb_coordinates("subSyNL.pdb",subSy.coords,subSy.types,subSy.symbols)

    #Get initial graph (from a neighbor list)
    graphNL = get_initial_graph(sy.coords,nl,sdc.rcut,sdc.maxDeg,True)

    fullGraph = np.zeros((sy.nats,sdc.maxDeg+1),dtype=int)
    fullGraph[:,:] = graphNL[:,:]

    return sdc,comm,rank,numranks,sy,hindex,fullGraph,nl

