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
            print("Using CPU")
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
if args.use_torch:
    nl = build_nlist_torch(sy.coords,sy.latticeVectors,5.0,device=args.device,rank=rank,numranks=numranks,verb=False)
else:    
    nl,nlTrX,nlTrY,nlTrZ = build_nlist(sy.coords,sy.latticeVectors,5.0,rank=rank,numranks=numranks,verb=False)
comm.Barrier()
toc = time.perf_counter()
print("Time for build_nlist", toc - tic,"(s)")
if rank == 0:
    with open('neighborinfo.txt','w') as of:
        for kk in range(sy.nats):
            print("Neighs (x-coords) of {} = ".format(kk),nl[kk,1:nl[kk,0]],"(",sy.coords[nl[kk,1:nl[kk,0]],0],")",file=of)
exit(0)
