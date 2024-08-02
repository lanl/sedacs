"""Main sedacs prototype driver"""

import argparse
import time

import numpy as np
from mpi4py import MPI
#from sedacs.graph import *
# from sedacs.graph_partition import *
from sedacs.io import read_coords_file
from sedacs.parser import Input
from sedacs.system import System, build_nlist

# from proxies.python.first_level import *

parser = argparse.ArgumentParser(description="Test driver for sedacs")

parser.add_argument("--use-torch", help="Use pytorch", required=False, action="store_true")

args = parser.parse_args()
if args.use_torch:
    try:
        import torch as tc

        if tc.cuda.is_available():
            print("Using CUDA")
            args.device = tc.device("cuda")
        elif tc.backends.mps.is_available():
            print("Using MPS")
            args.device = tc.device("mps")
        else:
            print("Using CPU")
            args.device = tc.device("cpu")
        from sedacs.torch import build_nlist_torch
    except ImportError:
        raise

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numranks = comm.Get_size()

# Initialize the code by reading the input file
sdc = Input("input.in", True)

# Read the coordinates
sy = System(1)
sy.latticeVectors, sy.symbols, sy.types, sy.coords = read_coords_file(sdc.coordsFileName, lib="None", verb=True)
sy.nats = len(sy.coords[:, 0])

tic = time.perf_counter()
if args.use_torch:
    nl = build_nlist_torch(
        sy.coords, sy.latticeVectors, 5.0, device=args.device, rank=rank, numranks=numranks, verb=False
    )
else:
    nl, nlTrX, nlTrY, nlTrZ = build_nlist(sy.coords, sy.latticeVectors, 5.0, rank=rank, numranks=numranks, verb=False)
comm.Barrier()
toc = time.perf_counter()
print("Time for build_nlist", toc - tic, "(s)")
if rank == 0:
    with open("neighborinfo.txt", "w") as of:
        for kk in range(sy.nats):
            nl_this = np.flip(np.sort(nl[kk, 1 : nl[kk, 0]]))
            print(
                "Neighs (x-coords) of {0} ({1})= ".format(kk, nl[kk, 0]), nl_this, "(", sy.coords[nl_this], ")", file=of
            )
