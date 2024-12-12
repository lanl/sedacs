"""Main sedacs prototype driver"""

import time
import sys
print(sys.version)
print(sys.executable)
#exit()
sys.path.insert(1, "/home/maxim/Projects/git2/PYSEQM_dev/")
import torch

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

import numpy as np
from sedacs.driver.graph_adaptive import get_adaptiveDM
from sedacs.driver.init import get_args, init
import os

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

torch.cuda.empty_cache()

tic = time.perf_counter()
# Pass arguments from comand line
args = get_args()

# Initialize sedacs
torch.set_printoptions(precision=4, linewidth=300, threshold = 20000 )
np.set_printoptions(precision=4, linewidth=300)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(args)
print("INITIALIZATION TIME {:>7.2f} (s)".format(time.perf_counter() - tic), rank)


#sdc.verb = True
#print("!!!", graphNL[0], graphNL.shape)

# Perform a graph-adaptive calculation of the density matrix

get_adaptiveDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL)
print("TOTAL TIME {:>7.2f} (s)".format(time.perf_counter() - tic), rank)


