"""Main sedacs prototype driver"""
import importlib.util
import numpy as np
import torch
import time
import sys
import os

# ADD PROXYA_PATH to PYTHONPATH
sedacs_base = importlib.util.find_spec("sedacs").submodule_search_locations[0]
target_path = os.path.abspath(os.path.join(sedacs_base, "../../"))
sys.path.append(target_path)

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

from sedacs.driver.graph_adaptive import get_adaptiveDM
from sedacs.driver.init import get_args, init

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
# Perform a graph-adaptive calculation of the density matrix
get_adaptiveDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL)
print("TOTAL TIME {:>7.2f} (s)".format(time.perf_counter() - tic), rank)
