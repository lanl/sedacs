"""Main sedacs prototype driver."""

import sys
import time

import numpy as np
import torch

from sedacs.driver.graph_adaptive import get_adaptiveDM
from sedacs.driver.init import get_args, init

# ADD PROXYA_PATH to PYTHONPATH
# sedacs_base = importlib.util.find_spec("sedacs").submodule_search_locations[0]
# target_path = os.path.abspath(os.path.join(sedacs_base, "../../"))
# sys.path.append(target_path)

# ADD PROXYA_PATH to PYTHONPATH
proxya_path = "/home/maxim/Projects/SEDACS_github/sedacs"
sys.path.append(proxya_path)

### ADD PATH TO PYSEQM ###
pyseqm_path = "/home/maxim/Projects/git2/PYSEQM_dev/"
sys.path.insert(1, pyseqm_path)

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

tic = time.perf_counter()

# Pass arguments from comand line
args = get_args()

# Initialize sedacs
torch.set_printoptions(precision=4, linewidth=300, threshold=20000)
np.set_printoptions(precision=4, linewidth=300)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graph_nl, _ = init(args)
print(f"INITIALIZATION TIME {time.perf_counter() - tic:>7.2f} (s)", rank)

# sdc.verb = True
# Perform a graph-adaptive calculation of the density matrix
get_adaptiveDM(sdc, eng, comm, rank, numranks, sy, hindex, graph_nl)
print(f"TOTAL TIME {time.perf_counter() - tic:>7.2f} (s)", rank)
