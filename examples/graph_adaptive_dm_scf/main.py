"""Main sedacs prototype driver"""

import sys

import numpy as np
from sedacs.driver.graph_adaptive_scf import get_adaptiveSCFDM
from sedacs.driver.init import get_args, init
from sedacs.charges import get_charges

# Pass arguments from comand line
args = get_args()

# Initialize sedacs

np.set_printoptions(threshold=sys.maxsize)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(args)

sdc.verb = True

# Perform a graph-adaptive calculation of the density matrix
get_adaptiveSCFDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL)

# Compute/retrieve the charges using the graph and calling the proxy 




