"""Main sedacs prototype driver"""

import sys

import numpy as np
from sedacs.driver.graph_adaptive import get_adaptiveDM
from sedacs.driver.init import get_args, init

# Pass arguments from comand line
args = get_args()

# Initialize sedacs

np.set_printoptions(threshold=sys.maxsize)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(args)

sdc.verb = True
print("!!!", graphNL[0], graphNL.shape)

# Perform a graph-adaptive calculation of the density matrix
get_adaptiveDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL)
