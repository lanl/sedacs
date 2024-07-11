"""Main sedacs prototype driver"""

import sys

import numpy as np
from sedacs.driver.classical import do_MD
from sedacs.driver.init import get_args, init

# Pass arguments from comand line
args = get_args()

# Initialize sedacs

np.set_printoptions(threshold=sys.maxsize)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(args)

sdc.verb = True
print("!!!", graphNL[0], graphNL.shape)

# test_ffield(sy.coords,sy.types,sy.symbols,sy.latticeVectors,nl,nlTrX,nlTrY,nlTrZ)
do_MD(sy.coords, sy.types, sy.symbols, sy.latticeVectors, nl, nlTrX, nlTrY, nlTrZ, sy.vels, 0.01, 10000, 0.0)
