"""Main sedacs prototype driver to perform a 
graph-based addaptive construction of the density 
matrix together with a full self-consistent charge 
optimization"""

import sys

import numpy as np
from sedacs.driver.latte_scf import get_adaptiveSCFDM
from sedacs.driver.init import get_args, init
from sedacs.charges import get_charges
import sedacs.globals as gl

# Pass arguments from comand line
args = get_args()

# Initialize sedacs

np.set_printoptions(threshold=sys.maxsize)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(args)

sdc.verb = True

mu = 0
# Perform a graph-adaptive calculation of the density matrix
sy.charges,parts,subSysOnRank = get_adaptiveSCFDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL)

#sy.energy,sy.forces = get_energy_and_forces(sdc, eng, comm, rank, numranks, sy, hindex, graphDH)





