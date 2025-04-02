"""Main sedacs prototype driver to perform a 
graph-based addaptive construction of the density 
matrix together with a full self-consistent charge 
optimization"""

import sys

import numpy as np
from sedacs.driver.graph_adaptive_scf import get_adaptiveSCFDM
from sedacs.driver.graph_adaptive_sp_energy_forces import get_sp_energy_forces
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

# Perform a graph-adaptive calculation of the density matrix
mu = 0.0
graphDH,sy.charges,mu,parts,subSysOnRank = get_adaptiveSCFDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu)

graphDH,sy.charges,EPOT,FTOT,mu,parts,subSysOnRank = get_sp_energy_forces(sdc, eng, comm, rank, numranks, sy, hindex, graphDH, mu)

print("total energy:", EPOT)
print("forces:", FTOT)
