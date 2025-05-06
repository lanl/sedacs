"""
main.py
====================================
Main sedacs prototype driver to perform a
graph-based addaptive construction of the density
matrix together with a full self-consistent charge
optimization, followed by single-point calculation 
of the energy and forces.

"""

import sys

import numpy as np
from sedacs.driver.graph_adaptive_scf import get_adaptiveSCFDM
from sedacs.driver.graph_adaptive_sp_energy_forces import get_adaptive_sp_energy_forces
from sedacs.driver.init import get_args, init
from sedacs.charges import get_charges
import sedacs.globals as gl


# Pass arguments from command line
args = get_args()

# Initialize sedacs

np.set_printoptions(threshold=sys.maxsize)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(
    args
)

sdc.verb = True

# Perform a graph-adaptive calculation of the density matrix through SCF cycles
mu = 0.0
graphDH, sy.charges, mu, parts, subSysOnRank = get_adaptiveSCFDM(
    sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu
)
# Perform a single-point graph-adaptive calculation of the energy and forces
graphDH, sy.charges, energy, forces, mu, parts, partsCoreHalo, subSysOnRank = get_adaptive_sp_energy_forces(
    sdc, eng, comm, rank, numranks, sy, hindex, graphDH, mu
)
print("total energy:", energy)
print("forces:", forces[0])
