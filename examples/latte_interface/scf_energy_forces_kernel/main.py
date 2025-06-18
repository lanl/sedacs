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
from sedacs.driver.graph_adaptive_kernel_scf import get_adaptive_KernelSCFDM 
from sedacs.driver.graph_adaptive_sp_energy_forces import get_adaptive_sp_energy_forces
from sedacs.driver.init import get_args, init
from sedacs.file_io import read_latte_tbparams
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

# Load the LATTE tight-binding parameters
latte_tbparams = read_latte_tbparams(
    "../../../parameters/latte/TBparam/electrons.dat"
)
# Get the Hubbard U values for each atom in the system
Hubbard_U = [latte_tbparams[symbol]["HubbardU"] for symbol in sy.symbols]
Hubbard_U = np.array(Hubbard_U)[sy.types]
sy.hubbard_u = Hubbard_U

# Perform a graph-adaptive calculation of the density matrix through SCF cycles
mu = 0.0
graphDH, sy.charges, mu, parts, partsCoreHalo, subSysOnRank = get_adaptive_KernelSCFDM(
    sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu
)
# Perform a single-point graph-adaptive calculation of the energy and forces
graphDH, sy.charges, energy, forces, mu, parts, partsCoreHalo, subSysOnRank = get_adaptive_sp_energy_forces(
    sdc, eng, comm, rank, numranks, sy, parts, partsCoreHalo, hindex, graphDH, mu
)
print("total energy:", energy)
print("forces:", forces[0])
