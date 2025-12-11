"""Main sedacs prototype driver to perform a 
graph-based addaptive construction of the density 
matrix together with a full self-consistent charge 
optimization"""

import sys

import numpy as np
from sedacs.driver.graph_adaptive_scf import get_adaptiveSCFDM
from sedacs.driver.init import get_args, init
from proxies.python.proxy_global import bring_tbparams
import sedacs.globals as gl

# Pass arguments from comand line
args = get_args()

# Initialize sedacs

np.set_printoptions(threshold=sys.maxsize)

# Initialize sdc parameters
sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(args)

sdc.verb = True

# Get the tight-binding parameters for the proxy code
proxy_tbparams = bring_tbparams()
# Get the Hubbard U values for each atom in the system
Hubbard_U = [proxy_tbparams[i][0].u for i in range(len(sy.symbols))]
Hubbard_U = np.array(Hubbard_U)[sy.types]
sy.hubbard_u = Hubbard_U

# Perform a graph-adaptive calculation of the density matrix
mu = 0.0
graphDH, sy.charges, mu, parts, partsCoreHalo, subSysOnRank = get_adaptiveSCFDM(
    sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu)






