import argparse
import time

from sedacs.driver.classical import *
from sedacs.driver.graph import *
from sedacs.driver.graphAdaptive import *
from sedacs.driver.init import *
from sedacs.parser import *
from sedacs.graph_partition import *
from sedacs.system import *
from sedacs.proxy_a import *

global mpiON
try:
    from mpi4py import MPI

    mpiON = True
except ModuleNotFoundError:
    mpiON = False

global tcAvail
try:
    import torch as tc

    tcAvail = True
    from sedacs.torch import *
except ModuleNotFoundError:
    tcAval = False
