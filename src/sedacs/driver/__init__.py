import argparse
import time

from sedacs.mods.sdc_parser import *
from sedacs.mods.sdc_system import *
from sedacs.proxy_a import *

global mpiON
try:
    from mpi4py import MPI

    mpiON = True
except ModuleNotFoundError:
    mpiON = False

from sedacs.driver.sdc_classical import *
from sedacs.driver.sdc_graph import *
from sedacs.driver.sdc_graphAdaptive import *
from sedacs.driver.sdc_init import *
from sedacs.mods.sdc_partition import *

global tcAvail
try:
    import torch as tc

    tcAvail = True
    from sedacs.mods.sdc_torch import *
except ModuleNotFoundError:
    tcAval = False
