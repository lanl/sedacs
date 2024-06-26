#!/usr/bin/env python3
""" Load modules 

"""

import argparse
from sdc_parser import *
from sdc_system import *
from proxy_a import *
import time
global mpiON
try:
    from mpi4py import MPI
    mpiON = True
except:
    mpiON = False
from sdc_graph import *
from sdc_partition import *
from sdc_init import *
from sdc_graphAdaptive import *
from sdc_classical import *
global tcAvail 
try:
    import torch as tc
    tcAvail = True
    from sdc_torch import *
except:
    tcAval = False

