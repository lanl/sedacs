#!/usr/bin/env python3
""" Load modules 

"""

import argparse
from sdc_parser import *
from sdc_system import *
from proxy_a import *
import time
try:
    from mpi4py import MPI
    mpiON = True
except ImportError as e:
    mpiON = False
from sdc_graph import *
from sdc_partition import *
from sdc_init import *
from sdc_graphSC import *

