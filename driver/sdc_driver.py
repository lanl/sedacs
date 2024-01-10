#!/usr/bin/env python3
""" Main sedacs prototype driver

"""

import argparse
from sdc_parser import *
from sdc_system import *
from proxy_a import *
import time
try:
    from mpi4py import MPI
    mpi = True
except ImportError as e:
    mpi = False
from sdc_graph import *
from sdc_partition import *
from sdc_init import *
from sdc_graphSC import *

parser = argparse.ArgumentParser(description='Test driver for sedacs')

parser.add_argument("--use-torch",help="Use pytorch",required=False,action="store_true")

args=parser.parse_args()
if args.use_torch:
    try:
        import torch as tc
        if tc.cuda.is_available():
            print("Using CUDA")
            args.device = tc.device('cuda')
        elif tc.backends.mps.is_available():
            print("Using MPS")
            args.device = tc.device('mps')
        else:
            args.device = tc.device('cpu')
        from sdc_torch import *
    except ImportError as e:
        raise ImportError("Unable to import pytorch")

sdc,comm,rank,numranks,sy,hindex,fullGraph,nl = init(args)
get_adaptiveDM(sdc,comm,rank,numranks,sy,hindex,fullGraph)



