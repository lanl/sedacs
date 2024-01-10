#!/usr/bin/env python3
""" Main sedacs prototype driver

"""

from sdc_loadMods import *

#Pass some command line variables
parser = argparse.ArgumentParser(description='Test driver for sedacs')
parser.add_argument("--use-torch",help="Use pytorch",required=False,action="store_true")
parser.add_argument("--input-file",help="Specify input file",required=False,type=str,default="input.in")

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

#Initialize sedacs 
sdc,comm,rank,numranks,sy,hindex,graphNL,nl = init(args)

#Perform a graph-adaptive calculation of the density matrix
get_adaptiveDM(sdc,comm,rank,numranks,sy,hindex,graphNL)



