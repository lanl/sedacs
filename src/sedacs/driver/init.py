"""Initialize sedac driver"""

import argparse
import time

import numpy as np

from sedacs.engine import Engine
from sedacs.file_io import read_coords_file, write_xyz_coordinates
from sedacs.graph import get_initial_graph
from sedacs.parser import Input
from sedacs.interface_modules import get_ppot_energy_expo, init_proxy
from sedacs.system import System, build_nlist, extract_subsystem, get_hindex, build_nlist_small
from sedacs.overlap import get_overlap
from sedacs.graph import collect_graph_from_rho

#from seqm.seqm_functions.pack import pack
import sedacs.globals 
import os


MPI = None
try:
    from mpi4py import MPI

    is_mpi_available = True
except ImportError:
    is_mpi_available = False
#torch = None
import torch
try:

    from sedacs.torch import build_nlist_torch

    is_torch_available = True
except ImportError:
    is_torch_available = False

is_torch_available = False


__all__ = ["available_device", "init"]


def available_device():
    if is_torch_available:
        if torch.cuda.is_available():
            print("Using CUDA")
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            print("Using MPS")
            return torch.device("mps")

        print("Using CPU")
        return torch.device("cpu")

    raise Exception("Pytorch is not available!")


## Getting arguments
# @brief This will get some arguments from command line. WARNING!!! This makes the code depending
# on the argparse library...
# @return args The argparse object (https://docs.python.org/3/library/argparse.html)
#
def get_args():
    parser = argparse.ArgumentParser(description="Test driver for sedacs")
    parser.add_argument("--use-torch", help="Use pytorch", required=False, action="store_true")
    parser.add_argument("--input-file", help="Specify input file", required=False, type=str, default="input.in")

    args = parser.parse_args()
    if args.use_torch:
        args.device = available_device()

    return args


## Initialize the driver
# @brief This will initialize all the input variables needed by the driver
# @param args The argparse object (https://docs.python.org/3/library/argparse.html)
# @return sdc SEDACS input variables. Example: sdc.threshold : Threshold vlue for the matrices.
# These variables are read from the input file.
# @return comm MPI communicator
# @return rank Current rank (= 0 if MPI is off)
# @return numranks Number of ranks (= 1 if MPI is off)
# @return sy System object (see `/mods/sdc_system.py`)
# @return hindex hindex Orbital index for each atom in the system
# @return fullGraph Initial atomic connectivity graph
# @return nl Neighbor list `nl[i,0]` = total number of neighbors.
# `nl[i,1:nl[i,0]]` = neigbors of i. Self neighbor i to i is not included explicitly.
#
def init(args):
    if is_mpi_available:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        numranks = comm.Get_size()
    else:
        comm = None
        rank = 0
        numranks = 1
    if rank == 0:    
        # Initialize the code by reading the input file
        sdc = Input(args.input_file, verb=True)

        # Initialize the engine (quantum chemistry code)
        eng = Engine(rank)
        eng.name = sdc.engine["Name"]
        eng.path = sdc.engine["Path"]
        eng.run = sdc.engine["Executable"]
        eng.interface = sdc.engine["InterfaceType"]

        eng.method = sdc.engine.get("RhoSolverMethod", None)
        eng.accel = sdc.engine.get("Accelerator", None)

        # Read the coordinates
        sy = System(1)
        sy.latticeVectors, sy.symbols, sy.types, sy.coords = read_coords_file(sdc.coordsFileName, lib="None", verb=True)
        sy.nats = len(sy.coords[:, 0])
        sy.vels = np.zeros((sy.nats, 3))
        sy.hubbard_u = np.zeros(sy.nats)
        sy.charges = np.zeros(sy.nats)

        # Get hindex (the orbital index for each atom in the system)
        sy.norbs, sy.orbs, hindex, sy.numel, sy.znuc = get_hindex(sdc.orbs, sy.symbols, sy.types)
        if eng.interface == "PySEQM": sy.numel = int(sy.numel/2)
        sy.numel -= sdc.charge
        if sdc.UHF:
            sy.nocc_alpha = sy.numel/2. + (sdc.mult-1)/2.
            sy.nocc_beta  = sy.numel/2. - (sdc.mult-1)/2.
            if ((sy.nocc_alpha%1 != 0) or (sy.nocc_beta%1 != 0)):
                raise ValueError("Invalid charge/multiplicity combination!")
            else:
                sy.nocc_alpha = np.int64(sy.nocc_alpha)
                sy.nocc_beta  = np.int64(sy.nocc_beta)
            sy.nocc = np.array([sy.nocc_alpha,sy.nocc_beta], dtype=np.int64)
        else:
            sy.nocc = sy.numel/2
            if (sy.nocc%1 != 0):
                raise ValueError("Odd number of electron in a closed shell!")
            else:
                sy.nocc = np.int64(sy.nocc)

    else:
        sdc = None
        eng = None
        sy = None
        hindex = None
    sdc = comm.bcast(sdc, root=0)    
    eng = comm.bcast(eng, root=0)
    sy = comm.bcast(sy, root=0)
    hindex = comm.bcast(hindex, root=0)

    tic = time.perf_counter()
    if(sy.nats > 100): 
        if args.use_torch:
            nl = build_nlist_torch(sy.coords, sy.latticeVectors, sdc.rcut, rank=rank, numranks=numranks, verb=False)
        else:
            nl, nlTrX, nlTrY, nlTrZ = build_nlist(
                sy.coords, sy.latticeVectors, sdc.rcut, api="old", rank=rank, numranks=numranks, verb=False
            )
            # nl,nlTrX,nlTrY,nlTrZ = build_nlist_integer(sy.coords,sy.latticeVectors,sdc.rcut,rank=rank,numranks=numranks,verb=False)
        if is_mpi_available:
            comm.Barrier()

        toc = time.perf_counter()
        if rank == 0: print("Time for build_nlist", toc - tic, "(s)")
        if rank == 0:
            with open("neighborinfo.txt", "w") as of:
                for kk in range(sy.nats):
                    print(
                        "Neighs (x-coords) of {} = ".format(kk),
                        nl[kk, 1 : nl[kk, 0]],
                        "(",
                        sy.coords[nl[kk, 1 : nl[kk, 0]], 0],
                        ")",
                        file=of,
                    )
    else:
        nl, nlTrX, nlTrY, nlTrZ = build_nlist_small(
                sy.coords, sy.latticeVectors, sdc.rcut, rank=rank, numranks=numranks, verb=False
            )

    # Get initial graph (from a neighbor list)
    if rank == 0:
        print('!!!!')
        if sdc.InitGraphType == "OverlapM":
            print('Creating overlap matrix for initial graph.')
            tic = time.perf_counter()
            
            sdc.overlap_whole = get_overlap(eng, sy.coords, sy.symbols, sy.types, hindex)
            #torch.save(sdc.overlap_whole, 'overlap_whole.pt')
            #sdc.overlap_whole = torch.load('overlap_whole.pt')
            
            print("Time to get overlap", time.perf_counter() - tic,"(s)")
            graphOnRank = None
            print('Creating initial graph.')
            tic = time.perf_counter()
            graphNL = collect_graph_from_rho(graphOnRank, sdc.overlap_whole, sdc.gthreshinit, sy.nats, sdc.maxDeg, [i for i in range(0,sy.nats)],hindex)
            print("Time to compute graph", time.perf_counter() - tic,"(s)")
            #del sdc.overlap_whole

        else:
            graphNL = get_initial_graph(sy.coords, nl, sdc.rcut, sdc.maxDeg, True)
    else:
        graphNL = None
    
    #comm.Barrier()
    graphNL = comm.bcast(graphNL, root=0)
    fullGraph = np.zeros((sy.nats, sdc.maxDeg + 1), dtype=int)
    fullGraph[:, :] = graphNL[:, :]

    if "Proxy" in eng.name:
        #Initialize proxy/guest code
        init_proxy(sy.symbols,sy.orbs)
    eng.up = True

    return sdc, eng, comm, rank, numranks, sy, hindex, fullGraph, nl, nlTrX, nlTrY, nlTrZ
