"""Initialize sedac driver"""

from sedacs.driver import *
from sedacs.engine import *
from sedacs.system import System


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
        if tcAvail:
            if tc.cuda.is_available():
                print("Using CUDA")
                args.device = tc.device("cuda")
            elif tc.backends.mps.is_available():
                print("Using MPS")
                args.device = tc.device("mps")
            else:
                args.device = tc.device("cpu")
        else:
            print("Pytorch is not available")
            exit(0)
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
    if mpiON:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        numranks = comm.Get_size()
    else:
        comm = None
        rank = 0
        numranks = 1

    # Initialize the code by reading the input file
    sdc = sdc_input(args.input_file, True)

    # Initialize the engine (quantum chemistry code)
    eng = engine(rank)
    eng.name = sdc.engine["Name"]
    eng.path = sdc.engine["Path"]
    eng.run = sdc.engine["Executable"]
    eng.interface = sdc.engine["InterfaceType"]

    # Read the coordinates
    sy = System(1)
    sy.latticeVectors, sy.symbols, sy.types, sy.coords = read_coords_file(sdc.coordsFileName, lib="None", verb=True)
    sy.nats = len(sy.coords[:, 0])
    sy.vels = np.zeros((sy.nats, 3))

    # Get hindex, the orbital index for each atom in the system
    sy.norbs, hindex = get_hindex(sdc.orbs, sy.symbols, sy.types)

    tic = time.perf_counter()
    if args.use_torch:
        nl = build_nlist_torch(sy.coords, sy.latticeVectors, sdc.rcut, rank=rank, numranks=numranks, verb=False)
    else:
        nl, nlTrX, nlTrY, nlTrZ = build_nlist(
            sy.coords, sy.latticeVectors, sdc.rcut, api="old", rank=rank, numranks=numranks, verb=False
        )
        # nl,nlTrX,nlTrY,nlTrZ = build_nlist_integer(sy.coords,sy.latticeVectors,sdc.rcut,rank=rank,numranks=numranks,verb=False)
    if mpiON:
        comm.Barrier()

    toc = time.perf_counter()
    print("Time for build_nlist", toc - tic, "(s)")
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

    # Get the neighbors of atom 1234
    AtToPrint = 0
    subSy = System(nl[AtToPrint, 0])
    subSy.symbols = sy.symbols
    subSy.coords, subSy.types = extract_subsystem(
        sy.coords, sy.types, sy.symbols, nl[AtToPrint, 1 : nl[AtToPrint, 0] + 1]
    )  # $$$ needs +1
    if rank == 0:
        write_xyz_coordinates("subSyNL.xyz", subSy.coords, subSy.types, subSy.symbols)

    # Get initial graph (from a neighbor list)
    graphNL = get_initial_graph(sy.coords, nl, sdc.rcut, sdc.maxDeg, True)

    fullGraph = np.zeros((sy.nats, sdc.maxDeg + 1), dtype=int)
    fullGraph[:, :] = graphNL[:, :]

    return sdc, eng, comm, rank, numranks, sy, hindex, fullGraph, nl, nlTrX, nlTrY, nlTrZ
