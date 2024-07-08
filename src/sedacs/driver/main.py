"""Main sedacs prototype driver"""

import argparse
import time

from sedacs.graph import *
from sedacs.graph_partition import *
from sedacs.parser import *
from sedacs.proxy_a import *
from sedacs.system import System

try:
    from mpi4py import MPI

    mpi = True
except ImportError:
    mpi = False

parser = argparse.ArgumentParser(description="Test driver for sedacs")

parser.add_argument("--use-torch", help="Use pytorch", required=False, action="store_true")
parser.add_argument("--verbose", help="Verbose output", required=False, action="store_true")

args = parser.parse_args()
if args.use_torch:
    try:
        import torch as tc

        if tc.cuda.is_available():
            print("Using CUDA")
            args.device = tc.device("cuda")
        elif tc.backends.mps.is_available():
            print("Using MPS")
            args.device = tc.device("mps")
        else:
            print("Using CPU")
            args.device = tc.device("cpu")
        from sedacs.torch import *
    except ImportError as e:
        raise ImportError("Unable to import pytorch")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numranks = comm.Get_size()

# Initialize the code by reading the input file
sdc = Input("input.in", True)

# Read the coordinates
sy = System(1)
sy.latticeVectors, sy.symbols, sy.types, sy.coords = read_coords_file(sdc.coordsFileName, lib="None", verb=True)
sy.nats = len(sy.coords[:, 0])

tic = time.perf_counter()
if args.use_torch:
    nl = build_nlist_torch(
        sy.coords, sy.latticeVectors, 5.0, device=args.device, rank=rank, numranks=numranks, verb=args.verbose
    )
else:
    nl, nlTrX, nlTrY, nlTrZ = build_nlist(
        sy.coords, sy.latticeVectors, 5.0, rank=rank, numranks=numranks, verb=args.verbose
    )
comm.Barrier()
toc = time.perf_counter()
print("Time for build_nlist", toc - tic, "(s)")
if rank == 0:
    with open("neighborinfo.txt", "w") as of:
        for kk in range(sy.nats):
            nl_this = np.flip(np.sort(nl[kk, 1 : nl[kk, 0]]))
            print(
                "Neighs (x-coords) of {0} ({1})= ".format(kk, nl[kk, 0]), nl_this, "(", sy.coords[nl_this], ")", file=of
            )

# Get the neighbors of atom 1234
# subSy = system(nl[1234,0])
# subSy.symbols = sy.symbols
# subSy.coords,subSy.types = extract_subsystem(sy.coords,sy.types,sy.symbols,nl[1234,1:nl[1234,0]])
# if rank == 0:
#    write_pdb_coordinates("subSy.pdb",subSy.coords,subSy.types,subSy.symbols)
# sys.exit(0)


# Get initial graph (from a neighbor list)
graph = get_initial_graph(sy.coords, nl, sdc.rcut, sdc.maxDeg, True)
print_graph(graph)

# Partition the graph
parts = partition(graph, sdc.partitionType, sdc.nparts, True)

njumps = 1
partsCoreHalo = []
numCores = []

print("\nCore and halos indices for every part:")
for i in range(sdc.nparts):
    coreHalo, nc, nh = get_coreHaloIndices(parts[i], graph, njumps)
    partsCoreHalo.append(coreHalo)
    numCores.append(nc)
    print("coreHalo for part", i, "=", coreHalo)

## Every rank will do a subset of the list of coreHalos
# @todo We will need to "reshuffle" the list so that the work-load
# gets distributed.
partsPerRank = int(sdc.nparts / numranks)

partIndex1 = rank * partsPerRank
partIndex2 = (rank + 1) * partsPerRank
print(rank, numranks, partIndex1, partIndex2)
for partIndex in range(partIndex1, partIndex2):
    print("Rank, part", rank, partIndex)
    subSy = System(len(partsCoreHalo[partIndex]))
    subSy.symbols = sy.symbols
    subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])
    partFileName = "subSy" + str(rank) + "_" + str(partIndex) + ".pdb"
    write_pdb_coordinates(partFileName, subSy.coords, subSy.types, subSy.symbols)
    ham = get_hamiltonian(subSy.coords, atomTypes=np.zeros((1), dtype=int), verb=False)
    norbs = subSy.nats
    ham = get_hamiltonian(subSy.coords)
    occ = int(float(norbs) / 2.0)  # Get the total occupied orbitals
    rho = get_densityMatrix(ham, occ)
