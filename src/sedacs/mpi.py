"""Some mpi wrappers using MPI4PY"""

import sys

import numpy as np

try:
    from mpi4py import MPI

    mpiLib = True
except ImportError as e:
    mpiLib = False

## Send and receive
# @brief Very simple send and receive mpi wrapper for integer numpy 2D arrays
# @param dataSend Date (2D numpy int array) to be sent
# @param fromRank The rank num from which data will be sent
# @param toRank The rank num from which data will be received
# @param rank The number for current mpi execution rank
# @param comm MPI communicator
# @return dataRecv Received data
#
def send_and_receive(dataSend, fromRank, toRank, rank, comm):
    dataRecv = None
    if rank == fromRank:
        comm.Send(dataSend, dest=toRank, tag=0)

    elif rank == toRank:
        dataRecv = np.empty((len(dataSend[:, 0]), len(dataSend[0, :])), dtype=int)
        comm.Recv(dataRecv, source=fromRank, tag=0)
    return dataRecv


## Collect matrix from chunks
# @brief This will collect a full matrix from severa "regular" chunks.
# @param chunk Matrix chunk to be collected
# @param nDim Dimension of the full matrix
# @param rowsPerChunk Number of rows in the chunk.Typically `rowsPerChunk = int(nDim/numranks)`.
# It can happen that the last chunk contains more than rowsPerChunk. Basically
# the last chunk will be used to adjust in case `nDim` is not divisible by numranks. In this case:
# the last chunk (corresponding to the last rank) will have: `nDim - rowsPerChunk*(numrank-1)`
# numbers of rows.
# @param rank The number for current mpi execution rank
# @param numranks Number of execution ranks
# @param comm MPI communicator
#
def collect_matrix_from_chunks(chunk, nDim, rowsPerChunk, rank, numranks, comm):
    if not mpiLib:
        raise ImportError("ERROR: Consider installing mpi4py")

    if nDim < rowsPerChunk:
        raise ValueError(f"ERROR: nDim should be larger than rowsPerChunk: {nDim, rowsPerChunk}")

    maxChunkDim = nDim - (numranks - 1) * rowsPerChunk

    mDim = len(chunk[0, :])
    chunkDim = len(chunk[:, 0])
    fullMat = np.empty([nDim, mDim], dtype=int)

    # Prepare buffers
    dataSend = np.empty((maxChunkDim, mDim), dtype=int)
    dataRecv = np.empty((maxChunkDim, mDim), dtype=int)
    dataSend[0:chunkDim, :] = chunk[0:chunkDim, :]

    # Do the "all gathers"
    for i in range(numranks):
        for j in range(numranks):
            if i != j:
                dataRecv = send_and_receive(dataSend, i, j, rank, comm)
            else:
                if i == rank:  # If i = j and rank = i rec = send
                    dataRecv = dataSend
                else:
                    dataRecv = None
            if isinstance(dataRecv, type(None)):
                pass
            else:
                if i < numranks - 1:
                    fullMat[i * rowsPerChunk : (i + 1) * rowsPerChunk, :] = dataRecv[0:rowsPerChunk, :]
                else:
                    fullMat[(numranks - 1) * rowsPerChunk : nDim, :] = dataRecv[
                        0 : nDim - (numranks - 1) * rowsPerChunk, :
                    ]
    return fullMat


def collect_and_sum_matrices(matOnRank, rank, numranks, comm):
    if not mpiLib:
        raise ImportError("ERROR: Consider installing mpi4py")

    nDim = len(matOnRank[:, 0])
    mDim = len(matOnRank[0, :])
    fullMat = np.zeros([nDim, mDim], dtype=int)

    # Prepare buffers
    dataSend = np.empty((nDim, mDim), dtype=int)
    dataRecv = np.empty((nDim, mDim), dtype=int)
    dataSend[:, :] = matOnRank[:, :]

    # Do the "all gathers" and sum
    for i in range(numranks):
        for j in range(numranks):
            if i != j:
                dataRecv = send_and_receive(dataSend, i, j, rank, comm)
            else:
                if i == rank:  # If i = j and rank = i rec = send
                    dataRecv = dataSend
                else:
                    dataRecv = None
            if isinstance(dataRecv, type(None)):
                pass
            else:
                fullMat[:, :] = dataRecv[:, :] + fullMat[:, :]

    return fullMat

def collect_and_sum_matrices_float(matOnRank, comm):
    if comm is None:
        raise ImportError("ERROR: Consider installing mpi4py and initializing MPI")

    # Initialize buffer for the result
    fullMat = np.zeros_like(matOnRank)

    # Perform element-wise sum across all ranks
    comm.Allreduce(matOnRank, fullMat, op=MPI.SUM)

    return fullMat 

def collect_and_sum_vectors_float(vectOnRank, rank, numranks, comm):

    if not mpiLib:

        raise ImportError("ERROR: Consider installing mpi4py")

    nDim = len(vectOnRank)

    fullVect = np.zeros(nDim, dtype=float)

    comm.Allreduce(vectOnRank,fullVect,op=MPI.SUM)

    return fullVect


def collect_and_concatenate_vectors(vectOnRank, comm):

    if not MPI.Is_initialized():

        raise ImportError("ERROR: Consider installing mpi4py")

    # Gather the sizes of each vectOnRank
    local_size = len(vectOnRank)
    sizes = comm.allgather(local_size)

    # Calculate the displacements for each rank
    displacements = np.cumsum([0] + sizes[:-1])

    # Create the full vector with the total size
    total_size = sum(sizes)
    fullVect = np.zeros(total_size, dtype=float)

    # Gather all vectors into fullVect
    comm.Allgatherv(vectOnRank, [fullVect, sizes, displacements, MPI.DOUBLE])

    return fullVect


def collect_matrix_from_chunks_v1(chunk, nDim, rowsPerChunk, rank, numranks, comm):
    if not mpiLib:
        raise ImportError("\nERROR: Consider installing mpi4py")

    if nDim < rowsPerChunk:
        raise ValueError(f"ERROR: nDim should be larger than rowsPerChunk: {nDim, rowsPerChunk}")

    maxChunkDim = nDim - (numranks - 1) * rowsPerChunk

    mDim = len(chunk[0, :])
    chunkDim = len(chunk[:, 0])
    fullMat = np.empty([nDim, mDim], dtype=int)

    comm.Barrier()

    displacements = []
    for i in range(numranks):
        displacements.append(i * rowsPerChunk * mDim)

    print(displacements)

    comm.Allgatherv(chunk, [fullMat, nDim * mDim, displacements, MPI.INT])

    return fullMat

if __name__ == "__main__":
    # Small test code
    n = len(sys.argv)
    if n == 1:
        print("Give the name of the function to be tested. Example: collect_matrix_from_chunks\n")
        sys.exit(0)
    else:
        test = str(sys.argv[1])

    if test == "collect_matrix_from_chunks":
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        numranks = comm.Get_size()

        row = [1, 2, 3, 4]
        chunk = np.zeros((5, 4), dtype=int)
        chunk[:, :] = row
        chunk[:, :] = chunk[:, :] + rank
        nDim = numranks * len(chunk[:, 0])
        rowsPerChunk = 5
        maxChunkDim = rowsPerChunk

        fullMat = collect_matrix_from_chunks(chunk, nDim, rowsPerChunk, rank, numranks, comm)

        if rank == 0:
            print(fullMat)

    if test == "collect_and_sum_matrices":
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        numranks = comm.Get_size()
        row = [0, 0, 0, 0]
        mat = np.zeros((5, 4), dtype=int)
        mat[:, :] = row
        mat[:, :] = mat[:, :] + rank
        print("Rank, mat", rank, mat)
        fullMat = collect_and_sum_matrices(mat, rank, numranks, comm)
        print("Full mat", fullMat)
