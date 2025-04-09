# Pytorch kernels

import sys
import time

MPI = None
try:
    from mpi4py import MPI
    is_mpi_available = True
except ImportError:
    is_mpi_available = False

import numpy as np
import torch
import torch.nn.functional as tf

from sedacs.system import collect_matrix_from_chunks, get_volBox
from typing import Union

## Neighbor list
# @brief It will bild a neighbor list using an "all to all" approach
# @param coords System coordinates. coords[7,1]: y-coordinate of atom 7.
# @param latticeVectors. Lattice vectors of the system box. latticeVectors[1,2]: z-coordinate of vector 1.
# @param nl neighbor list type: a simple 2D array indicating the neighbors of each atom.
# @param rank MPI rank
#
def build_nlist_torch(coords: np.ndarray,
                      latticeVectors: np.ndarray,
                      rcut: float,
                      device=torch.device("cpu"),
                      rank: int = 0,
                      numranks: int = 1,
                      verb: bool = False,
                      comm=None):
    
    """

    Builds a neighborlist as atorch tensor in an all-to-all fashion. Units between
    coords, latticeVectors, and rcut are not enforced to a particular convention,
    but they must be consistent with one another.

    Parameters
    ----------
    coords: np.ndarray (Natoms, 3)
        Cartesian coordinates for the atoms in the system.
    latticeVectors: np.ndarray (3, 3)
        Cell vectors. This only works for orthorhombic boxes (but still
        requires cells in full format).
    rcut: float
        Radial cutoff for the neighborlist.
    verb: bool
        Controls the verbosity of the neighblorlist generation routine.
    comm: None or an MPI communicator.
        MPI communicator.
        TODO handle typing better for this scenario where the user doesn't have a
        parallelized Python interpreter.

    Returns
    -------
    nl: torch.Tensor (Natoms, _)

    
    """

    if verb:
        print("Building neighbor list ...")

    nats = len(coords[:, 0])
    if numranks > 1:
        print("NUM RNAKS", numranks)
        comm = MPI.COMM_WORLD
    natsPerRank = int(nats / numranks)
    if rank == numranks - 1:
        natsInRank = nats - natsPerRank * (numranks - 1)
    else:
        natsInRank = natsPerRank

    #    nats_left = nats % numranks
    #    if (rank < nats_left):
    #        natsInRank = natsPerRank + 1
    #    else:
    #        natsInRank = natsPerRank

    # natsInBuff = max(natsInRank,nats - natsPerRank*(numranks - 1))

    # We will have approximatly [(4/3)*pi * rcut^3 * atomic density] number of neighbors.
    # A very large atomic density could be 1 atom per (1.0 Ang)^3 = 1 atoms per Ang^3
    volBox = get_volBox(latticeVectors, verb=False)
    density = 1.0
    maxneigh = int(3.14592 * (4.0 / 3.0) * density * rcut**3)
    boxSize = rcut

    # We assume the box is orthogonal
    nx = int(latticeVectors[0, 0] / boxSize)
    ny = int(latticeVectors[1, 1] / boxSize)
    nz = int(latticeVectors[2, 2] / boxSize)
    nBox = nx * ny * nz
    maxInBox = int(density * (boxSize) ** 3)  # Upper bound for the max number of atoms per box
    inbox = np.zeros((nBox, maxInBox), dtype=int)
    inbox[:, :] = -1
    totPerBox = np.zeros((nBox), dtype=int)
    totPerBox[:] = -1
    boxOfI = np.zeros((nats), dtype=int)
    xBox = np.zeros((nBox), dtype=int)
    yBox = np.zeros((nBox), dtype=int)
    zBox = np.zeros((nBox), dtype=int)
    ithFromXYZ = np.zeros((nx, ny, nz), dtype=int)
    neighbox = np.zeros((nBox, 27), dtype=int)

    minx = np.min(coords[:, 0])
    miny = np.min(coords[:, 1])
    minz = np.min(coords[:, 2])

    smallReal = 0.0
    # Search for the box coordinate and index of every atom

    for i in range(nats):
        # Index every atom respect to the discretized position on the simulation box.
        ix = int(coords[i, 0] / boxSize) % nx  # small box x-index of atom i
        iy = int(coords[i, 1] / boxSize) % ny  # small box x-index of atom i
        iz = int(coords[i, 2] / boxSize) % nz  # small box x-index of atom i

        ith = ix + iy * nx + iz * nx * ny  # Get small box index
        boxOfI[i] = ith

        # From index to box coordinates
        xBox[ith] = ix
        yBox[ith] = iy
        zBox[ith] = iz

        # From box coordinates to index
        ithFromXYZ[ix, iy, iz] = ith

        totPerBox[ith] = totPerBox[ith] + 1  # How many per box
        if totPerBox[ith] >= maxInBox:
            raise ValueError(f"Exceeding the max in box allowed: {totPerBox[ith]} >= { maxInBox}.")
        inbox[ith, totPerBox[ith]] = i  # Who is in box ith

    for i in range(nBox):  # Correcting - from indexing to
        totPerBox[i] = totPerBox[i] + 1

    # For each box get a flat list of neighboring boxes (including self)
    for i in range(nBox):
        neighbox[i, 0] = i
        j = 1
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    if not (ix == 0 and iy == 0 and iz == 0):
                        # Get neigh box coordinate
                        neighx = xBox[i] + ix
                        neighy = yBox[i] + iy
                        neighz = zBox[i] + iz
                        jxBox = neighx % nx
                        jyBox = neighy % ny
                        jzBox = neighz % nz

                        # Get the neigh box index
                        neighbox[i, j] = ithFromXYZ[jxBox, jyBox, jzBox]
                        j += 1
    # Move arrays to device
    tic = time.perf_counter()
    neighbox_d = torch.tensor(neighbox, device=device)
    inbox_d = torch.tensor(inbox, device=device)
    boxOfI_d = torch.tensor(boxOfI, device=device)
    latticeVectors_d = torch.tensor(latticeVectors.astype(np.float32), device=device)
    coords_d = torch.tensor(coords.astype(np.float32), device=device)
    t_copy = time.perf_counter() - tic
    if rank == 0 and verb:
        print("Time for copying arrays to device = ", t_copy, " sec")

    def get_neighs_of(i, coords, neighbox, boxOfI, inbox, latticeVectors):
        # print("atom",i)
        cnt = -1
        # Get the list of all atoms in neighboring boxes
        boxneighs = inbox[neighbox[boxOfI[i]]]
        # Shorten the long dimension for speedup on CPU
        max_nonzero_elems = torch.max(torch.count_nonzero(boxneighs != -1, axis=1))
        boxneighs = boxneighs[:, 0:max_nonzero_elems].flatten()
        # Filter and flatten the list
        # boxneighs = boxneighs[np.where(boxneighs != -1)]
        # Calculate the distances to all atoms in neighboring boxes
        dvec = torch.zeros((len(boxneighs), 3), dtype=torch.float64)
        #        dvec = dvec + coords[None,None,i]
        orthovec = torch.diagonal(latticeVectors)
        for k in range(3):
            dvec[:, k] = (coords[i, k] - coords[boxneighs, k] + orthovec[k] / 2.0) % orthovec[k] - orthovec[k] / 2.0
        distance = torch.linalg.norm(dvec, axis=1)
        # Filter the list according to the threshold
        nlVect = boxneighs[torch.where(distance < rcut)]
        # nlVect = boxneighs[np.where(np.logical_and(distance < rcut,distance > 1.0E-12))]
        nlVect = nlVect[nlVect != -1]
        nlVect = nlVect[nlVect != i]

        # cnt bug?
        print(nlVect.shape)
        cnt = len(nlVect[i])


        # Format and pad the list
        nlVect = tf.pad(nlVect, (1, maxneigh - cnt - 1), "constant", value=0)
        nlVect[0] = cnt
        nlVect = nlVect.cpu().numpy()
        return nlVect

    def get_neighs_of_range(i0, i1, coords, neighbox, boxOfI, inbox, latticeVectors):
        # print("atom",i)
        cnt = -1
        nats_this = i1 - i0 + 1

        # Get the list of all atoms in neighboring boxes
        tic = time.perf_counter()
        boxneighs = inbox[neighbox[boxOfI[i0 : i1 + 1]]]
        t_boxneighs = time.perf_counter() - tic

        # Shorten the long dimension for speedup on CPU
        max_nonzero_elems = torch.max(torch.count_nonzero(boxneighs != -1, axis=1))
        boxneighs = boxneighs[:, 0:max_nonzero_elems]
        orthovec = torch.diagonal(latticeVectors)

        # Build initial distance vector array from repeating coords rows
        tic = time.perf_counter()
        repeats = torch.tensor([boxneighs.shape[1] * boxneighs.shape[2]], device=device).repeat(nats_this)
        dvec = torch.repeat_interleave(coords[i0 : i1 + 1], repeats, axis=0)
        t_repeat_coords = time.perf_counter() - tic

        # Reshape boxneighs for vectorized distance calc
        boxneighs = boxneighs.reshape(boxneighs.shape[0] * boxneighs.shape[1] * boxneighs.shape[2])

        # Perform distance calc
        tic = time.perf_counter()
        for k in range(3):
            dvec[:, k] = (dvec[:, k] - coords[boxneighs, k] + orthovec[k] / 2.0) % orthovec[k] - orthovec[k] / 2.0
        distance = torch.linalg.norm(dvec, axis=1)
        t_distance = time.perf_counter() - tic

        # Reshape arrays to form neighbor list
        boxneighs = boxneighs.reshape(nats_this, int(len(boxneighs) / nats_this))
        distance = distance.reshape(nats_this, int(len(distance) / nats_this))

        # Build the neighbor list using a distance threshold mask
        tic = time.perf_counter()
        nlMask = distance < rcut
        nlVect = torch.where(nlMask, boxneighs, -1)
        nlVect, indices = torch.sort(nlVect, axis=1, descending=True)
        nlVect = tf.pad(nlVect, (1, maxneigh - nlVect.shape[1] - 1), "constant", value=0)

        # Just sum over the zero components in the +1 shifted tensor.
        nlVect[:, 0] = torch.count_nonzero(nlVect + 1, axis=1)
        # Leaving the old code in case there is some unintended change here
        # nlVect[:, 0] = torch.count_nonzero(nlMask, axis=1)

        t_build_nlvect = time.perf_counter() - tic

        # Copy the neighbor list back to the host
        tic = time.perf_counter()
        nlVect = nlVect.cpu()
        t_copy_nlvect = time.perf_counter() - tic

        # Convert to a numpy array
        tic = time.perf_counter()
        nlVect = nlVect.numpy()
        t_numpy = time.perf_counter() - tic

        if rank == 0 and verb:
            print("Time for building boxneighs = ", t_boxneighs, " sec")
            print("Time for repeating coords = ", t_repeat_coords, " sec")
            print("Time for distance calculation = ", t_distance, " sec")
            print("Time for building neighbor list vectors = ", t_build_nlvect, " sec")
            print("Time for copying nlVect to host = ", t_copy_nlvect, " sec")
            print("Time for converting nlVect to numpy = ", t_numpy, " sec")

        return nlVect

    #    nlChunk = np.empty([natsInRank,maxneigh],dtype=int)

    #   firstIdx = natsPerRank*(rank+1)
    #   for i in range(rank-1):
    #       if (i >= nats_left):
    #           firstIdx -= 1

    nlChunk = get_neighs_of_range(
        natsPerRank * rank,
        natsPerRank * rank + natsInRank - 1,
        coords_d,
        neighbox_d,
        boxOfI_d,
        inbox_d,
        latticeVectors_d,
    )

    # Gather the neighbor list
    nl = np.empty([nats, maxneigh], dtype=int)
    if is_mpi_available and comm is not None:
        tic = time.perf_counter()
        nl = collect_matrix_from_chunks(nlChunk, nats, natsPerRank, rank, numranks, comm)
        t_gather_nl = time.perf_counter() - tic
        if rank == 0 and verb:
            print("Time for gathering nl = ", t_gather_nl, " sec")
    else:
        nl = nlChunk

    # comm.Allgather(nlChunk,nl)
    # comm.Allgather(nlTrChunkX,nlTrX)
    # comm.Allgather(nlTrChunkY,nlTrY)
    # comm.Allgather(nlTrChunkZ,nlTrZ)

    return nl

