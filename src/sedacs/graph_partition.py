from sedacs.geometry import get_mic_distances, get_contact_map
from sedacs.graph import *
from sedacs.file_io import (
        read_xyz_file,
        write_xyz_coordinates,
        write_pdb_coordinates
)
#import matplotlib.pyplot as plt
#

import time

try:
    from sklearn.cluster import SpectralClustering
    SpectralClusteringLib = True 
except: SpectralClusteringLib = False

import os
from sedacs.periodic_table import PeriodicTable

try: 
    import metis
    metisLib = True
except: 
    metisLib = False

try:
    import networkx as nx
    nxLib = True
except ImportError as e:
    nxLib = False

try:
    import ase.io

    aseLib = True
except:
    aseLib = False

try:
    import torch
    torchLib = True
except:
    torchLib = False

import numpy as np
from warnings import warn

try:
    from numba import jit
except ImportError:
    # Dummy decorator if no numba.
    def jit(*args, **kwargs):
        return lambda f: f
    warn("Numba not installed, graph partitioning will be slow for systems above ~5k atoms")


##
# @brief
# @param
# @return
# FIXME need to make the jit depend on numbaFlag. Couldn't figure out how to 
# get that to work with the numba decorators.

DEBUG_LEVEL = 0  # > 0 gives assert statements, > 1 gives print statements.

## Partition
# @brief This will partition a graph based on a defined method
# @param graph Graph to be partition
# @param partitionType Method or type of partition to be used
# @param nparts Number of total parts
# @param verb Verbosity level
# @return parts Partition containing a "list of parts" where every
# part is a list of nodes
def graph_partition(sdc, eng, graph, partitionType, nparts, coords, verb=False):
    if partitionType == "Regular":
        parts = regular_partition(graph, nparts, verb)
    elif partitionType == "Metis":
        parts = metis_partition(graph, nparts, verb)
    elif partitionType == "MinCut":
        parts = mincut_partition(graph, nparts, verb)
    elif(partitionType == "SpectralClustering"):
        parts = spectral_clustering_partition(sdc, graph, nparts, coords, do_xyz=True)

    if(eng.interface == "PySEQM"):
        for i in range(len(parts)):
            parts[i] = sorted(parts[i])

    return parts


## get_cut
# @brief Gets the cut from the node partitions and graph.
# @param nodeIPartition (np.ndarray (n_nodes)) Partition node I belongs to.
# @param graph (np.ndarray (n_nodes,n_nodes)) Dense adjacency matrix.
# @return cut (int): The cut for the partitioning scheme.
# @jit(nopython=True)
def get_cut(nodeIPartition, graph):
    cut = 0
    n_nodes = graph.shape[0]
    for i in range(n_nodes):
        IK = nodeIPartition[i]

        # Look at the neighbors to see if they are in different part
        for j in range(1, graph[i, 0] + 1): # don't get this part

           
            # TODO: How is this anything but 1, or 0 if 
            # input graph is to be an adjacency matrix?
            index = graph[i, j]


            JK = nodeIPartition[index]
            if IK != JK:
                cut = cut + 1
    return cut

## get_balancing
# @brief Compute balance from a set of partition sizes.
# @param partitionKNumNodes (np.ndarray (number of partitions)): How many
#        node are in the Kth partiiton.
# @return balance (float): Balance as defined by largest/smallest part sizes.
def get_balance_from_partition_sizes(partitionKNumNodes):
    balance = np.max(partitionKNumNodes)/np.min(partitionKNumNodes)
    return balance


## get_balance_from_partitions
# @brief Compute balance from a set of partition sizes.
# @param parts (list<int>): Unpadded least of each partition's nodes.
# @return balance (float): Balance as defined by largest/smallest part sizes.
def get_balance_from_partitions(parts):
    lens = [len(part) for part in parts]
    maxPartSize = max(lens)
    minPartSize = min(lens)
    return maxPartSize/minPartSize

## get_balance
# @brief Compute balance from the array containing node i's current partition.
# @param nodeIPartitions (np.ndarray (n_nodes)): Each node's partition. E.g.
#        nodeParititons[i] returns the partition of node i.
# @return balance (float): Balance as defined by largest/smallest part sizes.
def get_balance(nodeIPartition, k):
    partsSizes = np.zeros((k), dtype=int)
    for i in range(len(nodeIPartition)):
        partsSizes[nodeIPartition[i]] = partsSizes[nodeIPartition[i]] + 1
    bal = np.max(partsSizes) / np.min(partsSizes)
    return bal


## initial_partition_from_coordinates
# @brief Computes the initial partitioning of a set of nodes using positional 
# info.
# @param systemFilename (str): System file name to parse for structural info.
# @param k (int): Number of partitions in the system.
# @param domainDecomp (List<int>): Number of domains to decomp along x, y, z.
#                                   nx*ny*nz must be divisor of k.
# @param
# @return
def _initial_partition_from_coordinates(
    systemFilename,
    k,
    domainDecomp,
    cutoff=1.8,
    device="cpu",
):

    assert (
        len(domainDecomp) == 3
        and domainDecomp[0] * domainDecomp[1] * domainDecomp[2] == k
    ), "Domain decomp must be [ngx, ngy, ngz] where all three are ints and ngx*ngy*ngz is a divisor of k"

    ngx, ngy, ngz = domainDecomp[0], domainDecomp[1], domainDecomp[2]

    # Extract info needed for MIC distance.
    # @return latticeVectors Lattice vectors. z-coordinate of the first
    #         vector = latticeVectors[0,2]
    # @return symbols Symbol for each atom type. Symbol for first atom
    #         type = symbols[0]
    # @return types Index type for each atom in the system. Type for first atom
    #         = type[0]
    # @return coords Position for every atoms. z-coordinate of
    #         atom 1 = coords[0,2]

    lat, syms, typs, coords = read_xyz_file(systemFilename, verb=False)
    if torchLib:
        R = torch.tensor(coords, device=device, dtype=torch.float32)
        cell = torch.tensor(lat, device=device, dtype=torch.float32)
    else:
        R = np.array(coords)
        cell = np.array(lat)

    n_atoms = R.shape[0]

    # Call functions in sedacs.geometry.py
    R_mic = get_mic_distances(R, cell, torchLib=torchLib)
    Gp = get_contact_map(R_mic, cutoff=cutoff, torchLib=torchLib)

    if torchLib:
        if device == "cpu":
            Gp = Gp.numpy()
            RPartition = R.numpy()
            cell = cell.numpy()
        else:
            Gp = Gp.cpu().numpy()
            RPartition = R.cpu().numpy()
            cell = cell.cpu().numpy()
    else:
        RPartition = R.copy()

    # Check that coordinates are wrapped. Orthorhombic box assumed.
    if DEBUG_LEVEL > 0:
        assert (R[:, 0] < cell[0, 0]).all(), "Wrap coordinates first."
        assert (R[:, 1] < cell[1, 1]).all(), "Wrap coordinates first."
        assert (R[:, 2] < cell[2, 2]).all(), "Wrap coordinates first."
        assert (R[:, 0] >= 0).all(), "Wrap coordinates first."
        assert (R[:, 1] >= 0).all(), "Wrap coordinates first."
        assert (R[:, 2] >= 0).all(), "Wrap coordinates first."

    # Partition length along the domain decomps.
    partition_length_x = cell[0, 0] / ngx
    partition_length_y = cell[1, 1] / ngy
    partition_length_z = cell[2, 2] / ngz

    # Determines the initial partition of the atoms by their location w.r.t.
    # the domain decomposition.
    RPartition[:, 0] = np.clip(np.floor(R[:, 0] / partition_length_x), a_min=0, a_max=k-1)
    RPartition[:, 1] = np.clip(np.floor(R[:, 1] / partition_length_y), a_min=0, a_max=k-1)
    RPartition[:, 2] = np.clip(np.floor(R[:, 2] / partition_length_z), a_min=0, a_max=k-1)

    # Flatten s.t. partition : R^3 -> R^1
    # Need indexing of image distances if that will be needed in the future.
    nodeIPartition = (
        RPartition[:, 0] * 1 +
        RPartition[:, 1] * ngx +
        RPartition[:, 2] * ngx * ngy
    ).astype(int)

    nNodes = n_atoms

    # BEGIN GRAPH PARTITIONING ROUTINE.

    nodeIDegree = np.sum(Gp, axis=1).astype(int)
    max_node_degree = int(np.max(nodeIDegree))
    nodeIConnections = np.full((nNodes, max_node_degree), nNodes)
    for i in range(nNodes):
        nodeIConnections[i, : nodeIDegree[i]] = np.where(Gp[i] == 1)[0]
        if DEBUG_LEVEL > 2:
            print(nodeIDegree[i])
            print(np.where(Gp[i] == 1)[0])
            print(nodeIConnections[i])

    if DEBUG_LEVEL > 1:
        assert np.sum(nodeIDegree) / 2 == float(
            G.number_of_edges()
        ), "Edges computed from row sum don't match NetworkX"

    nodeICuts = np.zeros(nNodes, dtype=int)
    partitionKNumNodes = np.zeros(k, dtype=int)

    # Populate the cuts mentioned below.
    for i in range(nNodes):
        iK = nodeIPartition[i]
        partitionKNumNodes[iK] += 1
        for j in range(max_node_degree):
            # Recall the 'empty' values are filled with nNodes
            # to indice an empty connection.
            if nodeIConnections[i, j] < nNodes:
                if nodeIPartition[j] != iK:
                    nodeICuts[i] += 1

    # One important thing here is that nothing is sparse and everything is
    # padded. This may be problem for materializing large dense graphs, and
    # in the future a conversion can be made from e.g. Gp -> csr, and
    # mitigating some of the below arrays.

    # == Overview of the key variables tracking metrics/data in this routine ==
    # Gp                       -> Dense adjacency matrix.
    # k                        -> Number of partitions.
    # nodeIDegree            -> Degree of node_i in the full graph, Gp.
    # nodeIConnections       -> The nodes that node i is connected to. *Padded*
    # nodeIPartition         -> The partition node i resides in.
    # nodes_in_partition_k     -> Number of nodes in partition k
    # partitionKNumNodes    -> Number of nodes in partition k
    # nodeICuts              -> Number of cuts on node i

    return (
        Gp,
        k,
        nodeIDegree,
        nodeIPartition,
        nodeIConnections,
        nodeICuts,
        partitionKNumNodes,
    )


## do_mitigate_large_partitions
# @brief Aims to mitigate large partitions by moving nodes to smaller ones.
# @param nodeIPartition (np.ndarray (nNodes)) containing node i's partition.
# @param nodeIConnections (np.ndarray (nNodes, n_max_degree)). Connections of
#        node I. This is padded with the value "nNodes"
# @param cutsIK (np.ndarray (n_nodes,k)) Cuts on I if it were in partition K.
# @param partitionKNumNodes (np.ndarray (k)). Number of core nodes in K.
# @param coreHaloSize (np.ndarray (k)). Core+Halo size of partition K.
# @param top_frac_search (float): We try to pull nodes from this % of the
#        largest partitions. Visual example below in docstrings.
# @param bot_frac_search (float): We try to push nodes to this % of the
#        smallest partitions. Visual example below in docstrings.
# @return halos (np.ndarray shape(k, nNodes/5) Holds the halos.
def do_mitigate_large_partitions(
    k,
    nodeIDegree,
    nodeIConnections,
    nodeIPartition,
    cutsIK,
    partitionKNumNodes,
    coreHaloSize,
    top_frac_search=0.30,
    bot_frac_search=0.40,
):
    """
    E.g. partition sizes (core nodes):
    k    = 0 1 2 3  4  5  6  7  8  9
    size = 7 8 9 10 12 11 15 18 20 21

    top_fraction_search = .2 would pick out partitions 8, 9
    bot_fraction search = .5 would pick out partitions 0, 1, 2, 3, 4

    Meaning we'd look for swap from 8->[0,1,2,3,4] and 9->[0,1,2,3,4]
    This portion scales roughly (assuming somewhat uniform paritition sizes) as top_frac*bot_frac*k^2.
    """

    indx = np.argsort(coreHaloSize)[::-1]
    # indx = np.argsort(partitionKNumNodes)[::-1]
    nk = int(top_frac_search * k)
    nNodes = nodeIPartition.shape[0]

    # Here we're sorting the indices of core+halo size in descending order.
    # E.g. the worst partition is indx[0], the best is indx[-1]
    # These correspond directly to the *actual* partition index, K.
    largestKs = indx[:nk]
    smallestKs = indx[int((1 - bot_frac_search) * k):]

    largePartitionNodes = partitionKNumNodes[largestKs]

    # We have to be careful not to pull too many nodes from decently balanced
    # partitions. Therefore we use this as a check, s.t. if large partition #
    # of nodes -> under this number we breaak that loop and stop pulling nodes
    # from it.
    largeCutoff = np.min(largePartitionNodes)

    # The logic here is that trading nodes from large partitions to small ones
    # where we reduce or even keep the cut equal is a net benefit (for
    # core/core+halo sizes).
    for cK in largestKs:
        # cK->current partition K
        cK_nodes = np.where(nodeIPartition == cK)[0]
        for cK_node_i in cK_nodes:
            if partitionKNumNodes[cK] < largeCutoff:
                break

            # Boolean array of partitions from bot_frac_search%ile  which
            # either reduce or keep equal the cut on node_i.
            res = 1
            bestPartition = None
            for pK in smallestKs:
                prop_res = cutsIK[cK_node_i, pK] - cutsIK[cK_node_i, cK]
                if prop_res < res:
                    res = prop_res
                    bestPartition = pK

            # If there are any such partitions, go to one with smallest cut.
            if bestPartition is not None:

                # Change the node's partition
                nodeIPartition[cK_node_i] = bestPartition

                partitionKNumNodes[cK] -= 1
                partitionKNumNodes[bestPartition] += 1

                # Actualize changes to the cut of cK_node_i's neighbors.
                for j in range(nodeIDegree[cK_node_i]):
                    iNeighborJ = nodeIConnections[cK_node_i, j]
                    cutsIK[iNeighborJ, cK] += 1
                    cutsIK[iNeighborJ, bestPartition] -= 1

                # Reset the smallestKs, so that we don't pile nodes into a small partition
                # and ruin the procedure.
                indx = np.argsort(partitionKNumNodes)[::-1]
                smallestKs = indx[int((1 - bot_frac_search) * k):]

    cut = 0
    assert (
        np.min(cutsIK) >= 0
    ), f"Cut can never be negative, but is currently as low as {np.min(cutsIK)}"

    for i in range(nNodes):
        iK = nodeIPartition[i]
        cut += cutsIK[i, iK]

    cut = int(cut/2)

    return cut, nodeIPartition, cutsIK, partitionKNumNodes


@jit(nopython=True)
def do_partition_flips(
    Gp,
    k,
    nodeIDegree,
    nodeIConnections,
    nodeIPartition,
    partitionKNumNodes,
):
    nNodes = Gp.shape[0]

    # Precompute cuts on node i if it were in partition K.
    cutsIK = np.zeros((nNodes, k), dtype=np.int32)

    for i in range(nNodes):
        # Degree of i is also max number of cuts, so initialize there and
        # subtract one for each neighbor in partition K.
        nodeIDeg = nodeIDegree[i]
        cutsIK[i, :] = nodeIDeg

        # Loop over each neigbhbor of node i.
        for j in range(nodeIDeg):
            neighJ = nodeIConnections[i, j]

            if DEBUG_LEVEL > 1:
                assert neighJ < nNodes, "Should never be this high"

            # Node i has neighbor j in partition K.
            # So if i->K then cutsIK gets decremented 1.
            neighJPartition = nodeIPartition[neighJ]

            cutsIK[i, neighJPartition] -= 1

    assert np.min(cutsIK) >= 0, "Cut should never be negative."
    assert np.min(nodeIDegree) >= 0, "Degree should never be negative."

    swaps = 0
    for i in range(nNodes):
        if cutsIK[i, nodeIPartition[i]] > 0:  # Leave a node with no cuts alone.
            for j in range(i + 1, nNodes):

                iK = nodeIPartition[i]
                jK = nodeIPartition[j]
                # For when nodes are not in same partition.
                if iK != jK:
                    # jK = nodeIPartition[j]
                    origCutsI = cutsIK[i, iK]
                    origCutsJ = cutsIK[j, jK]

                    newCutsI = cutsIK[i, jK]

                    # Ensure unisolated nodes upon switching
                    if newCutsI < nodeIDegree[i]:
                        newCutsJ = cutsIK[j, iK]
                        if newCutsJ < nodeIDegree[j]:
                            if (newCutsI + newCutsJ) < (origCutsJ + origCutsI):
                                swaps += 1

                                # Update the new partitions from the swap.
                                nodeIPartition[i] = jK
                                nodeIPartition[j] = iK

                                # Actualize changes to the cuts.
                                # i->j => i's neighbors in iK go UP 1, i's neighbors in jK go DOWN 1
                                for m in range(nodeIDegree[i]):
                                    iNeighborM = nodeIConnections[i, m]

                                    cutsIK[iNeighborM, iK] += 1
                                    cutsIK[iNeighborM, jK] -= 1

                                # Likewise, but flipped for j->i
                                for m in range(nodeIDegree[j]):
                                    jNeighborM = nodeIConnections[j, m]
                                    cutsIK[jNeighborM, iK] -= 1
                                    cutsIK[jNeighborM, jK] += 1

                                assert np.min(
                                    cutsIK >= 0
                                ), f"i:{i} j:{j}, iK:{iK}, jK:{jK}, i_neigh_k:{iNeighborM}, n_i_connections:{nodeIConnections[i,k]}, cuts_k_jK:{cutsIK[iNeighborM]}"

                # For when nodes are in same partition, we try to move ONLY I -> the smallest partition.
                elif iK == jK:
                    # Find the smallest partition, sK, and check for a swap there.
                    iK = nodeIPartition[i]
                    sK = np.argmin(partitionKNumNodes)

                    newCutsI = cutsIK[i, sK]
                    currCutsI = cutsIK[i, iK]

                    # If the node is already in the smallest partition, just go do the next node and do nothing.
                    if sK != iK:
                        # Ensure unisolated nodes upon switching
                        if newCutsI < nodeIDegree[i]:
                            if newCutsI <= currCutsI:
                                swaps += 1

                                partitionKNumNodes[sK] += 1
                                partitionKNumNodes[iK] -= 1

                                # Update the new partitions from the swap.
                                nodeIPartition[i] = sK

                                # Actualize changes to the cuts.
                                # i->j => i's neighbors in iK go UP 1, i's neighbors in sK go DOWN 1
                                for k in range(nodeIDegree[i]):
                                    iNeighborM = nodeIConnections[i, k]
                                    cutsIK[iNeighborM, iK] += 1
                                    cutsIK[iNeighborM, sK] -= 1

    cut = 0
    assert (
        np.min(cutsIK) >= 0
    ), f"Cut can never be negative, but is currently as low as {np.min(cutsIK)}"


    for i in range(nNodes):
        iK = nodeIPartition[i]
        cut += cutsIK[i, iK]
        if cutsIK[i, iK] < 0:
            raise ValueError("...")

    cut /= 2
    cut = int(cut)

    return cut, cutsIK, nodeIPartition, partitionKNumNodes, swaps


## get_core_halo
# @brief computes halos for th set of input core partitions.
# @param nodeIPartition (np.ndarray (nNodes)) containing node i's partition.
# @param nodeIConnections (np.ndarray (nNodes, n_max_degree)). Contains all of
#        connections of node i. This is padded with the value "nNodes".
# @param nodeIDegree np.ndarray(np.ndarray (nNodes)) contaning node i's degree.
# @param k (int) number of partitions for the graph.
# @param order (int) number of jumps along the graph walk to compute halos.
# @param maxHaloFraction (float): Max fraction of total nodes the halos can be.
#        This is set at a very safe 0.5, but can probably be much lower at 
#        ~.05 - .15 for very sparse graphs.
# @return halos (np.ndarray shape(k, nNodes/5) Holds the halo nodes for each
#         partition. May want another option to deal with extremely
#         dense graphs.
# @return halo_ct (np.ndarray (k)) Number of nodes in each partition's halo.
@jit(nopython=True)
def get_core_halo(nodeIPartition, nodeIConnections, nodeIDegree, k,
                      order=2, maxHaloFraction=.5):

    nNodes = nodeIPartition.shape[0]

    # Again these are filled with nNodes,
    # extract true halos with: halos < nNodes as a mask.
    halos = np.full((k, int(nNodes*maxHaloFraction)), nNodes, dtype=np.int32)
    halo_ct = np.zeros(k, dtype=np.int32)

    if order == 1:
        # First round of populating only the core's nearest neighbors.
        for i in range(nNodes):
            iK = nodeIPartition[i]
            deg_i = nodeIDegree[i]
            for j in range(deg_i):
                node_j = nodeIConnections[i, j]
                jK = nodeIPartition[node_j]

                # Checks if in core, checks if in halo
                if jK != iK and node_j not in halos[iK]:
                    halos[iK, int(halo_ct[iK])] = node_j
                    halo_ct[iK] += 1

    elif order > 1:
        # First round of populating only the core's nearest neighbors.
        for i in range(nNodes):
            iK = nodeIPartition[i]
            deg_i = nodeIDegree[i]
            for j in range(deg_i):
                node_j = nodeIConnections[i, j]
                jK = nodeIPartition[node_j]
                if jK != iK and node_j not in halos[iK]:
                    halos[iK, halo_ct[iK]] = node_j
                    halo_ct[iK] += 1

        # Populate halo's neighbors (order - 1) times to get
        # the 'order'th nearest neighbors.
        for ord in range(order - 1):
            # Loop over each halo
            for haloK in range(k):

                curr_halo = halos[haloK]
                nodes_in_curr_halo = halo_ct[haloK]

                # Loop over nodes in current halo.
                for halo_node in curr_halo[:nodes_in_curr_halo]:

                    # Loop over neighbors of current node.
                    for halo_node_neighbor in range(nodeIDegree[halo_node]):

                        neigh = nodeIConnections[halo_node, halo_node_neighbor]

                        # Checks that node isn't in core or already in halo.
                        if haloK != nodeIPartition[neigh] and neigh not in halos[haloK]:
                            halos[haloK, halo_ct[haloK]] = neigh
                            halo_ct[haloK] += 1
    return halos, halo_ct

    # Compute the core+halo sizes.

## Get the core and halo indices
# @brief Gets the halos given a list of cores and a graph
# @param core list of cores 
# @param graph Graph to extract the halos from
# @param njumps It will search the halos among the "njumps" nearest neighbors
#
def get_coreHaloIndices(eng, core,graph,njumps, *args):
    coreHalo = np.array(core.copy())
    nc = len(coreHalo)
    nch = nc
    nnodes = len(graph[:,0])
    nx = np.zeros((nnodes),dtype=bool)
    nx[:] = False # $$$ ??? what is nx ???
    nx[coreHalo] = True
    #Add halos from graph
    jump = 0
    jumps_done_but_looking_for_odd = False
    extraAtoms = []
    while jump < njumps:
        nc1 = nch 
        for k in range(nc1):
            i = coreHalo[k]

            nxFalseMask = (nx[graph[i,1:graph[i,0]+1]] == False)
            if jumps_done_but_looking_for_odd == False:
                nch += np.sum(nxFalseMask)
                coreHalo = np.append(coreHalo, graph[i,1:graph[i,0]+1][nxFalseMask])
                nx[graph[i,1:graph[i,0]+1]] = True
            # else:
            #     for kk in range(1, graph[i,0]+1):
            #         j = graph[i,kk]
            #         if (nx[j] == False) and (args[0].valency[args[1].symbols[args[1].types[j]]] ) % 2 == 1:
            #             nch = nch + 1
            #             coreHalo = np.append(coreHalo, int(j))
            #             extraAtoms.append(int(j))
            #             graph[core[0]][graph[core[0]][0]+1] = j
            #             graph[core[0]][0] += 1
            #             graph[core[0]][1:graph[core[0]][0]+1] = sorted(graph[core[0]][1:graph[core[0]][0]+1])
            #             print('APPENDED EXTRA', j)
            #             nx[j] = True
            #             if(eng.interface == "PySEQM"): coreHalo = sorted(coreHalo)
            #             return coreHalo, nc
                    
        # if jump == njumps - 1 and args:
        #     num_el = 0
        #     for II in range(len(coreHalo)):
        #         num_el += args[0].valency[args[1].symbols[args[1].types[coreHalo][II]]]
        #     #print('NumAt:', len(coreHalo), 'NumEl:', num_el)
        #     if num_el%2 != 0:
        #         #print('Odd NumEl:', num_el, '. Looking for an extra atom.')
        #         jumps_done_but_looking_for_odd = True
        #         jump -= 1
                
        jump += 1
            
                
    # if not jumps_done_but_looking_for_odd:
    #     print('\n')
    
    if(eng.interface == "PySEQM"): coreHalo = sorted(coreHalo)
    return coreHalo, nc

## coords_partition
# @brief Computes refined graph partitioning from either coordinates or 
#        the globally thresholded graph from DM (to be added)
# @param systemFilename (str): System file name to parse for structural info.
# @param k (int): Number of partitions in the system.
# @param domainDecomp (List<int>): Number of domains to decomp along x, y, z.
#                                   nx*ny*nz must be divisor of k.
# @param numSwapRuns (int): Number of runs to compute node partition flips.
# @param numMitRuns (int): Number of runs to mitigate large core+halo sizes.
# @param device (str): Device for torch tensores used in spatial partitioning.
# @param cutoff (float): Cutoff for edges between nodes. Units same as coords.
# @param order (int): Number of steps along graph walk to take for halos.
# @param visualize (bool): Whether to visualize the Initial/Final node
#                          distributions and Core/Core+Halo sizes.
# @return partitionKCore (np.ndarray shape(k, largestCorePartition)
# @return partitionKHalo (np.ndarray shape(k, largestHaloPartition)
# @return partitionKNumCore (int): Number of nodes in the core of partition k,
#         e.g. partition_k_core[k,:partition_k_num_core[k]] returns core nodes.
# @return partitionKNumHalo (int): Number of nodes in the halo of partition k,
#         e.g. partition_k_halo[k,:partition_k_num_halo[k]] returns halo nodes.
def coords_partition(
    systemFilename,
    k,
    domainDecomp,
    numSwapRuns=10,
    numMitRuns=10,
    device="cpu",
    cutoff=1.8,
    order=2,
    visualize=False,
    verbosity=1,
):
    # Carries out the initial placement of nodes into reasonable partitions.
    (
        Gp,
        k,
        nodeIDegree,
        nodeIPartition,
        nodeIConnections,
        nodeICuts,
        partitionKNumNodes,
    ) = _initial_partition_from_coordinates(
        systemFilename,
        k,
        domainDecomp,
        device="cpu",
        cutoff=cutoff,
    )
    p_k_nNodes0 = partitionKNumNodes.copy()
    # t = time.time()
    swapsTot = 0
    for i in range(numSwapRuns):
        if verbosity > 0 and i == 0:
            print("Beginning Flip Routine")
            cut = int(np.sum(nodeICuts)/2)
            balance = get_balance_from_partition_sizes(partitionKNumNodes)
            relcut = cut / np.sum(nodeIDegree)
            print(
                f"Graph Statistics:\nIteration {i}\tcut:\t{cut}\trelcut:\t{relcut:.2f}\tbalance:\t{balance:.2f}"
            )

        if DEBUG_LEVEL > 0:
            assert (
                np.sum(partitionKNumNodes) == nodeIDegree.shape[0]
            ), f"Number of nodes not adding up. SumPartitionSizes:{np.sum(partitionKNumNodes)} vs Nodes:{nodeIDegree.shape[0]}"

        cut, cutsIK, nodeIPartition, partitionKNumNodes, swaps = do_partition_flips(
            Gp,
            k,
            nodeIDegree,
            nodeIConnections,
            nodeIPartition,
            partitionKNumNodes,
        )
        assert (
            np.sum(partitionKNumNodes) == nodeIDegree.shape[0]
        ), f"Number of nodes not adding up. SumPartitionSizes:{np.sum(partitionKNumNodes)} vs Nodes:{nodeIDegree.shape[0]}"

        swapsTot += swaps

        halosK, haloKNumNodes = get_core_halo(
            nodeIPartition, nodeIConnections, nodeIDegree, k
        )
        coreHaloSize = haloKNumNodes+partitionKNumNodes
        coreHaloBalance = get_balance_from_partition_sizes(coreHaloSize)

        if verbosity > 0:
            relcut = cut / np.sum(nodeIDegree)
            balance = get_balance_from_partition_sizes(partitionKNumNodes)

            print(
                f"Iteration {i+1}\tcut:\t{cut}\trelcut:\t{relcut:.2f}\tbalance:\t{balance:.2f}\tswaps:{swaps}\tCH-Balance:{coreHaloBalance:.2f}\t\tLargest CH:{np.max(coreHaloSize)}"
            )

    for i in range(numMitRuns):
        cut, nodeIPartition, cutsIK, partitionKNumNodes = do_mitigate_large_partitions(
            k,
            nodeIDegree,
            nodeIConnections,
            nodeIPartition,
            cutsIK,
            partitionKNumNodes,
            coreHaloSize,
        )
        haloK, partitionKNumHalo = get_core_halo(
            nodeIPartition, nodeIConnections, nodeIDegree, k
        )
        coreHaloSize = partitionKNumHalo + partitionKNumNodes

        if verbosity > 0 and i == 0:
            print("Beginning Routine to Mitigate large C+H Partitions")
            relcut = cut / np.sum(nodeIDegree)
            balance = get_balance_from_partition_sizes(partitionKNumNodes)
            print(
                f"Iteration {i+1}\tcut:\t{cut}\trelcut:\t{relcut:.2f}\tbalance:\t{balance:.2f}\tswaps:{swaps}\tCH-Balance:{coreHaloBalance:.2f}\t\tLargest CH:{np.max(coreHaloSize)}"
            )
        elif verbosity > 0:
            relcut = cut / np.sum(nodeIDegree)
            balance = get_balance_from_partition_sizes(partitionKNumNodes)
            print(
                f"Iteration {i+1}\tcut:\t{cut}\trelcut:\t{relcut:.2f}\tbalance:\t{balance:.2f}\tswaps:{swaps}\tCH-Balance:{coreHaloBalance:.2f}\t\tLargest CH:{np.max(coreHaloSize)}"
            )

    if visualize:
        rngmin = min([np.min(p_k_nNodes0), np.min(partitionKNumNodes)])
        rngmax = max([np.max(p_k_nNodes0), np.max(partitionKNumNodes)])
        plt.figure(figsize=(14, 8))
        plt.yticks([])
        plt.subplot(2, 2, 1)
        plt.title("Initial Node Distribution")
        plt.xlabel("Nodes in Partition")
        plt.ylabel("Freqency")
        # plt.plot(list(range(k)), p_k_nNodes0)
        plt.hist(
            p_k_nNodes0,
            range=(rngmin, rngmax),
            rwidth=0.85,
            color="blue",
            edgecolor="k",
        )

        plt.subplot(2, 2, 2)
        plt.title("Final Node Distribution")
        plt.xlabel("Nodes in Partition")
        plt.ylabel("Freqency")
        plt.hist(
            partitionKNumNodes,
            range=(rngmin, rngmax),
            rwidth=0.85,
            color="blue",
            edgecolor="k",
        )
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title("Partition Sizes (Core and Core+Halo)")
        plt.ylabel("Number of nodes")
        plt.xlabel("Partition Number")
        plt.bar(list(range(k)), coreHaloSize, label="C+H", color="red", edgecolor="k")
        plt.bar(
            list(range(k)),
            partitionKNumNodes,
            label="Core",
            color="blue",
            edgecolor="k",
        )
        ch_max = np.max(coreHaloSize)
        plt.yticks(np.arange(0, ch_max, int(ch_max) / 8))
        plt.xlim(0, k)
        plt.legend()
        plt.show()

    # Gather everything to return partitions/sizes.
    haloK, partitionKNumHalo = get_core_halo(
        nodeIPartition, nodeIConnections, nodeIDegree, k
    )
    coreHaloSize = haloKNumNodes + partitionKNumNodes

    largestCorePartition = np.max(partitionKNumNodes)
    largestHaloPartition = np.max(partitionKNumHalo)
    partitionKCore = np.zeros((k, largestCorePartition), dtype = np.int32)
    partitionKNumCore = partitionKNumNodes

    # Trim this down
    partitionKHalo = haloK[:, :largestHaloPartition]

    coreKCounts = np.zeros(k, dtype=np.int32)
    for i in range(nodeIPartition.shape[0]):
        # Get node i partition.
        iK = nodeIPartition[i]
        # Put node i in partition K's list.
        partitionKCore[iK,coreKCounts[iK]] = i
        # Increment count in K.
        coreKCounts[iK] += 1

    return partitionKCore, partitionKHalo, partitionKNumCore, partitionKNumHalo



## ========================================================== ##
## The following three functions are direct copies from the 
## graph_partition.py file and completely unchanged for 
## consistency's sake.

## Metis partition
# @brief This will partition the graph according to the Metis method.
# Details about the metis method can be find in
# <a href="http://glaros.dtc.umn.edu/gkhome/views/metis">Metis site</a>
# @param graph Graph to be partition
# @nparts Number of total parts
# @param verb Verbosity level
# @return parts Partition containing a "list of parts" where every
# part is a list of nodes
#
def metis_partition(graph, nparts, verb=False):
    """Partitions using metis"""
    if metisLib == False:
        raise ImportError("Consider installing Metis library")
    if nxLib == False: 
        raise ImportError("Consider installing Network X library")
    if verb:
        print("\nMetis partition:")

    nnodes = len(graph[:, 0])
    if(nparts > 1):
        nxGraph = get_nx_graph(graph, 1.0)
        # Metis partition metis call
        # Metis returns nxParts which is a list of every's part (or "color")
        # to where they belong. Node "i" belongs to "metisParts[i]" part.
        edgecuts, metisParts = metis.part_graph(nxGraph, nparts)

        # The next lines will transform from metis to our partition format
        parts = []
        for k in range(nparts):
            parts.append([])
        for k in range(nnodes):
            parts[metisParts[k]].append(k)
        if verb:
            for i in range(nparts):
                print("part", i, "=", parts[i])

    else:
        
        parts = []
        parts.append(list(range(0,nnodes)))
        
    # plot_graph(nxGraph)
    return parts


## MinCut local partition optimization.
# @brief This will optimize a given partition based on a mincut algorithm.
# @param graph Graph to be partition **as adjacency matrix.
#              FIXME: Make work with format in "get_a_small_graph"
# @param nparts Number of total parts
# @param verb Verbosity level
# @param numSwapRuns (int): Number of runs to flip node partitions
# @return parts Partition containing a "list of parts" where every
# part is a list of nodes
def mincut_partition(graph, nparts, verb, numSwapRuns = 20):
    
    #np.save('graph_for_robert.npy', graph)
    #print('saved')
    #exit()
    #print(graph)

    nodeIDegree = np.sum(graph, axis = 0)
    # Do a first partition
    nNodes = len(graph[:, 0])
    parts = regular_partition(graph, nparts, verb)

    # Get part indices
    nodeIPartition = get_parts_indices(parts, nNodes)

    # Evaluate the cut
    cut = get_cut(nodeIPartition, graph)

    # Evaluate the balancing
    bal = get_balance_from_partitions(parts)

    cutOld = 10**10
    # Precompute some needed info
    max_node_degree = int(np.max(nodeIDegree))

    nodeIConnections = np.full((nNodes, max_node_degree), nNodes)
    print(nodeIConnections)

    if np.min(graph) < 0:
        nodeIDegree = np.zeros(nNodes)
        for i in range(nNodes):
            firstNonConnection = np.where(graph[i]==-1)[0][0]
            nodeIConnections[i,:firstNonConnection] = graph[i,:firstNonConnection]
            nodeIDegree[i] = firstNonConnection

    else:
        for i in range(nNodes):
            nodeIConnections[i, : nodeIDegree[i]] = np.where(graph[i] == 1)[0]
    partitionKNumNodes = np.zeros(nparts, dtype=int)
    for i in range(nNodes):
        iK = nodeIPartition[i]
        partitionKNumNodes[iK] += 1

    # whichPartNew = do_flips_precomp(whichPart, graph, nNodes, nparts, bal=bal)
    nodeIConnections = np.full((nNodes, max_node_degree), nNodes)
    for i in range(nNodes):
        nodeIConnections[i, : nodeIDegree[i]] = np.where(graph[i] == 1)[0]

    for i in range(numSwapRuns):
        (cut0,
         cutsIK,
         nodeIPartition,
         partitionKNumNodes,
         swaps) = do_partition_flips(graph,
                                     nparts,
                                     nodeIDegree,
                                     nodeIConnections,
                                     nodeIPartition,
                                     partitionKNumNodes)

        cut = get_cut(nodeIPartition, graph)
        bal = get_balance(nodeIPartition, nparts)
        if verb:
            print(f"Cut: {cut0}, Balance: {bal}")
        if cut == cutOld:
            break
        else:
            cutOld = cut

    parts = get_parts_from_indices(nodeIPartition, nparts)

    return parts

## Spectral Clustering partition
# @brief This will partition a graph according to the Spectral Clustering method
# @param graph Graph to be partition
# @param nparts Number of total parts
# @param verb Verbosity level
# @return parts Partition containing a "list of parts" where every
# part is a list of nodes
#

def spectral_clustering_partition(sdc, graph,nparts, coords, do_xyz, max_cluster_size=None, verb=False):
    if(SpectralClusteringLib == False):
         print("\n ERROR: Consider installing sklearn.cluster.SpectralClustering library \n")
         exit(0)
    if(verb):print("\nSpectral Clustering partition:")

    if do_xyz: # on xyz
        print('  Computing spectral_clustering_partition on XYZ data.')
        clustering = SpectralClustering(n_clusters=nparts, affinity="nearest_neighbors",random_state=0,
                                        n_neighbors=sdc.SpecClustNN, n_jobs=1, n_init=10) # or 'rbf' 'nearest_neighbors'
        cluster_labels = clustering.fit_predict(coords)
        parts = [np.where(cluster_labels == i)[0].tolist() for i in range(nparts)]

    else: # do on graph
        numNodes = graph.shape[0]  # Add this line
        adjacencyList = adjacency_matrix_to_graph(graph)
        
        # Convert adjacency list to adjacency matrix
        adjacencyMatrixGraph = np.zeros((numNodes, numNodes))
        for i, neighbors in enumerate(adjacencyList):
            adjacencyMatrixGraph[i, neighbors] = 1
            adjacencyMatrixGraph[neighbors, i] = 1

        # Convert the adjacency matrix to the Laplacian matrix
        laplacianMatrix = np.diag(np.sum(adjacencyMatrixGraph, axis=1)) - adjacencyMatrixGraph

        # Calculate the first `nparts` eigenvectors of the Laplacian matrix
        _, eigenvectors = np.linalg.eigh(laplacianMatrix)

        # Use k-means clustering on the selected eigenvectors
        clustering = SpectralClustering(n_clusters=nparts, affinity="rbf",random_state=0,
                                        n_neighbors=10, n_jobs=32, n_init=20) # or 'rbf' 'nearest_neighbors'
        cluster_labels = clustering.fit_predict(eigenvectors[:, :nparts])
        parts = [np.where(cluster_labels == i)[0].tolist() for i in range(nparts)]

    # Ensure clusters do not exceed max_cluster_size
    if max_cluster_size:
        parts = enforce_max_cluster_size_spectral(parts, adjacencyMatrixGraph, max_cluster_size)

    return parts

def adjacency_matrix_to_graph(adjacency_matrix):
    num_nodes = adjacency_matrix.shape[0]
    adjacency_list = [adjacency_matrix[i, 1:1 + adjacency_matrix[i, 0]] for i in range(num_nodes)]
    return adjacency_list

# Function to enforce max cluster size with iterative check for the closest cluster with available space
def enforce_max_cluster_size_spectral(clusters, adjacency_matrix, max_size):
    """
    Enforces a maximum cluster size by recursively applying spectral clustering
    to clusters that exceed the maximum size.
    """
    adjusted_clusters = []

    for cluster in clusters:
        if len(cluster) > max_size:
            # Extract subgraph for the oversized cluster
            subgraph = adjacency_matrix[np.ix_(cluster, cluster)]

            # Recursively apply spectral clustering to split the cluster
            sub_clusters = recursive_spectral_clustering(subgraph, max_size)

            # Map sub-clusters back to original node indices
            mapped_sub_clusters = [[cluster[node] for node in sub_cluster] for sub_cluster in sub_clusters]
            adjusted_clusters.extend(mapped_sub_clusters)
        else:
            adjusted_clusters.append(cluster)

    return adjusted_clusters

def recursive_spectral_clustering(subgraph, max_size):
    """
    Recursively splits a cluster using spectral clustering until all sub-clusters are within the size limit.
    """
    num_nodes = subgraph.shape[0]
    if num_nodes <= max_size:
        return [list(range(num_nodes))]

    # Compute the Laplacian matrix for the subgraph
    laplacianMatrix = np.diag(np.sum(subgraph, axis=1)) - subgraph

    # Compute eigenvectors of the Laplacian matrix
    _, eigenvectors = np.linalg.eigh(laplacianMatrix)

    # Perform Spectral Clustering
    clustering = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=0, n_neighbors=8, n_jobs=16, n_init=40)
    cluster_labels = clustering.fit_predict(eigenvectors[:, :2])

    # Divide nodes into two clusters
    cluster1 = np.where(cluster_labels == 0)[0].tolist()
    cluster2 = np.where(cluster_labels == 1)[0].tolist()

    # Recursively apply to each cluster if they exceed the max size
    final_clusters = []
    if len(cluster1) > max_size:
        final_clusters.extend(recursive_spectral_clustering(subgraph[np.ix_(cluster1, cluster1)], max_size))
    else:
        final_clusters.append(cluster1)

    if len(cluster2) > max_size:
        final_clusters.extend(recursive_spectral_clustering(subgraph[np.ix_(cluster2, cluster2)], max_size))
    else:
        final_clusters.append(cluster2)

    return final_clusters


## Regular partition
# @brief This will partition a graph in the most
# trivial way. Partition \f$ \Pi \f$ being:
# \f[
#    \Pi = \{\{1,...k\},\{k+1,...,2k\},...,\{(n-2)(k+1),...,(n-1)k\},\{(n-1)(k+1),...,N\}\}
# \f]
# where \f$ N = \f$ total nodes, and \f$ k = E(N/n) \f$.
# @param graph Graph to be partition
# @param nparts Number of total parts
# @param verb Verbosity level
# @return parts Partition containing a "list of parts" where every
# part is a list of nodes
#
def regular_partition(graph, nparts, verb=False):
    if verb:
        print("\nRegular partition:")
    nnodes = len(graph[:, 0])
    nnodesInPart = int(nnodes / nparts)
    parts = []
    for i in range(nparts):
        parti = []
        for k in range(i * nnodesInPart, (i + 1) * nnodesInPart):
            parti.append(k)
        parts.append(parti)
        if verb:
            print("part", i, "=", parti)
    if nnodesInPart * nparts < nnodes:
        for k in range(nnodesInPart * nparts, nnodes):
            parti.append(k)
        parts[nparts - 1] = parti

    return parts

## Get the partition list from the index vector
# @param whichPart part index vector for every node
# @param nparts Number of parts
# @return part Partition list. Every element of the list
# is a list of node on every part
#
def get_parts_from_indices(whichPart, nparts):
    parts = [[] for i in range(nparts)]
    for i in range(len(whichPart)):
        partIndex = whichPart[i]
        parts[partIndex].append(i)

    return parts


## Write partitions (core+halo) to structure files.
# @param symbols (list): List of unique atomic symbols,
#                        order consistency required with types.
# @param types (arraylike): Element types
# @param coords (arraylike): Atomic positions
# @param partition_K_core (ndarray): Padded array with core partitions.
# @param parittion_K_num_core (ndarray): Number of atoms in each core partition
# @param partition_K_halo (ndarray): Padded array with halo partitions.
# @param parittion_K_num_halo (ndarray): Number of atoms in each halo partition
# @param outputType (str): xyz, pdb, traj, or pyseqm.
# @param outputFolder (str): Name for the output folder, must exist.
# @return None
def write_partitions_to_files(symbols, types, coords,
                              partition_K_core, partition_K_halo,
                              partition_K_num_core, partition_K_num_halo,
                              outputType="xyz", outputFolder="structures"):

    if not os.path.isdir(outputFolder):
        raise ValueError(f"Folder: '{outputFolder}' does not exist to hold output.")

    numPartitions = partition_K_core.shape[0]

    all_coords = []
    all_types = []

    for k in range(numPartitions):
        coreIndices = partition_K_core[k, :partition_K_num_core[k]]
        haloIndices = partition_K_halo[k, :partition_K_num_halo[k]]
        partIndices = np.concat([coreIndices, haloIndices])

        coords_k = coords[partIndices]
        types_k = types[partIndices]
        all_coords.append(coords_k)
        all_types.append(types_k)

    if outputType == "xyz":
        for k in range(numPartitions):
            fileName = f"{outputFolder}/{k}.xyz"
            coords_k = all_coords[k]
            types_k = all_types[k]
            write_xyz_coordinates(fileName, coords_k, types_k, symbols)

    elif outputType == "pdb":
        for k in range(numPartitions):
            fileName = f"{outputFolder}/{k}.pdb"
            coords_k = all_coords[k]
            types_k = all_types[k]
            write_pdb_coordinates(fileName, coords_k, types_k, symbols)

    elif outputType == "traj":
        try:
            from ase.io.trajectory import Trajectory
            from ase import Atoms
        except ImportError:
            raise ImportError("ASE must be installed to write to .traj files")

        traj = Trajectory(f"{outputFolder}/partitions.traj", "w")

        for k in range(numPartitions):
            fileName = f"{outputFolder}/{k}.pdb"
            coords_k = all_coords[k]
            types_k = all_types[k]
            syms_k = [symbols[i] for i in types_k]
            atoms_k = Atoms(symbols=syms_k, positions=coords_k)
            traj.write(atoms_k)

    elif outputType.lower() == "pyseqm":
        ch_sizes = partition_K_num_core + partition_K_num_halo
        pad_dim = np.max(ch_sizes)

        R = np.zeros((numPartitions, pad_dim, 3))
        Z = np.zeros((numPartitions, pad_dim))

        pt = PeriodicTable()
        for k in range(numPartitions):
            Z_k = np.array([pt.get_atomic_number(symbols[i]) for i in all_types[k]])
            inds = np.argsort(Z_k)[::-1]
            ch_k = len(inds)

            assert len(Z_k) == len(all_coords[k])
            R[k, :ch_k] = np.array(all_coords[k])[inds]
            Z[k, :ch_k] = Z_k[inds]

        np.save(f"{outputFolder}/R.npy", R)
        np.save(f"{outputFolder}/Z.npy", Z)



## Get partition indices
# @brief Get a vector indicating which is the part index of a particular
# node.
# @param parts Partition containing a "list of parts" where every
# part is a list of nodes
# nnodes Number of nodes in the graph.
#
def get_parts_indices(parts, nnodes):
    whichPart = np.zeros((nnodes), dtype=int)
    partIndex = -1
    for part in parts:
        partIndex = partIndex + 1
        for node in part:
            whichPart[node] = partIndex
    return whichPart

## Get the core and halo indices
# @brief Gets the halos given a list of cores and a graph
# @param core list of cores
# @param graph Graph to extract the halos from
# @param njumps It will search the halos among the "njumps" nearest neighbors
#
def get_coreHaloIndices(core, graph, njumps, eng=None):

    # There are too many differences in how these things are computed across codes, at some point these
    # need to be universalized, or we need to ship functions like this off to interface specific python files
    # where the I/O is made consistent and we just have a simple function call here to handle the internal logic.
    if eng is not None:
        if eng.interface == "PySEQM":
            print("in pyseqm block")
            coreHalo = np.array(core.copy())
            nc = len(coreHalo)
            nch = nc
            nnodes = len(graph[:,0])
            nx = np.zeros((nnodes),dtype=bool)
            nx[:] = False # $$$ ??? what is nx ???
            nx[coreHalo] = True
            #Add halos from graph
            jump = 0
            jumps_done_but_looking_for_odd = False
            extraAtoms = []
            while jump < njumps:
                nc1 = nch 
                for k in range(nc1):
                    i = coreHalo[k]

                    nxFalseMask = (nx[graph[i,1:graph[i,0]+1]] == False)
                    if jumps_done_but_looking_for_odd == False:
                        nch += np.sum(nxFalseMask)
                        coreHalo = np.append(coreHalo, graph[i,1:graph[i,0]+1][nxFalseMask])
                        nx[graph[i,1:graph[i,0]+1]] = True
                jump += 1

            coreHalo = sorted(coreHalo)

            return coreHalo, nc, nch
    else:

        nc = len(core)
        coreHalo = []*nc
        coreHalo[:] = core[:]
        nch = nc
        nnodes = len(graph[:, 0])
        nx = np.zeros((nnodes), dtype=bool)
        nx[:] = False  # Logical mask

        for k in range(nc):
            i = coreHalo[k]
            if i != -1:
                nx[i] = True
        # Add halos from graph
        for jump in range(njumps):
            nc1 = nch
            for k in range(nc1):
                i = coreHalo[k]
                degI = len(graph[i, :])
                for kk in range(1, degI):
                    # $$$ also this cycles needs to be interrupted when reaching -1 ???
                    j = graph[i, kk]
                    if (j != -1) & (nx[j] == False):
                        # print(i,j)
                        nch = nch + 1
                        coreHalo.append(j)
                        nx[j] = True
        return coreHalo, nc, nch

def get_coreHaloIndicesPYSEQM(eng, core,graph,njumps, *args):
    coreHalo = np.array(core.copy())
    nc = len(coreHalo)
    nch = nc
    nnodes = len(graph[:,0])
    nx = np.zeros((nnodes),dtype=bool)
    nx[:] = False # $$$ ??? what is nx ???
    nx[coreHalo] = True
    #Add halos from graph
    jump = 0
    jumps_done_but_looking_for_odd = False
    extraAtoms = []
    while jump < njumps:
        nc1 = nch 
        for k in range(nc1):
            i = coreHalo[k]

            nxFalseMask = (nx[graph[i,1:graph[i,0]+1]] == False)
            if jumps_done_but_looking_for_odd == False:
                nch += np.sum(nxFalseMask)
                coreHalo = np.append(coreHalo, graph[i,1:graph[i,0]+1][nxFalseMask])
                nx[graph[i,1:graph[i,0]+1]] = True
            # else:
            #     for kk in range(1, graph[i,0]+1):
            #         j = graph[i,kk]
            #         if (nx[j] == False) and (args[0].valency[args[1].symbols[args[1].types[j]]] ) % 2 == 1:
            #             nch = nch + 1
            #             coreHalo = np.append(coreHalo, int(j))
            #             extraAtoms.append(int(j))
            #             graph[core[0]][graph[core[0]][0]+1] = j
            #             graph[core[0]][0] += 1
            #             graph[core[0]][1:graph[core[0]][0]+1] = sorted(graph[core[0]][1:graph[core[0]][0]+1])
            #             print('APPENDED EXTRA', j)
            #             nx[j] = True
            #             if(eng.interface == "PySEQM"): coreHalo = sorted(coreHalo)
            #             return coreHalo, nc
                    
        # if jump == njumps - 1 and args:
        #     num_el = 0
        #     for II in range(len(coreHalo)):
        #         num_el += args[0].valency[args[1].symbols[args[1].types[coreHalo][II]]]
        #     #print('NumAt:', len(coreHalo), 'NumEl:', num_el)
        #     if num_el%2 != 0:
        #         #print('Odd NumEl:', num_el, '. Looking for an extra atom.')
        #         jumps_done_but_looking_for_odd = True
        #         jump -= 1
                
        jump += 1
            
                
    # if not jumps_done_but_looking_for_odd:
    #     print('\n')
    
    if(eng.interface == "PySEQM"): coreHalo = sorted(coreHalo)
    return coreHalo, nc

