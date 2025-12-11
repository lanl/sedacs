"""graph
Some graph functions

"""

import sys

import mpi4py.MPI as MPI
from sedacs.mpi import collect_and_sum_matrices_float
from numba import njit
import numpy as np
import torch
import nvtx

global nxLib
try:
    import networkx as nx

    nxLib = True
except ImportError as e:
    nxLib = False
global pltLib
try:
    import matplotlib.pyplot as plt

    pltLib = True
except:
    pltLib = False


## Get an initial graph based on distance
# @brief This will give a graph based on distaces. Similar to
# a neighbor list.
# @param coords System coordinates
# @param nl Neighbor list `nl[i,0]` = total number of neighbors.
# `nl[i,1:nl[i,0]+1]` = neigbors of i. Self neighbor i to i is not included explicitly.
# @param radius Radius Cutoff to search for the neighbors
# @param maxDeg Max degrees allowed for each none
# @param verb Verbosity mode
# @return graph The graph consisting on a 2D integer numpy array.
# E.g, `graph[i,k]` is the kth neighbor of node i. NOTE: The 0 entry of
# every row is reserved to store the degree of every node.
#
def get_initial_graph(coords, nl, radius, maxDeg, LBox, graphweights=False, verb=False):
    nats = len(coords[:, 0])
    graph = np.full((nats, maxDeg + 1), -1, dtype=int)
    eweights = np.zeros((nats, maxDeg + 1), dtype=int) if graphweights else None
    if nl.shape[1] < maxDeg + 1:
        nl = np.pad(nl, ((0, 0), (0, maxDeg + 1 - nl.shape[1])), mode='constant', constant_values=-1)

    # Vectorized computation of distances
    delta = coords[:, np.newaxis, :] - coords[nl[:, 1:].astype(int)]
    delta -= LBox * np.round(delta / LBox)
    distances = np.linalg.norm(delta, axis=2)

    # Mask for valid neighbors within the radius
    valid_mask = (distances < radius) & (nl[:, 1:] >= 0)

    # Compute degrees (number of valid neighbors per atom)
    degs = valid_mask.sum(axis=1)

    # Initialize graph and eweights arrays
    graph[:, 0] = degs

    # Fill neighbor indices
    # We'll mask out invalid neighbors using valid_mask
    graph[:, 1:] = np.where(valid_mask, nl[:, 1:], -1)

    if graphweights:
        # Compute weights for all pairs at once
        weights = np.exp(-0.5 * distances**2)
        weights = np.round(100 * weights).astype(int)
        weights = np.maximum(weights, 1)

        # Mask invalid entries
        eweights[:, 1:] = np.where(valid_mask, weights, 0)

    return graph, eweights


## Print graph
# @brief Print the graph by showing the connection of every node.
# @param graph The graph consisting on a 2D numpy array.
# E.g, graph[i,k] is the kth neighbor of node i. NOTE: The 0 entry of
# every row is reserved to store the degree of every node.
#
def print_graph(graph):
    nnodes = len(graph[:, 0])
    print("\nGraph structure:")
    print("Number of nodes: ", len(graph[:, 0]))
    print("Max allowed degree per node: ", len(graph[0, :]))
    print("Number of edges: ", np.sum(graph[:, 0]))
    print("Connections of every node i follows: ")
    for i in range(nnodes):
        nodesList = []
        for k in range(1, graph[i, 0] + 1):
            if graph[i, k] != -1:
                nodesList.append(int(graph[i, k]))
        print(i, "(", graph[i, 0], ")", "-", nodesList)


## Get a networkX graph
# @param graph The graph in 2D numpy array where `graph[i,k]` is the kth neighbor
# of node i. NOTE: The 0 entry of
# every row is reserved to store the degree of every node.
# @param w The weight for the edges (tipically = 1.0)
# @return nxGraph networkX type of graph
#
def get_nx_graph(graph, w):
    if nxLib == False:
        sdc_error("get_nx_graph", "ERROR: Consider installing networkx")
    n = len(graph[:, 0])
    m = len(graph[0, :])
    nxGraph = nx.Graph()
    nxGraph.graph['edge_weight_attr'] = 'weight'
    if w is None:
        w = np.ones_like(graph, dtype=int)
    for i in range(0, n):
        nxGraph.add_node(i)
    for i in range(0, n):
        # nxGraph.add_nodes_from([i, i])
        for k in range(1, graph[i, 0] + 1):
            j = graph[i, k]
            if (j != -1) and (j != i):
                nxGraph.add_edge(i, j, weight=w[i, k])

    #print("graph", graph)
    #print("nxGraph", nxGraph)
    return nxGraph


## Get a regular graph from a nx graph.
# @brief From a networkx graph, this will construct a regular graph.
# @param nxGraph Networkx graph.
# @return graph The graph consisting on a 2D integer numpy array.
# E.g, `graph[i,k]` is the kth neighbor of node i. NOTE: The 0 entry of
# every row is reserved to store the degree of every node.
def get_graph_from_nx(nxGraph):
    if nxLib == False:
        sdc_error("get_nx_graph", "ERROR: Consider installing networkx")
    n = nxGraph.number_of_nodes()
    m = np.max(nxGraph.degree())
    graph = np.zeros((n, m + 1), dtype=int)
    for i in range(n):
        graph[i, 0] = nxGraph.degree()[i]
        jj = 0
        for j in nxGraph.neighbors(i):
            jj = jj + 1
            graph[i, jj] = j

    return graph


## To plot the resulting graph
# @brief Uses matplotlib to plot the nxGraph
# @param nxGraph NetworkX type of graph
# @param nodeColor The color of the nodes
# This will produce a "graph.png" file with the
# plot.
def plot_nx_graph(nxGraph, nodeColor="r"):
    labels = []
    n = nxGraph.number_of_nodes()
    for i in range(n):
        labels.append(str(i))
    pos1 = nx.spring_layout(nxGraph, scale=100.0, weight="weight")
    plt.figure(figsize=(30.0, 30.0))
    nx.draw_networkx_nodes(nxGraph, pos1, node_size=500, alpha=0.8, node_color=nodeColor)
    nx.draw_networkx_edges(nxGraph, pos1, width=1, alpha=0.5, edge_color="r")
    nx.draw_networkx_labels(nxGraph, pos1, font_size=16)
    plt.savefig("graph.png", dpi=400)
    plt.show()


## Collect a graph from DMs
# @brief This will build a graph from small DMs
# @param rho Density matrix. This is a 2D numpy array.
# @param nnodes Number of nodes of the full graph
# @param maxDeg Max degree parameter for the full "collected" graph
# @param indices list of nodes maping to every row and column of rho
# @param hindex A list of displacements maping every node in the full graph
# with a sequence of indices (orbitals) in the the full density matrix.
# The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`
# @return graph The graph in 2D numpy array where `graph[i,k]` is the kth neighbor
# of node i. NOTE: The 0 entry of every row is reserved to store the degree of every node.
#
def collect_graph_from_rho_PYSEQM(graph,rho,thresh,nnodes,maxDeg,indices,hindex=None,verb=False):
   
    rhoDim = len(rho[:,0])
    if (graph is None):
        graph = np.zeros((nnodes,maxDeg+1),dtype=np.int16) - 1
    
    #print('graph', graph[11])
    nats = len(indices)
    weights = np.zeros((nnodes))
    ki_ = 0
    if type(rho) is not np.ndarray:
        rho = rho.numpy().astype(np.float32)
    # Precompute the slice lengths for all j
    slice_lengths = hindex[np.array(indices) + 1] - hindex[indices]
    # Vectorize the extraction of slices from rho
    cumsum_lengths = np.cumsum(np.r_[0, slice_lengths[:-1]])
    max_length = np.max(slice_lengths)
    slice_indices = cumsum_lengths[:, None] + np.arange(max_length)
    # Mask to avoid out-of-bounds indexing
    valid_mask = slice_indices < cumsum_lengths[:, None] + slice_lengths[:, None]
    valid_indices = slice_indices[valid_mask]

    for i in range(nats):
        ii = indices[i]
        #Recovering the connections we already have
        weights[:] = 0.0

        ###
        j = np.arange(1, graph[ii,0]+1)
        weights[graph[ii,j]] = thresh

        ###
        ki_old = ki_
        ki_ = ki_ + hindex[ii+1] - hindex[ii]
        ki_ar = np.arange(ki_old, ki_,1)
        kj = 0  # Initialize kj
        
        flat_rho_slices = rho[ki_ar][:, kj + valid_indices]
        expanded_rho_slices = np.zeros((len(ki_ar),len(slice_lengths), max_length), dtype=rho.dtype)
        expanded_rho_slices[:,valid_mask] = flat_rho_slices
        abs_sums = np.sum(np.abs(expanded_rho_slices)**2, axis=(0,2))**0.5
        np.add.at(weights, indices, abs_sums)

        mask = (np.arange(nnodes) != ii) & (weights >= thresh)
        valid_jj_indices = np.nonzero(mask)[0]
        k = len(valid_jj_indices)
        if k > maxDeg:
            print("!!!ERROR: Max Degree parameter is too small")
            exit(0)
        graph[ii, 1:k+1] = valid_jj_indices[:maxDeg]
        graph[ii, 0] = k

        # if sum(abs(weights1-weights)) > 1e-14:
        #     np.save( "weights.npy", weights,)
        #     np.save( "indices.npy", indices,)
        #     np.save( "rho.npy", rho,)

        #     print('!!!NONZERO', sum(abs(weights1-weights)), ii, ki)
        #     exit(0)
        ###

        ##
        # for j in range(1,graph[ii,0]): # $$$ what does it do? It never enters this loop
        #     jj = graph[ii,j]             
        #     weights[jj] = thresh
        # ##

        # # Computing the new weights by rho 
        # ## $$$ vectorized this ###
        # for oi in range(hindex[ii],hindex[ii+1]):
        #     kj = 0
        #     for j in range(nats):
        #         jj = indices[j]
        #         for oj in range(hindex[jj],hindex[jj+1]):
        #             weights[jj] = weights[jj] + abs(rho[ki,kj])#**2
        #             kj = kj + 1
        #         #weights[jj] = weights[jj]**0.5
        #     ki = ki + 1
        # ##


        # # Reasigning the connections to ii by the merged weights (the ones computed 
        # # from rho and the ones already existing.

        # ## $$$ vectorized this ###
        # k = 0
        # for jj in range(nnodes): # $$$ ??? this cycle could be interrupted ???
        #     if ii ==0 and (jj == 4 or jj == 7):
        #         print(weights[jj])
        #     if((ii != jj) and (weights[jj] >= thresh)):
        #         k = k + 1
        #         if(k >= maxDeg + 1):
        #             print("!!!ERROR: Max Degree parameter is too small") # $$$ any way to use a warning instead of an error ?
        #             exit(0)
        #         graph[ii,k] = jj
        # if ii == 0:
        #     print('In graph',graph[ii][0:10])
        #     if verb:
        #         exit(0)
        # graph[ii,0] = k


    return graph

## Collect a graph from DMs
# @brief This will build a graph from small DMs
# @param rho Density matrix. This is a 2D numpy array.
# @param nnodes Number of nodes of the full graph
# @param maxDeg Max degree parameter for the full "collected" graph
# @param indices list of nodes maping to every row and column of rho
# @param hindex A list of displacements maping every node in the full graph
# with a sequence of indices (orbitals) in the the full density matrix.
# The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`
# @return graph The graph in 2D numpy array where `graph[i,k]` is the kth neighbor
# of node i. NOTE: The 0 entry of every row is reserved to store the degree of every node.
#
def collect_graph_from_rho(graph, rho, thresh, nnodes, maxDeg, indicesCoreHalos, ncores, hindex=None, verb=False):
    if graph is None:
        graph = np.zeros((nnodes, maxDeg + 1), dtype=int)
    nch = len(indicesCoreHalos)

    rho = np.abs(rho)
    reduced_rho = np.zeros((nch, nch), dtype=float)
    reduced_rho[:] = np.maximum.reduceat(
                    np.maximum.reduceat(rho, hindex[:-1], axis=0),
                    hindex[:-1], axis=1
                )

    for i in range(ncores):
        ii = indicesCoreHalos[i]
        # Recovering the connections we already have
        deg = int(graph[ii, 0])
        existing = graph[ii, 1:deg + 1] 

        # Contributions from rho for CH nodes for this core:
        # Any CH node with sum >= thresh becomes a candidate.
        cand_from_rho = np.array(indicesCoreHalos, dtype=int)[reduced_rho[i] >= thresh]

        # Union with existing neighbors (existing edges always retained at threshold)
        # Exclude self-loop if present.
        nbrs = np.unique(np.concatenate((existing, cand_from_rho)))
        nbrs = nbrs[(nbrs != ii)]

        # Degree check
        k = nbrs.size
        if k > maxDeg:
            raise ValueError(f"Max Degree parameter is too small: {maxDeg} (< {k}).")
        if len(existing) > k:
            raise ValueError("Existing degree larger than new degree, something went wrong.")

        # Fill graph row ii: header then neighbors (ascending by construction of np.unique)
        graph[ii, 0] = k
        if k:
            graph[ii, 1:k + 1] = nbrs
        # Clear the tail with -1s
        graph[ii, k + 1:] = -1

    return graph


## Get adjacency matrix
# @brief This will get an adyacency matrix for the graph.
# @param graph
# @para mat
def get_adjacencyMatrix(graph, mat):
    print("add func")


## Add/merge two graph (union operation)
# @brief This will merge or add two graphs
# @param graphA Graph to be merged
# @param graphB Graph to be merged
# @return graphC Resulting graph
#
def add_graphs(graphA, graphB):
    if len(graphA[:, 0]) != len(graphB[:, 0]):
        print("!!!ERROR: Graphs have different number of nodes")
    else:
        nnodes = len(graphA[:, 0])
        maxDeg = len(graphA[0, :])

    vectA = np.zeros((nnodes), dtype=bool)
    vectB = np.zeros((nnodes), dtype=bool)
    vectC = np.zeros((nnodes), dtype=bool)

    graphC = np.zeros((nnodes, maxDeg), dtype=int)
    graphC[:, :] = -1
    for i in range(nnodes):
        # Create a logical row from the neighbors of i in adj A
        vectA[:] = False
        for j in range(1, graphA[i, 0] + 1):
            vectA[graphA[i, j]] = True

        # Create a logical row from the neighbors of i in adj B
        vectB[:] = False
        for j in range(1, graphB[i, 0] + 1):
            vectB[graphB[i, j]] = True
        vectC[:] = vectA[:] + vectB[:]

        k = 0
        for j in range(0, nnodes):
            if vectC[j]:
                k = k + 1
                graphC[i, k] = j
        graphC[i, 0] = k

    return graphC

## Add/merge multiple graphs (union operation)
# @brief This will merge or add multiple graphs
# @param graphs Graphs to be merged
# @return graphC Resulting graph
#
def add_mult_graphs(graphs):
    # Ensure all graphs have the same number of nodes
    nnodes = graphs[0].shape[0]
    maxDeg = graphs[0].shape[1]
    
    if not all(graph.shape[0] == nnodes for graph in graphs):
        print("!!!ERROR: All graphs must have the same number of nodes")
        return None

    # Initialize the result graph `graphC` with -1s
    graphC = np.full((nnodes, maxDeg), -1, dtype=int)

    # Initialize a combined adjacency matrix for all graphs
    adjC = np.zeros((nnodes, nnodes), dtype=bool)

    # Populate combined adjacency matrix based on input graphs
    for graph in graphs:
        adj = np.zeros((nnodes, nnodes), dtype=bool)
        for i in range(nnodes):
            adj[i, graph[i, 1:graph[i, 0] + 1]] = True
        adjC = np.logical_or(adjC, adj)

    # Fill `graphC` based on the combined adjacency matrix `adjC`
    for i in range(nnodes):
        neighbors = np.where(adjC[i])[0]
        graphC[i, 0] = len(neighbors)  # Number of neighbors
        if len(neighbors) > 0:
            graphC[i, 1:len(neighbors) + 1] = neighbors

    return graphC
## Multiply two Adjacencies
# @brief The ij of the resulting graph will be connected
# if i in A and j in B have a common directly connected node k.
# @param graphA Initial adjacency
# @param graphB Initial adjacency
# @return graphC Multiplication result
def multiply_graphs(graphA, graphB):
    if len(graphA[:, 0]) != len(graphB[:, 0]):
        print("!!!ERROR: Graphs have different number of nodes")
    else:
        nnodes = len(graphA[:, 0])
        maxDeg = len(graphA[0, :])

    vectC = np.zeros((nnodes), dtype=bool)
    graphC = np.zeros((nnodes, maxDeg), dtype=int)
    graphC[:, :] = -1
    for i in range(nnodes):
        vectC[:] = False
        for j in range(1, graphB[i, 0] + 1):
            myK = graphB[i, j]
            vectC[myK] = True
        for j in range(1, graphA[i, 0] + 1):
            myK = graphA[i, j]  # All neighbors of i by A
            vectC[myK] = True
            for k in range(1, graphB[myK, 0] + 1):
                myJ = graphB[myK, k]  # All neighbors of myK by B
                if i != myJ:
                    # print(i, myJ)
                    vectC[myJ] = True

        k = 0
        for j in range(0, nnodes):
            if vectC[j]:
                k = k + 1
                graphC[i, k] = j
        graphC[i, 0] = k

    return graphC


# Get a small graph (>-<)
# @brief This will construct a small graph for testing purposes.
# This graph can be is trivially partitioned in two parts
# @return A 6 nodes graph that can be represented by the following
# picture:
#    0        3
#      \     /
#       1 - 4
#      /     \
#    2        5
def get_a_small_graph():
    nnodes = 6
    graph = np.zeros((nnodes, nnodes), dtype=int)
    graph[:, 0] = 1  # Every node has at least one neighbor
    graph[0, 1] = 1  # Node 0 is connected to 1
    graph[1, 0] = 3
    graph[1, 1] = 0
    graph[1, 2] = 2
    graph[1, 3] = 4  # Node 1 to 0,2,4
    graph[2, 1] = 1  # Node 2 to 1
    graph[4, 0] = 3  # Node 4 to 1,3,5
    graph[4, 1] = 1
    graph[4, 2] = 3
    graph[4, 3] = 5
    graph[3, 1] = 4  # Node 3 to 4
    graph[5, 1] = 4  # Node 5 to 4
    return graph

# Get a small graph as an adjacency matrix(>-<)
# @brief This will construct a small graph for testing purposes.
# This graph can be is trivially partitioned in two parts
# @return A 6 nodes graph that can be represented by the following
# picture:
#    0        3
#      \     /
#       1 - 4
#      /     \
#    2        5
def get_a_small_adjacency_matrix():
    nnodes = 6
    graph = np.zeros((nnodes, nnodes), dtype=int)

    # Node 0 to 1, 5
    graph[0, 1] = 1
    graph[0, 5] = 1

    # Node 1 to 2, 4 
    graph[1, 2] = 1
    graph[1, 4] = 1

    # Node 4 to 3, 5
    graph[4, 3] = 1

    graph += graph.T

    return graph

# Get a small graph as an adjacency matrix(>-<)
# @brief This will construct a random adjacency matrix wiht n_nodes.
# @param n_nodes (int): Number of nodes.
# @param density (float): Number between 0,1 represneting likelihood of 
#                   edge connections in the random graph.
# @param degreeOnDiagonal (bool): Whether or not to put the degree
#                                 of nodes on the diagonal. 0 if False.
# @return np.ndarray(n_nodes, n_nodes) of adjacency matrix.
def get_random_adjacency_matrix(n_nodes, density = .1, degreeOnDiagonal = False):
    gRaw = np.random.random((n_nodes, n_nodes))
    gBool = ((gRaw + gRaw.T)/2) < density
    gInt = gBool.astype(int)
    np.fill_diagonal(gInt, 0)

    assert np.all(gInt == gInt.T)

    if degreeOnDiagonal:
        diag = np.sum(gInt, axis = 0)
        np.fill_diagonal(gInt, diag)

    return gInt

# Update density matrix contraction based on the new graph of communities
# @brief.
# @param sy (obj): sedacs system of atoms.
# @param P_contr (tensor): Old density matrix.
# @param graph_for_pairs (list): old graph of communities.
# @param new_graph_for_pairs (list): new graph of communities.
def update_dm_contraction(sdc, sy, P_contr, graph_for_pairs, new_graph_for_pairs, device):
    P_contr_new = torch.zeros_like(P_contr, device=device)
    for i in range(sy.nats):
        tmp1 = graph_for_pairs[i][1:graph_for_pairs[i][0]+1]
        tmp2 = new_graph_for_pairs[i][1:new_graph_for_pairs[i][0]+1]
        pos = np.searchsorted(tmp1, tmp2)
        # Ensure the indices are within bounds
        pos = np.clip(pos, a_min=0, a_max=len(tmp1) - 1)
        # Check if the positions are valid and match
        mask_isin_n_in_o = (pos < len(tmp1)) & (tmp1[pos] == tmp2)
        #print('isin',(np.isin(tmp2, tmp1) == mask_isin_n_in_o).all())

        pos = np.searchsorted(tmp2, tmp1)
        # Ensure the indices are within bounds
        #pos = np.clip(pos, max=len(tmp2) - 1)
        # Check if the positions are valid and match
        mask_isin_o_in_n = (pos < len(tmp2)) & (tmp2[pos] == tmp1)
        #print('PC', (np.isin(tmp1, tmp2) == mask_isin_o_in_n).all())

        if sdc.UHF:
            P_contr_new[:,:,i][:,:new_graph_for_pairs[i][0]  ][:,   mask_isin_n_in_o   ] = \
                P_contr[:,:,i][:,:graph_for_pairs[i][0]][:,   mask_isin_o_in_n   ] 
        else:
            P_contr_new[:,i][:new_graph_for_pairs[i][0]  ][   mask_isin_n_in_o   ] = \
                P_contr[:,i][:graph_for_pairs[i][0]][   mask_isin_o_in_n   ] 
    P_contr[:] = P_contr_new[:]
    del P_contr_new

# Get a graph where each atom has all atoms from its CH as its neighbors, including itself.
# @brief .
# @param sdc (obj): sedacs driver.
# @param sy (obj): sedacs system of atoms.
# @param fullGraph (list): connectivity graph.
# @param parts (list): list of cores.
# @param partsCoreHalo (list): list of cores+halos.
# @return np.ndarray(n_nodes, MaxDeg) of CH.
def get_ch_graph(sdc, sy, fullGraph, parts, partsCoreHalo):
    new_graph_for_pairs = np.array(fullGraph.copy())
    for i in range(sy.nats):
        for sublist_idx in range(sdc.nparts):
            if i in parts[sublist_idx]:
                new_graph_for_pairs[i, 0] = len(partsCoreHalo[sublist_idx])
                new_graph_for_pairs[i, 1:new_graph_for_pairs[i][0]+1] = partsCoreHalo[sublist_idx]
                break
    return new_graph_for_pairs

# Get a mask of diagonal blocks for contracted density matrix.
# @brief .
# @param sdc (obj): sedacs driver.
# @param sy (obj): sedacs system of atoms.
# @param new_graph_for_pairs (list): graph of communities.
# @return np.ndarray(n_atoms).
def get_maskd(sdc, sy, graph_for_pairs):
    # Initialize an array to hold graph_maskd values
    graph_maskd = []
    # Track the position counter across rows in a vectorized way
    counter = 0
    for j in range(sy.nats):
        # Get neighbors for node j from graph_for_pairs
        neighbors = graph_for_pairs[j][1:graph_for_pairs[j][0] + 1]
        # Find positions where `i == j` (self-loops) in the neighbors list
        mask = np.where(neighbors == j)[0]
        # Calculate the absolute position for masked values and store them
        graph_maskd.extend(counter + mask)
        # Update the counter for the next row, adding the degree difference
        counter += len(neighbors) + int(sdc.maxDeg - graph_for_pairs[j][0])
    # Convert graph_maskd to a NumPy array
    graph_maskd = np.array(graph_maskd)
    return graph_maskd

# @brief Convert graphs into a square adjacency matrix.
# @param graph: Input graph to be converted to square adj matrix.
# @param graphType: Input graph type. Options:
#        'sedacs': NxN, first column = node degree. Rest are node connections.
#         E.g. 3rd row [3,1,5,8,0,0,0] => node3 has degree 3 and connections to
#         1, 5, 8.
#        'sklearn':NxX, array padded with -1s. Row i contains non -1 entries where
#         it has edges.
#         E.g. 3rd row [1,6,8,9,-1,-1,-1] => degree = 4, connections [1,6,8,9].
def convert_to_adjacency_matrix(graph, graphType='sedacs'):
    if graphType == 'sklearn':
        nNodes = graph.shape[0]
        adj = np.zeros((nNodes, nNodes), dtype = int)
        for i in range(nNodes):
            inds = graph[i]>-0.1
            adj[i,graph[i][inds]] = 1
            adj[graph[i][inds],i] = 1
        return adj
    elif graphType == 'sedacs':
        nNodes = graph.shape[0]
        adj = np.zeros((nNodes, nNodes), dtype = int)
        for i in range(nNodes):
            connections = graph[i,1:1+graph[i,0]]
            adj[i,connections] = 1
            adj[connections, i] = 1
        return adj


def convert_to_graph(adj, maxDeg):
    nNodes = adj.shape[0]

    graph = np.zeros((nNodes, maxDeg), dtype = int)
    for i in range(nNodes):
        connections = adj[i,:].nonzero()[0]
        graph[i,1:1+len(connections)] = connections[0:len(connections)]
        graph[i,0] = len(connections)

    return graph

# def symmetrize_graph(graph):
#     nnodes = graph.shape[0]
#     maxDeg = graph.shape[1] - 1

#     # Build adjacency list sets for symmetry
#     adj = [set() for _ in range(nnodes)]
#     for i in range(nnodes):
#         deg = graph[i, 0]
#         for j in range(deg):
#             nbr = graph[i, j+1]
#             adj[i].add(nbr)
#             adj[nbr].add(i)  # make symmetric

#     # Convert back to padded matrix
#     sym_graph = -np.ones((nnodes, maxDeg+1), dtype=int)
#     for i in range(nnodes):
#         neighbors = sorted(adj[i])
#         sym_graph[i, 0] = len(neighbors)
#         sym_graph[i, 1:len(neighbors)+1] = neighbors

#     return sym_graph

@njit
def symmetrize_graph(graph):
    nnodes = graph.shape[0]
    maxDeg = graph.shape[1] - 1

    # ----- Build inbound adjacency (CSR-like) -----
    indeg = np.zeros(nnodes, np.int64)
    total_edges = 0
    for i in range(nnodes):
        d = graph[i, 0]
        total_edges += d
        for j in range(d):
            indeg[graph[i, j + 1]] += 1

    offsets = np.empty(nnodes + 1, np.int64)
    offsets[0] = 0
    for i in range(nnodes):
        offsets[i + 1] = offsets[i] + indeg[i]

    inv = np.empty(total_edges, np.int64)
    fill = offsets.copy()
    for i in range(nnodes):
        d = graph[i, 0]
        for j in range(d):
            nbr = graph[i, j + 1]
            inv[fill[nbr]] = i
            fill[nbr] += 1

    # ----- First pass: count unique degree (timestamp trick) -----
    seen = np.full(nnodes, -1, np.int64)
    sym_deg = np.zeros(nnodes, np.int64)
    for i in range(nnodes):
        cnt = 0
        # out-neighbors
        d = graph[i, 0]
        for j in range(d):
            nbr = graph[i, j + 1]
            if seen[nbr] != i:
                seen[nbr] = i
                cnt += 1
        # inbound neighbors
        s, e = offsets[i], offsets[i + 1]
        for p in range(s, e):
            src = inv[p]
            if seen[src] != i:
                seen[src] = i
                cnt += 1
        sym_deg[i] = cnt
    # Check max degree constraint
    for i in range(nnodes):
        if sym_deg[i] > maxDeg:
            raise ValueError("Symmetrized graph has larger degree than original maxDeg")

    # ----- Allocate output with original maxDeg (no resize) -----
    sym_graph = np.full((nnodes, maxDeg + 1), -1, dtype=graph.dtype)

    # ----- Second pass: write neighbors in existing order (no sort) -----
    for i in range(nnodes):
        mark = i + nnodes  # different epoch from first pass
        w = 0

        # write out-neighbors first (preserve given order)
        d = graph[i, 0]
        for j in range(d):
            nbr = graph[i, j + 1]
            if seen[nbr] != mark:
                seen[nbr] = mark
                sym_graph[i, 1 + w] = nbr
                w += 1

        # then inbound neighbors (order of inv slice)
        s, e = offsets[i], offsets[i + 1]
        for p in range(s, e):
            src = inv[p]
            if seen[src] != mark:
                seen[src] = mark
                sym_graph[i, 1 + w] = src
                w += 1

        sym_graph[i, 0] = w  # by assumption w <= maxDeg

    return sym_graph

# @njit
# def update_graph(graph, adds, dels, indices):
#     nnodes = graph.shape[0]
#     if len(indices) != nnodes:
#         raise ValueError("Length of indices must match number of nodes in graph")
#     maxDeg = graph.shape[1] - 1

#     # O(1) membership with “epoch” trick — no clearing between iterations.
#     # mark_seen[v] == epoch  => v currently a neighbor
#     # mark_del[v] == epoch   => v requested to be deleted this iteration
#     mark_seen = np.zeros(nnodes, dtype=np.int64)
#     mark_del  = np.zeros(nnodes, dtype=np.int64)

#     # Scratch buffers reused every iteration
#     kept = np.empty(maxDeg, dtype=graph.dtype)

#     for ii in range(nnodes):
#         row = indices[ii]
#         epoch = ii + 1  # unique per-iteration tag; safe and fast

#         # Mark current neighbors
#         deg = graph[row, 0]
#         for j in range(deg):
#             v = graph[row, j + 1]
#             mark_seen[v] = epoch

#         # Mark deletions
#         d = dels[ii]
#         for j in range(d.size):
#             mark_del[d[j]] = epoch

#         # Keep neighbors not marked for deletion
#         t = 0
#         for j in range(deg):
#             v = graph[row, j + 1]
#             if mark_del[v] != epoch:
#                 kept[t] = v
#                 t += 1

#         # Add new neighbors if not already present
#         a = adds[ii]
#         for j in range(a.size):
#             v = a[j]
#             if mark_seen[v] != epoch:  # not already a neighbor
#                 if t < maxDeg:
#                     kept[t] = v
#                     t += 1
#                     mark_seen[v] = epoch
#                 # else: silently drop extras (or raise if you prefer)

#         # Write back
#         graph[row, 0] = t
#         if t > 0:
#             graph[row, 1:1+t] = kept[:t]

#     return graph


# @torch.compile
# def symmetrize_graph(graph, device="cpu"):
#     device = graph.device
#     nnodes = graph.shape[0]
#     maxDeg = graph.shape[1] - 1

#     degs = graph[:, 0]
#     neighs = graph[:, 1:]

#     # Mask valid neighbors
#     col_mask = torch.arange(maxDeg, device=device).expand(nnodes, maxDeg) < degs[:, None]
#     src = torch.arange(nnodes, device=device).repeat_interleave(degs)
#     dst = neighs[col_mask]

#     # Add reversed edges
#     src_all = torch.cat([src, dst])
#     dst_all = torch.cat([dst, src])

#     # deduplication
#     edges = torch.stack((src_all, dst_all), dim=1)
#     edges = edges[torch.argsort(edges[:, 0] * nnodes + edges[:, 1])]  # sort by (src, dst)
#     keep = torch.ones(edges.size(0), dtype=torch.bool, device=device)
#     keep[1:] = (edges[1:] != edges[:-1]).any(dim=1)
#     edges = edges[keep]

#     src_sorted, dst_sorted = edges[:, 0], edges[:, 1]

#     # Count new degrees
#     degs = torch.bincount(src_sorted, minlength=nnodes)
#     if degs.max() > maxDeg:
#         raise ValueError("Symmetrized graph has larger degree than original maxDeg")

#     # Compute rank within each src group
#     # rank is the index of each edge within its source node's edge list
#     starts = torch.cumsum(torch.cat([torch.tensor([0], device=device), degs[:-1]]), 0)
#     ranks = torch.arange(len(edges), device=device) - starts[src_sorted]

#     # Build output
#     sym_graph = torch.full((nnodes, maxDeg + 1), -1, dtype=graph.dtype, device=device)
#     sym_graph[:, 0] = degs
#     sym_graph[src_sorted, ranks + 1] = dst_sorted

#     return sym_graph.cpu().numpy()

def is_symmetric_graph(graph):
    nnodes = graph.shape[0]

    for i in range(nnodes):
        deg = graph[i, 0]
        for j in range(deg):
            nbr = graph[i, j+1]
            if nbr < 0:
                continue
            # check if i is in nbr's neighbor list
            nbr_deg = graph[nbr, 0]
            if i not in graph[nbr, 1:nbr_deg+1]:
                return False
    return True


def graph_to_adjlist(graph, graphweights=None):
    nnodes = graph.shape[0]
    adjlist = []

    if graphweights is None:
        for i in range(nnodes):
            deg = graph[i, 0]
            neighbors = []
            for j in range(deg):
                nbr = graph[i, j + 1]
                neighbors.append(nbr)
            adjlist.append(tuple(neighbors))
    else:
        for i in range(nnodes):
            deg = graph[i, 0]
            neighbors_weights = []
            for j in range(deg):
                nbr = graph[i, j + 1]
                weight = graphweights[i, j + 1]
                neighbors_weights.append((nbr, weight))
            adjlist.append(tuple(neighbors_weights))

    return adjlist

def adaptive_halo_expansion(graph, rho, thresh, nnodes, maxDeg, indicesCoreHalos, indicesCore, hindex, coords, latticeVectors, nl, alpha=0.7):
    """
    Adaptively expanding the size of halo regions by multiplying the 
    overlap matrix (estimated from exponential decay of neighboring distances) 
    from the out of core halo regions with the density matrix in the core halo regions.
    Dimension of the overlap matrix: Number of atoms (NA) in whole system/non core halo regions x NA in core halo regions.
    Dimension of the density matrix: Number of orbitals (NO) in core halo regions x NO in core halo regions
    Dimension of the reducted density matrix: NA in core halo regions x NA in core regions.
    Dimension of the SD matrix: NA in whole system/non core halo regions x NA in core regions.
    This function will return a new graph with the updated halo regions.

    Parameters
    ----------
    graph : np.ndarray
        The graph in 2D numpy array where `graph[i,k]` is the kth neighbor
        of node i. NOTE: The 0 entry of every row is reserved to store the degree of every node.
    rho : np.ndarray
        Density matrix from the subgraph. This is a 2D numpy array.
    thresh : float
        Threshold to determine significant connections.
    nnodes : int
        Number of nodes of the full graph.
    maxDeg : int
        Max degree parameter for the full graph.
    indicesCoreHalos : list
        List of core+halo indices.
    indicesCore : list
        List of core indices.
    hindex : list
        A list of displacements mapping every node in the full graph
        with a sequence of indices (orbitals) in the full density matrix.
        The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`.
    coords : np.ndarray
        Coordinates of the atoms in the system.
    latticeVectors : np.ndarray
        Lattice vectors of the system.
    nl : np.ndarray
        Neighbor list for the atoms in the system.
    alpha : float, optional
        Decay parameter for the overlap matrix, by default 0.7.
    expandonly : bool, optional
        If True, only expand the halo regions without modifying previous core+halo regions, by default True.
    
    Returns
    -------
    np.ndarray
        Updated graph with the new halo regions.
    """
    if coords is None or rho is None:
        raise ValueError("Coordinates and density matrix must be provided.")

    if graph is None:
        graph = np.zeros((nnodes, maxDeg + 1), dtype=int)
    weights = np.zeros((nnodes))
    ncores = len(indicesCore)
    nch = len(indicesCoreHalos)

    # Get the coordinates for the core halo regions
    corehalo_coords = coords[indicesCoreHalos]
    # Get the indices for the non core halo regions
    nonCoreHalo_indices = np.setdiff1d(np.arange(nnodes), indicesCoreHalos)
    # Identify the neighboring atoms within a certain cutoff distance
    # This will help to reduce the computational cost of the overlap matrix
    nl = nl[indicesCoreHalos]
    neighbor_indices = np.unique(nl[nl >= 0])
    nonCoreHalo_indices = np.intersect1d(neighbor_indices, nonCoreHalo_indices)
    if len(nonCoreHalo_indices) == 0:
        return graph
    coords = coords[nonCoreHalo_indices]
    # Initialize the overlap matrix with zeros
    overlap_matrix = np.zeros((coords.shape[0], corehalo_coords.shape[0]), dtype=float)
    # Calculate the distance between core and core halo regions with vectorized operations
    # Considering periodic boundary conditions
    delta = coords[:, np.newaxis] - corehalo_coords
    LBox = latticeVectors.diagonal()
    delta = delta - LBox[np.newaxis, np.newaxis, :] * np.round(delta / LBox[np.newaxis, np.newaxis, :])
    distances = np.linalg.norm(delta, axis=2)
    # Estimate the overlap matrix based on the distances
    overlap_matrix = np.exp(-alpha * distances ** 2)  # Exponential decay based on distance
    # overlap_matrix = np.where(overlap_matrix > thresh, overlap_matrix, 0)

    # Contract the density matrix from number of orbitals to number of atoms by selecting the max # density matrix elements for each atom.
    # hindex is the number of orbitals for each atom, so we can use it to slice the density matrix
    if hindex is None:
        raise ValueError("hindex must be provided to slice the density matrix.")
    if rho.shape[0] != hindex[-1]:
        raise ValueError("Density matrix shape does not match hindex length.")
    rho = np.abs(rho)  # Ensure the density matrix is non-negative
    # Create a reduced density matrix for the core halo regions
    reduced_rho = np.zeros((nch, ncores), dtype=float)
    # Vectorized max pooling to get the reduced density matrix
    reduced_rho[:] = np.maximum.reduceat(
                    np.maximum.reduceat(rho, hindex[:-1], axis=0),
                    hindex[:-1], axis=1
                )[:nch, :ncores]
    # reduced_rho = np.where(reduced_rho > thresh, reduced_rho, 0)
    # Matrix multiplication to get the new halo regions
    SD = overlap_matrix @ reduced_rho
    # assign indices
    indices = nonCoreHalo_indices 
    # Thresholding the SD matrix to get the new halo regions
    for i in range(ncores):
        ii = indicesCoreHalos[i]
        # Recovering the connections we already have
        weights[:] = 0.0
        if graph[ii, 0]:
            weights[graph[ii, 1:graph[ii, 0] + 1]] = thresh
        # Assign the weights
        weights[indices] += SD[:, i]
        # Vectorized selection of candidates above the threshold and not including self loop
        selected = np.where(weights >= thresh)[0]
        # selected = selected[selected != ii]
        k = len(selected)

        # Degree guard 
        if k > maxDeg:
            msg = (
                f"Max Degree parameter is too small, maxDeg: {maxDeg} "
                f"ActuallDeg: {k}"
            )
            raise ValueError(msg)

        # Write back neighbors
        graph[ii, 1:k + 1] = selected
        graph[ii, 0] = k
        graph[ii, k + 1:] = -1  # Fill the rest with -1s

    return graph


@njit
def compute_added(G1, G2, NNZ1, NNZ2, nnodes, maxToAddRemove=100):
    """
    Compute, for each vertex i, which neighbors in G2[i, :] are NOT already in G1[i, :].

    Parameters
    ----------
    G1 : numpy 2D array 
        Adjacency list (row i has up to NNZ1[i] valid entries).
    G2 : numpy 2D array 
        Second adjacency list (row i has up to NNZ2[i] valid entries).
    NNZ1 : numpy 1D array
        Counts per row for G1.
    NNZ2 : numpy 1D array
        Counts per row for G2.
    nnodes : int
        Number of nodes in the graph.
    maxToAddRemove : int, optional
        Maximum number of neighbors to add per row, by default 100.

    Returns
    -------
    G_added : numpy 2D array
        For each row i, the neighbors from G2 not in G1, packed in columns [0:N_added[i]).
    N_added : numpy 1D array
        Number of added neighbors per row.
    """
    N = G1.shape[0]
    M = G1.shape[1] # max possible new neighbors per row
    G_added = np.zeros((N, maxToAddRemove), dtype=G2.dtype)
    N_added = np.zeros(N, dtype=np.int64)

    # marker vector for membership checks
    v = np.zeros(nnodes, dtype=np.uint8)

    for i in range(N):
        # mark all G1 neighbors of i
        for j in range(NNZ1[i]):
            v[G1[i, j]] = 1

        # collect G2 neighbors not in G1
        k = 0
        for j in range(NNZ2[i]):
            b = G2[i, j]
            if v[b] == 0:
                G_added[i, k] = b
                k += 1
            if k > maxToAddRemove:
                break

        N_added[i] = k

        # clear marks for the next row
        for j in range(NNZ1[i]):
            v[G1[i, j]] = 0
        for j in range(NNZ2[i]):
            v[G2[i, j]] = 0

    return G_added, N_added

@njit
def compute_removed(G1, G2, NNZ1, NNZ2, nnodes, maxToAddRemove=100):
    """
    Compute, for each vertex i, which neighbors in G1[i, :] that are not in G2[i, :].

    Parameters
    ----------
    G1 : numpy 2D array 
        Adjacency list (row i has up to NNZ1[i] valid entries).
    G2 : numpy 2D array 
        Second adjacency list (row i has up to NNZ2[i] valid entries).
    NNZ1 : numpy 1D array
        Counts per row for G1.
    NNZ2 : numpy 1D array
        Counts per row for G2.
    nnodes : int
        Number of nodes in the graph.
    maxToAddRemove : int, optional
        Maximum number of neighbors to remove per row, by default 100.

    Returns
    -------
    G_removed : numpy 2D array
        For each row i, the neighbors from G1 not in G2, packed in columns [0:N_added[i]).
    N_removed : numpy 1D array
        Number of removed neighbors per row.
    """
    N = G1.shape[0]
    M = G1.shape[1]
    G_removed = np.zeros((N, maxToAddRemove), dtype=G1.dtype)
    N_removed = np.zeros(N, dtype=np.int64)

    # marker vector for membership checks
    v = np.zeros(nnodes, dtype=np.uint8)

    for i in range(N):
        # mark all G2 neighbors of i
        for j in range(NNZ2[i]):
            v[G2[i, j]] = 1

        # collect G1 neighbors not in G2
        k = 0
        for j in range(NNZ1[i]):
            b = G1[i, j]
            if v[b] == 0:
                G_removed[i, k] = b
                k += 1
            if k > maxToAddRemove:
                break

        N_removed[i] = k

        # clear marks for the next row
        for j in range(NNZ1[i]):
            v[G1[i, j]] = 0
        for j in range(NNZ2[i]):
            v[G2[i, j]] = 0

    return G_removed, N_removed

# Use G_removed and G_added to update from G1 to G2
@njit
def update_graph(G1, NNZ1, G_removed, N_removed, G_added, N_added):
    """
    Update G1 by removing neighbors in G_removed and adding neighbors in G_added.

    Parameters
    ----------
    G1 : int64[:, :]
        Adjacency list to be updated (row i has up to NNZ1[i] valid entries).
    G_removed : int64[:, :]
        Neighbors to be removed from G1 (row i has up to N_removed[i] valid entries).
    N_removed : int64[:]
        Number of neighbors to remove per row.
    G_added : int64[:, :]
        Neighbors to be added to G1 (row i has up to N_added[i] valid entries).
    N_added : int64[:]
        Number of neighbors to add per row.

    Returns
    -------
    G_updated : int64[:, :]
        Updated adjacency list.
    """
    N, maxDeg = G1.shape
    G_updated = np.full((N, maxDeg + 1), -1, dtype=G1.dtype)
    NNZ_updated = np.zeros(N, dtype=np.int64)

    for i in range(N):
        # thread-local marker vector (safe in parallel)
        v = np.zeros(N, dtype=np.uint8)

        # 1) mark existing neighbors
        nnz1 = NNZ1[i]
        for j in range(nnz1):
            v[G1[i, j]] = 1

        # 2) unmark neighbors to be removed
        nrem = N_removed[i]
        for j in range(nrem):
            v[G_removed[i, j]] = 0

        # 3) keep remaining neighbors
        cnt = 0
        for j in range(nnz1):
            b = G1[i, j]
            if v[b] == 1:
                G_updated[i, cnt + 1] = b
                cnt += 1

        # 4) add new neighbors
        nadd = N_added[i]
        NNZ_updated[i] = cnt + nadd
        for j in range(nadd):
            G_updated[i, cnt + j + 1] = G_added[i, j]

        G_updated[i, 0] = NNZ_updated[i]
        
        # 5) sort the neighbor list
        if NNZ_updated[i] > 1:
            neighbors = G_updated[i, 1:NNZ_updated[i] + 1]
            neighbors.sort()
            G_updated[i, 1:NNZ_updated[i] + 1] = neighbors

    return G_updated


def graph_diff_and_update(prevGraph, graphOnRank, partsOnRank, comm, maxToAddRemove=100):
    nnodes = prevGraph.shape[0]
    
    # Initialize added, removed, and updated graph
    G_added = np.zeros((nnodes, maxToAddRemove + 1), dtype=prevGraph.dtype)
    G_removed = np.zeros((nnodes, maxToAddRemove + 1), dtype=prevGraph.dtype)
    
    # Get the local graph on this rank
    localGraph = prevGraph[partsOnRank]
    localGraph_new = graphOnRank[partsOnRank].copy()

    # Compute added and removed neighbors
    G_added[partsOnRank, 1:], G_added[partsOnRank, 0] = compute_added(localGraph[:, 1:], localGraph_new[:, 1:], localGraph[:, 0], localGraph_new[:, 0], nnodes, maxToAddRemove=maxToAddRemove)
    G_removed[partsOnRank, 1:], G_removed[partsOnRank, 0] = compute_removed(localGraph[:, 1:], localGraph_new[:, 1:], localGraph[:, 0], localGraph_new[:, 0], nnodes, maxToAddRemove=maxToAddRemove)

    local_full_collection = False
    if max(G_added[:,0]) > maxToAddRemove or max(G_removed[:,0]) > maxToAddRemove:
        #raise ValueError("maxToAddRemove is too small to accommodate the number of added/removed neighbors.")
        print("maxToAddRemove is too small to accommodate the number of added/removed neighbors.\n")
        print("Do full collection instead\n")
        local_full_collection = True
    
    # Use logical OR reduction to see if ANY rank has True
    global_full_collection = comm.allreduce(1 if local_full_collection else 0, op=MPI.SUM) > 0 

    if global_full_collection:
        print("Collecting the full graph")  
        fullGraph = collect_and_sum_matrices_float(graphOnRank, comm)
        return fullGraph

    # MPI Allreduce to gather added and removed neighbors across all ranks
    comm.Allreduce(MPI.IN_PLACE, G_added, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, G_removed, op=MPI.SUM)

    nvtx.push_range("update_graph test")
    # Update the graph using the added and removed neighbors
    updatedGraph = update_graph(prevGraph[:, 1:], prevGraph[:, 0], G_removed[:, 1:], G_removed[:, 0], G_added[:, 1:], G_added[:, 0])
    nvtx.pop_range()

    return updatedGraph