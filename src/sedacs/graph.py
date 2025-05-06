"""graph
Some graph functions

"""

import sys

import numpy as np
import torch

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
def get_initial_graph(coords, nl, radius, maxDeg, verb=False):
    nats = len(coords[:, 0])
    graph = np.zeros((nats, maxDeg + 1), dtype=int)
    graph[:, :] = -1
    for i in range(nats):
        ik = 0
        degi = 0
        for j in range(1, nl[i, 0] + 1):
            jj = nl[i, j]

            distance = np.linalg.norm(coords[i, :] - coords[jj, :])
            if distance < radius:
                ik = ik + 1
                if ik < maxDeg + 1:
                    graph[i, ik] = jj
                    degi = degi + 1
                    # if i == 0:
                    #     print(nl[i,j])
                else:
                    print("!!!WARNING: at get_initial_graph. maxDeg exceeded. Consider increasing this number")
                    break

        graph[i, 0] = degi  # Storing the degrees

    return graph


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
    for i in range(0, n):
        nxGraph.add_nodes_from([i, i])
        for k in range(1, graph[i, 0] + 1):
            j = graph[i, k]
            if (j != -1) and (j != i):
                nxGraph.add_edge(i, j, weight=w)

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
    rhoDim = len(rho[:, 0])
    if graph is None:
        graph = np.zeros((nnodes, maxDeg + 1), dtype=int)
    weights = np.zeros((nnodes))
    nch = len(indicesCoreHalos)
    ki = 0

    for i in range(ncores):
        ii = indicesCoreHalos[i]
        # Recovering the connections we already have
        weights[:] = 0.0
        for j in range(1, graph[ii, 0]):
            jj = graph[ii, j]
            weights[jj] = thresh

        # Computing the new weights by rho
        for oi in range(hindex[ii], hindex[ii + 1]):
            kj = 0
            for j in range(nch):
                jj = indicesCoreHalos[j]
                for oj in range(hindex[jj], hindex[jj + 1]):
                    if abs(rho[ki, kj]) >= thresh: # Elementwise truncation test Anders
                        weights[jj] = weights[jj] + abs(rho[ki, kj])
                    kj = kj + 1
            ki = ki + 1

        # Reasigning the connections to ii by the merged weights (the ones computed
        # from rho and the ones already existing.
        k = 0
        for jj in range(nnodes):  # $$$ ??? this cycle could be interrupted ???
            if (ii != jj) and (weights[jj] >= thresh):
                k = k + 1
                if k >= maxDeg + 1:
                    raise ValueError(f"Max Degree parameter is too small: {maxDeg}")
                graph[ii, k] = jj

        graph[ii, 0] = k

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
        for j in range(nnodes):
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
