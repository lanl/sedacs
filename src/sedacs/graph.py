"""graph
Some graph functions

"""

import sys

import numpy as np

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
# `nl[i,1:nl[i,0]]` = neigbors of i. Self neighbor i to i is not included explicitly.
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
                nodesList.append(graph[i, k])
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
def collect_graph_from_rho(graph, rho, thresh, nnodes, maxDeg, indices, hindex=None, verb=False):
    rhoDim = len(rho[:, 0])
    if graph is None:
        graph = np.zeros((nnodes, maxDeg + 1), dtype=int)
    nats = len(indices)
    weights = np.zeros((nnodes))
    ki = 0
    for i in range(nats):
        ii = indices[i]
        # Recovering the connections we already have
        weights[:] = 0.0
        for j in range(1, graph[ii, 0]):
            jj = graph[ii, j]
            weights[jj] = thresh

        # Computing the new weights by rho
        for oi in range(hindex[ii], hindex[ii + 1]):
            kj = 0
            for j in range(nats):
                jj = indices[j]
                for oj in range(hindex[jj], hindex[jj + 1]):
                    weights[jj] = weights[jj] + abs(rho[ki, kj])
                    kj = kj + 1
            ki = ki + 1

        # Reasigning the connections to ii by the merged weights (the ones computed
        # from rho and the ones already existing.
        k = 0
        for jj in range(nnodes):  # $$$ ??? this cycle could be interrupted ???
            if (ii != jj) and (weights[jj] >= thresh):
                k = k + 1
                graph[ii, k] = jj
                if k >= maxDeg + 1:
                    raise ValueError(f"Max Degree parameter is too small: {maxDeg}")

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


## Multiply two Adjacencies
# @brief The ij of the resulting graph will be connected
# if i in A and j in B have a common directly connected node k.
# @param graphA Initial adjacency
# @param graphB Initial adjacency
# @return graphC Multiplication result
def multiply_graphs():
    if len(graphA[:, 0]) != len(graphB[:, 0]):
        print("!!!ERROR: Graphs have different number of nodes")
    else:
        nnodes = len(graphA[:, 0])
        maxDeg = len(graphA[0, :])

    for i in range(nnodes):
        myVect[:] = False
        for j in range(1, graphA[i, 0]):
            myK = graphA[i, j]  # All neighbors of i by A
            for k in range(1, graphB[myK, 0]):
                myJ = graphB[myK, k]  # All neighbors of myK by B
                if i != myJ:
                    vectC[myJ] = True

        k = 0
        for j in range(1, nnodes):
            if vectC[j]:
                k = k + 1
                graphC[i, k] = j
        graphC[i, 0] = k


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
