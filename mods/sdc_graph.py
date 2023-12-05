"""graph
Some graph functions
 
"""
import numpy as np
global nxLib
try:
    import networkx as nx
    nxLib = True
except ImportError as e:
    nxLib = False
global pltLib
try: import matplotlib.pyplot as plt ; pltLib = True
except: pltLib = False


## Get an initial graph based on distance
# @brief This will give a graph based on distaces. Similar to
# a neighbor list.
# @param coords System coordinates
# @param nl Neighbor list `nl[i,0]` = total number of neighbors.
# `nl[i,1:nl[i,0]]` = neigbors of i.
# @param radius Radius Cutoff to search for the neighbors
# @param maxDeg Max degrees allowed for each none
# @param verb Verbosity mode
# @return graph The graph consisting on a 2D integer numpy array. 
# E.g, `graph[i,k]` is the kth neighbor of node i
#
def get_initial_graph(coords,nl,radius,maxDeg,verb=False):
    nats = len(coords[:,0])
    graph = np.zeros((nats,maxDeg),dtype=int)
    graph[:,:] = -1
    for i in range(nats):
        ik = -1
        print("atom",i)
        for j in range(nl[i,0]):
            jj = nl[i,j]
            distance = np.linalg.norm(coords[i,:] - coords[jj,:])
            if(distance < radius):
                ik = ik + 1
                if(ik < maxDeg):
                    graph[i,ik] = jj
                else:
                    print("WARNING: at get_initial_graph. maxDeg exceeded. Consider increasing this number")
                    break

    return graph

## Print graph
# @brief Print the graph by showing the connection of every node.
# @param graph The graph consisting on a 2D numpy array. 
# E.g, graph[i,k] is the kth neighbor of node i. 
#
def print_graph(graph):
    nnodes = len(graph[:,0])
    mDeg = len(graph[0,:])
    print("\nGraph structure:")
    for i in range(nnodes):
        nodesList = []
        for k in range(mDeg):
            if(graph[i,k] != -1):
                nodesList.append(graph[i,k])
        print(i,"-",nodesList)

## Get a networkX graph
# @param graph The graph in 2D numpy array where `graph[i,k]` is the kth neighbor
# of node i
# @param w The weight for the edges (tipically = 1.0)
# @return nxGraph networkX type of graph
#
def get_nx_graph(graph,w):
    if(nxLib == False):
        sdc_error("get_nx_graph","ERROR: Consider installing networkx")
    n = len(graph[:,0])
    m = len(graph[0,:])
    nxGraph = nx.Graph()
    for i in range(0,n):
        nxGraph.add_nodes_from([i,i])
        for k in range(0,m):
            j = graph[i,k]
            if((j != -1) and (j != i)):
                nxGraph.add_edge(i,j,weight=w)
    return nxGraph


## To plot the resulting graph
# @brief Uses matplotlib to plot the nxGraph
# @param nxGraph NetworkX type of graph
# @param nodeColor The color of the nodes
# This will produce a "graph.png" file with the 
# plot.
def plot_nx_graph(nxGraph,nodeColor='r'):
    labels = []
    n = nxGraph.number_of_nodes()
    for i in range(n):
        labels.append(str(i))
    pos1 = nx.spring_layout(nxGraph,scale=100.0,weight='weight')
    plt.figure(figsize=(30.0, 30.0))
    nx.draw_networkx_nodes(nxGraph, pos1,node_size=500,alpha=0.8,node_color=nodeColor)
    nx.draw_networkx_edges(nxGraph, pos1,width=1, alpha=0.5, edge_color='r')
    nx.draw_networkx_labels(nxGraph, pos1, font_size=16)
    plt.savefig("graph.png", dpi=400)
    plt.show()

