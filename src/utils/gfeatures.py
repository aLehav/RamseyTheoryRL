import igraph as ig
import itertools
import concurrent.futures
import os
import multiprocessing
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils.guseful import *

# Since structures is read only, global initialization is fine
structures = {
    "K_{1,3}": ig.Graph.Full_Bipartite(1, 3),
    "P_4": ig.Graph(n=4, edges=[(i, i+1) for i in range(3)]),  # Path graph
    "C_4": ig.Graph.Ring(4),  # Cycle graph
    "K_4": ig.Graph.Full(4),  # Complete graph
}

# Add K_3+e
K_3_e = ig.Graph.Full(3)
K_3_e.add_vertex()
K_3_e.add_edge(2, 3)
structures["K_3+e"] = K_3_e

# Add K_4-e
K_4_e = ig.Graph.Full(4)
K_4_e.delete_edges((0, 1))
structures["K_4-e"] = K_4_e

# Add K_3
K_3 = ig.Graph.Full(3)
K_3.add_vertex()
structures["K_3"] = K_3

# Add P_3
# Path graph with 3 nodes + isolated node
P_3 = ig.Graph(n=4, edges=[(i, i+1) for i in range(2)])
structures["P_3"] = P_3

# Add K_2
K_2 = ig.Graph(n=4, edges=[(0, 1)])  # Edge + 2 isolated nodes
structures["K_2"] = K_2

# Add 2K_2
_2K_2 = ig.Graph(n=4, edges=[(0, 1), (2, 3)])  # Two edges + isolated nodes
structures["2K_2"] = _2K_2

# Add E_4
E_4 = ig.Graph(n=4)  # 4 isolated nodes
structures["E_4"] = E_4


# Count subgraph structures O(n^4)
def count_subgraph_structures(G):
    counters = {name: 0 for name in structures}
    for nodes in itertools.combinations(range(G.vcount()), 4):
        subgraph = G.subgraph(nodes)
        for name, structure in structures.items():
            if subgraph.isomorphic(structure):
                counters[name] += 1
    return counters

# Count subgraphs from edge O(n^2)
def count_subgraphs_from_edge(G, u, v):
    counters = {name: 0 for name in structures}
    nodes = set(range(G.vcount())) - {u, v}
    for node1, node2 in itertools.combinations(nodes, 2):
        subgraph = G.subgraph([u, v, node1, node2])
        for name, structure in structures.items():
            if subgraph.isomorphic(structure):
                counters[name] += 1
    return counters

# Update feature from edge
def update_feature_from_edge(G, u, v, counters):
    old_count = count_subgraphs_from_edge(G, u, v)
    # Change edge
    change_edge(G, (u,v))
    new_count = count_subgraphs_from_edge(G, u, v)
    # Make and edit a copy of counters such that we can pass it to all children
    new_counters = {}
    for name in counters:
        new_counters[name] = counters[name] + (new_count[name] - old_count[name])
    return new_counters

# Par
def count_subgraphs_from_edge_parBfs(G, u, v, to_change):
    counters = {name: 0 for name in structures}
    nodes = set(range(G.vcount())) - {u, v}
    for node1, node2 in itertools.combinations(nodes, 2):
        subgraph = G.subgraph([u, v, node1, node2])
        if to_change:
          if subgraph.are_connected(u, v):
              subgraph.delete_edges([(u, v)])
          else:
              subgraph.add_edge(u, v)
        for name, structure in structures.items():
            if subgraph.isomorphic(structure):
                counters[name] += 1
    return counters

# Par
def update_feature_from_edge_parBfs(G, u, v, counters):
    old_count = count_subgraphs_from_edge(G, u, v, False)
    # Change edge
    new_count = count_subgraphs_from_edge(G, u, v, True)
    # Make and edit a copy of counters such that we can pass it to all children
    new_counters = {}
    for name in counters:
        new_counters[name] = counters[name] + \
            (new_count[name] - old_count[name])
    return new_counters