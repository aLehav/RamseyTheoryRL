import random
import networkx as nx
import itertools

# Since structures is read only, global initialization is fine
structures = {
    "K_{1,3}": nx.complete_bipartite_graph(1, 3),
    "P_4": nx.path_graph(4),
    "C_4": nx.cycle_graph(4),
    "K_4": nx.complete_graph(4),
}
# Add K_3+e
K_3_e = nx.complete_graph(3)
K_3_e.add_edge(2, 3)
structures["K_3+e"] = K_3_e

# Add K_4-e
K_4_e = nx.complete_graph(4)
K_4_e.remove_edge(0, 1)
structures["K_4-e"] = K_4_e

# Add K_3
K_3 = nx.complete_graph(3)
K_3.add_node(3)
structures["K_3"] = K_3

# Add P_3
P_3 = nx.path_graph(3)
P_3.add_node(3)
structures["P_3"] = P_3

# Add K_2
K_2 = nx.Graph()
K_2.add_nodes_from(range(4))
K_2.add_edge(0, 1)
structures["K_2"] = K_2

# Add 2K_2
_2K_2 = nx.Graph()
_2K_2.add_nodes_from(range(4))
_2K_2.add_edge(0, 1)
_2K_2.add_edge(2, 3)
structures["2K_2"] = _2K_2

# Convert each structure graph to a canonical hash for comparison
for name, graph in structures.items():
    structures[name] = nx.weisfeiler_lehman_graph_hash(graph)

# Count number of occurences of features in table 2 of paper for every 4 subgraph
# O(n^4)
def count_subgraph_structures(G):
    # Initialize a counter for each structure
    counters = {name: 0 for name in structures}

    # Enumerate all 4-node subgraphs
    for nodes in itertools.combinations(G.nodes, 4):
        subgraph = nx.subgraph(G, nodes)

        # Compute the canonical hash of the subgraph
        hash = nx.weisfeiler_lehman_graph_hash(subgraph)

        # If the hash matches any of the structures, increment the corresponding counter
        for name, structure_hash in structures.items():
            counters[name] += (hash == structure_hash)

    return counters


# O(n^2) Counts graph structures for 4-subgraph with nodes u,v
def count_subgraphs_from_edge(G, u, v):
    # Initialize a counter for each structure
    counters = {name: 0 for name in structures}

    # Enumerate all 2-node subgraphs
    nodes = set(G.nodes) - {u, v}  # all nodes except u and v
    for node1, node2 in itertools.combinations(nodes, 2):
        # process pair of nodes node1 and node2
        subgraph = nx.subgraph(G, [u, v, node1, node2])
        # Compute the canonical hash of the subgraph
        hash = nx.weisfeiler_lehman_graph_hash(subgraph)

        # If the hash matches any of the structures, increment the corresponding counter
        for name, structure_hash in structures.items():
            counters[name] += (hash == structure_hash)

    return counters

# Assuming feature vector is a dictionary of feature counts for now.
# count = All subgraph counts without e(u,v)
# ncount = All subgraph counts with e(u,v).
# Updates feature vector by ncount - count for each field
def update_feature_from_edge(G, u, v, counters):
    count = count_subgraphs_from_edge(G, u, v)
    G.add_edge(u, v)
    ncount = count_subgraphs_from_edge(G, u, v)
    for name in counters:
        counters[name] += (ncount[name] - count[name])
