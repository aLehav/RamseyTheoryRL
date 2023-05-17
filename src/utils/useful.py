import itertools
import networkx as nx


def is_complete(G):
    n = G.number_of_nodes()
    return G.number_of_edges() == n * (n - 1) // 2

# Parallelize the generator 0_0
def has_independent_set_of_size_k(G, k):
    for sub_nodes in itertools.combinations(G.nodes, k):
        subgraph = G.subgraph(sub_nodes)
        if subgraph.number_of_edges() == 0:
            return True
    return False


def has_kn(G, k):
    for sub_nodes in itertools.combinations(G.nodes, k):
        subgraph = G.subgraph(sub_nodes)
        if is_complete(subgraph):
            return True
    return False
