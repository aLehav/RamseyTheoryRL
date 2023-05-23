import itertools
import igraph as ig


def is_complete(G):
    n = G.vcount()
    return G.ecount() == n * (n - 1) // 2


# TODO Parallelize
def has_independent_set_of_size_k(G, k):
    for sub_nodes in itertools.combinations(range(G.vcount()), k):
        subgraph = G.subgraph(sub_nodes)
        if subgraph.ecount() == 0:
            return True
    return False


# TODO Parallelize
def has_kn(G, k):
    for sub_nodes in itertools.combinations(range(G.vcount()), k):
        subgraph = G.subgraph(sub_nodes)
        if is_complete(subgraph):
            return True
    return False


def check_counterexample(G, s, t):
  return not has_kn(G, s) and not has_independent_set_of_size_k(G,t)