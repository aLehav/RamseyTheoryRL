from multiprocessing import Pool, Manager
import itertools
import igraph as ig


def is_complete(G):
    n = G.vcount()
    return G.ecount() == n * (n - 1) // 2


def check_independent_set(args):
    G, sub_nodes = args
    subgraph = G.subgraph(sub_nodes)
    return subgraph.ecount() == 0

# Parallelize


def has_independent_set_of_size_k(G, k):
    with Pool() as pool:
        args = [(G, sub_nodes)
                for sub_nodes in itertools.combinations(range(G.vcount()), k)]
        for result in pool.imap_unordered(check_independent_set, args):
            if result:
                return True
    return False


def check_complete(args):
    G, sub_nodes = args
    subgraph = G.subgraph(sub_nodes)
    return is_complete(subgraph)

# Parallelize


def has_kn(G, k):
    with Pool() as pool:
        args = [(G, sub_nodes)
                for sub_nodes in itertools.combinations(range(G.vcount()), k)]
        for result in pool.imap_unordered(check_complete, args):
            if result:
                return True
    return False


# Testing
G1 = ig.Graph.Full(5)
G2 = ig.Graph.Erdos_Renyi(n=5, m=10)

print(has_independent_set_of_size_k(G1, 3))  # Should print True
# May print True or False based on random graph generated
print(has_independent_set_of_size_k(G2, 3))

print(has_kn(G1, 3))  # Should print True
print(has_kn(G2, 3))  # May print True or False based on random graph generated
