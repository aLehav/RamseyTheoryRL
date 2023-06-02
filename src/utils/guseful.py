import itertools
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys

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

def change_edge(G, e):
    G.delete_edges([e]) if G.are_connected(*e) else G.add_edge(*e)

def are_graphs_isomorphic(G1, G2):
    return nx.is_isomorphic(G1, G2)

def consider_counterexample(G, counters, counter_path):
    nx_graph = nx.Graph(G.get_edgelist())
    is_unique = True
    for counter in counters:
        if are_graphs_isomorphic(nx_graph, counter):
            is_unique = False
            sys.stdout.write('\033[1m\033[92mCounterexample found but not unique.\033[0m\n')
            break
    if is_unique:
        sys.stdout.write('\033[1m\033[96mCounterexample found and added.\033[0m\n')
        counters.append(nx_graph)
        with open(counter_path, 'a') as file:
            file.write(nx.to_graph6_bytes(nx_graph, header=False).decode('ascii'))