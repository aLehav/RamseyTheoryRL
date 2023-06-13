import itertools
from functools import partial
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
from multiprocessing import Pool


def is_complete(G):
    n = G.vcount()
    return G.ecount() == n * (n - 1) // 2


def has_independent_set_of_size_k(G, k):
    for sub_nodes in itertools.combinations(range(G.vcount()), k):
        subgraph = G.subgraph(sub_nodes)
        if subgraph.ecount() == 0:
            return True
    return False

def has_independent_set_of_size_k_from_edge(G, k, e):
    chosen_nodes = list(e)
    other_nodes = set(range(G.vcount())) - set(chosen_nodes)
    for sub_nodes in itertools.combinations(other_nodes, k-2):
        sub_nodes = chosen_nodes + list(sub_nodes)
        subgraph = G.subgraph(sub_nodes)
        if subgraph.ecount() == 0:
            return True
    return False

def has_kn(G, k):
    for sub_nodes in itertools.combinations(range(G.vcount()), k):
        subgraph = G.subgraph(sub_nodes)
        if is_complete(subgraph):
            return True
    return False

def has_kn_from_edge(G, k, e):
    chosen_nodes = list(e)
    other_nodes = set(range(G.vcount())) - set(chosen_nodes)
    for sub_nodes in itertools.combinations(other_nodes, k-2):
        sub_nodes = chosen_nodes + list(sub_nodes)
        subgraph = G.subgraph(sub_nodes)
        if is_complete(subgraph):
            return True
    return False

# Par
def has_kn_parBfs(G, k, e):
    for sub_nodes in itertools.combinations(range(G.vcount()), k):
        subgraph = G.subgraph(sub_nodes)
        # Check if edge is in subgraph
        if all(node in sub_nodes for node in e):
          if subgraph.are_connected(e[0], e[1]):
            subgraph.delete_edges([(e[0], e[1])])
          else:
            subgraph.add_edge(e[0], e[1])
        if is_complete(subgraph):
            return True
    return False

# Par
def has_independent_set_of_size_k_parBfs(G, k, e):
    for sub_nodes in itertools.combinations(range(G.vcount()), k):
        subgraph = G.subgraph(sub_nodes)
        if all(node in sub_nodes for node in e):
          if subgraph.are_connected(e[0], e[1]):
            subgraph.delete_edges([(e[0], e[1])])
          else:
            subgraph.add_edge(e[0], e[1])
        if subgraph.ecount() == 0:
            return True
    return False


def check_counterexample(G, s, t, subgraph_counts):
    if s == 3:
        if subgraph_counts["K_4"] + subgraph_counts["K_4-e"] + subgraph_counts["K_3+e"] + subgraph_counts["K_3"] > 0: return False
    elif s == 4:
        if subgraph_counts["K_4"] > 0: return False
    else:
        if has_kn(G, s): return False

    if t == 4:
        if subgraph_counts["E_4"] > 0: return False
    else:
        if has_independent_set_of_size_k(G, t): return False
    
    return True
    # return not has_kn(G, s) and not has_independent_set_of_size_k(G, t)

def check_counterexample_from_edge(G, s, t, subgraph_counts, e, past_state):
    if s == 3:
        if subgraph_counts["K_4"] + subgraph_counts["K_4-e"] + subgraph_counts["K_3+e"] + subgraph_counts["K_3"] > 0: return False
    elif s == 4:
        if subgraph_counts["K_4"] > 0: return False
    else:
        if past_state == True:
            if has_kn_from_edge(G, s, e): return False
        else:
            if has_kn(G, s): return False

    if t == 4:
        if subgraph_counts["E_4"] > 0: return False
    else:
        if past_state == True:
            if has_independent_set_of_size_k_from_edge(G, t, e): return False
        else:
            if has_independent_set_of_size_k(G, t): return False
    
    return True

# Par
def check_counterexample_parBfs(G, s, t, e):
    return not has_kn_parBfs(G, s, e) and not has_independent_set_of_size_k_parBfs(G, t, e)


def change_edge(G, e):
    G.delete_edges([e]) if G.are_connected(*e) else G.add_edge(*e)


# Par
def are_graphs_isomorphic(G1, G2):
    return nx.is_isomorphic(G1, G2)

# Par
def consider_counterexample(G, counters, counter_path):
    nx_graph = nx.Graph(G.get_edgelist())
    is_unique = True
    for counter in counters:
        if are_graphs_isomorphic(nx_graph, counter):
            is_unique = False
            break
    if is_unique:
        sys.stdout.write(
            '\033[1m\033[96mCounterexample found and added.\033[0m\n')
        counters.append(nx_graph)
        with open(counter_path, 'a') as file:
            file.write(nx.to_graph6_bytes(
                nx_graph, header=False).decode('ascii'))
