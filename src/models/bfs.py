
import timeit
import random
import networkx as nx
import sys
sys.path.append("..")
from utils.useful import *


# Given a graph G and feature counts, add a node and explore around it, terminating if found counterexample
# Still very brute forcy
def add_vertex(G):
    G.add_node(max(G.nodes) + 1)


def bfs(G):
    nodes = list(G.nodes)
    startTime = timeit.default_timer()
    while has_independent_set_of_size_k(G, 6) or has_kn(G, 4):
        u, v = random.sample(nodes, 2)
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        else:
            G.add_edge(u, v)
        if not has_independent_set_of_size_k(G, 6) and not has_kn(G, 4):
            nx.write_graph6(G, "../../data/r(4,6,36)_graph.g6")
            print(f"Time elapsed: {timeit.default_timer()-startTime} seconds")
            return


if __name__ == '__main__':
  filename = '../../data/r46/r46_35some.g6'
  graphs = nx.read_graph6(filename)
  G = graphs[0]
  add_vertex(G)
  bfs(G)