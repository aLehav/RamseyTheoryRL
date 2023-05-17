import networkx as nx
import random
import timeit
import sys
sys.path.append("..")
from utils.useful import *


# Generate an r(3,9) graph with no K_3 and no K_10

def generate_graph(n):
    i = 0
    startTime = timeit.default_timer()
    while True:
        i += 1
        print(f"Graph {i}")
        p = random.random()
        G = nx.fast_gnp_random_graph(n, p)

        # Check if the graph now contains a K10 or an independent set of size 3
        if not has_kn(G,10) and not has_independent_set_of_size_k(G,3):
            # terminate and save graph
            nx.write_graph6(G, "../../data/r39_graph.g6")
            print(f"Time elapsed: {timeit.default_timer()-startTime} seconds")
            return


if __name__ == '__main__':
    generate_graph(36)
