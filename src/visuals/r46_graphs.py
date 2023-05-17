import networkx as nx
import matplotlib.pyplot as plt
import json
import sys
sys.path.append("..")
from utils.features import *
import timeit


# Obtain feature table for each of the 37 R(4,6,35) graphs

# 4 subgraph, K_2, P_3, K_{1,3}, P_4, K_3, K_3 + e, C_4, K_4 - e, K_4

# Right now we do it for graph 1

def obtain_counts():
  start_time = timeit.default_timer()
  filename = '../../data/r46/r46_35some.g6'
  graphs = nx.read_graph6(filename)
  for idx, g in enumerate(graphs):
    feature_counts = dict()
    counters = count_subgraph_structures(g)
    print(counters)
    G = graphs[0]
    feature_counts = dict()
    counters = count_subgraph_structures(G)
    print(counters)
    # Output to file
    with open('feature_counts.txt', 'a') as convert_file:
      convert_file.write(json.dumps(feature_counts))
  print(f"Time elapsed: {timeit.default_timer()-start_time} seconds")



def main():
  obtain_counts()


if __name__ == '__main__':
  main()
