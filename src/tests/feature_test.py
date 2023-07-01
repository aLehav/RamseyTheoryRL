import timeit
import random
import networkx as nx
import sys
sys.path.append("..")
from utils.features import *

# Deprecated
G = nx.complete_graph(3)
G.add_node(3)

count = count_subgraph_structures(G)
print(count)


update_feature_from_edge(G, 2,3, count)
print(count)

def main():
  totalStartTime = timeit.default_timer()
  print("Count_subgraph_structures Tests-----------------------------------------")
  G9 = nx.complete_graph(35)
  startTime = timeit.default_timer()
  count_subgraph_structures(G9)
  print(f"Single Threaded Time Elapsed {timeit.default_timer() - startTime}")

  # startTime = timeit.default_timer()
  # count_subgraph_structures_parallel(G9)
  # print(f"Multi Threaded Time Elapsed {timeit.default_timer() - startTime}\n")

  # Timing tests
  print("Count_subgraphs_from_edge Tests")
  startTime = timeit.default_timer()
  count_subgraphs_from_edge(G9, 3, 4)
  print(f"Single Threaded Time Elapsed {timeit.default_timer() - startTime}")

  # startTime = timeit.default_timer()
  # count_subgraphs_from_edge_parallel(G9, 3, 4)
  # print(f"Multi Threaded Time Elapsed {timeit.default_timer() - startTime}")
  print(f"Total Time elapsed: {timeit.default_timer() - totalStartTime}")

if __name__ == '__main__':
  main()