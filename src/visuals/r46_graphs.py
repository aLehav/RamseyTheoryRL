import networkx as nx
import matplotlib.pyplot as plt
import json
import sys
sys.path.append("..")
from utils.features import *


# Obtain feature table for each of the 37 R(4,6,35) graphs

# 4 subgraph, K_2, P_3, K_{1,3}, P_4, K_3, K_3 + e, C_4, K_4 - e, K_4

# Right now we do it for graph 1

def obtain_counts():
  print("here")
  filename = '../../data/r46/r46_35some.g6'
  graphs = nx.read_graph6(filename)
  G = graphs[0]
  feature_counts = dict()
  counters = count_subgraph_structures(G)
  print(counters)
  # features_counts['K_2'] = num_complete_subgraphs(G,2)
  # features_counts['P_3'] =
  # features_counts['K_{1,3}'] =
  # features_counts['P_4'] = 
  # features_counts['K_3'] = num_complete_subgraphs(G, 3)
  # features_counts['K_3 + e'] = num_extended_complete_subgraphs(G,3)
  # features_counts['C_4'] = num_cycle_subgraphs(4)
  # features_counts['K_4 - e'] = num_almost_complete_subgraphs(G,4)
  # features_counts['K_4'] = num_complete_subgraphs(G, 4)

  # Output to file
  # with open('feature_counts.txt', 'w') as convert_file:
  #    convert_file.write(json.dumps(feature_counts))


def main():
  obtain_counts()


if __name__ == '__main__':
  main()
