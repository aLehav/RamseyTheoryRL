
import timeit
import random
import networkx as nx
import sys
sys.path.append("..")
from utils.features import *

G = nx.complete_graph(3)
G.add_node(3)

count = count_subgraph_structures(G)
print(count)


update_feature_from_edge(G, 2,3, count)
print(count)