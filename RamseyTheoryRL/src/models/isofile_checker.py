import networkx as nx
import igraph as ig
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils.guseful import *


PATH1 = "data/found_counters/r4_6_35_isograph.g6"
PATH2 = "data/found_counters/scaled_dnn/r4_6_35_isograph.g6"

def path_to_nx_list(path):
    prior_counters = nx.read_graph6(path)
    prior_counters = [prior_counters] if type(prior_counters) != list else prior_counters
    return prior_counters

def nx_list_iso_comparison(list1, list2):
    i = 0
    for G2 in list2:
        unique = True
        for G1 in list1:
            if are_graphs_isomorphic(G1, G2):
                unique = False
                print(f"Graph {i} not unique")
                break
        if unique: print(f"Graph {i} unique")
        i += 1

def main():
    print(f"First path: {PATH1}")
    print(f"Second path: {PATH2}")
    list1 = path_to_nx_list(PATH1)
    print(f"First path has {len(list1)} counters")
    list2 = path_to_nx_list(PATH2)
    print(f"First path has {len(list2)} counters")
    nx_list_iso_comparison(list2, list1)

if __name__ == '__main__':
    main()

