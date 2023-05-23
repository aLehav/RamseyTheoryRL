import time
import networkx as nx
import igraph as ig
import pandas as pd
import csv
import os
import re
from utils.guseful import *
from utils.gfeatures import count_subgraph_structures

def get_files_for_path(path):
    return os.listdir(path)

def ramsey_entries_for_file(path, S, T, N):
    graphs = nx.read_graph6(path)
    graphs = [graphs] if type(graphs) != list else graphs
    data = []
    for G in graphs:
        # Compute count_subgraph_structures
        G = ig.Graph.TupleList(G.edges(), directed=False)
        subgraph_counts = count_subgraph_structures(G)

        # Check for independent sets and cliques, not needed for known counterexamples
        # has_no_independent_set = not has_independent_set_of_size_k(G, T)
        # has_no_clique = not has_kn(G, S)
        # is_counter = has_no_independent_set and has_no_clique
        data.append({**subgraph_counts, 'n': N, 's': S, 't': T, 'counter': True})
    return data
    
def ramsey_entries_for_path(path, time_path, df_path):
    data = []

    with open(time_path, 'w', newline='') as csvfile:
        fieldnames = ['n', 'runtime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file in get_files_for_path(path):
            match = re.search(r'r(\d)(\d+)_(\d+)\.g6', file)

            if match:
                S = int(match.group(1))
                T = int(match.group(2))
                N = int(match.group(3))
            else:
                raise ValueError(f"{file} is an improper file.")
            start_time = time.time()
            data += ramsey_entries_for_file(f'{path}/{file}', S=S, T=T, N=N)
            end_time = time.time()
            runtime = end_time - start_time          
            writer.writerow({'n': N, 'runtime': runtime})
            print(f'Generated all entries for n={N}. Runtime: {runtime} seconds.')

        df = pd.DataFrame(data)
        df.to_csv(df_path)