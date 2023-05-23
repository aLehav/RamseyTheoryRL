#########################
# Only works for n < 10 #
#########################
import time
import networkx as nx
import igraph as ig
import pandas as pd
import csv
from utils.guseful import *
from utils.gfeatures import count_subgraph_structures

def create_entries_for_n(n, max_s, max_t):
    data = []

    if n <= 9:
        graph_file_path = f'data/isomorphic_by_n/graph{n}.g6'
        #### Alternative that works for n <= 7 ####
        # for graph_id in nx.graph_atlas_g(n):    #
        #     G = nx.Graph(graph_id)              #
        ###########################################
        graphs = nx.read_graph6(graph_file_path)
        graphs = [graphs] if type(graphs) != list else graphs
        for G in graphs:
            # Compute count_subgraph_structures
            G = ig.Graph.TupleList(G.edges(), directed=False)
            subgraph_counts = count_subgraph_structures(G)

            # Check for independent sets and cliques
            for s in range(2, min(max_s + 1, n)):
                for t in range(s, min(max_t + 1, n)):
                    has_no_independent_set = not has_independent_set_of_size_k(G, t)
                    has_no_clique = not has_kn(G, s)
                    is_counter = has_no_independent_set and has_no_clique
                    data.append({**subgraph_counts, 'n': n, 's': s, 't': t, 'counter': is_counter})
    else:
       raise ValueError("create_entries_for_n using atlas graph generator. \nn must be less than 10.")

    return data

def create_entries_up_to_n(max_n, max_s, max_t, time_path, df_path):
    data = []

    with open(time_path, 'w', newline='') as csvfile:
        fieldnames = ['n', 'runtime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for n in range(2, max_n + 1):
            start_time = time.time()
            data += create_entries_for_n(n=n, max_s=max_s, max_t=max_t)
            end_time = time.time()
            runtime = end_time - start_time          
            writer.writerow({'n': n, 'runtime': runtime})
            print(f'Generated all entries for n={n}. Runtime: {runtime} seconds.')

        df = pd.DataFrame(data)
        df.to_csv(df_path)