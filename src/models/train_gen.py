import networkx as nx
import pandas as pd
import json
import time
import csv
import gzip
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils.useful import *
from utils.features import count_subgraph_structures

MAX_S = 4
MAX_T = 10
MAX_SIZE = 7

def unzip_gz(n):
    output_file_path = f'graph{n}.g6'
    gz_file_path = f'{output_file_path}.gz'
    print(f'Unzipping {gz_file_path} to {output_file_path}')
    with gzip.open(gz_file_path, 'rb') as gz_file:
        with open(output_file_path, 'wb') as output_file:
            output_file.write(gz_file.read())

def create_entries_for_n(n):
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
            subgraph_counts = count_subgraph_structures(G)

            # Check for independent sets and cliques
            for s in range(2, min(MAX_S + 1, n)):
                for t in range(s, min(MAX_T + 1, n)):
                    has_no_independent_set = not has_independent_set_of_size_k(G, t) if n >= t else True
                    has_no_clique = not has_kn(G, s) if n >= s else True
                    is_counter = has_no_independent_set and has_no_clique
                    data.append({**subgraph_counts, 'n': n, 's': s, 't': t, 'counter': is_counter})
    else:
        # TODO: Use another way to generate all isomorphic graphs for n > 7
        # Replace the code below with the appropriate implementation
        def generate_all_isomorphic_graphs(n):
            return None
        # Placeholder code
        for graph in generate_all_isomorphic_graphs(n):
            G = nx.Graph(graph)

            # Compute count_subgraph_structures
            subgraph_counts = count_subgraph_structures(G)

            # Check for independent sets and cliques
            for s in range(1, 5):
                for t in range(s, 11):
                    has_no_independent_set = not has_independent_set_of_size_k(G, t) if n >= t else True
                    has_no_clique = not has_kn(G, s) if n >= s else True
                    is_counter = has_no_independent_set and has_no_clique
                    data.append({**subgraph_counts, 'n': n, 's': s, 't': t, 'counter': is_counter})

    return data

def create_entries_up_to_n(n, time_path='train_gen_runtime.csv', df_path='train_gen.csv'):
    data = []

    with open(time_path, 'w', newline='') as csvfile:
        fieldnames = ['n', 'runtime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for n in range(2, n + 1):
            start_time = time.time()
            data += create_entries_for_n(n)
            end_time = time.time()
            runtime = end_time - start_time          
            writer.writerow({'n': n, 'runtime': runtime})
            print(f'Generated all entries for n={n}. Runtime: {runtime} seconds.')

        df = pd.DataFrame(data)
        df.to_csv(df_path)


create_entries_up_to_n(MAX_SIZE)