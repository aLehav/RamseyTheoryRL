import multiprocessing
import igraph as ig
import networkx as nx
import copy as cp
import random
import timeit
import pickle
from functools import partial
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils.gfeatures import *
from utils.guseful import *

def heuristic(G):
    return random.random()

# Load in a dictionary of vectorizations and heuristics, assume vectorization is a string TODO remove dict()
def load_vectorizations(PAST_path):
    return dict()
    with open(PAST_path, 'rb') as f:
        PAST = pickle.load(f)
    return PAST


def process_edge(e, G, PAST, used_edges, subgraph_counts, s, t, g6path_to_write_to, gEdgePath_to_write_to, path):
    check = str(e[0]) + ',' + str(e[1])
    if (check in used_edges.keys()):
      return None
    # Obtain new vectorization
    # G_prime = G.copy()
    new_subgraph_counts = update_feature_from_edge(G, e[0], e[1], subgraph_counts)
    is_counterexample = check_counterexample(G, s, t)

    # output to file
    if (is_counterexample):
       write_counterexample(G=G, 
                                     g6path_to_write_to=g6path_to_write_to,
                                     gEdgePath_to_write_to=gEdgePath_to_write_to,
                                     e=e,
                                     path=path)
    # Change back edited edge
    change_edge(G, e)

    vectorization = {**new_subgraph_counts, 'n': G.vcount(),
                      's': s, 't': t, 'counter': is_counterexample}
    vectorization_string = str(vectorization)
    # Assume keys in PAST are strings
    if vectorization_string not in PAST.keys():
        heuristic_val = heuristic(G)
        return (heuristic_val, e, new_subgraph_counts, vectorization_string)
        new_graphs.append((heuristic_val, e, G_prime))

# We are assuming python atomic list operations are thread-safe
def step_par(G, PAST, used_edges, edges, s, t, g6path_to_write_to, gEdgePath_to_write_to, subgraph_counts, path):
    new_edges = []
    process_edge_wrapper = partial(process_edge, G=G.copy(), PAST=PAST, used_edges=used_edges, subgraph_counts=subgraph_counts,
                                   s=s, t=t, g6path_to_write_to=g6path_to_write_to, gEdgePath_to_write_to=gEdgePath_to_write_to, path=path)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        new_graphs = pool.map(process_edge_wrapper, edges)

    new_graphs = [x for x in new_graphs if x is not None]
    if not new_graphs:
        return None
    
    new_graphs.sort(key=lambda x: x[0], reverse=True)  # sort by heuristic
    best_edge = (new_graphs[0][1][0], new_graphs[0][1][1]) # best edge
    path.append(best_edge)
    check = str(best_edge[0]) + ',' + str(best_edge[1])
    used_edges[check] = 1
    change_edge(G,best_edge)
    subgraph_counts.update(new_graphs[0][2])
    PAST[new_graphs[0][3]] = new_graphs[0][0]
    return G


def step(G, PAST, used_edges, edges, s, t, g6path_to_write_to, gEdgePath_to_write_to, subgraph_counts, path):
    new_graphs = []

    for e in edges:
        # check = str(e[0]) + ',' + str(e[1])
        # if check not in used_edges.keys():
            # Obtain new vectorization
            new_subgraph_counts = update_feature_from_edge(G, e[0], e[1], subgraph_counts)

            is_counterexample = check_counterexample(G, s, t)
            # output to file
            if (is_counterexample):
                write_counterexample(G=G, 
                                     g6path_to_write_to=g6path_to_write_to,
                                     gEdgePath_to_write_to=gEdgePath_to_write_to,
                                     e=e,
                                     path=path)

            vectorization = {**new_subgraph_counts, 'n': G.vcount(),
                             's': s, 't': t, 'counter': is_counterexample}
            vectorization_string = str(vectorization)
            # Assume keys in PAST are strings
            if vectorization_string not in PAST.keys():
                heuristic_val = heuristic(G)
                new_graphs.append((heuristic_val, e, new_subgraph_counts, vectorization_string))
            # Change back edited edge
            change_edge(G,e)

    if not new_graphs:
        print("No new graphs.")
        return None

    new_graphs.sort(key=lambda x: x[0], reverse=True)  # sort by heuristic
    best_edge = (new_graphs[0][1][0], new_graphs[0][1][1])
    path.append(best_edge)
    check = str(best_edge[0]) + ',' + str(best_edge[1])
    used_edges[check] = 1
    # return graph with max heuristic
    change_edge(G,best_edge)
    subgraph_counts.update(new_graphs[0][2])
    PAST[new_graphs[0][3]] = new_graphs[0][0]
    return G

# Assume we are only adding edges

def bfs(G, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, PARALLEL):
    # Maintain a list of edge additions for PATH
    path = []

    n = G.vcount()
    # we consider all edges
    edges = [(i, j) for i in range(n)
             for j in range(i+1, n)]
    # dictionary for used edges, e(u,v) has the key 'u,v'
    used_edges = dict()

    # for e in G.es:
    #     used_edges[str(e.source) + ',' + str(e.target)] = 1

    PAST = load_vectorizations(PAST_path)
    subgraph_counts = count_subgraph_structures(G)
    iterations = 0
    while G is not None:
        iterations += 1
        if (PARALLEL):
            G = step_par(G, PAST, used_edges, edges, s, t, g6path_to_write_to,
                         gEdgePath_to_write_to, subgraph_counts, path)
        else:
            G = step(G, PAST, used_edges, edges, s, t, g6path_to_write_to,
                     gEdgePath_to_write_to, subgraph_counts, path)
        if iterations % 50 == 0:
            print(f'{iterations} Iterations Completed')
    print("Total Iterations", iterations)
    print(path)
    # TODO Output PAST to file
    # with open(PAST_path, 'wb') as f:
    #     pickle.dump(PAST, f)


def main():
    G = ig.Graph(8)
    g6path_to_write_to = "data/found_counters/r34_graph.g6"
    gEdgePath_to_write_to = "data/found_counters/r34_path.txt"
    PAST_path = "none"
    s = 3
    t = 4
    PARALLEL = False
    startTime = timeit.default_timer()
    # bfs(G, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, PARALLEL)
    # print(f"Single Threaded Time Elapsed: {timeit.default_timer() - startTime}")
    # startTime = timeit.default_timer()
    # G2 = ig.Graph(7)
    # bfs(G2, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, True)
    # print(f"Multi Threaded Time Elapsed: {timeit.default_timer() - startTime}")
    get_isomorphic_graphs(g6path_to_read_from=g6path_to_write_to,
                          g6path_to_write_to='data/found_counters/r34_isograph.g6')
    print(f"Iso Updater Time Elapsed: {timeit.default_timer() - startTime}")


if __name__ == '__main__':
    main()

# TODO if we want this pausing scheme, we need to also store all metadata along with PAST_path to resume which adds overhead
# // point of storing PAST_path is we can run for a certain amount of steps max if needed, store last G visited as g6, load up G and PAST and start from where we left off. Note: if doing this, make sure G was last thing vectorized, and not G's derived from G.
