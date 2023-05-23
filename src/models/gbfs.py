import multiprocessing
import igraph as ig
import copy as cp
import random
import timeit
import pickle
from functools import partial
import sys
sys.path.append("..")
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


def process_edge(e, G, PAST, used_edges, subgraph_counts, s, t, g6path_to_write_to, gEdgePath_to_write_to, path, new_graphs):
    check = str(e[0]) + ',' + str(e[1])
    if check not in used_edges.keys():
      used_edges[check] = 1
    # Obtain new vectorization
    G_prime = G.copy()
    update_feature_from_edge(G_prime, e[0], e[1], subgraph_counts)

    is_counterexample = check_counterexample(G, s, t)
    # output to file TODO remove False
    if (is_counterexample and False):
        with open(g6path_to_write_to, 'a') as file:
            file.write(G.write(format='graph6'))
        # Add last edge
        temp_edges = cp.deepcopy(path)
        temp_edges.append(e)
        # Output edge path to file
        with open(gEdgePath_to_write_to, 'a') as file:
            file.write(' '.join(temp_edges))

    vectorization = {**subgraph_counts, 'n': G.vcount(),
                      's': s, 't': t, 'counter': is_counterexample}
    vectorization_string = str(vectorization)
    # Assume keys in PAST are strings
    if vectorization_string not in PAST.keys():
        heuristic_val = heuristic(G)
        PAST[vectorization_string] = heuristic_val
        new_graphs.append((heuristic_val, e, G_prime))

# We are assuming python atomic list operations are thread-safe

def step_par(G, PAST, used_edges, edges, s, t, g6path_to_write_to, gEdgePath_to_write_to, subgraph_counts, path):
    new_graphs = []
    new_edges = []

    process_edge_wrapper = partial(process_edge, G=G, PAST=PAST, used_edges=used_edges, subgraph_counts=subgraph_counts,
                                   s=s, t=t, g6path_to_write_to=g6path_to_write_to, gEdgePath_to_write_to=gEdgePath_to_write_to, path=path, new_graphs=new_graphs)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_edge_wrapper, edges)

    if not new_graphs:
        return None

    new_graphs.sort(key=lambda x: x[0], reverse=True)  # sort by heuristic
    best_edge = (new_graphs[0][1][0], new_graphs[0][1][1])
    path.append(*best_edge)
    return new_graphs[0][2]


def step(G, PAST, used_edges, edges, s, t, g6path_to_write_to, gEdgePath_to_write_to, subgraph_counts, path):
    new_graphs = []
    new_edges = []
    for e in edges:
        check = str(e[0]) + ',' + str(e[1])
        if check not in used_edges.keys():
            used_edges[check] = 1
            # Obtain new vectorization
            update_feature_from_edge(G, e[0], e[1], subgraph_counts)

            is_counterexample = check_counterexample(G, s, t)
            # output to file
            if (is_counterexample):
                with open(g6path_to_write_to, 'a') as file:
                    file.write(G.write(format='graph6'))
                # Add last edge
                temp_edges = cp.deepcopy(path)
                temp_edges.append(e)
                # Output edge path to file
                with open(gEdgePath_to_write_to, 'a') as file:
                    file.write(' '.join(temp_edges))

            vectorization = {**subgraph_counts, 'n': G.vcount(),
                             's': s, 't': t, 'counter': is_counterexample}
            vectorization_string = str(vectorization)
            # Assume keys in PAST are strings
            if vectorization_string not in PAST.keys():
                heuristic_val = heuristic(G)
                PAST[vectorization_string] = heuristic_val
                new_graphs.append((heuristic_val, e))
            # remove_edge is O(n^2)
            G.delete_edges((e[0], e[1]))

    if not new_graphs:
        return None

    new_graphs.sort(key=lambda x: x[0], reverse=True)  # sort by heuristic
    best_edge = (new_graphs[0][1][0], new_graphs[0][1][1])
    path.append(best_edge)
    # return graph with max heuristic
    G.add_edge(*best_edge)
    return G

# Assume we are only adding edges

def bfs(G, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, PARALLEL):
    # Maintain a list of edge additions for PATH
    path = []

    n = G.vcount()
    # we only consider edges not present in initial graph
    edges = [(i, j) for i in range(n)
             for j in range(i+1, n) if not G.are_connected(i, j)]
    # dictionary for used edges, e(u,v) has the key 'u,v'
    used_edges = dict()

    for e in G.es:
        used_edges[str(e.source) + ',' + str(e.target)] = 1

    PAST = load_vectorizations(PAST_path)
    subgraph_counts = count_subgraph_structures(G)
    while G is not None:
        if (PARALLEL):
            G = step_par(G, PAST, used_edges, edges, s, t, g6path_to_write_to,
                         gEdgePath_to_write_to, subgraph_counts, path)
        else:
            G = step(G, PAST, used_edges, edges, s, t, g6path_to_write_to,
                     gEdgePath_to_write_to, subgraph_counts, path)

    # TODO Output PAST to file
    # with open(PAST_path, 'wb') as f:
    #     pickle.dump(PAST, f)


def main():
    G = ig.Graph(35)
    g6path_to_write_to = "../../data/found_counters/r39_graph.g6"
    gEdgePath_to_write_to = "../../data/found_counters/r39_path.txt"
    PAST_path = "none"
    s = 3
    t = 9
    PARALLEL = False
    startTime = timeit.default_timer()
    bfs(G, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, PARALLEL)
    print(f"Single Threaded Time Elapsed: {timeit.default_timer() - startTime}")
    startTime = timeit.default_timer()
    bfs(G, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, True)
    print(f"Multi Threaded Time Elapsed: {timeit.default_timer() - startTime}")


if __name__ == '__main__':
    main()

# TODO if we want this pausing scheme, we need to also store all metadata along with PAST_path to resume which adds overhead
# // point of storing PAST_path is we can run for a certain amount of steps max if needed, store last G visited as g6, load up G and PAST and start from where we left off. Note: if doing this, make sure G was last thing vectorized, and not G's derived from G.
