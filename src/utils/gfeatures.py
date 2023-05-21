import igraph as ig
import itertools
import concurrent.futures
import os
import multiprocessing


node_list = [nodes for nodes in itertools.combinations(range(35), 4)]
max_workers = os.cpu_count()
# Since structures is read only, global initialization is fine
structures = {
    "K_{1,3}": ig.Graph.Full_Bipartite(1, 3),
    "P_4": ig.Graph(n=4, edges=[(i, i+1) for i in range(3)]),  # Path graph
    "C_4": ig.Graph.Ring(4),  # Cycle graph
    "K_4": ig.Graph.Full(4),  # Complete graph
}

# Add K_3+e
K_3_e = ig.Graph.Full(3)
K_3_e.add_vertex()
K_3_e.add_edge(2, 3)
structures["K_3+e"] = K_3_e

# Add K_4-e
K_4_e = ig.Graph.Full(4)
K_4_e.delete_edges((0, 1))
structures["K_4-e"] = K_4_e

# Add K_3
K_3 = ig.Graph.Full(3)
K_3.add_vertex()
structures["K_3"] = K_3

# Add P_3
# Path graph with 3 nodes + isolated node
P_3 = ig.Graph(n=4, edges=[(i, i+1) for i in range(2)])
structures["P_3"] = P_3

# Add K_2
K_2 = ig.Graph(n=4, edges=[(0, 1)])  # Edge + 2 isolated nodes
structures["K_2"] = K_2

# Add 2K_2
_2K_2 = ig.Graph(n=4, edges=[(0, 1), (2, 3)])  # Two edges + isolated nodes
structures["2K_2"] = _2K_2

# Add E_4
E_4 = ig.Graph(n=4)  # 4 isolated nodes
structures["E_4"] = E_4


# Count subgraph structures O(n^4)
def count_subgraph_structures(G):
    counters = {name: 0 for name in structures}
    for nodes in itertools.combinations(range(G.vcount()), 4):
        subgraph = G.subgraph(nodes)
        for name, structure in structures.items():
            if subgraph.isomorphic(structure):
                counters[name] += 1
    return counters

# Count subgraphs from edge O(n^2)
def count_subgraphs_from_edge(G, u, v):
    counters = {name: 0 for name in structures}
    nodes = set(range(G.vcount())) - {u, v}
    for node1, node2 in itertools.combinations(nodes, 2):
        subgraph = G.subgraph([u, v, node1, node2])
        for name, structure in structures.items():
            if subgraph.isomorphic(structure):
                counters[name] += 1
    return counters

# Update feature from edge
def update_feature_from_edge(G, u, v, counters):
    count = count_subgraphs_from_edge(G, u, v)
    G.add_edge(u, v)
    ncount = count_subgraphs_from_edge(G, u, v)
    for name in counters:
        counters[name] += (ncount[name] - count[name])


def process_subgraph(G, nodes, structures):
    counters = {name: 0 for name in structures}
    subgraph = G.subgraph(nodes)
    for name, structure in structures.items():
        if subgraph.isomorphic(structure):
            counters[name] += 1
    return counters

def process_subgraph_par(nodes):
  # print(nodes)
  return nodes

# apple silicon does not support hyperthreading, so we have at most os.cpu_count()=8 processes
def count_subgraph_structures_parallel(G):
    counters = {name: 0 for name in structures}

    # This is the function that will be submitted to the executor
    # def process_subgraph(nodes):
    #     subgraph = G.subgraph(nodes)
    #     local_counters = {name: 0 for name in structures}
    #     for name, structure in structures.items():
    #         if subgraph.isomorphic(structure):
    #             local_counters[name] += 1
    #     return local_counters

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
      results = pool.map(process_subgraph_par, node_list)
    
    return counters


    # with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     future_to_nodes = {executor.submit(
    #         process_subgraph, i): i for i in range(n/max_workers)}
    #     print(future_to_nodes)
    #     for future in concurrent.futures.as_completed(future_to_nodes):
    #         nodes = future_to_nodes[future]
    #         try:
    #             local_counters = future.result()
    #             for name in local_counters:
    #                 counters[name] += local_counters[name]
    #         except Exception as exc:
    #             print('Generated an exception: %s' % (exc))
    # return counters


def count_subgraphs_from_edge_parallel(G, u, v):
    counters = {name: 0 for name in structures}
    nodes = set(range(G.vcount())) - {u, v}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_subgraph, G, [u, v, node1, node2], structures)
                   for node1, node2 in itertools.combinations(nodes, 2)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            for name in result:
                counters[name] += result[name]
    return counters
