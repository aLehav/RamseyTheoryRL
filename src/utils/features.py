import random
import networkx as nx
import itertools

# Count occurences of feature: Exponential runtime


def num_complete_subgraphs(G, i):
    return len(list(nx.enumerate_all_cliques(G))) if i > 2 else len([clq for clq in nx.enumerate_all_cliques(G) if len(clq) == i])


def num_paths_length_i(G, i):
    paths = nx.all_pairs_shortest_path(G)
    count = sum(1 for _, paths_dict in paths if i in map(
        len, paths_dict.values()))
    return count


def num_almost_complete_subgraphs(G, i):
    count = 0
    for sub_nodes in itertools.combinations(G.nodes(), i):
        subg = G.subgraph(sub_nodes)
        if subg.number_of_edges() == (i*(i-1))//2 - 1:
            count += 1
    return count


def num_extended_complete_subgraphs(G, i):
    count = 0
    for sub_nodes in itertools.combinations(G.nodes(), i):
        subg = G.subgraph(sub_nodes)
        if subg.number_of_edges() == (i*(i-1))//2 + 1:
            count += 1
    return count


def num_cycle_subgraphs(G, i):
    cycles = nx.cycle_basis(G)
    return sum(1 for cycle in cycles if len(cycle) == i)


def num_bipartite_subgraphs(G, m, n):
    count = 0
    for sub_nodes in itertools.combinations(G.nodes(), m + n):
        subg = G.subgraph(sub_nodes)
        if nx.is_bipartite(subg):
            nodes1, nodes2 = nx.bipartite.sets(subg)
            if len(nodes1) in [m, n] and len(nodes2) in [m, n]:
                count += 1
    return count

# Count number of occurences of features in table 2 of paper for every 4 subgraph


# O(n^4)
def count_subgraph_structures(G):
    # Define the structures to look for
    structures = {
        "K_2": nx.complete_graph(2),
        "P_3": nx.path_graph(3),
        "K_{1,3}": nx.complete_bipartite_graph(1, 3),
        "P_4": nx.path_graph(4),
        "K_3": nx.complete_graph(3),
        "C_4": nx.cycle_graph(4),
        "K_4": nx.complete_graph(4),
    }
    # Add K_3+e
    K_3_e = nx.complete_graph(3)
    K_3_e.add_edge(2, 3)
    structures["K_3+e"] = K_3_e

    # Add K_4-e
    K_4_e = nx.complete_graph(4)
    K_4_e.remove_edge(0, 1)
    structures["K_4-e"] = K_4_e

    # Convert each structure graph to a canonical hash for comparison
    for name, graph in structures.items():
        structures[name] = nx.weisfeiler_lehman_graph_hash(graph)

    # Initialize a counter for each structure
    counters = {name: 0 for name in structures}

    # Enumerate all 4-node subgraphs
    for nodes in itertools.combinations(G.nodes, 4):
        subgraph = nx.subgraph(G, nodes)

        # Compute the canonical hash of the subgraph
        hash = nx.weisfeiler_lehman_graph_hash(subgraph)

        # If the hash matches any of the structures, increment the corresponding counter
        for name, structure_hash in structures.items():
            match name:
                case "K_2":
                    counters["K_2"] += subgraph.number_of_edges() == 1
                case "P_3":
                    for subnodes in itertools.combinations(nodes, 3):
                        sub_subgraph = nx.subgraph(G, subnodes)
                        sub_hash = nx.weisfeiler_lehman_graph_hash(
                            sub_subgraph)
                        if sub_hash == structures["P_3"] and subgraph.number_of_edges() == 2:
                            counters["P_3"] += 1
                            break
                case "K_3":
                    for subnodes in itertools.combinations(nodes, 3):
                        sub_subgraph = nx.subgraph(G, subnodes)
                        sub_hash = nx.weisfeiler_lehman_graph_hash(
                            sub_subgraph)
                        if sub_hash == structures["K_3"] and subgraph.number_of_edges() == 3:
                            counters["K_3"] += 1
                            break
                case "K_3+e":
                    for subnodes in itertools.combinations(nodes, 3):
                        sub_subgraph = nx.subgraph(G, subnodes)
                        sub_hash = nx.weisfeiler_lehman_graph_hash(
                            sub_subgraph)
                        if sub_hash == structures["K_3"] and subgraph.number_of_edges() == 4:
                            counters["K_3+e"] += 1
                            break
                case _:
                    counters[name] += (hash == structure_hash)

    return counters

# Update graph given an increase/decrease in a particular feature


def add_K3(G):
    # Find nonadjacent pairs of nodes
    nonadjacent_pairs = [(u, v) for u, v in itertools.combinations(
        G.nodes, 2) if not G.has_edge(u, v)]

    # Try to find a pair that can form a triangle
    for u, v in nonadjacent_pairs:
        common_neighbors = list(nx.common_neighbors(G, u, v))
        if common_neighbors:  # If there's a common neighbor, we can form a triangle
            G.add_edge(u, v)
            return
    print("Couldn't add a K3 subgraph.")


def remove_K3(G):
    # Find all triangles in the graph
    triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]

    if triangles:  # If there are any triangles
        # Pick a random triangle and remove a random edge from it
        triangle = random.choice(triangles)
        u, v = random.sample(triangle, 2)
        G.remove_edge(u, v)
    else:
        print("Couldn't remove a K3 subgraph.")


def add_K4(G):
    # Find all triangles in the graph
    triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]

    # Try to find a triangle and a node that are only missing one edge to form a K4
    for triangle in triangles:
        for u in G.nodes:
            if u not in triangle and all(G.has_edge(v, u) for v in triangle):
                # We've found a triangle and a node that only need one more edge to form a K4
                G.add_edge(u, random.choice(triangle))  # Add that edge
                return
    print("Couldn't add a K4 subgraph.")


def remove_K4(G):
    # Find all K4 subgraphs in the graph
    K4s = [c for c in nx.enumerate_all_cliques(G) if len(c) == 4]

    if K4s:  # If there are any K4 subgraphs
        # Pick a random K4 and remove a random edge from it
        K4 = random.choice(K4s)
        u, v = random.sample(K4, 2)
        G.remove_edge(u, v)
    else:
        print("Couldn't remove a K4 subgraph.")

# Add functions to count subgraph structures added/removed when adding/removing an edge