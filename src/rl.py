import networkx as nx

def W_clique(clique_len):
    """
    Computes the reward for a clique of size clique_len.
    """
    return -clique_len**2

def W_K(K_len):
    """
    Computes the reward for a complete subgraph of size K_len.
    """
    return -K_len

def heuristic(G, W_clique=W_clique, W_K=W_K):
    """
    Computes the reward of state G achieved by graph G to be W_clique(len(clique_i)) for all cliques and W_K(len(clique_k)) for all cliques W_K.
    """
    # Find all cliques in G
    cliques = list(nx.find_cliques(G))

    # Check for independent sets by creating the complement graph
    complement = nx.complement(G)
    Ks = list(nx.find_cliques(complement))

    # TODO pos or negative reward?
    # Compute the reward of state G
    reward = sum([W_clique(-len(c)) for c in cliques])
    reward += sum([W_K(-len(k)) for k in Ks if len(k) == 2])

    return reward

def action(G, c_size):
    # Find all cliques of size c_size in G
    cliques = [c for c in nx.find_cliques(G) if len(c) == c_size]
    
    if len(cliques) == 0:
        raise ValueError(f"No clique of size {c_size} exists in the graph.")

    # Remove an edge from the graph
    edge = cliques[0][:2]
    G.remove_edge(*edge)
    
    return edge