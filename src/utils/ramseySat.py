import networkx as nx

def ramseySat(G: nx.Graph, k: int, l: int) -> bool:
    """
    Check if a graph G contains a clique of size k or an independent set of size l
    using the Ramsey's theorem.

    Parameters:
    G (networkx.Graph): The graph to be checked.
    k (int): The size of the desired clique.
    l (int): The size of the desired independent set.

    Returns:
    bool: True if G contains a clique of size k or an independent set of size l, False otherwise.
    """
    cliques = list(nx.find_cliques(G))
    max_cliques = [clique for clique in cliques if len(clique) >= k]
    if any(max_cliques):
        return True

    # TODO remove?
    if l == 1:
        return False

    # Check for independent sets by creating the complement graph
    complement = nx.complement(G)
    cliques = list(nx.find_cliques(complement))
    max_cliques = [clique for clique in cliques if len(clique) >= l]
    if any(max_cliques):
        return True

    return False
