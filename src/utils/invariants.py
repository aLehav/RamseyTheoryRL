import networkx as nx
import numpy as np


# O(1)
def num_vertices_edges(G):
    return G.number_of_nodes(), G.number_of_edges()

# O(nlogn)
def degree_sequence(G):
    return sorted([d for n, d in G.degree()], reverse=True)

# O(n^2)
def avg_clustering_coefficient(G):
    return nx.average_clustering(G)

# O(n^3)
def avg_shortest_path_length(G):
    # Note: this function assumes the graph is connected
    return nx.average_shortest_path_length(G)

# O(1)
def graph_density(G):
    return nx.density(G)

# O(n^2) for degree and closeness centrality,
# O(n^3) for betweenness and eigenvector centrality
def centralities(G):
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    return degree_centrality, closeness_centrality, betweenness_centrality, eigenvector_centrality

# O(n^3)
def adjacency_spectrum(G):
    adjacency_matrix = nx.adjacency_matrix(G)
    eigenvalues = np.linalg.eigvals(adjacency_matrix.A)
    return sorted(eigenvalues, reverse=True)
