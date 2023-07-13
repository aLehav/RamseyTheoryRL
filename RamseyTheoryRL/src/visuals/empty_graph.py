import networkx as nx


for i in range(35,41):
    G = nx.empty_graph(i)
    path = f"../../data/empty_graphs/v{i}.g6"
    nx.write_graph6(G, path)