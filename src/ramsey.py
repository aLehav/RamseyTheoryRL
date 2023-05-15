import networkx as nx
from ramseySat import ramseySat

# Test symmetric cases
G1 = nx.complete_graph(5)
assert ramseySat(G1, 3, 3) == True
assert ramseySat(G1, 4, 4) == True
assert ramseySat(G1, 5, 5) == True
assert ramseySat(G1, 6, 6) == False

G2 = nx.cycle_graph(5)
assert ramseySat(G2, 3, 3) == False
assert ramseySat(G2, 4, 4) == False
assert ramseySat(G2, 5, 5) == False
assert ramseySat(G2, 6, 6) == False

# Test known true cases
G3 = nx.Graph()
G3.add_nodes_from([1, 2, 3, 4, 5])
G3.add_edges_from([(1, 2), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5)])
assert ramseySat(G3, 3, 3) == True
assert ramseySat(G3, 4, 4) == False
assert ramseySat(G3, 5, 5) == False

G4 = nx.Graph()
G4.add_nodes_from([1, 2, 3, 4, 5])
G4.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)])
assert ramseySat(G4, 3, 3) == True
assert ramseySat(G4, 4, 4) == False
assert ramseySat(G4, 5, 5) == False

# Test known false cases
G5 = nx.Graph()
G5.add_nodes_from([1, 2, 3, 4, 5])
G5.add_edges_from([(1, 2), (1, 4), (2, 3), (2, 5), (3, 4)])
assert ramseySat(G5, 3, 3) == True
assert ramseySat(G5, 4, 4) == False
assert ramseySat(G5, 5, 5) == False

G6 = nx.Graph()
G6.add_nodes_from([1, 2, 3, 4, 5])
G6.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
assert ramseySat(G6, 3, 3) == True
assert ramseySat(G6, 4, 4) == False
assert ramseySat(G6, 5, 5) == False

# Wikipedia Star
G7 = nx.Graph()
G7.add_nodes_from([1, 2, 3, 4, 5])
G7.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
assert ramseySat(G7, 3, 3) == False
assert ramseySat(G7, 4, 4) == False
assert ramseySat(G7, 5, 5) == False

print("All tests passed!")
