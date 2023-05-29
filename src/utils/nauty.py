from pynauty import *

# Create a graph with n vertices
n = 5
G = Graph(n)

# Generate all non-isomorphic graphs of size n
canon_list, labelling = refine(G)

# Extract the list of non-isomorphic graphs
graphs = [from_nauty_graph(c) for c in canon_list]

# Print the number of non-isomorphic graphs and their canonical labellings
print(len(graphs))
for i, g in enumerate(graphs):
    print(i, canonical_labeling(g))
