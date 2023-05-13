import networkx as nx
import matplotlib.pyplot as plt
from random import choice
import matplotlib.animation as animation

# Define the parameters
n = 5   # Number of nodes
l = 1   # Number of edges to add
k = 1   # Number of edges to remove

# Create the complete graph
G = nx.empty_graph(n)

# Generate the initial circular layout
pos = nx.circular_layout(G)

# Define empty edges and added edges lists
empty_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
added_edges = []



def default_graph():
    # Draw the graph
    nx.draw_networkx_edges(G, pos, edgelist=empty_edges, edge_color='black', style='solid', alpha=0.3)
    nx.draw_networkx_nodes(G, pos, edgecolors='black', node_color='white')
    # nx.draw_networkx_labels(G, pos)

    # Set the plot title
    plt.title("Empty Graph with {} Nodes".format(n))

# Define a function to update the graph with a new edge
def update_graph(i):

    global empty_edges, added_edges
    
    # Pick a random edge to add
    if empty_edges:
        new_edge = choice(empty_edges)
    else:
        ani.event_source.stop()
        return

    #  Remove the edge from empty_edges and add it to added_edges
    empty_edges.remove(new_edge)
    added_edges.append(new_edge)

    # Add the new edge to the graph
    G.add_edge(*new_edge)

    # Draw the graph
    plt.clf()  # Clear the figure
    nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='blue', style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=empty_edges, edge_color='black', style='solid', alpha=0.3)
    nx.draw_networkx_nodes(G, pos, edgecolors='black', node_color='white')
    # nx.draw_networkx_labels(G, pos)

    # Set the plot title to show the current step
    plt.title("Adding edge ({}, {})".format(*new_edge))

# Define a function to call either the default graph or the update graph function
def update_func(i):
    if i == 0:
        default_graph()
    else:
        update_graph(i - 1)

# Create the animation object
ani = animation.FuncAnimation(plt.gcf(), update_func, init_func=default_graph, interval=1000)

# Save the animation
ani.save('test.gif', writer='pillow')