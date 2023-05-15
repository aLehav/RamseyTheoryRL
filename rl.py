import networkx as nx
from ramseySat import ramseySat
from typing import List, Tuple, Dict
from tensorflow import keras
from tensorflow.keras import layers

def heuristic(G: nx.Graph, k: int, l: int) -> int:
    """
    Computes the reward of state G achieved by graph G as the value of a call to ramseySat.
    A positive value indicates a case that disproves R(k,l) = n.
    """
    return int(not ramseySat(G, k, l))

def add_edge(G: nx.Graph, edge: tuple, k: int, l: int) -> int:
    """
    Consider add_edge to G to be an estimator for Q(s,a) with state of G being the degree
    sequence and a to be adding the edge.
    """
    if not G.has_edge(*edge):  # Check if the edge already exists
        G.add_edge(*edge)  # Add the edge if it doesn't already exist
    
    return heuristic(G, k, l)

def G_to_rl_input(G: nx.Graph) -> Tuple[List[float], Dict[int, int]]:
    """
    Returns the degree sequence of the graph G sorted in decreasing order and scaled by a factor of n-1.
    """
    degrees = ((v, d) for v, d in G.degree())
    degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
    max_degree = len(degrees) - 1
    degree_vals = [deg[1]/max_degree for deg in degrees]
    degree_locs = [deg[0] for deg in degrees]
    return degree_vals, degree_locs


G = nx.cycle_graph(4)
G.add_edge(*(1,3))

print(G_to_rl_input(G))

def init_model() -> keras.models.Model:
    input_size = 200  # Size of the input sequence

    # Define the LSTM model
    model = keras.Sequential()
    model.add(layers.LSTM(32, input_shape=(input_size, 1)))  # 64 is the number of LSTM units
    model.add(layers.Dense(input_size, activation='sigmoid'))  # Output layer with the same length as input

    # Compile the model
    model.compile(loss='mse', optimizer='adam')

    # Print the model summary
    model.summary()

    return model

init_model()