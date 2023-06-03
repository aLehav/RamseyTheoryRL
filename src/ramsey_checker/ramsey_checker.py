from abc import ABC, abstractmethod
import igraph as ig
import itertools
import networkx as nx

class RamseyChecker(ABC):
    def generate_structures(cls):
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
    
    structures = generate_structures()

    # Count subgraph structures O(n^4)
    def count_subgraph_structures(cls, G):
        counters = {name: 0 for name in cls.structures}
        for nodes in itertools.combinations(range(G.vcount()), 4):
            subgraph = G.subgraph(nodes)
            for name, structure in cls.structures.items():
                if subgraph.isomorphic(structure):
                    counters[name] += 1
        return counters

    def is_complete(self, G):
        n = G.vcount()
        return G.ecount() == n * (n - 1) // 2

    def change_edge(self, G, e):
        G.delete_edges([e]) if G.are_connected(*e) else G.add_edge(*e)
        
    def are_graphs_isomorphic(self, G1, G2):
        return nx.is_isomorphic(G1, G2)

    @abstractmethod
    def has_independent_set_of_size_k(self, G, k):
        pass

    @abstractmethod
    def has_kn(self, G, k):
        pass

    @abstractmethod
    def check_counterexample(self, G, s, t):
        pass
      
    @abstractmethod
    def consider_counterexample(self, G, counters, counter_path):
        pass

    @abstractmethod
    def count_subgraphs_from_edge(self, G, u, v):
        pass

    @abstractmethod
    def update_feature_from_edge(self, G, u, v, counters):
        pass
