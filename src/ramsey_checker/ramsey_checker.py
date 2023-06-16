from abc import ABC, abstractmethod
import igraph as ig
import itertools
import networkx as nx


class RamseyChecker(ABC):
    def __init__(self):
        self.structures = {
            "K_{1,3}": ig.Graph.Full_Bipartite(1, 3),
            # Path graph
            "P_4": ig.Graph(n=4, edges=[(i, i+1) for i in range(3)]),
            "C_4": ig.Graph.Ring(4),  # Cycle graph
            "K_4": ig.Graph.Full(4),  # Complete graph
        }

        # Add K_3+e
        K_3_e = ig.Graph.Full(3)
        K_3_e.add_vertex()
        K_3_e.add_edge(2, 3)
        self.structures["K_3+e"] = K_3_e

        # Add K_4-e
        K_4_e = ig.Graph.Full(4)
        K_4_e.delete_edges((0, 1))
        self.structures["K_4-e"] = K_4_e

        # Add K_3
        K_3 = ig.Graph.Full(3)
        K_3.add_vertex()
        self.structures["K_3"] = K_3

        # Add P_3
        # Path graph with 3 nodes + isolated node
        P_3 = ig.Graph(n=4, edges=[(i, i+1) for i in range(2)])
        self.structures["P_3"] = P_3

        # Add K_2
        K_2 = ig.Graph(n=4, edges=[(0, 1)])  # Edge + 2 isolated nodes
        self.structures["K_2"] = K_2

        # Add 2K_2
        # Two edges + isolated nodes
        _2K_2 = ig.Graph(n=4, edges=[(0, 1), (2, 3)])
        self.structures["2K_2"] = _2K_2

        # Add E_4
        E_4 = ig.Graph(n=4)  # 4 isolated nodes
        self.structures["E_4"] = E_4

        # Add table
        self.edge_bounds = [
        [-1 for i in range(0,17)],
        [-1 for i in range(0, 17)],
        [-1 for i in range(0, 17)],
        [*[-1 for i in range(3)], 1, *[-1 for i in range (0,13)]],
        [*[-1 for i in range(3)], 2, 1, *[-1 for i in range(0, 12)]],
        [*[-1 for i in range(3)], 5, 2, 1, *[-1 for i in range(0, 11)]], #5
        [*[-1 for i in range(4)], 3, 2, 1, *[-1 for i in range(0, 10)]],
        [*[-1 for i in range(4)], 6, 3, 2, 1, *[-1 for i in range(0, 9)]],
        [*[-1 for i in range(4)], 10, 4, 3, 2, 1, *[-1 for i in range(0, 8)]],
        [*[-1 for i in range(5)], 7, 4, 3, 2, 1, *[-1 for i in range(0, 7)]],
        [*[-1 for i in range(5)], 10, 5, 4, 3, 2, 1, *[-1 for i in range(0, 6)]], #10
        [*[-1 for i in range(5)], 15, 8, 5, 4, 3, 2, 1, *[-1 for i in range(0, 5)]],
        [*[-1 for i in range(5)], 20, 11, 6, 5, 4, 3, 2, 1, *[-1 for i in range(0, 4)]],
        [*[-1 for i in range(5)], 26, 15, 9, 6, 5, 4, 3, 2, 1, *[-1 for i in range(0, 3)]],
        [*[-1 for i in range(6)], 20, 12, 7, 6, 5, 4, 3, 2, 1, *[-1 for i in range(0, 2)]],
        [*[-1 for i in range(6)], 25, 15, 10, 7, 6, 5, 4, 3, 2, 1, *[-1 for i in range(0, 1)]], #15
        [*[-1 for i in range(6)], 32, 20, 13, 8, 7, 6, 5, 4, 3, 2, 1],
        [*[-1 for i in range(6)], 40, 25, 16, 11, 8, 7, 6, 5, 4, 3, 2],
        [*[-1 for i in range(7)], 30, 20, 14, 9, 8, 7, 6, 5, 4, 3],
        [*[-1 for i in range(7)], 37, 25, 17, 12, 9, 8, 7, 6, 5, 4],
        [*[-1 for i in range(7)], 44, 30, 20, 15, 10, 9, 8, 7, 6, 5], #20
        [*[-1 for i in range(7)], 51, 35, 25, 18, 13, 10, 9, 8, 7, 6],
        [*[-1 for i in range(7)], 60, 42, 30, 21, 16, 11, 10, 9, 8, 7],
        [*[-1 for i in range(8)], 49, 35, 25, 19, 14, 11, 10, 9, 8],
        [*[-1 for i in range(8)], 56, 40, 30, 22, 17, 12, 11, 10, 9],
        [*[-1 for i in range(8)], 65, 46, 35, 25, 20, 15, 12, 11, 10], #25
        [*[-1 for i in range(8)], 73, 52, 40, 30, 23, 18, 13, 12, 11],
        [*[-1 for i in range(8)], 85, 61, 45, 35, 26, 21, 16, 13, 12],
        [*[-1 for i in range(9)], 68, 51, 40, 30, 24, 19, 14, 13],
        [*[-1 for i in range(9)], 77, 58, 45, 35, 27, 22, 17, 14],
        [*[-1 for i in range(9)], 86, 66, 50, 40, 30, 25, 20, 15], #30
        [*[-1 for i in range(9)], 95, 73, 56, 45, 35, 28, 23, 18]]

    # Count subgraph structures O(n^4)
    def count_subgraph_structures(self, G):
        counters = {name: 0 for name in self.structures}
        for nodes in itertools.combinations(range(G.vcount()), 4):
            subgraph = G.subgraph(nodes)
            for name, structure in self.structures.items():
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

    def has_kn_from_edge(self, G, k, e):
        chosen_nodes = list(e)
        other_nodes = set(range(G.vcount())) - set(chosen_nodes)
        for sub_nodes in itertools.combinations(other_nodes, k-2):
            sub_nodes = chosen_nodes + list(sub_nodes)
            subgraph = G.subgraph(sub_nodes)
            if self.is_complete(subgraph):
                return True
        return False

    def has_independent_set_of_size_k_from_edge(G, k, e):
        chosen_nodes = list(e)
        other_nodes = set(range(G.vcount())) - set(chosen_nodes)
        for sub_nodes in itertools.combinations(other_nodes, k-2):
            sub_nodes = chosen_nodes + list(sub_nodes)
            subgraph = G.subgraph(sub_nodes)
            if subgraph.ecount() == 0:
                return True
        return False

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
