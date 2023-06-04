from ramsey_checker import RamseyChecker
import itertools
import networkx as nx
import sys
import tqdm
from functools import partial
import multiprocessing

class RamseyCheckerMultiThread(RamseyChecker):

    def has_kn(self, G, k, e):
        for sub_nodes in itertools.combinations(range(G.vcount()), k):
            subgraph = G.subgraph(sub_nodes)
            # Check if edge is in subgraph
            if all(node in sub_nodes for node in e):
              if subgraph.are_connected(e[0], e[1]):
                subgraph.delete_edges([(e[0], e[1])])
              else:
                subgraph.add_edge(e[0], e[1])
            if self.is_complete(subgraph):
                return True
        return False

    def has_independent_set_of_size_k(self, G, k, e):
        for sub_nodes in itertools.combinations(range(G.vcount()), k):
            subgraph = G.subgraph(sub_nodes)
            if all(node in sub_nodes for node in e):
              if subgraph.are_connected(e[0], e[1]):
                subgraph.delete_edges([(e[0], e[1])])
              else:
                subgraph.add_edge(e[0], e[1])
            if subgraph.ecount() == 0:
                return True
        return False

    def check_counterexample(self, G, s, t, e):
        return not self.has_kn(G, s, e) and not self.has_independent_set_of_size_k(G, t, e)

    def consider_counterexample(self, G, counters, counter_path, e):
        nx_graph = nx.Graph(G.get_edgelist())
        if G.are_connected(e[0], e[1]):
          nx_graph.remove_edge(*e)
        else:
          nx_graph.add_edge(*e)
        is_unique = True
        for counter in counters:
            if self.are_graphs_isomorphic(nx_graph, counter):
                is_unique = False
                sys.stdout.write(
                    '\033[1m\033[92mCounterexample found but not unique.\033[0m\n')
                break
        if is_unique:
            sys.stdout.write(
                '\033[1m\033[96mCounterexample found and added.\033[0m\n')
            counters.append(nx_graph)
            with open(counter_path, 'a') as file:
                file.write(nx.to_graph6_bytes(
                    nx_graph, header=False).decode('ascii'))

    def count_subgraphs_from_edge(self, G, u, v, to_change):
        counters = {name: 0 for name in self.structures}
        nodes = set(range(G.vcount())) - {u, v}
        for node1, node2 in itertools.combinations(nodes, 2):
            subgraph = G.subgraph([u, v, node1, node2])
            if to_change:
              if subgraph.are_connected(u, v):
                  subgraph.delete_edges([(u, v)])
              else:
                  subgraph.add_edge(u, v)
            for name, structure in self.structures.items():
                if subgraph.isomorphic(structure):
                    counters[name] += 1
        return counters

    def update_feature_from_edge(self, G, u, v, counters):
        old_count = self.count_subgraphs_from_edge(G, u, v, False)
        # Change edge
        new_count = self.count_subgraphs_from_edge(G, u, v, True)
        # Make and edit a copy of counters such that we can pass it to all children
        new_counters = {}
        for name in counters:
            new_counters[name] = counters[name] + \
                (new_count[name] - old_count[name])
        return new_counters

    def process_edge(self, e, G, PAST, used_edges, subgraph_counts, s, t, unique_path, heuristic):
        check = str(e[0]) + ',' + str(e[1])
        if (check in used_edges.keys()):
          return None
        # Obtain new vectorization
        # G_prime = G.copy()
        new_subgraph_counts = self.update_feature_from_edge(
            G, e[0], e[1], subgraph_counts)
        is_counterexample = self.check_counterexample(G, s, t, e)

        # output to file
        if (is_counterexample):
          self.consider_counterexample(G=G, counter_path=unique_path, e=e)
        # Change back edited edge
        # change_edge(G, e)

        vectorization = {**new_subgraph_counts, 'n': G.vcount(),
                        's': s, 't': t, 'counter': is_counterexample}
        vectorization_string = str(vectorization)
        # Assume keys in PAST are strings
        if vectorization_string not in PAST.keys():
            heuristic_val = heuristic(vectorization)
            return (heuristic_val, e, new_subgraph_counts, vectorization_string)
            new_graphs.append((heuristic_val, e, G_prime))

    def step_par(self, G, PAST, used_edges, edges, s, t, unique_path, subgraph_counts, heuristic):
        new_edges = []
        process_edge_wrapper = partial(self.process_edge, G=G.copy(), PAST=PAST, used_edges=used_edges, subgraph_counts=subgraph_counts,
                                      s=s, t=t,  unique_path=unique_path, heuristic=heuristic)

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            new_graphs = pool.map(process_edge_wrapper, edges)

        new_graphs = [x for x in new_graphs if x is not None]
        if not new_graphs:
            return None

        new_graphs.sort(key=lambda x: x[0], reverse=True)  # sort by heuristic
        best_edge = (new_graphs[0][1][0], new_graphs[0][1][1])  # best edge
        check = str(best_edge[0]) + ',' + str(best_edge[1])
        used_edges[check] = 1
        self.change_edge(G, best_edge)
        subgraph_counts.update(new_graphs[0][2])
        PAST[new_graphs[0][3]] = new_graphs[0][0]
        return G

    def bfs(self, g, unique_path, past, counters, s, t, n, iter_batch, update_model, heuristic, update_running, oldIterations=0, batches=None):
        # we consider all edges
        edges = [(i, j) for i in range(n)
                for j in range(i+1, n)]

        # Will store a list of vectors either expanded or found to be counterexamples, and upate a model after a given set of iterations
        training_data = []
        subgraph_counts = self.count_subgraph_structures(g)
        iterations = oldIterations
        progress_bar = tqdm.tqdm(
            initial=iterations, total=iterations+iter_batch, leave=False)
        while g is not None:
            g = self.step_par(g, past, dict(), edges, s, t,
                        unique_path, subgraph_counts, training_data, heuristic)

            iterations += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                iterations=f'{progress_bar.n}/{progress_bar.total} Iterations Completed')
            if iterations % iter_batch == 0:
                if batches is not None:
                    if iterations / iter_batch == batches:
                        break
                update_model(training_data, past, g)
                update_running(iterations, len(counters))
                training_data = []
                progress_bar = tqdm.tqdm(
                    initial=iterations, total=iterations+iter_batch, leave=False)
        update_model(training_data, past, g)
        update_running(iterations, len(counters))
        progress_bar.close()
        print("Total Iterations", iterations)

        return iterations
