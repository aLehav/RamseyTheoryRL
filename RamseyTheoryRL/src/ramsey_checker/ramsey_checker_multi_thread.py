from .ramsey_checker import RamseyChecker
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
            if all(node in sub_nodes for node in e):
                # Map the original node IDs to the indices in the sub_nodes list
                node_map = {node_id: i for i, node_id in enumerate(sub_nodes)}
                # Get the edge in the subgraph corresponding to e in the original graph
                subgraph_edge = tuple(node_map[node_id] for node_id in e)
                self.change_edge(subgraph, subgraph_edge)

            if self.is_complete(subgraph):
                return True
        return False

    def has_independent_set_of_size_k(self, G, k, e):
        for sub_nodes in itertools.combinations(range(G.vcount()), k):
            subgraph = G.subgraph(sub_nodes)
            if all(node in sub_nodes for node in e):
                # Map the original node IDs to the indices in the sub_nodes list
                node_map = {node_id: i for i, node_id in enumerate(sub_nodes)}

                # Get the edge in the subgraph corresponding to e in the original graph
                subgraph_edge = tuple(node_map[node_id] for node_id in e)
                self.change_edge(subgraph, subgraph_edge)
            if subgraph.ecount() == 0:
                return True
        return False

    def has_independent_set_of_size_k_start(self, G, k):
        for sub_nodes in itertools.combinations(range(G.vcount()), k):
            subgraph = G.subgraph(sub_nodes)
            if subgraph.ecount() == 0:
                return True
        return False

    def check_counterexample(self, G, s, t, subgraph_counts):
        if s == 3:
            if subgraph_counts["K_4"] + subgraph_counts["K_4-e"] + subgraph_counts["K_3+e"] + subgraph_counts["K_3"] > 0:
                return False
            # Check table and inequality
            edge_count = G.ecount()
            n = G.vcount()
            bound = self.edge_bounds[n][t]
            if (bound != -1 and edge_count != bound):
                return False
            # e(3,k+1,n) >= (40n-91k)/6
            first_bound = (40*n - 91*(t-1))/6
            # e(3,k+1,n) >= 6n-13k
            second_bound = 6*n - 13*(t-1)
            best_bound = min(first_bound, second_bound)
            if (edge_count < best_bound):
                return False
        elif s == 4:
            if subgraph_counts["K_4"] > 0:
                return False
        else:
            # Check table and inequality
            if self.has_kn(G, s):
                return False

        if t == 4:
            if subgraph_counts["E_4"] > 0:
                return False
        else:
            # Check table and inequality

            if self.has_independent_set_of_size_k_start(G, t):
                return False

        return True

    def check_counterexample_from_edge(self, G, s, t, subgraph_counts, e, past_state):
        if s == 3:
            if subgraph_counts["K_4"] + subgraph_counts["K_4-e"] + subgraph_counts["K_3+e"] + subgraph_counts["K_3"] > 0:
                return False
            # Check table and inequality
            if (G.are_connected(*e)):
                offset = -1
            else:
                offset = 1
            edge_count = G.ecount() + offset
            edge_count = G.ecount()
            n = G.vcount()
            bound = self.edge_bounds[n][t]
            if (bound != -1 and edge_count != bound):
                return False
            # e(3,k+1,n) >= (40n-91k)/6
            first_bound = (40*n - 91*(t-1))/6
            # e(3,k+1,n) >= 6n-13k
            second_bound = 6*n - 13*(t-1)
            best_bound = min(first_bound,second_bound)
            if (edge_count < best_bound):
                return False
        elif s == 4:
            if subgraph_counts["K_4"] > 0:
                return False
        else:
            if past_state == True:
                if self.has_kn_from_edge(G, s, e):
                    return False
            else:
                if self.has_kn(G, s):
                    return False
        if t == 4:
            if subgraph_counts["E_4"] > 0:
                return False
        else:
            if past_state == True:
                if self.has_independent_set_of_size_k_from_edge(G, t, e):
                    return False
            else:
                if self.has_independent_set_of_size_k(G, t, e):
                    return False

        return True

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
                self.change_edge(subgraph, (0, 1))
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

    def process_edge(self, e, g, past, subgraph_counts, s, t, unique_path, training_data, counters, past_state):
        new_subgraph_counts = self.update_feature_from_edge(
            g, e[0], e[1], subgraph_counts)
        is_counterexample = self.check_counterexample_from_edge(g, s, t, new_subgraph_counts, e, past_state)
        vectorization = {**new_subgraph_counts, 'n': g.vcount(),
                         's': s, 't': t, 'counter': is_counterexample}

        if (is_counterexample):
            self.consider_counterexample(
                G=g, counters=counters, counter_path=unique_path, e=e)

        training_data.append(vectorization)

        # Assume keys in PAST are strings
        if str(vectorization) not in past.keys():
            return (e, new_subgraph_counts, vectorization)

    def step_par(self, g, past, edges, s, t, unique_path, subgraph_counts, training_data, counters, heuristic, past_state):
        process_edge_wrapper = partial(self.process_edge, g=g, past=past, subgraph_counts=subgraph_counts,
                                       s=s, t=t,  unique_path=unique_path, training_data=training_data, counters=counters, past_state=past_state)

        cpu_count = multiprocessing.cpu_count()
        cpu_count = max(cpu_count-1,1)
        with multiprocessing.Pool(processes=cpu_count) as pool:
            new_graphs = pool.map(process_edge_wrapper, edges)

        new_graphs = [x for x in new_graphs if x is not None]

        if not new_graphs:
            return None

        heuristic_values = heuristic(
            [vectorization for (_, _, vectorization) in new_graphs])
        max_index = max(range(len(heuristic_values)),
                        key=heuristic_values.__getitem__)
        best = new_graphs[max_index]
        self.change_edge(g, best[0])
        subgraph_counts.update(best[1])
        past[str(best[2])] = heuristic_values[max_index]
        return g

    def bfs(self, g, unique_path, past, counters, s, t, n, iter_batch, update_model, heuristic, update_running, edges, oldIterations=0, batches=None):
        # Will store a list of vectors either expanded or found to be counterexamples, and upate a model after a given set of iterations
        training_data = []
        subgraph_counts = self.count_subgraph_structures(g)
        iterations = oldIterations
        state = self.check_counterexample(
            G=g, s=s, t=t, subgraph_counts=subgraph_counts)
        progress_bar = tqdm.tqdm(
            initial=iterations, total=iterations+iter_batch, leave=False)
        while g is not None:
            g = self.step_par(g, past, edges, s, t,
                              unique_path, subgraph_counts, training_data, counters, heuristic, state)

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
