from ramsey_checker import RamseyChecker
import itertools
import sys
import networkx as nx
import tqdm


class RamseyCheckerSingleThread(RamseyChecker):

    def has_kn(self, G, k):
        for sub_nodes in itertools.combinations(range(G.vcount()), k):
            subgraph = G.subgraph(sub_nodes)
            if self.is_complete(subgraph):
                return True
        return False

    def has_independent_set_of_size_k(self, G, k):
        for sub_nodes in itertools.combinations(range(G.vcount()), k):
            subgraph = G.subgraph(sub_nodes)
            if subgraph.ecount() == 0:
                return True
        return False

    def check_counterexample(self, G, s, t, subgraph_counts):
        if s == 3:
            if subgraph_counts["K_4"] + subgraph_counts["K_4-e"] + subgraph_counts["K_3+e"] + subgraph_counts["K_3"] > 0:
                return False
        elif s == 4:
            if subgraph_counts["K_4"] > 0:
                return False
        else:
            if self.has_kn(G, s):
                return False

        if t == 4:
            if subgraph_counts["E_4"] > 0:
                return False
        else:
            if self.has_independent_set_of_size_k(G, t):
                return False

        return True
    
    def check_counterexample_from_edge(self, G, s, t, subgraph_counts, e, past_state):
        if s == 3:
            if subgraph_counts["K_4"] + subgraph_counts["K_4-e"] + subgraph_counts["K_3+e"] + subgraph_counts["K_3"] > 0:
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
                if self.has_independent_set_of_size_k(G, t):
                    return False

        return True

    def consider_counterexample(self, G, counters, counter_path):
        nx_graph = nx.Graph(G.get_edgelist())
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

    def count_subgraphs_from_edge(self, G, u, v):
        counters = {name: 0 for name in self.structures}
        nodes = set(range(G.vcount())) - {u, v}
        for node1, node2 in itertools.combinations(nodes, 2):
            subgraph = G.subgraph([u, v, node1, node2])
            for name, structure in self.structures.items():
                if subgraph.isomorphic(structure):
                    counters[name] += 1
        return counters

    def update_feature_from_edge(self, G, u, v, counters):
        old_count = self.count_subgraphs_from_edge(G, u, v)
        # Change edge
        self.change_edge(G, (u, v))
        new_count = self.count_subgraphs_from_edge(G, u, v)
        # Make and edit a copy of counters such that we can pass it to all children
        new_counters = {}
        for name in counters:
            new_counters[name] = counters[name] + \
                (new_count[name] - old_count[name])
        return new_counters

    def step(self, g, past, edges, s, t, unique_path, subgraph_counts, past_state, training_data: list, counters: list, heuristic):
        new_graphs = []
        vectorizations = []

        for e in edges:
            new_subgraph_counts = self.update_feature_from_edge(
                g, e[0], e[1], subgraph_counts)
            is_counterexample = self.check_counterexample(
                g, s, t, subgraph_counts=new_subgraph_counts)
            vectorization = {**new_subgraph_counts, 'n': g.vcount(),
                            's': s, 't': t, 'counter': is_counterexample}
            vectorizations.append(vectorization)

            if (is_counterexample):
                self.consider_counterexample(
                    G=g, counters=counters, counter_path=unique_path)

            if str(vectorization) not in past.keys():
                new_graphs.append((e, new_subgraph_counts, vectorization))

            self.change_edge(g, e)

        training_data.extend(vectorizations)

        if not new_graphs:
            return None, None

        heuristic_values = heuristic(
            [vectorization for (_, _, vectorization) in new_graphs])
        max_index = max(range(len(heuristic_values)),
                        key=heuristic_values.__getitem__)
        best = new_graphs[max_index]
        self.change_edge(g, best[0])
        subgraph_counts.update(best[1])
        past[str(best[2])] = heuristic_values[max_index]
        new_state = best[2]['counter']

        return g, new_state

    def bfs(self, g, unique_path, past, counters, s, t, n, iter_batch, update_model, heuristic, update_running, edges, oldIterations=0, batches=None):
        # Will store a list of vectors either expanded or found to be counterexamples, and upate a model after a given set of iterations
        training_data = []
        subgraph_counts = self.count_subgraph_structures(g)
        iterations = oldIterations
        progress_bar = tqdm.tqdm(
            initial=iterations, total=iterations+iter_batch, leave=False)
        state = self.check_counterexample(
            G=g, s=s, t=t, subgraph_counts=subgraph_counts)
        while g is not None:
            g, state = self.step(g, past, edges, s, t, unique_path,
                    subgraph_counts, state, training_data, counters, heuristic)

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
        print("\nTotal Iterations", iterations)

        return iterations
