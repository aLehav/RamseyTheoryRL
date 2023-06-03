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

    def check_counterexample(self, G, s, t):
        return not self.has_kn(G, s) and not self.has_independent_set_of_size_k(G, t)

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

    def count_subgraphs_from_edge(cls, self, G, u, v):
        counters = {name: 0 for name in cls.structures}
        nodes = set(range(G.vcount())) - {u, v}
        for node1, node2 in itertools.combinations(nodes, 2):
            subgraph = G.subgraph([u, v, node1, node2])
            for name, structure in cls.structures.items():
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

    def step(self, g, past, edges, s, t, unique_path, subgraph_counts, training_data: list, counters: list, heuristic):
        new_graphs = []
        vectorizations = []

        for e in edges:
            new_subgraph_counts = self.update_feature_from_edge(
                g, e[0], e[1], subgraph_counts)
            is_counterexample = self.check_counterexample(g, s, t)
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

    def bfs(cls, self, g, unique_path, past, counters, s, t, n, iter_batch, update_model, heuristic, update_running, oldIterations=0, batches=None):
        # we consider all edges
        edges = [(i, j) for i in range(n)
                 for j in range(i+1, n)]

        # Will store a list of vectors either expanded or found to be counterexamples, and upate a model after a given set of iterations
        training_data = []
        subgraph_counts = cls.count_subgraph_structures(g)
        iterations = oldIterations
        progress_bar = tqdm.tqdm(
            initial=iterations, total=iterations+iter_batch, leave=False)
        while g is not None:
            # if (parallel):
            #     # TODO Update to match step functionality
            #     g = step_par(g, past, dict(), edges, s, t,
            #                  unique_path, subgraph_counts)
            # else:
            g = self.step(g, past, edges, s, t, unique_path,
                    subgraph_counts, training_data, counters, heuristic)

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
