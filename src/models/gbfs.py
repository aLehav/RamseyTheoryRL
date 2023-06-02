import multiprocessing
import igraph as ig
import numpy as np
import tensorflow as tf
import networkx as nx
import random
import timeit
import pickle
from functools import partial
from neptune.integrations.tensorflow_keras import NeptuneCallback
import math
import tqdm
import pickle
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils.gfeatures import *
from utils.guseful import *
import utils.heuristic.train_heuristic as train
import utils.heuristic.handle_neptune as hn
import utils.heuristic.create_heuristic as ch
from models.heuristic import load_model_by_id

PROJECT = "alehav/RamseyRL"
MODEL_NAME = "RAM-HEUR"
LOAD_MODEL = False
# Choose from RANDOM, DNN, SCALED_DNN
HEURISTIC_TYPE = "SCALED_DNN"
PARAMS = {'epochs': 1, 'batch_size':32, 'optimizer':'adam', 'loss':tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.2),'last_activation':'sigmoid','pretrain':True}
N = 8
S = 3
T = 4

# TODO: Update parallel threaded processes 
def process_edge(e, G, PAST, used_edges, subgraph_counts, s, t, g6path_to_write_to, gEdgePath_to_write_to, path, heuristic):
    check = str(e[0]) + ',' + str(e[1])
    if (check in used_edges.keys()):
      return None
    # Obtain new vectorization
    # G_prime = G.copy()
    new_subgraph_counts = update_feature_from_edge(G, e[0], e[1], subgraph_counts)
    is_counterexample = check_counterexample(G, s, t)

    # output to file
    if (is_counterexample):
       write_counterexample(G=G, g6path_to_write_to=g6path_to_write_to, gEdgePath_to_write_to=gEdgePath_to_write_to, e=e, path=path)
    # Change back edited edge
    change_edge(G, e)

    vectorization = {**new_subgraph_counts, 'n': G.vcount(),
                      's': s, 't': t, 'counter': is_counterexample}
    vectorization_string = str(vectorization)
    # Assume keys in PAST are strings
    if vectorization_string not in PAST.keys():
        heuristic_val = heuristic(vectorization)
        return (heuristic_val, e, new_subgraph_counts, vectorization_string)
        new_graphs.append((heuristic_val, e, G_prime))

# We are assuming python atomic list operations are thread-safe
def step_par(G, PAST, used_edges, edges, s, t, g6path_to_write_to, gEdgePath_to_write_to, subgraph_counts, path, heuristic):
    new_edges = []
    process_edge_wrapper = partial(process_edge, G=G.copy(), PAST=PAST, used_edges=used_edges, subgraph_counts=subgraph_counts, s=s, t=t,  g6path_to_write_to=g6path_to_write_to,  gEdgePath_to_write_to=gEdgePath_to_write_to, path=path)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        new_graphs = pool.map(process_edge_wrapper, edges)

    new_graphs = [x for x in new_graphs if x is not None]
    if not new_graphs:
        return None
    
    new_graphs.sort(key=lambda x: x[0], reverse=True)  # sort by heuristic
    best_edge = (new_graphs[0][1][0], new_graphs[0][1][1]) # best edge
    path.append(best_edge)
    check = str(best_edge[0]) + ',' + str(best_edge[1])
    used_edges[check] = 1
    change_edge(G,best_edge)
    subgraph_counts.update(new_graphs[0][2])
    PAST[new_graphs[0][3]] = new_graphs[0][0]
    return G


def step(g, past, edges, s, t, g6path_to_write_to, subgraph_counts, training_data: list, heuristic):
    new_graphs = []
    vectorizations = []

    for e in edges:
        new_subgraph_counts = update_feature_from_edge(g, e[0], e[1], subgraph_counts)
        is_counterexample = check_counterexample(g, s, t)
        vectorization = {**new_subgraph_counts, 'n': g.vcount(), 's': s, 't': t, 'counter': is_counterexample}
        vectorizations.append(vectorization)
        
        if (is_counterexample):
            write_counterexample(G=g, g6path_to_write_to=g6path_to_write_to)
 
        if str(vectorization) not in past.keys():
            new_graphs.append((e, new_subgraph_counts, vectorization))

        # Change back edited edge
        change_edge(g,e)

    training_data.extend(vectorizations)

    if not new_graphs:
        return None

    heuristic_values = heuristic([vectorization for (_, _, vectorization) in new_graphs])
    max_index = max(range(len(heuristic_values)), key=heuristic_values.__getitem__)
    best = new_graphs[max_index]
    change_edge(g,best[0])
    subgraph_counts.update(best[1])
    past[str(best[2])] = heuristic_values[max_index]

    # new_graphs = [(heuristic_val, e, subgraph_counts, vec) for (heuristic_val, (e, subgraph_counts, vec)) in zip(heuristic_values, new_graphs)]
    # new_graphs.sort(key=lambda x: x[0], reverse=True)  # sort by heuristic
    # change_edge(g,new_graphs[0][1])
    # subgraph_counts.update(new_graphs[0][2])
    # past[new_graphs[0][3]] = new_graphs[0][0]

    return g

# Assume we are only adding edges

def bfs(g, g6path_to_write_to, past, s, t, PARALLEL, iter_batch, update_model, heuristic, run, model_version):
    n = g.vcount()
    # we consider all edges
    edges = [(i, j) for i in range(n)
             for j in range(i+1, n)]

    # Will store a list of vectors either expanded or found to be counterexamples, and upate a model after a given set of iterations
    training_data = []

    subgraph_counts = count_subgraph_structures(g)
    iterations = 0
    progress_bar = tqdm.tqdm(total=iter_batch, leave=False)
    while g is not None:
        if (PARALLEL):
            # TODO Update to match step functionality
            g = step_par(g, past, dict(), edges, s, t, g6path_to_write_to, subgraph_counts)
        else:
            g = step(g, past, edges, s, t, g6path_to_write_to, subgraph_counts, training_data, heuristic)

        iterations += 1
        progress_bar.update(1)
        progress_bar.set_postfix(iterations=f'{progress_bar.n}/{progress_bar.total} Iterations Completed')
        if iterations % iter_batch == 0:
            update_model(training_data, past, g)
            training_data = []
            progress_bar = tqdm.tqdm(initial=iterations, total=iterations+iter_batch, leave=False)
        
    progress_bar.close()
    print("Total Iterations", iterations)
    # TODO Output PAST to file
    # with open(PAST_path, 'wb') as f:
    #     pickle.dump(PAST, f)

    return iterations


def main():
    if LOAD_MODEL:
        MODEL_ID = "RAM-HEUR-37"
        RUN_ID = "RAM-46"
        run, model_version, model = load_model_by_id(project=PROJECT,
                                model_name=MODEL_NAME,
                                model_id=MODEL_ID,
                                run_id=RUN_ID)
    else:
        run, model_version = hn.init_neptune(params=PARAMS,
                                            project=PROJECT,
                                            model_name=MODEL_NAME)
        MODEL_ID = model_version["sys/id"].fetch()
        RUN_ID = run["sys/id"].fetch()
        model = ch.create_model(PARAMS)
        if PARAMS['pretrain']:
            TRAIN_PATH = 'data/csv/scaled/'
            # CSV_LIST = ['all_leq9','ramsey_3_4','ramsey_3_5','ramsey_3_6','ramsey_3_7','ramsey_3_9','ramsey_4_4']
            CSV_LIST = ['all_leq6']
            TRAIN_CSV_LIST = [f'{TRAIN_PATH}{CSV}.csv' for CSV in CSV_LIST]
            train_X, train_y = train.split_X_y_list(TRAIN_CSV_LIST)
            print(f"Pretraining on {train_X.shape[0]} samples.")
            neptune_cbk = hn.get_neptune_cbk(run=run)
            train.train(model=model, train_X=train_X, train_y=train_y, params=PARAMS, neptune_cbk=neptune_cbk)
            print(f"Pretrained on {train_X.shape[0]} samples.")
        train.save_trained_model(model_version=model_version, model=model)
            

    if HEURISTIC_TYPE == "RANDOM":
        def heuristic(vectorization):
            return random.random()
    elif HEURISTIC_TYPE == "DNN":
        def heuristic(vectorizations):
            X = np.array([list(vec.values())[:-1] for vec in vectorizations])
            predictions = model.predict(X, verbose=0)
            return [prediction[0] for prediction in predictions]
    elif HEURISTIC_TYPE == "SCALED_DNN":
        scaler = float(math.comb(N, 4))
        def heuristic(vectorizations):
            X = np.array([list(vec.values())[:-1] for vec in vectorizations]).astype(float)
            X[:11] /= scaler
            predictions = model.predict(X, verbose=0)
            return [prediction[0] for prediction in predictions]

    if HEURISTIC_TYPE == "RANDOM":
        def update_model(*args, **kwargs):
            pass
        PAST, G = dict(), ig.Graph.GRG(N, N/2/(N-1))
    else:
        neptune_cbk = hn.get_neptune_cbk(run)
        def save_past_and_g(past, g):
                np.save
                with open('past.pkl','wb') as file:
                    pickle.dump(past, file)
                run['running/PAST'].upload('past.pkl')

                nx_graph = nx.Graph(g.get_edgelist())
                nx.write_graph6(nx_graph, 'G.g6', header=False)
                run['running/G'].upload('G.g6')
        def load_past_and_g():
            run['running/PAST'].download('past.pkl')
            with open('past.pkl', 'rb') as file:
                past = pickle.load(file)

            run['running/G'].download('G.g6')
            g = ig.Graph.from_networkx(nx.read_graph6('G.g6'))

            return past, g
        if LOAD_MODEL:
            PAST, G = load_past_and_g()
        else:
            PAST, G = dict(), ig.Graph.GRG(N, N/2/(N-1))
        def update_model(training_data, past, g):
            X = np.array([list(vec.values())[:-1] for vec in training_data])
            y = np.array([list(vec.values())[-1] for vec in training_data])
            model.fit(X, y, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], callbacks=[neptune_cbk], verbose=1)
            train.save_trained_model(model_version, model)  
            save_past_and_g(past, g)
    n = N
    s = S
    t = T
    run['running/N'] = n
    run['running/S'] = s
    run['running/T'] = t
    ITER_BATCH = 200
    counter_path = f'data/found_counters/scaled_dnn'
    write_path = f"{counter_path}/r{s}_{t}_{n}_graph.g6"
    unique_path = f'{counter_path}/r{s}_{t}_{n}_isograph.g6'
    if os.path.exists(write_path):
        os.remove(write_path)
    if os.path.exists(unique_path):
        os.remove(unique_path)
    PARALLEL = False
    startTime = timeit.default_timer()
    iterations = bfs(G, write_path, PAST, s, t, PARALLEL, ITER_BATCH, update_model, heuristic, run, model_version)
    print(f"Single Threaded Time Elapsed: {timeit.default_timer() - startTime}")
    # startTime = timeit.default_timer()
    # G2 = ig.Graph(7)
    # bfs(G2, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, True)
    # print(f"Multi Threaded Time Elapsed: {timeit.default_timer() - startTime}")
    if os.path.exists(write_path):
        isoStartTime = timeit.default_timer()
        counters_found = get_unique_graphs(read_path=write_path,
                            write_path=unique_path)
        print(f"Iso Updater Time Elapsed: {timeit.default_timer() - isoStartTime}")
        run['running/counter_count'] = counters_found
        run['running/counters'].upload(unique_path)
        run['running/redundant_counters'].upload(write_path)
    else:
        run['running/counter_count'] = 0
    run['running/time'] = timeit.default_timer() - startTime
    run['running/iterations'] = iterations
    
    run.stop()
    model_version.stop()


if __name__ == '__main__':
    main()

# TODO if we want this pausing scheme, we need to also store all metadata along with PAST_path to resume which adds overhead
# // point of storing PAST_path is we can run for a certain amount of steps max if needed, store last G visited as g6, load up G and PAST and start from where we left off. Note: if doing this, make sure G was last thing vectorized, and not G's derived from G.
