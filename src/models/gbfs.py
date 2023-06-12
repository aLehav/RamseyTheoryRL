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

PROJECT = "rinzzier/RamseyRL"
MODEL_NAME = "RAM-HEUR"
LOAD_MODEL = False
# Choose from RANDOM, 4PATH, DNN, SCALED_DNN
HEURISTIC_TYPE = "4PATH"
# Steps to take before updating model data / weights
ITER_BATCH = 200 
PARAMS = {'heuristic_type':HEURISTIC_TYPE, 'iter_batch':ITER_BATCH}
if HEURISTIC_TYPE in ["DNN", "SCALED_DNN"]:
    DNN_PARAMS = {'training_epochs': 1, 'epochs': 1, 'batch_size':32, 'optimizer':'adam', 'loss':tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.2),'last_activation':'sigmoid','pretrain':True}
    PARAMS.update(DNN_PARAMS)
    if PARAMS['pretrain']:
        CSV_LIST = ['all_leq6','ramsey_3_4']
        PARAMS.update({'pretrain_data':CSV_LIST})

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
    new_subgraph_counts = update_feature_from_edge_parBfs(G, e[0], e[1], subgraph_counts)
    is_counterexample = check_counterexample_parBfs(G, s, t, e)

    # output to file
    if (is_counterexample):
       consider_counterexample(G=G, g6path_to_write_to=g6path_to_write_to, gEdgePath_to_write_to=gEdgePath_to_write_to, e=e, path=path)
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


def step(g, past, edges, s, t, unique_path, subgraph_counts, training_data: list, counters: list, heuristic):
    new_graphs = []
    vectorizations = []

    for e in edges:
        new_subgraph_counts = update_feature_from_edge(g, e[0], e[1], subgraph_counts)
        is_counterexample = check_counterexample(g, s, t)
        vectorization = {**new_subgraph_counts, 'n': g.vcount(), 's': s, 't': t, 'counter': is_counterexample}
        vectorizations.append(vectorization)
        
        if (is_counterexample):
            consider_counterexample(G=g, counters=counters, counter_path=unique_path)
 
        if str(vectorization) not in past.keys():
            new_graphs.append((e, new_subgraph_counts, vectorization))

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

    return g

def bfs(g, unique_path, past, counters, s, t, n, parallel, iter_batch, update_model, heuristic, update_running, oldIterations=0, batches=None):
    # we consider all edges
    edges = [(i, j) for i in range(n)
             for j in range(i+1, n)]

    # Will store a list of vectors either expanded or found to be counterexamples, and upate a model after a given set of iterations
    training_data = []
    subgraph_counts = count_subgraph_structures(g)
    iterations = oldIterations
    progress_bar = tqdm.tqdm(initial=iterations, total=iterations+iter_batch, leave=False)
    while g is not None:
        if (parallel):
            # TODO Update to match step functionality
            g = step_par(g, past, dict(), edges, s, t, unique_path, subgraph_counts)
        else:
            g = step(g, past, edges, s, t, unique_path, subgraph_counts, training_data, counters, heuristic)

        iterations += 1
        progress_bar.update(1)
        progress_bar.set_postfix(iterations=f'{progress_bar.n}/{progress_bar.total} Iterations Completed')
        if iterations % iter_batch == 0:
            if batches is not None:
                if iterations / iter_batch == batches: 
                    break
            update_model(training_data, past, g)
            update_running(iterations, len(counters))
            training_data = []
            progress_bar = tqdm.tqdm(initial=iterations, total=iterations+iter_batch, leave=False)
    update_model(training_data, past, g)
    update_running(iterations, len(counters))
    progress_bar.close()
    print("Total Iterations", iterations)

    return iterations


def main():
    if LOAD_MODEL:
        MODEL_ID = "RAM-HEUR-85"
        RUN_ID = "RAM-94"
        print(f"Loading {MODEL_ID} and {RUN_ID}.")
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
        if HEURISTIC_TYPE == "SCALED_DNN" or HEURISTIC_TYPE == 'DNN':
            model = ch.create_model(PARAMS)
            if PARAMS['pretrain']:
                TRAIN_PATH = 'data/csv/scaled/'                
                train_csv_list = [f'{TRAIN_PATH}{CSV}.csv' for CSV in PARAMS['pretrain_data']]
                train_X, train_y = train.split_X_y_list(train_csv_list)
                print(f"Pretraining on {train_X.shape[0]} samples.")
                neptune_cbk = hn.get_neptune_cbk(run=run)
                train.train(model=model, train_X=train_X, train_y=train_y, params=PARAMS, neptune_cbk=neptune_cbk)
            train.save_trained_model(model_version=model_version, model=model)
            

    if HEURISTIC_TYPE == "RANDOM":
        def heuristic(vectorizations):
            return [random.random() for vec in vectorizations]
    elif HEURISTIC_TYPE == "4PATH":
        def heuristic(vectorizations):
            return [vec["P_4"] for vec in vectorizations]
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

    def save_past_and_g(past, g):
                np.save
                with open('past.pkl','wb') as file:
                    pickle.dump(past, file)
                run['running/PAST'].upload('past.pkl')
                if g is not None:
                    nx_graph = nx.Graph(g.get_edgelist())
                    nx.write_graph6(nx_graph, 'G.g6', header=False)
                    run['running/G'].upload('G.g6')

    if HEURISTIC_TYPE in ["RANDOM","4PATH"]:
        def update_model(training_data, past, g):
            save_past_and_g(past, g)
        PAST = dict()
        COUNTERS = []
        G = ig.Graph.GRG(N, N/2/(N-1))
        oldIterations = 0
        timeOffset = 0
    elif HEURISTIC_TYPE in ["SCALED_DNN","DNN"]:
        neptune_cbk = hn.get_neptune_cbk(run)
        def load_data():
            run['running/PAST'].download('past.pkl')
            with open('past.pkl', 'rb') as file:
                past = pickle.load(file)

            run['running/G'].download('G.g6')
            g = ig.Graph.from_networkx(nx.read_graph6('G.g6'))

            if run.exists('running/counters'):
                run['running/counters'].download('counters.g6')
                counters = nx.read_graph6('counters.g6')
                counters = [counters] if type(counters) != list else counters
            else:
                counters = []
            
            oldIterations = run['running/iterations'].fetch_last()
            timeOffset = run['running/time'].fetch_last()
            return past, counters, g, oldIterations, timeOffset
        if LOAD_MODEL:
            PAST, COUNTERS, G, oldIterations, timeOffset = load_data()
        else:
            PAST = dict()
            COUNTERS = []
            G = ig.Graph.GRG(N, N/2/(N-1))
            oldIterations = 0
            timeOffset = 0
        def update_model(training_data, past, g):
            X = np.array([list(vec.values())[:-1] for vec in training_data])
            y = np.array([list(vec.values())[-1] for vec in training_data])
            model.fit(X, y, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], callbacks=[neptune_cbk], verbose=1)
            train.save_trained_model(model_version, model)  
            save_past_and_g(past, g)
    
    run['running/N'] = N
    run['running/S'] = S
    run['running/T'] = T
    # BATCHES = 5
    BATCHES = None
    counter_path = f'data/found_counters/scaled_dnn'
    unique_path = f'{counter_path}/r{S}_{T}_{N}_isograph.g6'
    if os.path.exists(unique_path):
        os.remove(unique_path)
    PARALLEL = False
    startTime = timeit.default_timer()
    def update_run_data(unique_path, startTime):
        def update_running(iterations, counter_count):
            if os.path.exists(unique_path):
                run['running/counters'].upload(unique_path)
            run['running/counter_count'].append(counter_count)
            run['running/time'].append(timeit.default_timer() - startTime + timeOffset)
            run['running/iterations'].append(iterations)
        return update_running
    update_running = update_run_data(unique_path, startTime)
    bfs(g=G, unique_path=unique_path, past=PAST, counters=COUNTERS, s=S, t=T, n=N, parallel=PARALLEL, iter_batch=PARAMS['iter_batch'], update_model=update_model, heuristic=heuristic, update_running=update_running, oldIterations=oldIterations, batches=BATCHES)
    print(f"Single Threaded Time Elapsed: {timeit.default_timer() - startTime}")
    # startTime = timeit.default_timer()
    # G2 = ig.Graph(7)
    # bfs(G2, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, True)
    # print(f"Multi Threaded Time Elapsed: {timeit.default_timer() - startTime}")    
    run.stop()
    model_version.stop()


if __name__ == '__main__':
    main()