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
import random
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

PROJECT = f"{os.environ.get('NEPTUNE_NAME')}/RamseyRL"
MODEL_NAME = "RAM-HEUR"
LOAD_MODEL = False

PARAMS = {'heuristic_type':"SCALED_DNN", # Choose from RANDOM, 4PATH, DNN, SCALED_DNN
          'iter_batch':1, # Steps to take before updating model data / weights
          'iter_batches':20, # None if no stopping value, else num. of iter_batches
          'starting_graph':"FROM_CURRENT"} # Choose from RANDOM, FROM_PRIOR, FROM_CURRENT, EMPTY
if PARAMS['heuristic_type'] in ["DNN", "SCALED_DNN"]:
    DNN_PARAMS = {'training_epochs': 5, 'epochs': 1, 'batch_size':32, 'optimizer':'adam', 'loss':tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.2), 'loss_info':'BinaryCrossentropy(from_logits=False, label_smoothing=0.2)', 'last_activation':'sigmoid','pretrain':True}
    PARAMS.update(DNN_PARAMS)
    if PARAMS['pretrain']:
        CSV_LIST = ['all_leq6', 'ramsey_3_4', 'ramsey_3_5', 'ramsey_3_6', 'ramsey_3_7', 'ramsey_3_9']
        PARAMS.update({'pretrain_data':CSV_LIST})
if PARAMS['starting_graph'] in ["FROM_PRIOR", "FROM_CURRENT"]:
    STARTING_GRPAPH_PARAMS = {'starting_graph_path':'data/found_counters/r3_10_35_isograph.g6',
    'starting_graph_index':0 # 0 is default
    }
    PARAMS.update(STARTING_GRPAPH_PARAMS)

N = 35
S = 4
T = 6


def step(g, past, edges, s, t, unique_path, subgraph_counts, past_state,  training_data: list, counters: list, heuristic):
    new_graphs = []
    vectorizations = []

    for e in edges:
        new_subgraph_counts = update_feature_from_edge(g, e[0], e[1], subgraph_counts)
        is_counterexample = check_counterexample_from_edge(G=g, s=s, t=t, subgraph_counts=new_subgraph_counts, e=e, past_state=past_state)
        vectorization = {**new_subgraph_counts, 'n': g.vcount(), 's': s, 't': t, 'counter': is_counterexample}
        vectorizations.append(vectorization)
        
        if (is_counterexample):
            consider_counterexample(G=g, counters=counters, counter_path=unique_path)
 
        if str(vectorization) not in past.keys():
            new_graphs.append((e, new_subgraph_counts, vectorization))

        change_edge(g,e)

    training_data.extend(vectorizations)

    if not new_graphs:
        return None, False

    heuristic_values = heuristic([vectorization for (_, _, vectorization) in new_graphs])
    max_index = max(range(len(heuristic_values)), key=heuristic_values.__getitem__)
    best = new_graphs[max_index]
    change_edge(g,best[0])
    subgraph_counts.update(best[1])
    past[str(best[2])] = heuristic_values[max_index]
    new_state = best[2]['counter']

    return g, new_state

def bfs(g, unique_path, past, counters, s, t, n, parallel, iter_batch, update_model, heuristic, update_running, edges, oldIterations=0, batches=None):
    # Will store a list of vectors either expanded or found to be counterexamples, and upate a model after a given set of iterations
    training_data = []
    subgraph_counts = count_subgraph_structures(g)
    iterations = oldIterations
    state = check_counterexample(G=g, s=s, t=t, subgraph_counts=subgraph_counts)
    progress_bar = tqdm.tqdm(initial=iterations, total=iterations+iter_batch, leave=False)
    while g is not None:
        if (parallel):
            # TODO Update to match step functionality
            g = step_par(g, past, dict(), edges, s, t, unique_path, subgraph_counts)
        else:
            g, state = step(g, past, edges, s, t, unique_path, subgraph_counts, state, training_data, counters, heuristic)

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
        run = hn.init_neptune_run(params=PARAMS, project=PROJECT)
        RUN_ID = run["sys/id"].fetch()

        if PARAMS['heuristic_type'] in ["SCALED_DNN", 'DNN']:
            model_version = hn.init_neptune_model_version(params=PARAMS, project=PROJECT, model_name=MODEL_NAME)
            model = ch.create_model(PARAMS)
            MODEL_ID = model_version["sys/id"].fetch()
            if PARAMS['pretrain']:
                TRAIN_PATH = 'data/csv/scaled/'                
                train_csv_list = [f'{TRAIN_PATH}{CSV}.csv' for CSV in PARAMS['pretrain_data']]
                train_X, train_y = train.split_X_y_list(train_csv_list)
                print(f"Pretraining on {train_X.shape[0]} samples.")
                neptune_cbk = hn.get_neptune_cbk(run=run)
                train.train(model=model, train_X=train_X, train_y=train_y, params=PARAMS, neptune_cbk=neptune_cbk)
            train.save_trained_model(model_version=model_version, model=model)
      
    if PARAMS['starting_graph'] == "RANDOM":
        def generate_starting_graph():
            return ig.Graph.GRG(N, N/2/(N-1))
    elif PARAMS['starting_graph'] == "EMPTY":
        def generate_starting_graph():
            return ig.Graph(n=N)
    elif PARAMS['starting_graph'] == "FROM_PRIOR":
        prior_counters = nx.read_graph6(PARAMS['starting_graph_path'])
        prior_counters = [prior_counters] if type(prior_counters) != list else prior_counters
        prior_counter = ig.Graph.from_networkx(prior_counters[PARAMS['starting_graph_index']])
        prior_counter.add_vertex()
        random.seed(42)
        for vertex_index in range(N-1):
            if random.random() <= 0.5:
                prior_counter.add_edge(N-1, vertex_index)
        def generate_starting_graph():
            return prior_counter
    elif PARAMS['starting_graph'] == "FROM_CURRENT":
        prior_counters = nx.read_graph6(PARAMS['starting_graph_path'])
        prior_counters = [prior_counters] if type(prior_counters) != list else prior_counters
        prior_counter = ig.Graph.from_networkx(prior_counters[PARAMS['starting_graph_index']])
        def generate_starting_graph():
            return prior_counter
    else:
        raise ValueError("Use a valid starting_graph.")
    
    if PARAMS['starting_graph'] == "FROM_PRIOR":
        EDGES = [(N-1,i) for i in range(N-1)]
    else:
        EDGES = [(i, j) for i in range(N)
             for j in range(i+1,N)]  
        
    if PARAMS['heuristic_type'] == "RANDOM":
        def heuristic(vectorizations):
            return [random.random() for vec in vectorizations]
    elif PARAMS['heuristic_type'] == "4PATH":
        def heuristic(vectorizations):
            return [vec["P_4"] for vec in vectorizations]
    elif PARAMS['heuristic_type'] == "DNN":
        def heuristic(vectorizations):
            X = np.array([list(vec.values())[:-1] for vec in vectorizations])
            predictions = model.predict(X, verbose=0)
            return [prediction[0] for prediction in predictions]
    elif PARAMS['heuristic_type'] == "SCALED_DNN":
        scaler = float(math.comb(N, 4))
        def heuristic(vectorizations):
            X = np.array([list(vec.values())[:-1] for vec in vectorizations]).astype(float)
            X[:11] /= scaler
            predictions = model.predict(X, verbose=0)
            return [prediction[0] for prediction in predictions]
    else:
        raise ValueError("Use a valid heuristic_type.")

    def save_past_and_g(past, g):
        np.save
        with open('past.pkl','wb') as file:
            pickle.dump(past, file)
        run['running/PAST'].upload('past.pkl')
        if g is not None:
            nx_graph = nx.Graph(g.get_edgelist())
            nx.write_graph6(nx_graph, 'G.g6', header=False)
            run['running/G'].upload('G.g6')

    if PARAMS['heuristic_type'] in ["RANDOM","4PATH"]:
        def update_model(training_data, past, g):
            save_past_and_g(past, g)
        PAST = dict()
        COUNTERS = []
        G = generate_starting_graph()
        oldIterations = 0
        timeOffset = 0
    elif PARAMS['heuristic_type'] in ["SCALED_DNN","DNN"]:
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
            G = generate_starting_graph()
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
    bfs(g=G, unique_path=unique_path, past=PAST, counters=COUNTERS, s=S, t=T, n=N, parallel=PARALLEL, iter_batch=PARAMS['iter_batch'], update_model=update_model, heuristic=heuristic, update_running=update_running, oldIterations=oldIterations, batches=PARAMS['iter_batches'], edges=EDGES)
    print(f"Single Threaded Time Elapsed: {timeit.default_timer() - startTime}")
    # startTime = timeit.default_timer()
    # G2 = ig.Graph(7)
    # bfs(G2, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, True)
    # print(f"Multi Threaded Time Elapsed: {timeit.default_timer() - startTime}")    
    run.stop()
    if PARAMS['heuristic_type'] in ["SCALED_DNN", "DNN"]:
        model_version.stop()


if __name__ == '__main__':
    main()