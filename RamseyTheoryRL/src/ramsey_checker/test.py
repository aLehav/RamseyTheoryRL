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
from .ramsey_checker_multi_thread import RamseyCheckerMultiThread
from .ramsey_checker_single_thread import RamseyCheckerSingleThread
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from models.heuristic import load_model_by_id
import utils.heuristic.create_heuristic as ch
import utils.heuristic.handle_neptune as hn
import utils.heuristic.train_heuristic as train
from utils.guseful import *
from utils.gfeatures import *
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir)

class NeptuneRunner:
    def get_default_params():
        PARAMS = {'heuristic_type': "SCALED_DNN",  # Choose from RANDOM, 4PATH, DNN, SCALED_DNN
          'iter_batch': 1,  # Steps to take before updating model data / weights
          'iter_batches': 1000,  # None if no stopping value, else num. of iter_batches
          'starting_graph': "FROM_CURRENT"}  # Choose from RANDOM, FROM_PRIOR, FROM_CURRENT, EMPTY
        if PARAMS['heuristic_type'] in ["DNN", "SCALED_DNN"]:
            DNN_PARAMS = {'training_epochs': 5, 'epochs': 1, 'batch_size': 32, 'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(
                from_logits=False, label_smoothing=0.2), 'loss_info': 'BinaryCrossentropy(from_logits=False, label_smoothing=0.2)', 'last_activation': 'sigmoid', 'pretrain': True}
            PARAMS.update(DNN_PARAMS)
            if PARAMS['pretrain']:
                CSV_LIST = ['all_leq6', 'ramsey_3_4', 'ramsey_3_5',
                            'ramsey_3_6', 'ramsey_3_7', 'ramsey_3_9']
                PARAMS.update({'pretrain_data': CSV_LIST})
        if PARAMS['starting_graph'] in ["FROM_PRIOR", "FROM_CURRENT"]:
            STARTING_GRPAPH_PARAMS = {'starting_graph_path': sys.path[-1] + '/data/found_counters/r3_5_12_isograph.g6', # Mac: Absolute path
                                    'starting_graph_index': 0  # 0 is default
                                    }
            PARAMS.update(STARTING_GRPAPH_PARAMS)
        return PARAMS
    def __init__(self, n, s, t, project = f"{os.enself.viron.get('NEPTUNE_NAME')}/RamseyRL", model_name = 'RAM-HEUR', load_model=False, params=get_default_params()):
        self.N = n
        self.S = s
        self.T = t
        self.PROJECT = project
        self.MODEL_NAME = model_name
        self.LOAD_MODEL = load_model
        self.PARAMS = params
    def run(self):
        ramsey_checker = RamseyCheckerSingleThread()
        ramsey_multi = RamseyCheckerMultiThread()
        if self.LOAD_MODEL:
            MODEL_ID = "RAM-HEUR-85"
            RUN_ID = "RAM-94"
            print(f"Loading {MODEL_ID} and {RUN_ID}.")
            run, model_version, model = load_model_by_id(project=self.PROJECT,
                                                        model_name=self.MODEL_NAME,
                                                        model_id=MODEL_ID,
                                                        run_id=RUN_ID)
        else:
            run = hn.init_neptune_run(params=self.PARAMS, project=self.PROJECT)
            RUN_ID = run["sys/id"].fetch()

            if self.PARAMS['heuristic_type'] in ["SCALED_DNN", 'DNN']:
                model_version = hn.init_neptune_model_version(
                    params=self.PARAMS, project=self.PROJECT, model_name=self.MODEL_NAME)
                model = ch.create_model(self.PARAMS)
                MODEL_ID = model_version["sys/id"].fetch()
                if self.PARAMS['pretrain']:
                    # MAC: must provide absolute path
                    TRAIN_PATH = sys.path[-1] + '/data/csv/scaled/'
                    train_csv_list = [
                        f'{TRAIN_PATH}{CSV}.csv' for CSV in self.PARAMS['pretrain_data']]
                    train_X, train_y = train.split_X_y_list(train_csv_list)
                    print(f"Pretraining on {train_X.shape[0]} samples.")
                    neptune_cbk = hn.get_neptune_cbk(run=run)
                    train.train(model=model, train_X=train_X, train_y=train_y,
                                params=self.PARAMS, neptune_cbk=neptune_cbk)
                train.save_trained_model(model_version=model_version, model=model)

        if self.PARAMS['starting_graph'] == "RANDOM":
            def generate_starting_graph():
                return ig.Graph.GRG(self.N, self.N/2/(self.N-1))
        elif self.PARAMS['starting_graph'] == "EMPTY":
            def generate_starting_graph():
                return ig.Graph(n=self.N)
        elif self.PARAMS['starting_graph'] == "FROM_PRIOR":
            prior_counters = nx.read_graph6(self.PARAMS['starting_graph_path'])
            prior_counters = [prior_counters] if type(
                prior_counters) != list else prior_counters
            prior_counter = ig.Graph.from_networkx(
                prior_counters[self.PARAMS['starting_graph_index']])
            prior_counter.add_vertex()
            random.seed(42)

            # TODO add missing nodes
            while (prior_counter.vcount() < self.N):
                prior_counter.add_vertex()
            
            # TODO remove?
            for vertex_index in range(self.N-1):
                if random.random() <= 0.5:
                    prior_counter.add_edge(self.N-1, vertex_index)

            def generate_starting_graph():
                return prior_counter
        elif self.PARAMS['starting_graph'] == "FROM_CURRENT":
            prior_counters = nx.read_graph6(self.PARAMS['starting_graph_path'])
            prior_counters = [prior_counters] if type(
                prior_counters) != list else prior_counters
            prior_counter = ig.Graph.from_networkx(
                prior_counters[self.PARAMS['starting_graph_index']])

            def generate_starting_graph():
                return prior_counter
        else:
            raise ValueError("Use a valid starting_graph.")

        if self.PARAMS['starting_graph'] == "FROM_PRIOR":
            EDGES = [(self.N-1, i) for i in range(self.N-1)]
        else:
            EDGES = [(i, j) for i in range(self.N)
                    for j in range(i+1, self.N)]

        if self.PARAMS['heuristic_type'] == "RANDOM":
            def heuristic(vectorizations):
                return [random.random() for vec in vectorizations]
        elif self.PARAMS['heuristic_type'] == "4PATH":
            def heuristic(vectorizations):
                return [vec["P_4"] for vec in vectorizations]
        elif self.PARAMS['heuristic_type'] == "DNN":
            def heuristic(vectorizations):
                X = np.array([list(vec.values())[:-1] for vec in vectorizations])
                predictions = model.predict(X, verbose=0)
                return [prediction[0] for prediction in predictions]
        elif self.PARAMS['heuristic_type'] == "SCALED_DNN":
            scaler = float(math.comb(self.N, 4))

            def heuristic(vectorizations):
                X = np.array([list(vec.values())[:-1]
                            for vec in vectorizations]).astype(float)
                if (scaler != 0):
                    X[:11] /= scaler
                predictions = model.predict(X, verbose=0)
                return [prediction[0] for prediction in predictions]
        else:
            raise ValueError("Use a valid heuristic_type.")

        def save_past_and_g(past, g):
            np.save
            with open('past.pkl', 'wb') as file:
                pickle.dump(past, file)
            run['running/PAST'].upload('past.pkl')
            if g is not None:
                nx_graph = nx.Graph(g.get_edgelist())
                nx.write_graph6(nx_graph, 'G.g6', header=False)
                run['running/G'].upload('G.g6')

        if self.PARAMS['heuristic_type'] in ["RANDOM", "4PATH"]:
            def update_model(training_data, past, g):
                save_past_and_g(past, g)
            PAST = dict()
            COUNTERS = []
            G = generate_starting_graph()
            oldIterations = 0
            timeOffset = 0
        elif self.PARAMS['heuristic_type'] in ["SCALED_DNN", "DNN"]:
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
            if self.LOAD_MODEL:
                PAST, COUNTERS, G, oldIterations, timeOffset = load_data()
            else:
                PAST = dict()
                COUNTERS = []
                G = generate_starting_graph()
                oldIterations = 0
                timeOffset = 0

            def update_model(training_data, past, g):
                # Clean up training data
                print('cleaned_data-----------------------------------------', all(training_data))
                # print(training_data)
                if (training_data is None or len(training_data) == 0):
                    return
                X = np.array([list(vec.values())[:-1] for vec in training_data])
                y = np.array([list(vec.values())[-1] for vec in training_data])
                model.fit(X, y, epochs=self.PARAMS['epochs'], batch_size=self.PARAMS['batch_size'], callbacks=[
                        neptune_cbk], verbose=1)
                train.save_trained_model(model_version, model)
                save_past_and_g(past, g)

        run['running/N'] = self.N
        run['running/S'] = self.S
        run['running/T'] = self.T
        # Mac: Absolute path
        counter_path = sys.path[-1] + f'/data/found_counters/scaled_dnn'
        unique_path = f'/{counter_path}/r{self.S}_{self.T}_{self.N}_isograph.g6'
        if os.path.exists(unique_path):
            os.remove(unique_path)
        PARALLEL = False
        startTime = timeit.default_timer()

        def update_run_data(unique_path, startTime):
            def update_running(iterations, counter_count):
                if os.path.exists(unique_path):
                    run['running/counters'].upload(unique_path)
                run['running/counter_count'].append(counter_count)
                run['running/time'].append(timeit.default_timer() -
                                        startTime + timeOffset)
                run['running/iterations'].append(iterations)
            return update_running
        update_running = update_run_data(unique_path, startTime)
        ramsey_checker.bfs(g=G, unique_path=unique_path, past=PAST, counters=COUNTERS, s=self.S, t=self.T, n=self.N,
            iter_batch=self.PARAMS['iter_batch'], update_model=update_model, heuristic=heuristic, update_running=update_running, oldIterations=oldIterations, batches=self.PARAMS['iter_batches'], edges=EDGES)
        print(
            f"Single Threaded Time Elapsed: {timeit.default_timer() - startTime}")
        # startTime = timeit.default_timer()
        # G2 = ig.Graph(7)
        # bfs(G2, g6path_to_write_to, gEdgePath_to_write_to, PAST_path, s, t, True)
        # print(f"Multi Threaded Time Elapsed: {timeit.default_timer() - startTime}")
        run.stop()
        if self.PARAMS['heuristic_type'] in ["SCALED_DNN", "DNN"]:
            model_version.stop()


if __name__ == '__main__':
    runner = NeptuneRunner()
    runner.run()
