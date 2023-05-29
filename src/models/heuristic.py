import sys
import os
import zipfile
from tensorflow import keras
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import utils.heuristic.create_heuristic as ch
import utils.heuristic.train_heuristic as train
import utils.heuristic.test_heuristic as test
import utils.heuristic.handle_neptune as hn

def load_model_by_id(project,
                    model_name,
                    model_id,
                    run_id,
                    for_running=False):
    if not for_running:
        run, model_version = hn.load_neptune(project=project,
                                    model_name=model_name,
                                    model_id=model_id,
                                    run_id=run_id,
                                    for_running=for_running)
        
        model_version['saved_model'].download('my_model')
        with zipfile.ZipFile('my_model/saved_model.zip', 'r') as zip_ref:
            zip_ref.extractall('/')
        model = keras.models.load_model('my_model')
        return run, model_version, model
    else:
        model_version = hn.load_neptune(project=project,
                                    model_name=model_name,
                                    model_id=model_id,
                                    run_id=run_id,
                                    for_running=for_running)
        
        model_version['saved_model'].download('my_model')
        with zipfile.ZipFile('my_model/saved_model.zip', 'r') as zip_ref:
            zip_ref.extractall('/')
        model_version.stop()
        model = keras.models.load_model('my_model')
        return model


def main():
    PROJECT = "alehav/RamseyRL"
    MODEL_NAME = "RAM-HEUR"
    PARAMS = {'epochs': 100, 'batch_size':16, 'optimizer':'adam'}
    # Run with one CSV_PATH for 80/20 test/train split, and TRAIN_CSV_LIST and TEST_CSV_LIST otherwise
    BATCH_TRAIN_TEST = True

    TRAIN = False
    TEST = True
    MODEL_ID = "RAM-HEUR-31"
    RUN_ID = "RAM-40"

    if BATCH_TRAIN_TEST:
        if TRAIN:
            TRAIN_CSV_LIST = ['data/csv/all_leq6.csv',
                        'data/csv/ramsey_3_4.csv',
                        'data/csv/ramsey_3_5.csv',
                        'data/csv/ramsey_3_6.csv',
                        'data/csv/ramsey_3_7.csv']
            train_X, train_y = train.split_X_y_list(TRAIN_CSV_LIST)
            print(f"train_X shape: {train_X.shape}, train_y shape: {train_y.shape}.")
        if TEST:
            TEST_CSV_LIST = ['data/csv/all_leq6.csv',
                             'data/csv/ramsey_3_6.csv',
                             'data/csv/ramsey_3_7.csv',
                             'data/csv/ramsey_3_9.csv']
            test_X, test_y = train.split_X_y_list(TEST_CSV_LIST)
            print(f"test_X shape: {test_X.shape}, test_y shape: {test_y.shape}.")
            print(f"test_y distribution: {pd.Series(test_y).value_counts()}")
    else:
        CSV_PATH = 'train_gen.csv'

        train_X, test_X, train_y, test_y = train.split_test_train(csv_path=CSV_PATH)
    neptune_objects = []
    if TRAIN:
        run, model_version = hn.init_neptune(params=PARAMS,
                                        project=PROJECT,
                                        model_name=MODEL_NAME)
        if TEST:
            RUN_ID = run["sys/id"].fetch()
            MODEL_ID = model_version["sys/id"].fetch()
        neptune_objects += [run, model_version]
        neptune_cbk = hn.get_neptune_cbk(run=run)
        model = ch.create_model()

        train.train(model=model,
                train_X=train_X,
                train_y=train_y,
                params=PARAMS,
                neptune_cbk=neptune_cbk)
        train.save_trained_model(model_version=model_version,
                                model=model)
    if TEST:
        run, model_version, model = load_model_by_id(project=PROJECT,
                                                    model_name=MODEL_NAME,
                                                    model_id=MODEL_ID,
                                                    run_id=RUN_ID)
        if not TRAIN:
            neptune_objects += [run, model_version]
        test.test(model=model,
                test_X=test_X,
                test_y=test_y,
                run=run)

    hn.stop_neptune_objects(neptune_objects=neptune_objects)

if __name__ == '__main__':
    main()