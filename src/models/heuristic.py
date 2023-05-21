import sys
import os
import zipfile
from tensorflow import keras
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import utils.heuristic.create_heuristic as ch
import utils.heuristic.train_heuristic as train
import utils.heuristic.test_heuristic as test
import utils.heuristic.handle_neptune as hn

PROJECT = "alehav/RamseyRL"
MODEL_NAME = "RAM-HEUR"
PARAMS = {'epochs': 10, 'batch_size':16, 'optimizer':'adam'}
CSV_PATH = 'train_gen.csv'
TRAIN = False
TEST = True
MODEL_ID = "RAM-HEUR-26"
RUN_ID = "RAM-35"

train_X, test_X, train_y, test_y = train.split_test_train(csv_path=CSV_PATH)

neptune_objects = []
if TRAIN:
    run, model_version = hn.init_neptune(params=PARAMS,
                                     project=PROJECT,
                                     model_name=MODEL_NAME)
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
    run, model_version = hn.load_neptune(project=PROJECT,
                                model_name=MODEL_NAME,
                                model_id=MODEL_ID,
                                run_id=RUN_ID)
    neptune_objects += [run, model_version]
    model_version['saved_model'].download('my_model')
    with zipfile.ZipFile('my_model/saved_model.zip', 'r') as zip_ref:
        zip_ref.extractall('/')
    model = keras.models.load_model('my_model')
    test.test(model=model,
              test_X=test_X,
              test_y=test_y,
              run=run)

hn.stop_neptune_objects(neptune_objects=neptune_objects)
