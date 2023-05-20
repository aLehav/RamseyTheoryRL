import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import neptune
import os
from neptune.integrations.tensorflow_keras import NeptuneCallback
import glob


# Retrieve the API token from the environment variable
api_token = os.environ.get('NEPTUNE_API_TOKEN')

run = neptune.init_run(
    project="alehav/RamseyRL",
    api_token=api_token
)
neptune_cbk = NeptuneCallback(run=run, base_namespace="training")

# neptune_model = neptune.init_model(
#     name="Heuristic estimator",
#     key='HEUR',
#     project="alehav/RamseyRL",
#     api_token=api_token
# )
neptune_model = neptune.init_model(
    with_id='RAM-HEUR',
    project='alehav/RamseyRL',
    api_token=api_token)
model_version = neptune.init_model_version(
    model='RAM-HEUR',
    project='alehav/RamseyRL',
    api_token=api_token)


params = {'epochs': 10, 'batch_size':16, 'optimizer':'adam'}
run['parameters'] = params

def create_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(12,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def split_test_train(csv_path):
    data = pd.read_csv(csv_path, index_col=0)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    return train_X, test_X, train_y, test_y

def train(model, train_X, train_y):
    model.fit(train_X, 
            train_y, 
            epochs=params['epochs'], 
            batch_size=params['batch_size'],
            callbacks=[neptune_cbk])

def test(model, test_X, test_y):
    eval_metrics = model.evaluate(test_X, test_y, verbose=0)
    for j, metric in enumerate(eval_metrics):
        run["eval/{}".format(model.metrics_names[j])] = metric

# Example usage
csv_path = 'train_gen.csv'
train_X, test_X, train_y, test_y = split_test_train(csv_path)
model = create_model()
train(model, train_X, train_y)
test(model, test_X, test_y)

model.save("my_model")
model_version["saved_model"].upload("my_model/saved_model.pb")

for name in glob.glob("variables/*"):
    model_version[name].upload(name)


run.stop()
neptune_model.stop()

