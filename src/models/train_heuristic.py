import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
import glob

def split_test_train(csv_path):
    data = pd.read_csv(csv_path, index_col=0)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    return train_X, test_X, train_y, test_y

def train(model: keras.Sequential, 
          train_X, 
          train_y, 
          params, 
          neptune_cbk: NeptuneCallback):
    model.fit(train_X, 
            train_y, 
            epochs=params['epochs'], 
            batch_size=params['batch_size'],
            callbacks=[neptune_cbk])
    
def save_trained_model(run: neptune.Run, 
                       model_version: neptune.ModelVersion,
                       model: keras.Sequential):
    # Save the model path in Neptune run
    run['model_path'] = 'my_model'
    model_path = run['model_path']
    model.save(model_path)
    # Save the model itself in Neptune model
    model_version["saved_model"].upload("my_model/saved_model.pb")
    for name in glob.glob("variables/*"):
        model_version[name].upload(name)
    print(f"Saved model to: {model_path}")