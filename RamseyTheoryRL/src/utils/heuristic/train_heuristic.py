import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
import glob
import numpy as np

def split_X_y(csv_path):
    data = pd.read_csv(csv_path, index_col=0)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int)
    return X, y

def split_X_y_list(csv_paths):
    X_list = []
    y_list = []

    for csv_path in csv_paths:
        X, y = split_X_y(csv_path)
        X_list.append(X)
        y_list.append(y)

    # X_concatenated = pd.concat(X_list, ignore_index=True)
    # y_concatenated = pd.concat(y_list, ignore_index=True)
    X_concatenated = np.concatenate(X_list)
    y_concatenated = np.concatenate(y_list)
    return X_concatenated, y_concatenated

def split_test_train(csv_path):
    X, y = split_X_y(csv_path=csv_path)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    return train_X, test_X, train_y, test_y

def train(model: keras.Sequential, 
          train_X, 
          train_y, 
          params, 
          neptune_cbk: NeptuneCallback):
    def compute_class_weights(train_y):
        class_counts = np.bincount(train_y)
        total_samples = np.sum(class_counts)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = class_weights / np.sum(class_weights)
        class_weights = {i: weight for i, weight in enumerate(class_weights)}
        return class_weights
    class_weights = compute_class_weights(train_y)

    model.fit(train_X, 
            train_y, 
            epochs=params['training_epochs'], 
            batch_size=params['batch_size'],
            callbacks=[neptune_cbk],
            class_weight=class_weights,
            verbose=1)
    
def save_trained_model(model_version: neptune.ModelVersion,
                       model: keras.Sequential):
    # Save the model itself in Neptune model
    model.save('my_model')
    model_version["saved_model"].upload_files("my_model")
    for name in glob.glob("variables/*"):
        model_version[name].upload(name)

    