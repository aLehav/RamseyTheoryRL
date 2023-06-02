from tensorflow import keras
from keras import layers

def create_model(params):
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(14,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation=params['last_activation'])
    ])
    model.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=['accuracy'])
    return model