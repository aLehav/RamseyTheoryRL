from tensorflow import keras
from keras import layers

def create_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(14,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model