import tensorflow as tf
import numpy as np


def train_model(x, y, size, epochs, name):
    mse = 'mean_squared_error'

    normalization_in = tf.keras.layers.Normalization()
    normalization_in.adapt(x)

    model = tf.keras.models.Sequential()
    model.add(normalization_in)
    model.add(tf.keras.layers.Input(3))
    model.add(tf.keras.layers.Dense(
        size, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.relu))
    model.compile(optimizer='adam', loss=mse, metrics=[mse])
    model.fit(x, y, epochs=epochs, verbose=0)
    model.save(f"./trained_models/{name}.h5")

    return model

def load_model(name):
    model = tf.keras.models.load_model(f"./trained_models/{name}.h5")
    return model

def check_results(x, y, model):
    y_pred = model.predict(x)
    ERA = np.abs(1 - y_pred / y)
    MAPE = np.average(ERA) * 100
    MSE = np.square(np.subtract(y, y_pred)).mean()
    return (MAPE, MSE, y_pred)
