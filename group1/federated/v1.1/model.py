from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_federated as tff
import tensorflow as tf


def build_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(5,4)))
    model.add(Dense(1))
    return model

def create_tff_model(train_datasets):
      return tff.learning.from_keras_model(build_model(), 
                                       input_spec=train_datasets[0].element_spec,
                                       loss=tf.keras.losses.MeanAbsoluteError(),
                                       metrics=[tf.keras.metrics.MeanAbsoluteError()])
                                       

def create_iterative_process(train_datasets):
    iterative_process = tff.learning.build_federated_averaging_process(model_fn=create_tff_model(train_datasets),
                                                                   client_optimizer_fn = lambda: tf.keras.optimizers.SGD(0.002))
    return iterative_process