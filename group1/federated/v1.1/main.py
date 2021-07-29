from preprocess import preprocess_federated
from preprocess import get_data_federated
from preprocess import arrange_y_x_federated
from preprocess import prepapre_all_federated_data
from model import create_iterative_process
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_federated as tff
import tensorflow as tf
from model import create_tff_model
from model import build_model
import time

dataPath = 'Data/2016-2019(NO2, NOX, PM2.5)'

print("Getting training and testing data")
list_of_training,list_of_testing= preprocess_federated(get_data_federated(dataPath))

print("arranging federated data")
x_train_list,y_train_list=arrange_y_x_federated(list_of_training)
x_test_list,y_test_list=arrange_y_x_federated(list_of_testing)
train_datasets, test_datasets = prepapre_all_federated_data(x_train_list,y_train_list,x_test_list,y_test_list)

print("Creating iterative process")
def create_tff_model():
      return tff.learning.from_keras_model(build_model(), 
                                       input_spec=train_datasets[0].element_spec,
                                       loss=tf.keras.losses.MeanAbsoluteError(),
                                       metrics=[tf.keras.metrics.MeanAbsoluteError()])
                                       

iterative_process = tff.learning.build_federated_averaging_process(model_fn=create_tff_model,
                                                                   client_optimizer_fn = lambda: tf.keras.optimizers.SGD(0.002))


print("Initzialize averaging process")
state = iterative_process.initialize()
begin = time.time() 

print("Start iterations")
for _ in range(10):
  state, metrics = iterative_process.next(state, train_datasets)
  print('metrics={}'.format(metrics))
  
end = time.time()
print(f"Time elapsed for training: {end-begin} Seconds")

print("Global model evaluated over all clients")
evaluation = tff.learning.build_federated_evaluation(model_fn=create_tff_model)
test_metrics = evaluation(state.model, test_datasets)
print(test_metrics)
