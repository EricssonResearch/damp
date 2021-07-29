#!/usr/bin/env python
# coding: utf-8

# required libraries
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_federated as tff
import collections
# import functions from other .py files
from model import Build_model
from Preprocessing import Read_feature_selection, Add_features, BatchPreprocess


# create federated model
def Create_tff_model():
    return tff.learning.from_keras_model(Build_model(),
                                         input_spec=train_datasets[0].element_spec,
                                         loss=tf.keras.losses.MeanSquaredError(),
                                         metrics=[tf.keras.metrics.MeanAbsoluteError()])


# function parameters
column_names = ['DateTimestamp','National Station Code',
                'Classification','NO2','NOX as NO2','PM2.5','PM10','CO',
                'Black Carbon','O3','Air temperature']
replace_NA = True
replace_negative = True

# read data and select features
df = Read_feature_selection(column_names,replace_NA,replace_negative) # file_path,columns_selected,replace_NA,replace_negative
df.info()


# add features to processed data
df1 = Add_features(df)
df1.info()


# resample the data
to_be_normalized_columns = df1.select_dtypes(float).columns
target_column = 'Predict PM10'
scaler = MinMaxScaler(feature_range = (0, 1))
df1[to_be_normalized_columns] = scaler.fit_transform(df1[to_be_normalized_columns])
df1.describe()

# create train test partition data for tff nodel training
temp = df1.set_index(df1['DateTimestamp'])
temp = temp[temp['Station Group'] == 0]
temp = temp.sort_index()
train = temp['2016-01-01 00:00:00':'2018-12-31 23:59:59']
test  = temp['2019-01-01 00:00:00':'2019-12-31 23:59:59']
print('Train Dataset:',train.shape)
print('Test Dataset:',test.shape)


NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

shape = [train.shape,test.shape]

dfs = [x for _, x in df1.groupby('Station Group')]
# empty place holder
train_datasets = []
test_datasets = []

# timestep variable
lag = 1

i = 0
for dataframe in dfs:
    i=i+1
    
    # generate y(predictor/label)
    dataframe['Predict PM10'] = dataframe['PM10'].shift(periods=-1)

    # drop last row after shift
    dataframe = dataframe[:-1]
    
    # remove unwanted columns
    target = dataframe.pop('Predict PM10')
    dataframe.pop('Station Group')
    dataframe.pop('DateTimestamp')
    
    # generate and reshape (train,test) 3D [samples, timesteps, features] 
    X_train = dataframe[0:shape[0][0]].values.reshape(-1,lag,shape[0][1]-2)
    y_train = target[0:shape[0][0]].values.reshape(-1,lag,1)
    
    X_test = dataframe[shape[0][0]:shape[0][0]+shape[1][0]].values.reshape(-1,lag,shape[1][1]-2)
    y_test = target[shape[0][0]:shape[0][0]+shape[1][0]].values.reshape(-1,lag,1)
    
    print("Dataframe Number:",i)
    print("X_train_shape: ",X_train.shape)
    print("y_train_shape: ",y_train.shape)
    print("X_test_shape: ",X_test.shape)
    print("y_test_shape: ",y_test.shape)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(({'x': X_train, 'y': y_train}))
        
    test_dataset = tf.data.Dataset.from_tensor_slices(({'x': X_test, 'y': y_test}))
    
    preprocessed_train_dataset = BatchPreprocess(train_dataset,NUM_EPOCHS,BATCH_SIZE,SHUFFLE_BUFFER,PREFETCH_BUFFER)
    preprocessed_test_dataset = BatchPreprocess(test_dataset,NUM_EPOCHS,BATCH_SIZE,SHUFFLE_BUFFER,PREFETCH_BUFFER)

    train_datasets.append(preprocessed_train_dataset)
    test_datasets.append(preprocessed_test_dataset)
    
    print()
       
# create the tff models
print("Create TFF Models")
iterative_process = tff.learning.build_federated_averaging_process(model_fn=Create_tff_model,
                                                                   client_optimizer_fn = lambda: tf.keras.optimizers.Adam())

import nest_asyncio
nest_asyncio.apply()

print("Initzialize averaging process")
state = iterative_process.initialize()

print("Start iterations")
for _ in range(10):
    state, metrics = iterative_process.next(state, train_datasets)
    print('metrics={}'.format(metrics))

# Global model evaluated over all clients
evaluation = tff.learning.build_federated_evaluation(model_fn=Create_tff_model)
test_metrics = evaluation(state.model, test_datasets)
print(test_metrics)

# Global model evaluated per individual client
for i in range(len(test_datasets)):
    test_metrics = evaluation(state.model, [test_datasets[i]])
    print(test_metrics)






