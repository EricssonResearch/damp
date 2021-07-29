from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import numpy as np
import re
import math
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_federated as tff
import collections
import tensorflow as tf
from sklearn.model_selection import train_test_split

import nest_asyncio
nest_asyncio.apply()

def get_data_federated(dataPath):
    
    listOfDataFrames=[]
    stations=os.listdir(dataPath)
    for station in stations:
        print("Processing year: "+ dataPath +" station: "+station)
        airQualityData=pd.read_csv(dataPath+'/'+station, header=12,sep=';').rename(columns={'Start':'Start','Slut':'Stop'})
        airQualityData.rename(columns = lambda x: re.sub('NOX.*','NOX',x), inplace = True)
        airQualityData.rename(columns = lambda x: re.sub('PM10.*','PM10',x), inplace = True)
        airQualityData.rename(columns = lambda x: re.sub('PM2.5.*','PM2_5',x), inplace = True)
        airQualityData.rename(columns = lambda x: re.sub('NO2.*','NO2',x), inplace = True)
        listOfDataFrames.append(airQualityData)
    return listOfDataFrames

def preprocess_federated(listOfDataFrames):
    list_of_training=[]
    list_of_testing=[]
    for index, airData in enumerate(listOfDataFrames):
        df= airData
        #df['origin']="station{}".format(index)

        df.loc[(df['PM10'] <= 0, 'PM10')]=np.nan
        df.loc[(df['NO2'] <= 0, 'NO2')]=np.nan
        df.loc[(df['PM2_5'] <= 0, 'PM2_5')]=np.nan
        df.loc[(df['NOX'] <= 0, 'NOX')]=np.nan
        df=df.fillna(0)


        sc = MinMaxScaler(feature_range = (0, 1))
        scaled_down=df.copy()
        scaled_down['PM10']=sc.fit_transform(scaled_down['PM10'].values.reshape(-1, 1))
        scaled_down['NO2']=sc.fit_transform(scaled_down['NO2'].values.reshape(-1, 1))
        scaled_down['PM2_5']=sc.fit_transform(scaled_down['PM2_5'].values.reshape(-1, 1))
        scaled_down['NOX']=sc.fit_transform(scaled_down['NOX'].values.reshape(-1, 1))

        train=scaled_down[(scaled_down['Start']<= "2018-12-31 23:00:00")]
        test=scaled_down[(scaled_down['Start'] >= "2019-01-01 00:00:00")]
        train=train.drop('Start',axis = 1)
        test=test.drop('Start',axis = 1)
        test= test.reset_index().drop('index',axis=1)

        list_of_training.append(train)
        list_of_testing.append(test)
    return list_of_training,list_of_testing
    
def arrange_y_x_federated(list_of_data):
    x_list=[]
    y_list=[]
    for dataset in list_of_data:
        data_set=dataset[['PM10','NO2','PM2_5','NOX']]
        X = []
        y = []
        for i in range(5, len(data_set)):
            X.append(data_set.iloc[i-5:i].values)
            y.append(data_set['PM10'].iloc[i])

        X, y = np.array(X), np.array(y)
#         X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
#         X_train=X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        print(X.shape)
        x_list.append(X)
        y_list.append(y)
    return x_list,y_list

NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):

  def batch_format_fn(element):
      return collections.OrderedDict(x=element['x'], y=element['y'])

  return dataset.batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

# produce datasets for each origin
def make_federated_data(X_train, y_train, X_test, y_test):
        
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'x': X_train, 'y': y_train}))
        
        test_dataset = tf.data.Dataset.from_tensor_slices(
            ({'x': X_test, 'y': y_test}))

        preprocessed_train_dataset = preprocess(train_dataset)
        preprocessed_test_dataset = preprocess(test_dataset)
        
        
        
        return preprocessed_train_dataset, preprocessed_test_dataset 

def prepapre_all_federated_data(x_train_list,y_train_list,x_test_list,y_test_list):
    train_datasets = []
    test_datasets = []
    for X_train,y_train, X_test, y_test in zip(x_train_list,y_train_list,x_test_list,y_test_list):
        preprocessed_train_dataset,preprocessed_test_dataset=make_federated_data(X_train, y_train, X_test, y_test)
        train_datasets.append(preprocessed_train_dataset)
        test_datasets.append(preprocessed_test_dataset)    
    
    return train_datasets, test_datasets
