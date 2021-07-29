#!/usr/bin/env python
# coding: utf-8

def Build_model():
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    
    lag = 1
    drop_out = 0.2
    lstm_units = 24
    
    ## LSTM Architecture
    # input layer
    model = Sequential()
    # 1
    model.add(LSTM(units=lstm_units,return_sequences = False,
                   input_shape=(1,16)))
    model.add(Dropout(drop_out))
    # output layer
    model.add(Dense(1))
    # model architecture
    model.summary()
    return model

