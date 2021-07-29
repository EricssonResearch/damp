#!/usr/bin/env python
# coding: utf-8
import collections

def Read_feature_selection(column_names,replace_NA,replace_negative):
    import os 
    from glob import glob
    import pandas as pd
    curdir = os.getcwd()
    
    # read csv files
    df = pd.read_csv(curdir+str('/')+str(glob('*.csv')[0]))

    # selecting station #Stockholm Torkel Knutssongatan
    df = df.loc[(df['Station Name'] == "#Stockholm Sveavägen 59 Gata") |
                (df['Station Name'] == "#Stockholm Hornsgatan 108 Gata") |
                (df['Station Name'] == "#Stockholm Torkel Knutssongatan") | 
                (df['Station Name'] == "#Stockholm E4/E20 Lilla Essingen")]

    # convert columns to date
    df['DateTimestamp'] = pd.to_datetime(df['DateTimestamp'])

    # Generating Year
    df['Year'] = df['DateTimestamp'].dt.year

    # drop 2020 data
    df = df[df['Year'] !=  2020]

    # selectin years of data
    df = df.loc[df['Year'].isin([2016,2017,2018,2019])]

    # select columns
    df = df[column_names]
    
    if replace_NA == True:
        # replace na with zero
        df = df.fillna(0)
    
    if replace_negative == True:
        # select float columns apart from Air tempterature
        columns_to_remove_negative = df.select_dtypes(include=['float64']).columns[df.select_dtypes(include=['float64']).columns != 'Air temperature']
        # replace negative with zero
        temp = df[columns_to_remove_negative].copy()
        temp[temp<0] = 0
        df[columns_to_remove_negative] = temp
    
    return(df)

def Add_features(df):
    import pandas as pd
    # Generating weektype from day of week
    df_2 = df.copy()
    df_2['Weektype'] = pd.DatetimeIndex(df_2['DateTimestamp']).dayofweek
    # Replace weedays labels WeekDays and WeekEnds 
    df_2['Weektype'] = df_2['Weektype'].replace([0, 1, 2, 3, 4], 'WeekDay')
    df_2['Weektype'] = df_2['Weektype'].replace([5, 6], 'WeekEnd')

    # Generating Hour class 'Transition', 'Night', 'Traffic', 'Peak'
    df_3 = df_2.copy()
    df_3 = df_3.assign(Hour=pd.cut(df_3['DateTimestamp'].dt.hour,[0,1,5,6,9,16,20,22,24],labels=['Transition','Night','Transition','Traffic','Peak','Traffic','Peak','Transition'],ordered=False))
    # Replace timestamp 00:00 with Transition
    df_3['Hour'] = df_3['Hour'].fillna('Transition')
    df_3['Hour'] = df_3.Hour.astype(str)

    ## onehot encoding catergories column
    # Get dummies
    df_4 = pd.get_dummies(df_3)
    
    # convert station name to categorical
    codes, uniques = pd.factorize(df_4['National Station Code'])
    df_4['Station Group'] = codes
    
    # drop station code column
    df_4 = df_4.drop(columns = ['National Station Code'])
    
    return(df_4)

def BatchPreprocess(dataset,NUM_EPOCHS,BATCH_SIZE,SHUFFLE_BUFFER,PREFETCH_BUFFER):
    def Batch_format_fn(element):
        return collections.OrderedDict(x=element['x'], y=element['y'])
    return dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE).map(
        Batch_format_fn).prefetch(PREFETCH_BUFFER)





