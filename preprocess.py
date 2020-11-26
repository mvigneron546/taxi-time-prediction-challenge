import pandas as pd
import numpy as np
import datetime as dt

from sklearn.preprocessing import StandardScaler
from collections import deque
from sklearn.impute import SimpleImputer

def calc_TO_time(df):
    '''Substracts AOBT from ATOT to get TO in seconds '''
    cols = ['Flight Datetime', 'AOBT', 'ATOT']
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    #df[['Flight Datetime', 'AOBT', 'ATOT']] = df[['Flight Datetime', 'AOBT', 'ATOT']].apply(pd.to_datetime)
    df =df[df['ATOT']>=df['AOBT']]
    df['TO'] = (df['ATOT'] - df['AOBT']).dt.seconds
    return df

def ohe(df,columns):
    '''one-hot-encode the listed columns in the DataFrame df'''
    for column in columns:
        df=df.join(pd.get_dummies(df[column], prefix=column)).drop(column,axis=1)
    return df

def get_date_attributes(df_airport):
    '''Extracts date attributes from the Timestamps Flight Datetime and AOBT'''
    df_airport['flight_weekday']=df_airport['Flight Datetime'].dt.weekday
    df_airport['flight_hour']=df_airport['Flight Datetime'].dt.hour
    df_airport['flight_minute']=df_airport['Flight Datetime'].dt.minute

    df_airport['aobt_year']=df_airport['AOBT'].dt.year
    df_airport['aobt_month']=df_airport['AOBT'].dt.month
    df_airport['aobt_day']=df_airport['AOBT'].dt.weekday
    df_airport['aobt_hour']=df_airport['AOBT'].dt.hour
    df_airport['aobt_minute']=df_airport['AOBT'].dt.minute
    
    return df_airport

def scale_data(df, column_list):
    '''Takes in a dataframe and a list of column names to transform
     returns a dataframe of scaled values'''
    df_to_scale = df[column_list]
    x = df_to_scale.values
    standard_scaler = StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    df_to_scale = pd.DataFrame(x_scaled, columns=df_to_scale.columns)
    return df_to_scale

def get_previous_taxi_times(df,n):
    '''Get the taxi out times of the last n airplanes 
    that have already taken off'''
    df=df.sort_values(by='AOBT').reset_index(drop=True)
    last_n_taxi=deque(n*[None], n)
    candidates = []
    candidates_taxi=[]
    last_n_taxi_list=[]
    for aobt, atot ,to in df[['AOBT','ATOT','TO']].itertuples(index=False):
        decrement = 0
        recent_takeoffs=[]
        candidates.append(atot)
        candidates_taxi.append(to)
        len_candidates = len(candidates)
        for i in range(0,len_candidates):
            if aobt <= candidates[i-decrement]:
                pass
            else:
                recent_takeoffs.append((candidates[i-decrement],candidates_taxi[i-decrement]))
                del candidates[i-decrement]
                del candidates_taxi[i-decrement]
                decrement +=1
        recent_takeoffs=sorted(recent_takeoffs,key=lambda x:x[0])
        for i in range(0,len(recent_takeoffs)):
            last_n_taxi.appendleft(recent_takeoffs[i][1])
        
        last_n_taxi_list.append(list(last_n_taxi))
    col_names=['TO -{}'.format(i+1) for i in range(n)]
    df[col_names]=pd.DataFrame(last_n_taxi_list,columns=col_names)
    df[col_names]=df[col_names].fillna(method='bfill')
    df=df.sort_values(by='Flight Datetime').reset_index(drop=True)
    return df

def encode(df, col, max_val):
    '''Get polar coordinates of date attributes given in arguments'''
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

def get_previous_taxi_times_by_runway(df,n):
    '''For each airplane, get the taxi out times of the last n airplanes
    that have already taken off from the same runway'''
    appended_data = []
    h=0
    for runway in df['Runway'].unique():
        df_2=df[df['Runway']==runway].sort_values(by='AOBT').reset_index(drop=True)
        last_n_taxi=deque(n*[None], n)
        candidates = []
        candidates_taxi=[]
        last_n_taxi_list=[]
        for aobt, atot ,to in df_2[['AOBT','ATOT','TO']].itertuples(index=False):
            decrement = 0
            recent_takeoffs=[]
            candidates.append(atot)
            candidates_taxi.append(to)
            len_candidates = len(candidates)
            for i in range(0,len_candidates):
                if aobt <= candidates[i-decrement]:
                    pass
                else:
                    recent_takeoffs.append((candidates[i-decrement],candidates_taxi[i-decrement]))
                    del candidates[i-decrement]
                    del candidates_taxi[i-decrement]
                    decrement +=1
            recent_takeoffs=sorted(recent_takeoffs,key=lambda x:x[0])
            for i in range(0,len(recent_takeoffs)):
                last_n_taxi.appendleft(recent_takeoffs[i][1])
            
            last_n_taxi_list.append(list(last_n_taxi))
        col_names=['TO Runway -{}'.format(i+1) for i in range(n)]
        df_2[col_names]=pd.DataFrame(last_n_taxi_list,columns=col_names)
        df_2[col_names]=df_2[col_names].fillna(method='bfill')
        df_2=df_2.sort_values(by='Flight Datetime').reset_index(drop=True)
        appended_data.append(df_2)
        h +=1
    df_final = pd.concat(appended_data)
    return df_final

def get_ma(df,n):
    '''Get the average TO of the last n airplanes 
    that have already taken off'''
    df=df.sort_values(by='AOBT').reset_index(drop=True)
    last_n_taxi=deque(n*[np.nan], n)
    candidates = []
    candidates_taxi=[]
    last_n_taxi_list=[]
    for aobt, atot ,to in df[['AOBT','ATOT','TO']].itertuples(index=False):
        decrement = 0
        recent_takeoffs=[]
        candidates.append(atot)
        candidates_taxi.append(to)
        len_candidates = len(candidates)
        for i in range(0,len_candidates):
            if aobt <= candidates[i-decrement]:
                pass
            else:
                recent_takeoffs.append((candidates[i-decrement],candidates_taxi[i-decrement]))
                del candidates[i-decrement]
                del candidates_taxi[i-decrement]
                decrement +=1
        recent_takeoffs=sorted(recent_takeoffs,key=lambda x:x[0])
        for i in range(0,len(recent_takeoffs)):
            last_n_taxi.appendleft(recent_takeoffs[i][1])
        
        last_n_taxi_list.append(np.nanmean(list(last_n_taxi)))
    df['MA_{}'.format(n)]=last_n_taxi_list
    df['MA_{}'.format(n)]=df['MA_{}'.format(n)].fillna(method='bfill')
    
    return df

def treat_NA(df):
    '''Treat the NaN of the distance column by replacing 
    them by a simple mean of the distance column '''
    imputer = SimpleImputer()
    df = imputer.fit_transform(df)
    return df

