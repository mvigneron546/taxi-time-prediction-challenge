import pandas as pd
import numpy as np
import datetime as dt
from scipy.sparse.construct import random

from sklearn.preprocessing import StandardScaler
from collections import deque
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer

def calc_TO_time(df):
    cols = ['Flight Datetime', 'AOBT', 'ATOT']
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    #df[['Flight Datetime', 'AOBT', 'ATOT']] = df[['Flight Datetime', 'AOBT', 'ATOT']].apply(pd.to_datetime)
    df =df[df['ATOT']>=df['AOBT']]
    df['TO'] = (df['ATOT'] - df['AOBT']).dt.seconds
    return df

def ohe(df,columns):
    for column in columns:
        df=df.join(pd.get_dummies(df[column], prefix=column)).drop(column,axis=1)
    return df

def get_date_attributes(df_airport):
    df_airport['flight_weekday']=df_airport['Flight Datetime'].dt.weekday
    df_airport['flight_hour']=df_airport['Flight Datetime'].dt.hour
    df_airport['flight_minute']=df_airport['Flight Datetime'].dt.minute

    df_airport['aobt_year']=df_airport['AOBT'].dt.year
    df_airport['aobt_month']=df_airport['AOBT'].dt.month
    df_airport['aobt_day']=df_airport['AOBT'].dt.weekday
    df_airport['aobt_hour']=df_airport['AOBT'].dt.hour
    df_airport['aobt_minute']=df_airport['AOBT'].dt.minute
    # df_airport.drop(['Flight Datetime', 'AOBT', 'ATOT'],axis=1,inplace=True)
    
    return df_airport

def log_column(df, column):
    df['log(' + column + ')'] = np.log10(df[column], where = (df[column] != 0))
    return df

def scale_data(df, column_list):
    """Takes in a dataframe and a list of column names to transform
     returns a dataframe of scaled values"""
    df_to_scale = df[column_list]
    x = df_to_scale.values
    standard_scaler = StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    df_to_scale = pd.DataFrame(x_scaled, columns=df_to_scale.columns)
    return df_to_scale

def get_previous_taxi_times(df,n):
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

def reject_outliers(data, TO_bound):
    #l = np.percentile(data["TO"], [25, 75])[0]
    #u = np.percentile(data["TO"], [25, 75])[1]
    data_filtered = data[data["TO"]<= TO_bound]
    return data_filtered

def encode(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

def get_previous_taxi_times_by_runway(df,n):
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
        # print(appended_data[h].head())
        h +=1
    df_final = pd.concat(appended_data)
    return df_final

def get_ma(df,n):
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

def get_runway_traffic(df):
    candidates = []
    l = []
    df=df.sort_values(by='AOBT').reset_index(drop=True)
    for a, b in df[['AOBT','ATOT']].itertuples(index=False):
        count = 0
        decrement = 0
        candidates.append(b)
        len_candidates = len(candidates)
        for i in range(0,len_candidates):
            if a <= candidates[i-decrement]:
                count += 1
            else:
                del candidates[i-decrement]
                decrement +=1
        l.append(count)
    df['runway_traffic'] = l
    df.sort_values('AOBT')
    df=df.sort_values(by='Flight Datetime').reset_index(drop=True)
    return df

def treat_NA(df):
    imputer = SimpleImputer()
    df = imputer.fit_transform(df)
    return df

