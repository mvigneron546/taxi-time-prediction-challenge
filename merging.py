import datetime
import pandas as pd
import numpy as np
from geopy.distance import great_circle
from preprocess import get_ma

def merge_traffic(df):
    '''Calculates the amount of airplanes taxiing at AOBT'''

    df=df.sort_values(by='AOBT').reset_index(drop=True)
    candidates = []
    l = []
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
            i +=1
        l.append(count)
    df['traffic'] = l
    df=df.sort_values(by='Flight Datetime').reset_index(drop=True)
    return df

def get_estimated_ATOT(df):
    '''Computes an estimation of ATOT based on the rolling mean 
    of TO of the last 10 planes taking off '''

    df = get_ma(df, 10)
    df['MA_Estimated_ATOT'] = df['AOBT'] + df['MA_10'].apply(lambda x: datetime.timedelta(seconds = x))

    return df

def get_Q(df):
    '''Calculates the amount of planes taking off in the 
    interval [AOBT, MA_Estimated_ATOT].
    
    First estimate the ATOT_x of the considered airplane x. Then estimate
    the ATOT_y of the planes y taxiing at time AOBT_x.
    Finally, count the number of plane y for which the estimated 
    ATOT_y <= ATOT_x
    '''

    df=df.sort_values(by='AOBT').reset_index(drop=True)
    df = get_estimated_ATOT(df)
    candidates = []
    l = []
    for a, b in df[['AOBT', 'MA_Estimated_ATOT']].itertuples(index=False):
        count = 0
        decrement = 0
        candidates.append(b)
        len_candidates = len(candidates)
        for i in range(0,len_candidates-1):
            if (a <= candidates[i-decrement]):
                count += 1
            elif a > candidates[i-decrement]:
                del candidates[i-decrement]
                decrement +=1
            i +=1
        l.append(count)
    df['Q'] = l
    df=df.sort_values(by='Flight Datetime').reset_index(drop=True)#.drop(['MA_Estimated_ATOT', 'SMA'], axis = 1)
    return df

def distance_calc(row):
    '''Calculates the length of the straight line binding 
    two points given by their coordinates'''
    start = (row['Lat_runway'], row['Lng_runway'])
    stop = (row['Lat_stand'], row['Lng_stand'])

    return great_circle(start, stop).meters


def calculate_distance(filename):
    '''Creates a csv file containing the distance for every
    combination (stand, runway)
    '''
    df_geographic = pd.read_csv('Taxi time - eleven Data Challenge/0. Airport data/geographic_data.csv')
    df_geographic['new_runway'] = 'RUNWAY_'+(df_geographic['runway'].str[-1].astype(int)+1).astype(str)
    df_geographic['stand'] = df_geographic['stand'].str.upper()

    # Remove duplicates rows
    df_geographic.drop_duplicates(inplace = True)

    df_geographic_runway =  df_geographic[['new_runway','Lat_runway','Lng_runway']].drop_duplicates()
    df_geographic_stand =  df_geographic[['stand','Lat_stand','Lng_stand']].drop_duplicates()

    df_geographic_runway['key'] = 1
    df_geographic_stand['key'] = 1

    df_geographic_2 = pd.merge(df_geographic_runway, df_geographic_stand,on='key')

    df_geographic_2['distance'] = df_geographic_2.apply(lambda row: distance_calc (row),axis=1)
    df_geographic_2.to_csv(filename,index=False)

def merge_distance(df):
    '''Gets the distance variable for the complete DataFrame'''
    df_geographic=pd.read_csv('Taxi time - eleven Data Challenge/0. Airport data/geographic_data_updated.csv')
    df_merged = pd.merge(df, df_geographic[['new_runway','stand','distance']],how='left',left_on=['Runway','Stand'],right_on=['new_runway','stand'])
    df_merged.drop(columns=['new_runway','stand'],inplace=True)
    return df_merged

def get_weather_data(df):
    '''Gets the weather variables for the complete DataFrame. We resampled
    the weather DataFrame to contain one row per minute and fill the rows by 
    linear interpolation.
    '''
    df_weather_train = pd.read_csv('Taxi time - eleven Data Challenge/2. Weather data/weather_data_train_set.csv')
    df_weather_test = pd.read_csv('Taxi time - eleven Data Challenge/2. Weather data/test_set_weather_data.csv')
    df_weather=pd.concat([df_weather_train,df_weather_test]).reset_index(drop=True)
    #prepare and clean weather data
    df_weather_unique = df_weather.drop_duplicates()
    df_weather_unique.reset_index(inplace=True)
    df_weather_unique.drop(columns=['index'],inplace=True)
    df_weather_unique.set_index('time_hourly',inplace=True)
    df_weather_unique.index = pd.to_datetime(df_weather_unique.index)  
    df_weather_unique = df_weather_unique.resample('1T').asfreq()  
    df_weather_unique.drop(columns=['icon','precipType'],inplace=True)
    df_weather_unique.interpolate(method='linear',inplace=True)
    df_weather_unique.fillna(method='ffill',inplace=True)
    #convert Flight Date to datetime to make match possible with airport dataset
    df['Flight Datetime'] = pd.to_datetime(df['Flight Datetime'])
    df_merged = pd.merge(df, df_weather_unique, how='left', left_on = 'Flight Datetime', right_on = 'time_hourly')
    return df_merged

def merge_tech(df):
    '''Merges the DataFrame with the technical characteristics of each plane
    df_technical is a manually labelled and cleaned version of ACchar
    '''
    df_technical = pd.read_csv('Taxi time - eleven Data Challenge/1. AC characteristics/df_technical_merged.csv')

    df_merged = pd.merge(df, df_technical, how = 'left', left_on = 'Aircraft Model', right_on = 'Final')

    return df_merged

