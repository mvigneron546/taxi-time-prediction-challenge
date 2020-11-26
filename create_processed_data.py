'''Script to create the dataset with all features to be used for modeling '''
import pandas as pd
import numpy as np
import merging
import preprocess

# Import and merge train test split
df_airport_train = pd.read_csv('Taxi time - eleven Data Challenge/0. Airport data/training_set_airport_data.csv')
df_airport_test = pd.read_csv('Taxi time - eleven Data Challenge/0. Airport data/test_set_airport_data.csv')
df_airport=pd.concat([df_airport_train, df_airport_test]).reset_index(drop=True)
print('Data Imported')

# Calculates the taxi out time
df_airport = preprocess.calc_TO_time(df_airport)
# Adds distance between stand and runway
df_airport = merging.merge_distance(df_airport)
# Calculates traffic feature
df_airport = merging.merge_traffic(df_airport)
# Calculates Q (queue size)
df_airport = merging.get_Q(df_airport)
print('Feature engineering in process')
# Merges weather features
df_airport = merging.get_weather_data(df_airport)
# Calculates the last known taxi times for the aircrafts
df_airport = preprocess.get_previous_taxi_times(df_airport,3)
print('Feature engineering still in process')
# Calculates the last known taxi times by runway
df_airport = preprocess.get_previous_taxi_times_by_runway(df_airport,3)
# Seperates the datetime columns into its attributes
df_airport = preprocess.get_date_attributes(df_airport)
# Converts datetime variables into cyclic features
df_airport = preprocess.encode(df_airport, 'aobt_hour', 23)
df_airport = preprocess.encode(df_airport, 'aobt_month', 12)
df_airport = preprocess.encode(df_airport, 'aobt_day', 365)
# Merge Aircraft characteristic features
df_airport = merging.merge_tech(df_airport)

# Outputs the full dataset with all features to be used for modelling
df_airport.to_csv('df_preprocessed_2015-2019.csv', index=False)
print('File created')