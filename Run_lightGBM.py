#Import all the modules
import lightgbm as lgb
import pandas as pd
import numpy as np
import merging
import preprocess
import scores
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import model_selection

#Train and Test Splitting function
def create_train_test(df):
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_ ]+', '', x))
    train=df[df['aobt_year']!=2019]
    test=df[df['aobt_year']==2019]
    X_train=train.drop('TO',axis=1)
    X_test=test.drop('TO',axis=1)
    y_train=train['TO']
    y_test=test['TO']
    
    return X_train,y_train,X_test,y_test

#Load the preprocessed file and drop some columns before inputting to the model
df = pd.read_csv('df_preprocessed_2015-2019.csv')
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df.drop(['Unnamed0','AircraftModel_x', 'Stand', 'Runway', 'summary', 'Manufacturer',
       'Model', 'WakeCategory', 'Final', 'AircraftModel_y', 'OldMovementType'],axis=1,inplace=True)
df = preprocess.ohe(df,['aobt_month','aobt_hour','aobt_day'])

#Choosing the features to be run
features_var = ['flight_hour','flight_weekday','cloudCover','windSpeed','windBearing','traffic', 'Q',
                'TO1', 'TO2', 'TO3',  'TORunway1',
                'TORunway2', 'TORunway3',
                'aobt_year', 'aobt_month_1', 'aobt_month_2',
                'aobt_month_3', 'aobt_month_4', 'aobt_month_5', 'aobt_month_6',
                'aobt_month_7', 'aobt_month_8', 'aobt_month_9', 'aobt_month_10',
                'aobt_month_11', 'aobt_month_12', 'aobt_hour_0', 'aobt_hour_1',
                'aobt_hour_2', 'aobt_hour_3', 'aobt_hour_4', 'aobt_hour_5',
                'aobt_hour_6', 'aobt_hour_7', 'aobt_hour_8', 'aobt_hour_9',
                'aobt_hour_10', 'aobt_hour_11', 'aobt_hour_12', 'aobt_hour_13',
                'aobt_hour_14', 'aobt_hour_15', 'aobt_hour_16', 'aobt_hour_17',
                'aobt_hour_18', 'aobt_hour_19', 'aobt_hour_20', 'aobt_hour_21',
                'aobt_hour_22', 'aobt_hour_23', 'aobt_day_0', 'aobt_day_1',
                'aobt_day_2', 'aobt_day_3', 'aobt_day_4', 'aobt_day_5', 'aobt_day_6',
                'precipAccumulation',   'Lengthft','TO']
df_lightGBM_features = df[features_var]

df_lightGBM = df_lightGBM_features.rename(columns={'TO1': 'Taxi_Out_lag1','TO2':'Taxi_Out_lag2','TO3': 'Taxi_Out_lag3',
                            'TORunway1': 'Taxi_Out_lag1_by_runway','TORunway2':'Taxi_Out_lag2_by_runway',
                           'TORunway3':'Taxi_Out_lag3_by_runway','Q':'Queue'})

#Set the parameters obtained from hyperparameter tuning
params = {'num_leaves': 67, 'lambda_l1': 1.0034340430533997, 'lambda_l2': 6.273266822048027, 'bagging_fraction': 0.9441048263180452, 'feature_fraction': 0.7741800625243584,'n_jobs':1}

#Create the train/Test split and 
X_train,y_train,X_test,y_test = create_train_test(df_lightGBM)
train_set = lgb.Dataset(X_train, label=y_train)
#Train the model
print('Training begins')
model = lgb.train(params,train_set)

#Test the model
preds = model.predict(X_test)

#print the results
scores=scores.get_scores(y_test,preds,metrics=['rmse','mae','first_quartile_error','third_quartile_error'])
print("Scores on test data\n")
for k,v in scores.items():
    print("{}: {:0.2f}".format(k,v))



