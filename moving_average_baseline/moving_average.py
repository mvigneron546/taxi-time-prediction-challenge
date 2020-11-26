import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

def simple_moving_average(df,window_size=10,fill_na=True):
    '''Calculates simple moving average for a given window size and adds the predictions as a new column'''
    df['SMA']=df.TO.rolling(window_size).mean().shift()
    if fill_na:
        df['SMA']=df['SMA'].fillna(method='bfill')
    return(df)

def exponential_moving_average(df,alpha=0.1,fill_na=True):
    '''Calculates the exponential moving average for a given alpha and adds the predictions as a new column'''
    df['EMA']=df.TO.ewm(alpha=alpha).mean().shift()
    if fill_na:
        df['EMA']=df['EMA'].fillna(method='bfill')
    return(df)

def get_best_window_size(df,window_sizes,metric='rmse'):
    '''Calculates the best window_size for a simple moving average by runway'''
    scores=[]
    for ws in window_sizes:
        df_sma=df.groupby('Runway').apply(lambda x: simple_moving_average(x,ws))
        y_true=df_sma['TO']
        y_pred=df_sma['SMA']
        scores_dict=get_scores(y_true,y_pred,[metric])
        scores.append(scores_dict[metric])
    
    minimize=['rmse','mae','first_quartile_error','third_quartile_error']
    if metric in minimize:
        return(scores.index(min(scores))+1)
    else:
        return(scores.index(max(scores))+1)


def get_scores(y_true,y_pred,metrics=['rmse']):
    '''Calculates the scores for the metrics in the input and returns a dictionary of the scores'''
    scores_dict={}
    errors=abs(y_true-y_pred)
    for metric in metrics:
        if metric=='rmse':
            scores_dict[metric]=np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric=='mae':
            scores_dict[metric]=errors.mean()
        elif metric=='r2':
            scores_dict[metric]=r2_score(y_true, y_pred)
        elif metric=='first_quartile_error':
            scores_dict[metric]=np.percentile(errors,[25,75])[0]
        elif metric=='third_quartile_error':
            scores_dict[metric]=np.percentile(errors,[25,75])[1]
        else:
            scores_dict[metric]:'Invalid metric'
    return scores_dict
    